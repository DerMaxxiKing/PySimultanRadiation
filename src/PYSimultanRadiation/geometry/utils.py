import pygmsh
from pygmsh.common.polygon import Polygon
import gmsh
import numpy as np
import logging
from pygmsh.helpers import extract_to_meshio
import trimesh

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger('PySimultanRadiation')


def mesh_planar_face(face, mesh_size=0.1):

    with pygmsh.geo.Geometry() as geom:

        holes = None
        if face.holes.__len__() > 0:
            holes = [Polygon(geom, x.points,
                             holes=None,
                             mesh_size=mesh_size,
                             ) for x in face.holes]

        geom.add_polygon(
            face.points,
            holes=holes,
            mesh_size=mesh_size,
        )

        if face.holes.__len__() > 0:
            [geom.remove(x) for x in holes]

        mesh = geom.generate_mesh(dim=2, verbose=False)

        # mesh.write('face_with_opening.vtk')
    return mesh


def generate_mesh(*args, **kwargs):

    vertices = kwargs.get('vertices', [])
    edges = kwargs.get('edges', [])
    edge_loops = kwargs.get('edge_loops', [])
    faces = kwargs.get('faces', [])
    volumes = kwargs.get('volumes', [])
    model_name = kwargs.get('model_name', None)
    lc = kwargs.get('lc', 1)    # Node default mesh size
    min_size = kwargs.get('min_size', 0.5)
    max_size = kwargs.get('max_size', 10)
    verbose = kwargs.get('verbose', False) # show gmsh-output
    mesh_size_from_faces = kwargs.get('mesh_size_from_faces', True)
    dim = kwargs.get('dim', 3)  # dimension of the mesh: 2:surface-mesh, 3:volume-mesh

    gmsh.initialize([])
    gmsh.model.add(f'{model_name}')

    try:

        geo = gmsh.model.geo

        vertex_lookup = dict(zip(vertices, range(vertices.__len__())))
        point_mesh_sizes = np.full(vertices.__len__(), lc)

        if mesh_size_from_faces:
            for face in faces:
                try:
                    vertex_indices = np.array([vertex_lookup[x] for x in face.vertices])
                    point_mesh_sizes[vertex_indices[point_mesh_sizes[vertex_indices] > face.mesh_size]] = face.mesh_size
                except Exception as e:
                    logger.error(f'Error adding mesh_size for face {face.name}, {face.id}: {e}')

        lc_p = lc
        for i, vertex in enumerate(vertices):
            if mesh_size_from_faces:
                lc_p = point_mesh_sizes[i]
            geo.addPoint(vertex.position[0], vertex.position[1], vertex.position[2], lc_p, i+1)

        gmsh.model.geo.synchronize()

        edge_lookup = dict(zip(edges, range(edges.__len__())))
        for i, edge in enumerate(edges):
            try:
                geo.addLine(vertex_lookup[edge.vertices[0]]+1, vertex_lookup[edge.vertices[1]]+1, i+1)
            except Exception as e:
                logger.error(f'Error adding edge {edge}:\n{e}')
        gmsh.model.geo.synchronize()

        edge_loop_lookup = dict(zip(edge_loops, range(edge_loops.__len__())))
        for i, edge_loop in enumerate(edge_loops):
            try:
                ids = np.array([edge_lookup[x]+1 for x in edge_loop.edges]) * np.array(edge_loop.edge_orientations)
                geo.addCurveLoop(ids, i+1)
            except Exception as e:
                logger.error(f'Error creating CurveLoop for {edge_loop.id}, {edge_loop.gmsh_id}:\n{e}')
        gmsh.model.geo.synchronize()

        face_lookup = dict(zip(faces, range(faces.__len__())))
        for i, face in enumerate(faces):
            try:
                edge_loop_ids = [edge_loop_lookup[face.boundary]+1] + [edge_loop_lookup[x]+1 for x in face.holes]
                geo.addPlaneSurface(edge_loop_ids, i+1)
            except Exception as e:
                logger.error(f'Error creating CurveLoop for {face.name},{face.id}, {face.gmsh_id}:\n{e}')
        gmsh.model.geo.synchronize()

        volume_lookup = dict(zip(volumes, range(volumes.__len__())))
        for i, volume in enumerate(volumes):
            geo.addSurfaceLoop([face_lookup[x]+1 for x in faces], i+1)
            geo.addVolume([i+1], i+1)

        gmsh.model.geo.synchronize()

        # add physical domains for faces:
        for face in faces:
            ps = gmsh.model.addPhysicalGroup(2, [face_lookup[face]+1])
            gmsh.model.setPhysicalName(2, ps, str(face.id))

        # add physical domains for volumes:
        for volume in volumes:
            ps = gmsh.model.addPhysicalGroup(3, [volume_lookup[volume]+1])
            gmsh.model.setPhysicalName(3, ps, str(volume.id))

        gmsh.model.geo.synchronize()

        gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", min_size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", max_size)

        # Finally, while the default "Frontal-Delaunay" 2D meshing algorithm
        # (Mesh.Algorithm = 6) usually leads to the highest quality meshes, the
        # "Delaunay" algorithm (Mesh.Algorithm = 5) will handle complex mesh size fields
        # better - in particular size fields with large element size gradients:
        # gmsh.option.setNumber("Mesh.Algorithm", 5)

        if volumes.__len__() == 0:
            dim = 2

        gmsh.model.mesh.generate(dim)
        mesh = extract_to_meshio()

    except Exception as e:
        logger.error(f'Error creating mesh for {model_name}')
        gmsh.finalize()
        raise e

    gmsh.finalize()

    return mesh


def generate_surface_mesh(*args, **kwargs):

    vertices = kwargs.get('vertices', [])
    edges = kwargs.get('edges', [])
    edge_loops = kwargs.get('edge_loops', [])
    faces = kwargs.get('faces', [])
    model_name = kwargs.get('model_name', None)
    lc = kwargs.get('lc', 1)
    min_size = kwargs.get('min_size', 0.5)
    max_size = kwargs.get('max_size', 10)
    verbose = kwargs.get('verbose', False)
    mesh_size_from_faces = kwargs.get('mesh_size_from_faces', True)
    method = kwargs.get('method', '')

    if method == 'robust':
        with pygmsh.geo.Geometry() as geom:
            model = geom.__enter__()

            polys = {}

            for face in faces:
                holes = []
                if face.holes.__len__() > 0:
                    holes = [geom.add_polygon(x.points,
                                              holes=None,
                                              mesh_size=face.mesh_size,
                                              ) for x in face.holes]

                poly = geom.add_polygon(
                    face.points,
                    holes=holes,
                    mesh_size=face.mesh_size,
                )
                polys[str(face.id)] = poly

                if face.holes.__len__() > 0:
                    [geom.remove(x) for x in holes]

            [model.add_physical(value, key) for key, value in polys.items()]

            mesh = geom.generate_mesh(dim=2, verbose=verbose)

    else:
        mesh = generate_mesh(vertices=vertices,
                             edges=edges,
                             edge_loops=edge_loops,
                             faces=faces,
                             model_name=model_name,
                             lc=lc,                      # Node default mesh size
                             dim=2,                     # dimension of the mesh: 2:surface-mesh, 3:volume-mesh
                             min_size=min_size,
                             max_size=max_size,
                             verbose=verbose,           # show gmsh-output
                             mesh_size_from_faces=mesh_size_from_faces)

    return mesh


def generate_terrain(hull_mesh, terrain_height):
    from .base_geometry import Vertex, Edge, EdgeLoop, Face, Terrain

    surf_mesh = trimesh.Trimesh(vertices=hull_mesh.points,
                                faces=hull_mesh.cells[1].data)

    surf_mesh.remove_unreferenced_vertices()
    surf_mesh.merge_vertices()
    surf_mesh.remove_duplicate_faces()
    trimesh.repair.fix_winding(surf_mesh)

    path = surf_mesh.section(np.array([0, 0, 1]), np.array([0, 0, terrain_height]))

    # create terrain face:

    x_ext = surf_mesh.bounds[1, 0] - surf_mesh.bounds[0, 0]
    y_ext = surf_mesh.bounds[1, 1] - surf_mesh.bounds[0, 1]

    # outer loop:
    p0 = Vertex(position=np.array([surf_mesh.bounds[0, 0] - x_ext, surf_mesh.bounds[0, 1] - y_ext, terrain_height]))
    p1 = Vertex(position=np.array([surf_mesh.bounds[1, 0] + x_ext, surf_mesh.bounds[0, 1] - y_ext, terrain_height]))
    p2 = Vertex(position=np.array([surf_mesh.bounds[1, 0] + x_ext, surf_mesh.bounds[1, 1] + y_ext, terrain_height]))
    p3 = Vertex(position=np.array([surf_mesh.bounds[0, 0] - x_ext, surf_mesh.bounds[1, 1] + y_ext, terrain_height]))

    e0 = Edge(vertices=[p0, p1])
    e1 = Edge(vertices=[p1, p2])
    e2 = Edge(vertices=[p2, p3])
    e3 = Edge(vertices=[p3, p0])

    el0 = EdgeLoop(edges=[e0, e1, e2, e3], edge_orientations=[1] * 4)

    # holes
    points = [None] * path.vertices.shape[0]
    for i, vertex in enumerate(path.vertices):
        points[i] = Vertex(position=np.array(vertex))

    holes = []
    all_edges = [e0, e1, e2, e3]
    edge_loops = [el0]

    for line in path.entities:
        # create edges:
        edges = [None] * (line.nodes.shape[0])
        for i, node in enumerate(line.nodes):
            edges[i] = Edge(vertices=[points[node[0]], points[node[1]]])
        all_edges.extend(edges)
        edge_loop = EdgeLoop(edges=edges, edge_orientations=[1] * edges.__len__())
        edge_loops.append(edge_loop)
        holes.append(edge_loop)

    terrain_face = Face(name='Terrain', boundary=el0, holes=holes)

    terrain = Terrain(vertices=[*points, p0, p1, p2, p3],
                      edges=all_edges,
                      edge_loops=edge_loops,
                      faces=[terrain_face])

    return terrain
