import pygmsh
from pygmsh.common.polygon import Polygon
import gmsh
import numpy as np
import logging
from itertools import count
import meshio
from pygmsh.helpers import extract_to_meshio

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


def generate_mesh(vertices,
                  edges=[],
                  edge_loops=[],
                  faces=[],
                  volumes=[],
                  model_name=[],
                  lc=1,             # Node default mesh size
                  dim=3,
                  verbose=False):

    gmsh.initialize([])
    gmsh.model.add(f'{model_name}')

    geo = gmsh.model.geo

    vertex_lookup = dict(zip(vertices, range(vertices.__len__())))
    point_mesh_sizes = np.full(vertices.__len__(), lc)
    for face in faces:
        vertex_indices = np.array([vertex_lookup[x] for x in face.vertices])
        point_mesh_sizes[vertex_indices[point_mesh_sizes[vertex_indices] > face.mesh_size]] = face.mesh_size

    for i, vertex in enumerate(vertices):
        geo.addPoint(vertex.position[0], vertex.position[1], vertex.position[2], point_mesh_sizes[i], i+1)
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

    # Finally, while the default "Frontal-Delaunay" 2D meshing algorithm
    # (Mesh.Algorithm = 6) usually leads to the highest quality meshes, the
    # "Delaunay" algorithm (Mesh.Algorithm = 5) will handle complex mesh size fields
    # better - in particular size fields with large element size gradients:
    # gmsh.option.setNumber("Mesh.Algorithm", 5)

    if volumes.__len__() == 0:
        dim = 2

    gmsh.model.mesh.generate(dim)
    mesh = extract_to_meshio()

    # open the mesh in gmsh
    # gmsh.fltk.run()

    gmsh.finalize()

    return mesh
