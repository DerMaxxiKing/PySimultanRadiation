from PySimultan.geo_default_types import geometry_types
from .utils import mesh_planar_face, generate_mesh
from trimesh import Trimesh
import logging
import numpy as np
import pygmsh
import meshio
import gmsh
from itertools import count
from random import randint

from pygmsh.helpers import extract_to_meshio

logger = logging.getLogger('PySimultanRadiation')


class ExtendedVertex(geometry_types.vertex):

    newid = count(start=1, step=1).__next__

    def __init__(self, *args, **kwargs):
        geometry_types.vertex.__init__(self, *args, **kwargs)
        self.gmsh_id = ExtendedVertex.newid()

    def add_to_gmsh_geo(self, geo, mesh_size):
        try:
            geo.addPoint(self.position[0], self.position[1], self.position[2], mesh_size, self.gmsh_id)
        except Exception as e:
            logger.error(f'Error creating point for {self.id}, {self.gmsh_id}:\n{e}')


class ExtendedEdge(geometry_types.edge):

    newid = count(start=1, step=1).__next__

    def __init__(self, *args, **kwargs):
        geometry_types.edge.__init__(self, *args, **kwargs)
        self.gmsh_id = ExtendedEdge.newid()

    def add_to_gmsh_geo(self, geo):
        try:
            geo.addLine(self.vertices[0].gmsh_id, self.vertices[1].gmsh_id, self.gmsh_id)
        except Exception as e:
            logger.error(f'Error creating edge for {self.id}, {self.gmsh_id}:\n{e}')


class ExtendedEdgeLoop(geometry_types.edge_loop):

    newid = count(start=1, step=1).__next__

    def __init__(self, *args, **kwargs):
        geometry_types.edge_loop.__init__(self, *args, **kwargs)
        self.gmsh_id = ExtendedEdgeLoop.newid()

    def add_to_gmsh_geo(self, geo):
        try:
            ids = np.array([x.gmsh_id for x in self.edges]) * np.array(self.edge_orientations)
            geo.addCurveLoop(ids, self.gmsh_id)
        except Exception as e:
            logger.error(f'Error creating CurveLoop for {self.id}, {self.gmsh_id}:\n{e}')


class ExtendedFace(geometry_types.face):

    newid = count(start=1, step=1).__next__

    def __init__(self, *args, **kwargs):
        geometry_types.face.__init__(self, *args, **kwargs)

        self.gmsh_id = ExtendedFace.newid()

        self._mesh = None
        self._trimesh = None
        self._mesh_size = None
        self._vertices = None

        self.mesh_size = kwargs.get('mesh_size', randint(1, 6))
        self.is_hull_face = kwargs.get('is_hull_face', True)

        self.side1 = kwargs.get('side1_volume', None)
        self.side2 = kwargs.get('side2_volume', None)

    @property
    def vertices(self):
        if self._vertices is None:
            self._vertices = self.get_vertices()
        return self._vertices

    @property
    def trimesh(self):
        if self._trimesh is None:
            self._trimesh = Trimesh(vertices=self.mesh.points, faces=self.mesh.cells[1].data)
        return self._trimesh

    @property
    def mesh_size(self):
        return self._mesh_size

    @mesh_size.setter
    def mesh_size(self, value):
        self._mesh_size = value
        self._mesh = None
        self._trimesh = None

    @property
    def mesh(self):
        if self._mesh is None:
            self.mesh = mesh_planar_face(self, mesh_size=self.mesh_size)
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = value

    def create_mesh(self):
        return mesh_planar_face(self, mesh_size=self.mesh_size)

    def export_vtk(self, file_name):
        self.mesh.write(file_name)

    def add_to_gmsh_geo(self, geo):
        try:
            edge_loop_ids = [self.boundary.gmsh_id] + [x.gmsh_id for x in self.holes]
            geo.addPlaneSurface(edge_loop_ids, self.gmsh_id)
        except Exception as e:
            logger.error(f'Error creating CurveLoop for {self.name},{self.id}, {self.gmsh_id}:\n{e}')

    def get_vertices(self):

        edge_loops = []
        edge_loops.append(self.boundary)
        edge_loops.extend(self.holes)
        edge_loops = set(edge_loops)

        edges = []
        [edges.extend(x.edges) for x in edge_loops]
        edges = set(edges)
        vertices = []
        [vertices.extend([x.vertices[0], x.vertices[1]]) for x in edges]
        vertices = set(vertices)

        return vertices


class ExtendedVolume(geometry_types.volume):

    newid = count(start=1, step=1).__next__

    def __init__(self, *args, **kwargs):
        geometry_types.volume.__init__(self, *args, **kwargs)
        self._mesh = None
        self.gmsh_id = ExtendedVolume.newid()

    @property
    def mesh(self):
        if self._mesh is None:
            self.mesh = self.create_mesh()
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = value

    def create_mesh(self, method='gmsh'):

        # examples found:
        # https://programtalk.com/vs2/python/9379/pygmsh/pygmsh/geometry.py/

        if (self.faces is None) or (self.faces.__len__() == 0):
            logger.error(f'{self.name}; {self.id}: Scene has no faces')
            return

        # old version; did not work because returned cell_sets were wrong!
        mesh = None

        if method == 'from_model':

            try:

                logger.debug(f'Creating mesh for {self.name}: {self.id} with method: {method}')
                with pygmsh.geo.Geometry() as geom:
                    model = geom.__enter__()

                    polys = {}

                    for face in self.faces:
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

                    # mesh = geom.generate_mesh(dim=2, verbose=True)
                    # mesh.write('volume_rest.vtk')
                    model.synchronize()

                    surf_loop = geom.add_surface_loop([x.curve_loop for x in polys.values()])

                    model.synchronize()
                    volume = geom.add_volume(surf_loop)
                    # model.add_physical(volume, str(self.id))

                    # model.synchronize()
                    #
                    # [geom.remove(x) for x in polys.values()]
                    # n = gmsh.model.getDimension()
                    # s = gmsh.model.getEntities(n)

                    geom.save_geometry('test.geo_unrolled')

                    mesh = geom.generate_mesh(dim=3, verbose=True)
                    mesh = self.add_mesh_properties(mesh)
            except Exception as e:
                logger.error(f'Error while creating mesh for {self.name}: {self.id}: \n {e}')

        elif method == 'from_faces':
            logger.warning("Mesh creation with method 'from_faces' not recommended. Use of 'from_model' recommended.")
            import trimesh

            # assemble mesh from the mesh of all faces:

            mesh = trimesh.Trimesh()
            cell_sets = {}
            cell_sets_dict = {}

            for face in self.faces:
                num_elem0 = mesh.faces.shape[0]
                num_elem1 = mesh.faces.shape[0] + face.trimesh.faces.shape[0]
                elem_ids = np.array(range(num_elem0, num_elem1))
                cell_sets[str(face.id)] = [elem_ids]
                cell_sets_dict[str(face.id)] = {'triangle': elem_ids}
                mesh = trimesh.util.concatenate(mesh, face.trimesh)

            cells = [
                ("triangle", mesh.faces)
            ]

            # create meshio:
            mesh = meshio.Mesh(points=mesh.vertices, cells=cells, cell_sets=cell_sets)
            mesh = self.add_mesh_properties(mesh)

            logger.info(f'Mesh creation for {self.name}: {self.id} successful')

        elif method == 'gmsh':

            try:

                faces = self.faces
                edge_loops = []
                [(edge_loops.append(x.boundary), edge_loops.extend(x.holes)) for x in faces]
                edge_loops = set(edge_loops)
                edges = []
                [edges.extend(x.edges) for x in edge_loops]
                edges = set(edges)
                vertices = []
                [vertices.extend([x.vertices[0], x.vertices[1]]) for x in edges]
                vertices = set(vertices)

                mesh = generate_mesh(vertices,
                                     edges,
                                     edge_loops,
                                     faces,
                                     [self],
                                     str(self.id),
                                     5)
            except Exception as e:
                logger.error(f'{self.name}; {self.id}: Error while creating mesh:\n{e}')

        return mesh

    def create_topology(self, face):
        pass

    def export_mesh(self, file_name):
        self.mesh.write(file_name)

    def add_to_gmsh_geo(self, geo):
        geo.addSurfaceLoop([x.gmsh_id for x in self.faces], self.gmsh_id)
        geo.addVolume([self.gmsh_id], self.gmsh_id)
