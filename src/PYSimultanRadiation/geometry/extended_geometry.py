from PySimultan.geo_default_types import geometry_types
from .utils import mesh_planar_face, generate_mesh, generate_surface_mesh
from trimesh import Trimesh
import logging
import numpy as np
from itertools import count
from copy import copy

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

        self.mesh_size = kwargs.get('mesh_size', 100)
        self.hull_face = kwargs.get('hull_face', True)
        self.internal_face = kwargs.get('internal_face', False)

        self.side1 = kwargs.get('side1', None)
        self.side2 = kwargs.get('side2', None)

    @property
    def normal(self):
        return np.array([self._wrapped_obj.Normal.X, self._wrapped_obj.Normal.Y, self._wrapped_obj.Normal.Z])

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

        try:

            faces = [self]
            edge_loops = []
            [(edge_loops.append(x.boundary), edge_loops.extend(x.holes)) for x in faces]
            edge_loops = set(edge_loops)
            edges = []
            [edges.extend(x.edges) for x in edge_loops]
            edges = set(edges)
            vertices = []
            [vertices.extend([x.vertices[0], x.vertices[1]]) for x in edges]
            vertices = set(vertices)

            mesh = generate_mesh(vertices=vertices,
                                 edges=edges,
                                 edge_loops=edge_loops,
                                 faces=faces,
                                 volumes=[],
                                 model_name=str(self.id),
                                 lc=self.mesh_size,
                                 dim=2)
        except Exception as e:
            logger.error(f'{self.name}; {self.id}: Error while creating mesh:\n{e}')
            return

        return mesh

        # return mesh_planar_face(self, mesh_size=self.mesh_size)

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

        self.mesh_size = kwargs.get('mesh_size', 1)
        self.mesh_min_size = kwargs.get('mesh_min_size', 0.1)
        self.mesh_max_size = kwargs.get('mesh_max_size', 10)

        self.surface_mesh_method = kwargs.get('surface_mesh_method', 'robust')

        geometry_types.volume.__init__(self, *args, **kwargs)
        self._mesh = None
        self._surface_mesh = None
        self._surface_trimesh = None
        self.gmsh_id = ExtendedVolume.newid()

    @property
    def surface_trimesh(self):
        if self._surface_trimesh is None:
            self._surface_trimesh = Trimesh(vertices=self.surface_mesh.points, faces=self.surface_mesh.cells[1].data)
        return self._surface_trimesh

    @property
    def is_watertight(self):
        return self.surface_trimesh.is_watertight

    @property
    def mesh(self):
        if self._mesh is None:
            self.mesh = self.create_mesh(dim=3)
        return self._mesh

    @property
    def surface_mesh(self):
        if self._surface_mesh is None:
            self._surface_mesh = self.create_surface_mesh()
        return self._surface_mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = value

    def create_surface_mesh(self):

        faces = copy(self.faces)
        edge_loops = []
        [(edge_loops.append(x.boundary), edge_loops.extend(x.holes)) for x in faces]
        edge_loops = set(edge_loops)
        edges = []
        [edges.extend(x.edges) for x in edge_loops]
        edges = set(edges)
        vertices = []
        [vertices.extend([x.vertices[0], x.vertices[1]]) for x in edges]
        vertices = set(vertices)

        try:
            mesh = generate_surface_mesh(vertices=vertices,
                                         edges=edges,
                                         edge_loops=edge_loops,
                                         faces=faces,
                                         model_name=str(self.id),
                                         lc=self.mesh_size,
                                         min_size=self.mesh_min_size,
                                         max_size=self.mesh_max_size,
                                         method=self.surface_mesh_method)
        except Exception as e:
            logger.error(f'{self.name}; {self.id}: Error while creating surface mesh:\n{e}')
            return

        if mesh is not None:
            mesh = self.add_mesh_properties(mesh)

        return mesh

    def create_mesh(self, dim=3):

        try:

            faces = copy(self.faces)
            edge_loops = []
            [(edge_loops.append(x.boundary), edge_loops.extend(x.holes)) for x in faces]
            edge_loops = set(edge_loops)
            edges = []
            [edges.extend(x.edges) for x in edge_loops]
            edges = set(edges)
            vertices = []
            [vertices.extend([x.vertices[0], x.vertices[1]]) for x in edges]
            vertices = set(vertices)

            mesh = generate_mesh(vertices=vertices,
                                 edges=edges,
                                 edge_loops=edge_loops,
                                 faces=faces,
                                 volumes=[self],
                                 model_name=str(self.id),
                                 lc=self.mesh_size,
                                 min_size=self.mesh_min_size,
                                 max_size=self.mesh_max_size,
                                 dim=dim)
        except Exception as e:
            logger.error(f'{self.name}; {self.id}: Error while creating mesh:\n{e}')
            return

        return mesh

    def create_topology(self, face):
        pass

    def export_mesh(self, file_name):
        self.mesh.write(file_name)

    def add_to_gmsh_geo(self, geo):
        geo.addSurfaceLoop([x.gmsh_id for x in self.faces], self.gmsh_id)
        geo.addVolume([self.gmsh_id], self.gmsh_id)

    def export_surf_mesh(self, file_name):
        self.surface_mesh.write(file_name)

    def add_mesh_properties(self, mesh):
        return mesh
