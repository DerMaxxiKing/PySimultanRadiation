from uuid import uuid4
import numpy as np
from .. import logger
from .utils import generate_mesh, generate_surface_mesh, generate_terrain
import trimesh
from collections import Counter
from copy import copy


class Scene(object):

    def __init__(self, *args, **kwargs):
        """
        Class with the geometry which should be ray_traced; creates mesh with all necessary informations.

        @keyword name: Name of the template; type: str
        @keyword id: id of the template; type: uuid4; default: new generated uuid4
        @keyword faces: list of PySimultanRadiation.Geometry.extended_face.ExtendedFace; default: []
        """

        self._mesh = None
        self._hull_mesh = None
        self._surface_mesh = None
        self.mesh_size = kwargs.get('mesh_size', 1)
        self.mesh_min_size = kwargs.get('mesh_min_size', 0.5)
        self.mesh_max_size = kwargs.get('mesh_max_size', 10)

        self.surface_mesh_method = kwargs.get('surface_mesh_method', 'robust')

        self.name = kwargs.get('name', None)
        self.id = kwargs.get('id', uuid4())

        self.vertices = kwargs.get('vertices', [])
        self.edges = kwargs.get('edges', [])
        self.edge_loops = kwargs.get('edge_loops', [])
        self.faces = kwargs.get('faces', [])
        self.volumes = kwargs.get('volumes', [])

        self._hull_faces = kwargs.get('_hull_faces', None)
        self._internal_faces = kwargs.get('_internal_faces', None)

        self._face_ids = None

        self.topo_done = False

        self.terrain_height = kwargs.get('terrain_height', 0)
        self._terrain = None
        self.terrain = kwargs.get('terrain', None)

        logger.debug(f'Created new scene {self.name}; {self.id}')

    @property
    def terrain(self):
        if self._terrain is None:
            self.terrain = self.generate_terrain()
        return self.terrain

    @terrain.setter
    def terrain(self, value):
        self._terrain = value

    @property
    def hull_faces(self):
        if self._hull_faces is None:
            if not self.topo_done:
                self.create_topology()
            self._hull_faces = [x for x in self.faces if x.hull_face]
        return self._hull_faces

    @property
    def internal_faces(self):
        if self._internal_faces is None:
            if not self.topo_done:
                self.create_topology()
            self._internal_faces = [x for x in self.faces if x.internal_faces]
        return self._internal_faces

    @property
    def hull_mesh(self):
        if self._hull_mesh is None:
            self.hull_mesh = self.create_hull_surface_mesh()
        return self._hull_mesh

    @hull_mesh.setter
    def hull_mesh(self, value):
        self._hull_mesh = value

    @property
    def face_ids(self):
        if self._face_ids is None:
            self._face_ids = np.array([x.id for x in self.faces])
        return self._face_ids

    @property
    def faces_with_construction(self):
        return [x for x in self.faces if (x.construction is not None)]

    @property
    def faces_with_undefined_construction(self):
        return [x for x in self.faces if (x.construction is None)]

    @property
    def walls(self):
        return [x for x in self.faces_with_construction if (not x.construction.is_window)]

    @property
    def windows(self):
        return [x for x in self.faces_with_construction if x.construction.is_window]

    @property
    def mesh(self):
        if self._mesh is None:
            self.mesh = self.create_mesh(dim=3)
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = value

    @property
    def surface_mesh(self):
        if self._surface_mesh is None:
            self.surface_mesh = self.create_surface_mesh()
        return self._surface_mesh

    @surface_mesh.setter
    def surface_mesh(self, value):
        self._surface_mesh = value

    def create_hull_surface_mesh(self):

        try:
            mesh = generate_surface_mesh(vertices=self.vertices,
                                         edges=self.edges,
                                         edge_loops=self.edge_loops,
                                         faces=self.hull_faces,
                                         model_name=str(self.id),
                                         lc=self.mesh_size,
                                         min_size=self.mesh_min_size,
                                         max_size=self.mesh_max_size,
                                         method=self.surface_mesh_method)

        except Exception as e:
            logger.error(f'{self.name}; {self.id}: Error while creating hull surface mesh:\n{e}')
            return

        return mesh

    def create_surface_mesh(self):

        try:
            mesh = generate_surface_mesh(vertices=self.vertices,
                                         edges=self.edges,
                                         edge_loops=self.edge_loops,
                                         faces=self.faces,
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
            mesh = generate_mesh(vertices=self.vertices,
                                 edges=self.edges,
                                 edge_loops=self.edge_loops,
                                 faces=self.faces,
                                 volumes=self.volumes,
                                 model_name=str(self.id),
                                 lc=self.mesh_size,
                                 min_size=self.mesh_min_size,
                                 max_size=self.mesh_max_size,
                                 dim=dim)
        except Exception as e:
            logger.error(f'{self.name}; {self.id}: Error while creating mesh:\n{e}')
            return

        if mesh is not None:
            mesh = self.add_mesh_properties(mesh)

        return mesh

    def export_face_mesh_vtk(self, face, filename=None):

        import meshio

        if filename is None:
            filename = face.name + '.vtk'

        # replace Umlaute
        special_char_map = {ord('ä'): 'ae', ord('ü'): 'ue', ord('ö'): 'oe', ord('ß'): 'ss'}
        filename = filename.translate(special_char_map)

        points = self.mesh.points
        cells = [("triangle",
                  self.mesh.cells[1].data[self.mesh.cell_sets[str(face.id)][1], :])]

        mesh = meshio.Mesh(
            points,
            cells
        )
        mesh.write(
            filename,  # str, os.PathLike, or buffer/open file
            # file_format="vtk",  # optional if first argument is a path; inferred from extension
        )

    def add_mesh_properties(self, mesh=None):

        if mesh is None:
            mesh = self.mesh

        if not self.topo_done:
            self.create_topology()

        materials_ids = {}
        room_ids = {}
        room_id = 0
        mat_id = 0

        # num_elem_types = mesh.cells.__len__()
        tri_elem_type_index = [i for i, x in enumerate(mesh.cells) if x.type == 'triangle'][0]

        mesh.cells = [mesh.cells[tri_elem_type_index]]

        material_data = np.full(mesh.cells[0].__len__(), np.NaN, dtype=float)
        is_window = np.full(mesh.cells[0].__len__(), np.NaN, dtype=float)
        side1_zone = np.full(mesh.cells[0].__len__(), np.NaN, dtype=float)
        side2_zone = np.full(mesh.cells[0].__len__(), np.NaN, dtype=float)
        hull_face = np.full(mesh.cells[0].__len__(), 1, dtype=float)
        internal_face = np.full(mesh.cells[0].__len__(), 0, dtype=float)
        g_value = np.full(mesh.cells[0].__len__(), 0, dtype=float)
        eps = np.full(mesh.cells[0].__len__(), 0, dtype=float)

        for key, value in mesh.cell_sets.items():
            face_id = np.argwhere(self.face_ids == int(key))[0, 0]
            face = self.faces[face_id]
            cell_ids = value[tri_elem_type_index]

            g_value[cell_ids] = face.g_value
            eps[cell_ids] = face.eps

            hull_face[cell_ids] = int(face.hull_face)
            internal_face[cell_ids] = int(face.internal_face)

            room1 = face.side1
            if room1 is not None:
                if room1 not in room_ids.keys():
                    room_ids[room1] = room_id
                    room_id += 1
                side1_zone[cell_ids] = room_ids[room1]

            room2 = face.side2
            if room2 is not None:
                if room2 not in room_ids.keys():
                    room_ids[room2] = room_id
                    room_id += 1
                side2_zone[cell_ids] = room_ids[room2]

            construction = self.faces[face_id].construction
            if construction is None:
                print(f'face without construction: {face.name}, {face.id}')
                material_data[cell_ids] = np.NaN
            else:
                if construction not in materials_ids.keys():
                    materials_ids[construction] = mat_id
                    mat_id += 1
                material_data[cell_ids] = materials_ids[construction]
                is_window[cell_ids] = int(not construction.is_window)

        mesh.cell_data['material'] = [material_data]
        mesh.cell_data['opaque'] = [is_window]
        mesh.cell_data['hull_face'] = [hull_face]
        mesh.cell_data['internal_face'] = [internal_face]
        mesh.cell_data['side1_zone'] = [side1_zone]
        mesh.cell_data['side2_zone'] = [side2_zone]
        mesh.cell_data['g_value'] = [g_value]
        mesh.cell_data['eps'] = [eps]

        return mesh

    def export_mesh(self, file_name):
        if self.mesh is None:
            logger.error(f'{self}: mesh is None')
            return
        self.mesh.write(file_name)

    def export_surf_mesh(self, file_name):
        if self.surface_mesh is None:
            logger.error(f'{self}: surface_mesh is None')
            return
        self.surface_mesh.write(file_name)

    def create_topology(self):

        faces = copy(self.faces)
        [faces.extend(x.faces) for x in self.volumes]
        occurrences = Counter(faces)

        hull_faces = [k for (k, v) in occurrences.items() if v in [1, 2]]
        inside_faces = [k for (k, v) in occurrences.items() if v == 3]
        # no_occurance = [k for (k, v) in occurrences.items() if v == 1]
        # np.array([x.hull_face for x in self.faces])

        for face in hull_faces:
            face.hull_face = True
            face.internal_face = True

        for face in inside_faces:
            face.internal_face = True
            face.hull_face = False

        self.topo_done = True

        # np.array([x.hull_face for x in self.faces])
        # generate_surface_mesh(faces=no_occurance, method='robust').write('no_occurrence.vtk')

    def generate_face_side_topology(self):

        for volume in self.volumes:
            if not volume.is_watertight:
                logger.error(f'{self.name}, {self.id}: Volume is not watertight')

            if not volume.is_watertight:
                trimesh.repair.fix_winding(volume.surface_trimesh)

            surface_normals = np.array([x.normal for x in volume.faces])
            first_cell_ids = [volume.surface_mesh.cell_sets[str(x.id)][1][0] for x in volume.faces]
            origins = volume.surface_trimesh.triangles_center[first_cell_ids, :]

            inside = volume.surface_trimesh.contains(origins + 0.05 * surface_normals)

            for i, face in enumerate(volume.faces):
                if inside[i]:
                    face.side2 = volume
                else:
                    face.side1 = volume

    def generate_terrain(self):

        return generate_terrain(self.hull_mesh, self.terrain_height)
