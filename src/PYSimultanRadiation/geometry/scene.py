from uuid import uuid4
import pygmsh
import numpy as np
import meshio
from itertools import count
from .. import logger


class Scene(object):

    def __init__(self, *args, **kwargs):
        """
        Class with the geometry which should be ray_traced; creates mesh with all necessary informations.

        @keyword name: Name of the template; type: str
        @keyword id: id of the template; type: uuid4; default: new generated uuid4
        @keyword faces: list of PySimultanRadiation.Geometry.extended_face.ExtendedFace; default: []
        """

        self._mesh = None

        self.name = kwargs.get('name', None)
        self.id = kwargs.get('id', uuid4())

        self.vertices = kwargs.get('vertices', [])
        self.edges = kwargs.get('edges', [])
        self.edge_loops = kwargs.get('edge_loops', [])
        self.faces = kwargs.get('faces', [])
        self.volumes = kwargs.get('volumes', [])

        self._face_ids = None

        logger.debug(f'Created new scene {self.name}; {self.id}')

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
            self.mesh = self.create_mesh()
        return self._mesh

    @mesh.setter
    def mesh(self, value):
        self._mesh = value

    def create_mesh(self, method='gmsh'):

        if (self.faces is None) or (self.faces.__len__() == 0):
            logger.error(f'{self.name}; {self.id}: Scene has no faces')
            return

        # old version; did not work because returned cell_sets were wrong!
        mesh = None

        if method == 'from_model':
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

                mesh = geom.generate_mesh(dim=2, verbose=True)
                mesh = self.add_mesh_properties(mesh)

        elif method == 'from_faces':
            logger.warn("Mesh creation with method 'from_faces' not recommended. Use of 'from_model' recommended.")
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
            import gmsh

            gmsh.initialize()
            gmsh.model.add("t2")

            import gmsh
            # If sys.argv is passed to gmsh.initialize(), Gmsh will parse the command line
            # in the same way as the standalone Gmsh app:
            gmsh.initialize()
            gmsh.model.add("t2")
            # Copied from t1.py...
            lc = 1
            for vertex in self.vertices:
                gmsh.model.geo.addPoint(vertex.position[0], vertex.position[1], vertex.position[2], lc, vertex.gmsh_id)

            for edge in self.edges:
                gmsh.model.geo.addLine(edge.vertices[0].gmsh_id, edge.vertices[1].gmsh_id, edge.gmsh_id)

            for edge_loop in self.edge_loops:
                try:
                    ids = np.array([x.gmsh_id for x in edge_loop.edges]) * np.array(edge_loop.edge_orientations)
                    gmsh.model.geo.addCurveLoop(ids, edge_loop.gmsh_id)
                except Exception as e:
                    logger.error(f'Error creating CurveLoop for {edge_loop.id}, {edge_loop.gmsh_id}:\n{e}')

            for face in self.faces:
                try:
                    edge_loop_ids = [face.boundary.gmsh_id] + [x.gmsh_id for x in face.holes]
                    gmsh.model.geo.addPlaneSurface(edge_loop_ids, face.gmsh_id)
                except Exception as e:
                    logger.error(f'Error creating CurveLoop for {face.name},{face.id}, {face.gmsh_id}:\n{e}')

            for volume in self.volumes:
                gmsh.model.geo.addSurfaceLoop([x.gmsh_id for x in volume.faces], volume.gmsh_id)
                gmsh.model.geo.addVolume([volume.gmsh_id], volume.gmsh_id)

            gmsh.model.geo.synchronize()
            gmsh.model.mesh.generate(2)

            #Exception: Unable to recover the edge 7120 (1/4) on curve 1454 (on surface 177)

            print('done')


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

        materials_ids = {}
        mat_id = 0

        # num_elem_types = mesh.cells.__len__()
        tri_elem_type_index = [i for i, x in enumerate(mesh.cells) if x.type == 'triangle'][0]

        mesh.cells = [mesh.cells[tri_elem_type_index]]

        material_data = np.full(mesh.cells[0].__len__(), np.NaN, dtype=float)
        is_window = np.full(mesh.cells[0].__len__(), np.NaN, dtype=float)

        for key, value in mesh.cell_sets.items():
            face_id = np.argwhere(self.face_ids == int(key))[0, 0]
            face = self.faces[face_id]
            construction = self.faces[face_id].construction
            if construction is None:
                print(f'face without construction: {face.name}, {face.id}')
                material_data[value[tri_elem_type_index]] = np.NaN
            else:
                if construction not in materials_ids.keys():
                    materials_ids[construction] = mat_id
                    mat_id += 1
                material_data[value[tri_elem_type_index]] = materials_ids[construction]
                is_window[value[tri_elem_type_index]] = int(construction.is_window)

        mesh.cell_data['material'] = [material_data]
        mesh.cell_data['is_window'] = [is_window]

        return mesh

    def export_mesh(self, file_name):
        self.mesh.write(file_name)

    def create_topology(self):

        pass
