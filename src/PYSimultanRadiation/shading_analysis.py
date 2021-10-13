import copy
import os
# import socket
# import numpy as np
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

import trimesh
import pandas as pd
from meshio import Mesh as MioMesh
from . import TemplateParser, Template, yaml, DataModel
from .geometry.scene import Scene
from .radiation.location import Location
from .gui.dialogs import askComboValue
from .radiation.utils import create_sun_window, npyAppendableFile, calc_timestamp
from .client.client import Client, get_free_port, next_free_port
from .docker.docker_manager import ShadingService
from . import logger
import asyncio
from functools import partial
from tqdm.asyncio import tqdm as async_tqdm
# import time
from trimesh import Trimesh
import meshio
from tqdm import tqdm, trange

import psycopg2
from psycopg2 import Error
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
import sqlalchemy

from PySimultan import logger as py_sim_logger

from pandas.io import sql

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 importlib_resources.
    import importlib_resources as pkg_resources

from . import resources

with pkg_resources.path(resources, 'shading_analysis_template.yml') as r_path:
    template_filename = str(r_path)


import numpy as np
from psycopg2.extensions import register_adapter, AsIs
psycopg2.extensions.register_adapter(np.float32, AsIs)
psycopg2.extensions.register_adapter(np.ndarray, AsIs)


def create_shading_template():
    shading_template = Template(template_name='ShadingAnalysis',
                                template_id='1',
                                content=['EndDate',
                                         'MeshSize',
                                         'NorthAngle',
                                         'RayResolution',
                                         'ExportDirectory',
                                         'StartDate',
                                         'TerrainHeight',
                                         'FacesOfInterest',
                                         'Weather',
                                         'GeometryModel',
                                         'NumTimesteps',
                                         'TimestepSize',
                                         'TimestepUnit',
                                         'AddTerrain',
                                         'NumWorkers',
                                         'WriteVTK',
                                         'LogLevel',
                                         'WriteXLSX'],
                                documentation='',
                                units={},
                                types={'EndDate': 'str',
                                       'MeshSize': 'float',
                                       'NorthAngle': 'float',
                                       'RayResolution': 'float',
                                       'ExportDirectory': 'str',
                                       'StartDate': 'str',
                                       'TerrainHeight': 'float',
                                       'TimestepUnit': 'str',
                                       'TimestepSize': 'float',
                                       'AddTerrain': 'bool',
                                       'NumTimesteps': 'float',
                                       'NumWorkers': 'int',
                                       'WriteVTK': 'bool',
                                       'LogLevel': 'str',
                                       'WriteXLSX': 'bool'},
                                slots={'FacesOfInterest': 'Undefined Slot_00',
                                       'Weather': 'Undefined Slot_01'}
                                )

    weather_template = Template(template_name='Weather',
                                template_id='2',
                                content=[''],
                                documentation='',
                                units={},
                                types={},
                                slots={})

    foi_template = Template(template_name='FOIS',
                            inherits_from='ReferenceList',
                            template_id='3',
                            content=[''],
                            documentation='',
                            units={},
                            types={},
                            slots={})

    with open(template_filename,
              mode='w',
              encoding="utf-8") as f_obj:
        yaml.dump([shading_template, foi_template], f_obj)


class ShadingAnalysis(object):

    def __init__(self, *args, **kwargs):

        self._location = None
        self._geo_model = None
        self._scene = None
        self._mesh = None
        self._foi_mesh = None
        self._hull_mesh = None
        self._dti = None

        self.user_name = kwargs.get('user_name', 'admin')
        self.password = kwargs.get('password', 'admin')

        self.project_filename = kwargs.get('project_filename')
        self.template_filename = kwargs.get('template_filename', template_filename)

        self.template_parser = None
        self.data_model = None
        self.typed_data = None

        self.setup_component = None

    def load_project(self):
        self.template_parser = TemplateParser(template_filepath=self.template_filename)
        self.data_model = DataModel(project_path=self.project_filename,
                                    user_name=self.user_name,
                                    password=self.password)

        self.typed_data = self.data_model.get_typed_data(template_parser=self.template_parser, create_all=False)

        self.setup_component = list(self.template_parser.template_classes['ShadingAnalysis']._cls_instances)[0]

        try:
            logger.setLevel(self.setup_component.LogLevel)
            py_sim_logger.setLevel(self.setup_component.LogLevel)
        except Exception as e:
            logger.error(f'Error setting LogLevel:\n{e}')

        self.update_fois()

        print('done')

    @property
    def id(self):
        if self.setup_component is None:
            return

        return self.setup_component.id

    @property
    def dti(self):
        if self._dti is None:
            # https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases

            # pd.Series(pd.date_range(start=None,
            #                         end=None,
            #                         periods=None,
            #                         freq=None,
            #                         tz=None))

            if None not in [self.setup_component.StartDate,
                            self.setup_component.NumTimesteps,
                            self.setup_component.TimestepSize,
                            self.setup_component.TimestepUnit]:

                self._dti = pd.Series(pd.date_range(self.setup_component.StartDate,
                                                    periods=self.setup_component.NumTimesteps,
                                                    freq=f"{self.setup_component.TimestepSize}{self.setup_component.TimestepUnit}"))
            elif None not in [self.setup_component.StartDate,
                              self.setup_component.EndDate,
                              self.setup_component.NumTimesteps]:
                self._dti = pd.date_range(start=self.setup_component.StartDate,
                                          end=self.setup_component.EndDate,
                                          periods=self.setup_component.NumTimesteps)

        return self._dti

    @property
    def scene(self):
        if self._scene is None:
            self._scene = self.create_scene()
        return self._scene

    @property
    def mesh(self):
        if self._mesh is None:
            self._mesh = self.generate_mesh()
        return self._mesh

    @property
    def foi_faces(self):
        if self.setup_component is None:
            return

        if self.setup_component.FacesOfInterest is None:
            return

        fois = set()
        [fois.update(x.geo_instances) for x in self.setup_component.FacesOfInterest]
        return fois

    @property
    def foi_mesh(self):
        if self._foi_mesh is None:
            self._foi_mesh = self.generate_foi_mesh()
        return self._foi_mesh

    @property
    def hull_mesh(self):
        if self._hull_mesh is None:
            self._hull_mesh = self.generate_hull_mesh()
        return self._hull_mesh

    @property
    def location(self):
        if self._location is None:
            self._location = Location(file_name=self.setup_component.Weather.weather_file_name,
                                      north_angle=self.setup_component.NorthAngle)
        return self._location

    @property
    def geo_model(self):
        if self._geo_model is None:
            self._geo_model = self.select_geo_model()
        return self._geo_model

    @property
    def db_name(self):
        return str(self.id.GlobalId)

    def select_geo_model(self):
        model_names = dict((i, x.filename) for i, x in self.template_parser.typed_geo_models.items() if x is not None)
        if model_names.__len__() > 1:
            model_name = askComboValue(question='Select Geometry model', values=model_names.values())
            geo_key = list(model_names.keys())[list(model_names.values()).index(model_name)]
        else:
            geo_key = list(model_names.keys())[0]
        geo_model = self.template_parser.typed_geo_models[geo_key]
        return geo_model

    def create_scene(self):
        scene = Scene(vertices=self.geo_model.vertices,
                      edges=self.geo_model.edges,
                      edge_loops=self.geo_model.edge_loops,
                      faces=self.geo_model.faces,
                      volumes=self.geo_model.volumes,
                      terrain_height=self.setup_component.TerrainHeight)
        return scene

    def generate_mesh(self, mesh_size=None):
        if mesh_size is None:
            mesh_size = self.setup_component.MeshSize

        return self.scene.generate_shading_analysis_mesh(mesh_size=mesh_size,
                                                         add_terrain=self.setup_component.AddTerrain)

    def write_mesh(self):
        if self.mesh is not None:
            self.mesh.write(os.path.join(self.setup_component.ExportDirectory, f'{self.setup_component.name}_mesh.vtk'))

    def generate_foi_mesh(self):
        foi_mesh = trimesh.Trimesh(vertices=self.mesh.points,
                                   faces=self.mesh.cells_dict['triangle'][np.where(self.mesh.cell_data['foi'][0]), :][0])
        return foi_mesh

    def update_fois(self):

        if self.foi_faces is None:
            return

        if self.foi_faces.__len__() > 0:
            for face in self.geo_model.faces:
                face.foi = False

            for face in self.foi_faces:
                face.foi = True

    def run(self):

        logger.info(f'Starting shading analysis {self.id}...')

        df = pd.DataFrame(index=self.dti, columns=[])

        logger.info(f'Calculating irradiation vectors')
        irradiation_vector = self.location.generate_irradiation_vector(self.dti)
        df['irradiation_vector'] = irradiation_vector['irradiation_vector']

        logger.info(f'Calculating sun windows')
        sun_window = create_sun_window(self.foi_mesh, np.stack(df['irradiation_vector'].values))
        df['windows'] = [x for x in sun_window]

        # num_hull_cells = [1, self.mesh.cells_dict['triangle'][np.where(self.mesh.cell_data['hull_face'][0]), :][0].shape[0]]
        # cell_ids = np.where(self.mesh.cell_data['hull_face'][0])

        num_cells = self.mesh.cells_dict['triangle'].shape[0]

        logger.info(f'Getting free ports...')
        port = get_free_port()
        db_port = next_free_port(port=port+1, max_port=65535)
        logger.info(f'Server-port {port}; Database-port: {db_port}')

        serv_work_workdir = os.path.join(self.setup_component.ExportDirectory, 'serv_work_workdir')
        if not os.path.isdir(serv_work_workdir):
            os.makedirs(serv_work_workdir, exist_ok=True)

        my_shading_service = ShadingService(workdir=serv_work_workdir,
                                            port=port,
                                            db_port=db_port,
                                            num_workers=int(self.setup_component.NumWorkers),
                                            user=self.user_name,
                                            password=self.password,
                                            db_name=self.db_name)

        print(my_shading_service.db_dir)

        my_client = Client(ip=f'tcp://localhost:{port}')

        my_shading_service.write_compose_file('docker_compose_test.yml')
        # f_sh_np_file_name = os.path.join(self.setup_component.ExportDirectory, 'f_sh.npy')
        # f_sh_numpy_file = npyAppendableFile(f_sh_np_file_name, True)

        logger.info(f'Starting shading service...')
        with my_shading_service:
            logger.info(f'Shading service started')

            # check if database ready:
            logger.info(f'Initializing database')
            tablename = 'f_sh'
            engine = create_engine(f'postgresql://{self.user_name}:{self.password}@localhost:{db_port}/{str(self.id.GlobalId)}')

            # delete existing table:
            sql.execute('DROP TABLE IF EXISTS %s' % tablename, engine)
            # sql.execute('VACUUM', engine)

            # part_fcn = partial(calc_timestep_async,
            #                    client=my_client,
            #                    sample_dist=self.setup_component.RayResolution,
            #                    num_cells=num_cells,
            #                    db_engine=engine,
            #                    tablename=tablename
            #                    )

            logger.info(f'Sending mesh to clients')
            my_client.send_mesh(self.mesh)

            logger.info(f'Starting calculation...')

            # https://leimao.github.io/blog/Python-tqdm-AsyncIO/
            # async def run_loops(df):
            #
            #     tasks = []
            #
            #     for index, row in df.iterrows():
            #         task = asyncio.create_task(part_fcn(sun_window=row['windows'],
            #                                             irradiation_vector=row['irradiation_vector'],
            #                                             date=index))
            #         tasks.append(task)
            #
            #     [await f for f in tqdm(asyncio.as_completed(tasks), total=len(tasks))]
            #
            # asyncio.run(run_loops(df))

            # part_fcn = partial(calc_timestep_async,
            #                    client=my_client,
            #                    sample_dist=self.setup_component.RayResolution,
            #                    num_cells=num_cells,
            #                    db_engine=engine,
            #                    tablename=tablename
            #                    )

            loop = asyncio.get_event_loop()
            coros = [calc_timestep_async(client=my_client,
                                         db_engine=engine,
                                         sun_window=df_row['windows'],
                                         sample_dist=self.setup_component.RayResolution,
                                         irradiation_vector=df_row['irradiation_vector'],
                                         num_cells=num_cells,
                                         date=ts_date,
                                         tablename=tablename) for (ts_date, df_row) in df.iterrows()]
            all_groups = async_tqdm.gather(*coros)
            results = loop.run_until_complete(all_groups)
            loop.close()

            # await asyncio.gather(*coros)
            # loop = asyncio.get_event_loop()
            # loop.run_until_complete()
            #
            #
            #     coros = [part_fcn(sun_window=row['windows'],
            #                       irradiation_vector=row['irradiation_vector'],
            #                       date=index) for i, (index, row) in enumerate(df.iterrows())]
            #     # await asyncio.gather(*coros)
            #     await async_tqdm.gather(*coros, total=df.shape[0])
            # loop = asyncio.get_event_loop()
            # loop.run_until_complete(run_loops())
            #
            #
            # loop = asyncio.get_event_loop()
            # futures = [await(loop.create_task(calc_timestep_async(sun_window=row['windows'],
            #                                                 irradiation_vector=row['irradiation_vector'],
            #                                                 date=index,
            #                                                 client=my_client,
            #                                                 sample_dist=self.setup_component.RayResolution,
            #                                                 num_cells=num_cells,
            #                                                 db_engine=engine,
            #                                                 tablename=tablename
            #                                                 )) for i, (index, row) in enumerate(df.iterrows())]
            # loop.run_until_complete(futures)
            # # for f in futures:
            #     f.cancel()

            logger.info(f'Calculation finished')

            logger.info(f'Getting results from database')
            with engine.connect() as con:
                con.execute(f"ALTER TABLE {tablename} ADD PRIMARY KEY (\"date\");")

            f_sh = pd.read_sql_query(f"select * from \"{tablename}\"", con=engine, index_col='date').sort_values(by='date')

            # ----------------------------------------------------------------------------------------------------------
            # create f_sh_for named faces:
            logger.info(f'Aggregating results')
            face_f_sh = pd.DataFrame(f_sh.index.values, columns=['date'])
            face_f_sh.set_index('date', inplace=True)
            tri_mesh = Trimesh(vertices=self.mesh.points,
                               faces=self.mesh.cells_dict['triangle'])

            areas = tri_mesh.area_faces

            face_names = dict(zip([x.id for x in self.scene.faces if x.components],
                                  [x.components[0].name for x in self.scene.faces if x.components]))

            # for every face calculate the mean f_sh
            for key, value in self.mesh.cell_sets.items():
                f_areas = areas[value[1]]
                f_areas_sum = sum(areas[value[1]])

                def aggregate(x):
                    return np.sum(np.array(x)[value[1]] * f_areas) / f_areas_sum

                # face_f_sh[key] = f_sh['f_sh'].apply(lambda x: sum(np.array(x)[value[1]] * areas[value[1]]) / sum(areas[value[1]]))
                face_f_sh[key] = f_sh['f_sh'].apply(aggregate)

            logger.info(f'Writing aggregated results to database')
            # write to database:
            sql.execute('DROP TABLE IF EXISTS %s' % 'face_f_sh', engine)
            face_f_sh.to_sql('face_f_sh',
                             engine,
                             if_exists='append',
                             index=False)

            # ---------------------------------------------------------------------------------------------------------
            # write vtk
            # ---------------------------------------------------------------------------------------------------------

            if bool(self.setup_component.WriteVTK):

                vtk_mesh = copy.deepcopy(self.mesh)
                vtk_mesh.cell_data = {}

                logger.info(f'Writing .vtk files')

                # ------------------------------------------------------------------------------------------------------

                logger.info(f'Writing raw mesh results')
                vtk_path = os.path.join(self.setup_component.ExportDirectory, 'vtk', 'raw')
                if not os.path.isdir(vtk_path):
                    os.makedirs(vtk_path, exist_ok=True)
                for i, (index, row) in enumerate(tqdm(f_sh.iterrows(),
                                                      total=f_sh.shape[0],
                                                      colour='green',
                                                      desc="Writing raw mesh results")):
                # for i, (index, row) in enumerate(f_sh.iterrows()):
                    if f_sh.iloc[i]['irradiation_vector'][2] > 0:
                        continue
                    vtk_mesh.cell_data['f_sh'] = [np.array(f_sh['f_sh'].values[i])]
                    meshio.vtk.write(os.path.join(vtk_path, f"shading_{index.strftime('%Y%m%d_%H%M%S')}.vtk"),
                                     vtk_mesh,
                                     binary=True)

                # ------------------------------------------------------------------------------------------------------

                logger.info(f'Writing f_sh_mean')
                vtk_path = os.path.join(self.setup_component.ExportDirectory, 'vtk')
                vtk_mesh.cell_data = {}
                vtk_mesh.cell_data['f_sh_mean'] = [np.stack(f_sh['f_sh'].values,
                                                            axis=1).sum(axis=1) / f_sh['f_sh'].__len__()]

                meshio.vtk.write(os.path.join(vtk_path, f"f_sh_mean.vtk"),
                                 vtk_mesh,
                                 binary=True)

                # ------------------------------------------------------------------------------------------------------

                logger.info(f'Writing f_sh_faces')
                vtk_mesh.cell_data = {}
                vtk_path = os.path.join(self.setup_component.ExportDirectory, 'vtk', 'face_f_sh')
                if not os.path.isdir(vtk_path):
                    os.makedirs(vtk_path, exist_ok=True)

                face_f_sh_vec = np.zeros([face_f_sh.shape[0], self.mesh.cells[0].data.shape[0]])
                # for key, value in self.mesh.cell_sets.items():
                for key, value in tqdm(self.mesh.cell_sets.items(),
                                       total=f_sh.shape[0],
                                       colour='green',
                                       desc="Writing f_sh for faces:"):

                    values = np.array(face_f_sh[key])
                    face_f_sh_vec[:, self.mesh.cell_sets[key][1]] = np.broadcast_to(values,
                                                                                    (self.mesh.cell_sets[key][1].shape[
                                                                                         0],
                                                                                     values.shape[0])
                                                                                    ).T

                # for i in range(face_f_sh.shape[0]):
                for i in trange(face_f_sh.shape[0],
                                total=f_sh.shape[0],
                                colour='green',
                                desc="Writing irradiation vector vtks:"):

                    if f_sh['irradiation_vector'].iloc[i][2] > 0:
                        continue
                    vtk_mesh.cell_data['face_f_sh'] = [face_f_sh_vec[i, :]]
                    meshio.vtk.write(os.path.join(vtk_path,
                                                  f"shading_{face_f_sh.index[i].strftime('%Y%m%d_%H%M%S')}.vtk"
                                                  ),
                                     vtk_mesh,
                                     binary=True)

                # ------------------------------------------------------------------------------------------------------

                logger.info(f'Writing f_sh_faces_mean')
                vtk_path = os.path.join(self.setup_component.ExportDirectory, 'vtk')
                if not os.path.isdir(vtk_path):
                    os.makedirs(vtk_path, exist_ok=True)
                vtk_mesh.cell_data = {}
                vtk_mesh.cell_data['f_sh_mean'] = [np.stack(face_f_sh_vec,
                                                            axis=1).sum(axis=1) / face_f_sh_vec.shape[0]]

                meshio.vtk.write(os.path.join(vtk_path, f"f_sh_faces_mean.vtk"),
                                 vtk_mesh,
                                 binary=True)

            # ---------------------------------------------------------------------------------------------------------
            # write xls
            # ---------------------------------------------------------------------------------------------------------

            if bool(self.setup_component.WriteXLSX):

                xlsx_path = os.path.join(self.setup_component.ExportDirectory, 'xls')
                if not os.path.isdir(xlsx_path):
                    os.makedirs(xlsx_path, exist_ok=True)

                with pd.ExcelWriter(os.path.join(xlsx_path, 'output.xlsx')) as writer:

                    workbook = writer.book

                    summary_df = pd.DataFrame(data={'Analysis ID': self.id.LocalId,
                                                    'Analysis Name': self.setup_component.name,
                                                    'North Angle': self.setup_component.NorthAngle,
                                                    'Weather File': os.path.basename(self.setup_component.Weather.weather_file_name),
                                                    'Mesh Size': self.setup_component.MeshSize,
                                                    'Ray Resolution': self.setup_component.RayResolution,
                                                    'Start Date': self.setup_component.StartDate,
                                                    'Number of Timesteps': self.setup_component.NumTimesteps,
                                                    'Timestep Size': self.setup_component.TimestepSize,
                                                    'Timestep Unit': self.setup_component.TimestepUnit,
                                                    'AddTerrain': self.setup_component.AddTerrain,
                                                    'Terrain Height': self.setup_component.TerrainHeight,
                                                    'Mesh': '',
                                                    'Num Triangles': self.mesh.cells[0].data.shape[0]
                                                    }, index=[0])

                    summary_df.T.to_excel(writer,
                                          sheet_name='Summary',
                                          index=True,
                                          header=True,
                                          startrow=1,
                                          startcol=1)

                    irradiation_vectors_df = pd.DataFrame(data={'x': df['irradiation_vector'].apply(lambda x: x[0]),
                                                                'y': df['irradiation_vector'].apply(lambda x: x[1]),
                                                                'z': df['irradiation_vector'].apply(lambda x: x[2])})

                    irradiation_vectors_df.to_excel(writer,
                                                    sheet_name='Irradiation Vectors',
                                                    index=True,
                                                    startrow=0,
                                                    startcol=0)


                    face_f_sh.to_excel(writer,
                                       sheet_name='f_sh',
                                       index=True,
                                       startrow=1,
                                       startcol=0
                                       )

                    worksheet = workbook['f_sh']

                    for i in range(self.scene.faces.__len__()):
                        c1 = worksheet.cell(row=1, column=i+2)
                        if self.scene.faces[i].components:
                            c1.value = self.scene.faces[i].components[0].name
                        else:
                            c1.value = ''

                    writer.save()




    def generate_hull_mesh(self):

        hull_mesh = MioMesh(points=self.mesh.points,
                            cells=[("triangle",
                                    self.mesh.cells_dict['triangle'][np.where(self.mesh.cell_data['hull_face'][0]), :][0])
                                   ])
        return hull_mesh

    def evaluate_shading_results(self):
        pass


async def calc_timestep_async(client=None,
                        db_engine=None,
                        sun_window=None,
                        sample_dist=None,
                        irradiation_vector=None,
                        num_cells=None,
                        date=None,
                        tablename=None):

    start_time = time.time()

    # logger.info(f'processing timestep: {date}')

    f_sh = np.zeros([num_cells])

    if irradiation_vector[2] < 0:
        rt_start_time = time.time()
        count = client.rt_sun_window(scene='hull',
                                     sun_window=sun_window,
                                     sample_dist=sample_dist,
                                     irradiation_vector=irradiation_vector)
        f_sh[0:count.shape[0]] = count
        rt_end_time = time.time()
        # logger.info(f'RayTracing needed: {rt_end_time - rt_start_time}')

    # write to database
    df0 = pd.DataFrame({'date': date,
                        'irradiation_vector': [irradiation_vector.tolist()],
                        'f_sh': [f_sh.tolist()]})

    # logger.info(f'writing results for timestep: {date}')
    df0.to_sql(tablename,
               db_engine,
               if_exists='append',
               index=False,
               dtype={'date': sqlalchemy.TIMESTAMP(),
                      'irradiation_vector': postgresql.ARRAY(sqlalchemy.types.FLOAT),
                      'f_sh': postgresql.ARRAY(sqlalchemy.types.FLOAT)
                      }
               )

    end_time = time.time()
    # logger.info(f'processing needed: {end_time - start_time}')
