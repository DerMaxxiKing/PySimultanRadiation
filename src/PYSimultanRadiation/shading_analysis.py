import copy
import os
import sys
import time
import traceback
from multiprocessing import Pool, cpu_count
from scipy import integrate

import meshio
import numpy as np
import pandas as pd
import psycopg2
import sqlalchemy
import trimesh
from PySimultan import logger as py_sim_logger
from meshio import Mesh as MioMesh
from pandas.io import sql
from psycopg2.extensions import register_adapter, AsIs
from psycopg2.extras import Json
from sqlalchemy import Table, Column, MetaData
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from tqdm import tqdm, trange
from trimesh import Trimesh

from . import TemplateParser, Template, yaml, DataModel
from . import logger
from .client.client import Client, next_free_port
from .docker.docker_manager import ShadingService, DatabaseService
from .geometry.scene import Scene
from .geometry.utils import calc_aoi
from .gui.dialogs import askComboValue
from .radiation.location import Location
from .radiation.utils import create_sun_window
from .utils import df_interpolate_at, write_face_results
from .db_utils import DBInterface


try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 importlib_resources.
    import importlib_resources as pkg_resources

from . import resources

with pkg_resources.path(resources, 'shading_analysis_template.yml') as r_path:
    template_filename = str(r_path)

psycopg2.extensions.register_adapter(np.float32, AsIs)
psycopg2.extensions.register_adapter(np.ndarray, postgresql.ARRAY(sqlalchemy.types.FLOAT))
register_adapter(dict, Json)


def create_shading_template():

    shading_template = Template(template_name='ShadingAnalysis',
                                template_id='1',
                                content=['geometry',
                                         'location',
                                         'results',
                                         'run_configuration',
                                         'simulation_setup',
                                         'Active'],
                                slots={'geometry': 'Element_01',
                                       'location': 'Element_02',
                                       'results': 'Element_03',
                                       'run_configuration': 'Element_04',
                                       'simulation_setup': 'Element_05'}
                                )

    sa_geometry_template = Template(template_name='SAGeometry',
                                    template_id='2',
                                    content=['additional_geometry',
                                             'excluded_face_list',
                                             'faces_of_interest',
                                             'geometry',
                                             'AddTerrain',
                                             'TerrainHeight'],
                                    slots={'additional_geometry': 'Element_01',
                                           'excluded_face_list': 'Element_02',
                                           'faces_of_interest': 'Element_03',
                                           'geometry': 'Element_04'}
                                    )

    sa_location_template = Template(template_name='SALocation',
                                    template_id='3',
                                    content=['Altitude',
                                             'FromWeatherFile',
                                             'Latitude',
                                             'Longitude',
                                             'NorthAngle',
                                             'Timezone',
                                             'Name',
                                             'Weather'],
                                    slots={'Weather': 'Element_00'},
                                    types={'Name': 'str'}
                                    )

    sa_run_configuration = Template(template_name='SARunConfiguration',
                                    template_id='4',
                                    content=['ExportDirectory',
                                             'LogLevel',
                                             'NumWorkers',
                                             'PersistDB',
                                             'RunMeshing',
                                             'RunShadingFactorCalculation',
                                             'RunViewFactorCalculation',
                                             'WriteViewFactors',
                                             'WriteVTK',
                                             'WriteWeather',
                                             'WriteXLSX'],
                                    types={'ExportDirectory': 'str',
                                           'LogLevel': 'str'}
                                    )

    sa_sim_setup = Template(template_name='SASimulationSetup',
                            template_id='5',
                            content=['ShadingSetup',
                                     'ViewFactorSetup'],
                            slots={'ShadingSetup': 'Element_01',
                                   'ViewFactorSetup': 'Element_02'}
                            )

    sa_shading_setup = Template(template_name='SAShadingSetup',
                                template_id='6',
                                content=['MeshSize',
                                         'NumTimesteps',
                                         'RayResolution',
                                         'StartDate',
                                         'TimestepSize',
                                         'TimestepUnit',
                                         'ResultExport'],
                                types={'StartDate': 'str',
                                       'TimestepUnit': 'str'},
                                slots={'ResultExport': 'Element_01'},
                                )

    sa_shading_export_setup = Template(template_name='SAShadingExportSetup',
                                       template_id='7',
                                       content=['WriteAbsoluteIrradiation',
                                                'WriteAngleOfIncidence',
                                                'WriteIrradiatedAmountOfHeat',
                                                'WriteIrradiationVectors',
                                                'WriteMeanShadingFactors',
                                                'WriteShadingFactors',
                                                'WriteSpecificIrradiation',
                                                'WriteSummary',
                                                'WriteZeroRes']
                                       )

    sa_view_factor_setup = Template(template_name='SAViewFactorSetup',
                                    template_id='8',
                                    content=['NRay',
                                             'OnlyInsideZone',
                                             'SampleDistance']
                                    )


    # shading_template = Template(template_name='ShadingAnalysis',
    #                             template_id='1',
    #                             content=['EndDate',
    #                                      'MeshSize',
    #                                      'NorthAngle',
    #                                      'RayResolution',
    #                                      'ExportDirectory',
    #                                      'StartDate',
    #                                      'TerrainHeight',
    #                                      'FacesOfInterest',
    #                                      'Weather',
    #                                      'GeometryModel',
    #                                      'NumTimesteps',
    #                                      'TimestepSize',
    #                                      'TimestepUnit',
    #                                      'AddTerrain',
    #                                      'NumWorkers',
    #                                      'WriteVTK',
    #                                      'LogLevel',
    #                                      'WriteXLSX'],
    #                             documentation='',
    #                             units={},
    #                             types={'EndDate': 'str',
    #                                    'MeshSize': 'float',
    #                                    'NorthAngle': 'float',
    #                                    'RayResolution': 'float',
    #                                    'ExportDirectory': 'str',
    #                                    'StartDate': 'str',
    #                                    'TerrainHeight': 'float',
    #                                    'TimestepUnit': 'str',
    #                                    'TimestepSize': 'float',
    #                                    'AddTerrain': 'bool',
    #                                    'NumTimesteps': 'float',
    #                                    'NumWorkers': 'int',
    #                                    'WriteVTK': 'bool',
    #                                    'LogLevel': 'str',
    #                                    'WriteXLSX': 'bool'},
    #                             slots={'FacesOfInterest': 'Undefined Slot_00',
    #                                    'Weather': 'Undefined Slot_01'}
    #                             )
    #
    # weather_template = Template(template_name='Weather',
    #                             template_id='2',
    #                             content=[''],
    #                             documentation='',
    #                             units={},
    #                             types={},
    #                             slots={})

    if not os.path.isfile(template_filename):
        open(template_filename, 'a').close()
    with open(template_filename,
              mode='w',
              encoding="utf-8") as f_obj:
        yaml.dump([shading_template,
                   sa_geometry_template,
                   sa_location_template,
                   sa_run_configuration,
                   sa_sim_setup,
                   sa_shading_setup,
                   sa_shading_export_setup,
                   sa_view_factor_setup], f_obj)


class ProjectLoader(object):

    def __init__(self, *args, **kwargs):

        self.user_name = kwargs.get('user_name', 'admin')
        self.password = kwargs.get('password', 'admin')

        self.project_filename = kwargs.get('project_filename')
        self.template_filename = kwargs.get('template_filename', template_filename)

        self.template_parser = None
        self.data_model = None
        self.typed_data = None

        self.setup_components = []

    def load_project(self):

        logger.setLevel('INFO')
        self.template_parser = TemplateParser(template_filepath=self.template_filename)

        logger.info(f'Loading SIMULTAN Project: {self.project_filename}')
        self.data_model = DataModel(project_path=self.project_filename,
                                    user_name=self.user_name,
                                    password=self.password)
        logger.info(f'SIMULTAN Project loaded successfully')

        logger.info(f'Creating typed data model...')
        self.typed_data = self.data_model.get_typed_data(template_parser=self.template_parser, create_all=True)
        logger.info(f'Typed data model created successfully')

        logger.info(f'Searching active ShadingAnalysis components...')
        self.setup_components = [x for x in self.template_parser.template_classes['ShadingAnalysis']._cls_instances if bool(x.Active)]
        logger.info(f'Found {self.setup_components.__len__()} active Shading Analyses')

        print('done')

    def run(self):

        for setup_component in self.setup_components:

            try:
                print('\n\n')
                logger.info(f'Running Analysis {setup_component.name}, {setup_component.id}... ')

                shading_analysis = ShadingAnalysis(project_filename=self.project_filename,
                                                   user_name=self.user_name,
                                                   password=self.password,
                                                   template_parser=self.template_parser,
                                                   data_model=self.data_model,
                                                   typed_data=self.typed_data,
                                                   setup_component=setup_component)

                shading_analysis.db_service.keep_running = True
                shading_analysis.write_mesh()
                shading_analysis.run()

                logger.info(f'Running Analysis {setup_component.name}, {setup_component.id} finished successfully')
                print('\n\n')
            except Exception as e:
                logger.error(f'Error while running Analysis {setup_component.name}, {setup_component.id}:\n{e}\n{traceback.format_exc()}\n{sys.exc_info()[2]}')

        logger.info('Finished all active analyses')


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

        self.template_parser = kwargs.get('template_parser', None)
        self.data_model = kwargs.get('data_model', None)
        self.typed_data = kwargs.get('typed_data', None)

        self.setup_component = kwargs.get('setup_component', None)

        self._shading_service = None
        self._db_service = None

        self._db_interface = None

    def load_project(self):
        self.template_parser = TemplateParser(template_filepath=self.template_filename)
        self.data_model = DataModel(project_path=self.project_filename,
                                    user_name=self.user_name,
                                    password=self.password)

        self.typed_data = self.data_model.get_typed_data(template_parser=self.template_parser, create_all=True)

        self.setup_component = list(self.template_parser.template_classes['ShadingAnalysis']._cls_instances)[0]

        try:
            logger.setLevel(self.setup_component.run_configuration.LogLevel)
            py_sim_logger.setLevel(self.setup_component.run_configuration.LogLevel)
        except Exception as e:
            logger.error(f'Error setting LogLevel:\n{e}\n{traceback.format_exc()}\n{sys.exc_info()[2]}')

        self.update_fois()

        print('done')

    @property
    def shading_setup(self):
        return self.setup_component.simulation_setup.ShadingSetup

    @property
    def db_service(self):
        if self._db_service is None:
            serv_work_workdir = os.path.join(self.setup_component.run_configuration.ExportDirectory, 'serv_work_workdir')
            port = next_free_port(port=9006, max_port=65535)
            logger.debug(f'database port is {port}')

            self._db_service = DatabaseService(port=port,
                                               user=self.user_name,
                                               password=self.password,
                                               db_name=self.db_name,
                                               log_dir=os.path.join(serv_work_workdir, 'logs'),
                                               logging_mode=self.setup_component.run_configuration.LogLevel,
                                               persist_volume=self.setup_component.run_configuration.PersistDB)
        return self._db_service

    @db_service.setter
    def db_service(self, value):
        self._db_service = value

    @property
    def db_interface(self):
        if self._db_interface is None:
            self._db_interface = DBInterface(engine=self.db_service.engine)
        return self._db_interface

    @property
    def shading_service(self):
        if self._shading_service is None:
            serv_work_workdir = os.path.join(self.setup_component.run_configuration.ExportDirectory, 'serv_work_workdir')
            port = next_free_port(port=10006, max_port=65535)
            if not os.path.isdir(serv_work_workdir):
                os.makedirs(serv_work_workdir, exist_ok=True)

            self._shading_service = ShadingService(workdir=serv_work_workdir,
                                                   port=port,
                                                   num_workers=int(self.setup_component.run_configuration.NumWorkers),
                                                   logging_mode=self.setup_component.run_configuration.LogLevel)
        return self._shading_service

    @shading_service.setter
    def shading_service(self, value):
        self._shading_service = value

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

            shading_setup_component = self.setup_component.simulation_setup.ShadingSetup

            if None not in [shading_setup_component.StartDate,
                            shading_setup_component.NumTimesteps,
                            shading_setup_component.TimestepSize,
                            shading_setup_component.TimestepUnit]:

                start_date = pd.to_datetime(shading_setup_component.StartDate, format='%d.%m.%Y %H:%M:%S')

                self._dti = pd.Series(pd.date_range(start_date,
                                                    periods=shading_setup_component.NumTimesteps,
                                                    freq=f"{shading_setup_component.TimestepSize}{shading_setup_component.TimestepUnit}",
                                                    tz='UTC'),
                                      )
            elif None not in [shading_setup_component.StartDate,
                              shading_setup_component.EndDate,
                              shading_setup_component.NumTimesteps]:
                start_date = pd.to_datetime(shading_setup_component.StartDate, format='%d.%m.%Y %H:%M:%S')
                end_date = pd.to_datetime(shading_setup_component.EndDate, format='%d.%m.%Y %H:%M:%S')

                self._dti = pd.date_range(start=start_date,
                                          end=end_date,
                                          periods=shading_setup_component.NumTimesteps,
                                          tz='UTC')

        return self._dti

    @property
    def scene(self):
        if self._scene is None:
            self._scene = self.create_scene()
        return self._scene

    @property
    def mesh(self):
        if self._mesh is None:
            if bool(self.setup_component.run_configuration.PersistDB) and not (bool(self.setup_component.run_configuration.RunMeshing)):
                mesh = None
                try:
                    mesh = self.read_mesh_from_db()
                except Exception as e:
                    logger.error(f'Error reading mesh from database:\n{e}\n{traceback.format_exc()}\n{sys.exc_info()[2]}')
                if isinstance(mesh, meshio.Mesh):
                    logger.info(f'Found mesh: Triangles: {mesh}')
                    self._mesh = mesh
                else:
                    self._mesh = self.generate_mesh()
                    self.write_mesh_to_db(self._mesh)
            else:
                self._mesh = self.generate_mesh()
                if bool(self.setup_component.run_configuration.PersistDB):
                    self.write_mesh_to_db(self._mesh)

        return self._mesh

    @property
    def foi_faces(self):
        if self.setup_component is None:
            return

        if self.setup_component.geometry is None:
            return

        return self.setup_component.geometry.faces_of_interest

        # fois = set()
        # [fois.update(x.geo_instances) for x in self.setup_component.geometry.faces_of_interest]
        # return fois

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
            self._location = Location(file_name=self.setup_component.location.Weather.weather_file_name,
                                      setup_component=self.setup_component.location,
                                      north_angle=self.setup_component.location.NorthAngle)
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

        if self.setup_component.geometry.geometry._wrapped_obj.ReferencedAssets.Items.Items.__len__() == 0:
            logger.error(f'Could not find a geometry model in: {self.setup_component.name}, {self.setup_component.id}. Please reference a .simgeo file in Geometry -> SimultanGeometry')
            model_names = dict((i, x.filename) for i, x in self.template_parser.typed_geo_models.items() if x is not None)
            if model_names.__len__() > 1:
                model_name = askComboValue(question='Select Geometry model', values=model_names.values())
                geo_key = list(model_names.keys())[list(model_names.values()).index(model_name)]
            else:
                geo_key = list(model_names.keys())[0]
            geo_model = self.template_parser.typed_geo_models[geo_key]

        else:
            simgeo_file = self.setup_component.geometry.geometry._wrapped_obj.ReferencedAssets.Items.Items[0].Resource.CurrentRelativePath
            geo_model = next((x for i, x in self.template_parser.typed_geo_models.items() if x is not None and (x.filename == simgeo_file)), None)

        if geo_model is None:
            logger.error(f'Could not find a geometry model: {self.setup_component.name}, {self.setup_component.id}')
            raise ValueError(f'Could not find a geometry model: {self.setup_component.name}, {self.setup_component.id}')
        return geo_model

    def create_scene(self):

        faces = set(self.geo_model.faces)

        # update faces of interest
        self.update_fois()

        # exclude faces
        try:
            faces.difference_update(self.setup_component.geometry.excluded_face_list)
        except Exception as e:
            logger.error(f'Error removing faces in ExcludedFaceList: {e}\nFaces: {faces}\nFaces to exclude: {self.setup_component.geometry.excluded_face_list}')

        scene = Scene(vertices=self.geo_model.vertices,
                      edges=self.geo_model.edges,
                      edge_loops=self.geo_model.edge_loops,
                      faces=self.geo_model.faces,
                      volumes=self.geo_model.volumes,
                      terrain_height=self.setup_component.geometry.TerrainHeight)
        return scene

    def generate_mesh(self, mesh_size=None):
        if mesh_size is None:
            mesh_size = self.shading_setup.MeshSize

        if mesh_size < 0:
            logger.error(f'Error: Mesh size cannot be less than 0')

        return self.scene.generate_shading_analysis_mesh(mesh_size=mesh_size,
                                                         add_terrain=self.setup_component.geometry.AddTerrain)

    def write_mesh(self):

        if self.mesh is not None:
            logger.info(f'Writing analyse {self.setup_component.name}, {self.setup_component.id} mesh to {self.setup_component.run_configuration.ExportDirectory}')
            if not os.path.isdir(self.setup_component.run_configuration.ExportDirectory):
                os.makedirs(self.setup_component.run_configuration.ExportDirectory, exist_ok=True)
            self.mesh.write(os.path.join(self.setup_component.run_configuration.ExportDirectory, f'{self.setup_component.name}_mesh.vtk'))

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
        if self.setup_component.run_configuration.RunShadingFactorCalculation:
            self.run_shading_analysis()
            self.export_results()

    def run_shading_analysis(self):

        logger.info(f'Starting shading analysis {self.id}...')

        self.db_service.keep_running = True
        with self.db_service:

            engine = create_engine(
                f'postgresql://{self.user_name}:{self.password}@localhost:{self.db_service.port}/{str(self.id.GlobalId)}')
            engine.dispose()

            base_df = pd.DataFrame(index=self.dti, columns=[])

            logger.info(f'Calculating irradiation vectors')
            irradiation_vector = self.location.generate_irradiation_vector(self.dti)
            base_df['irradiation_vector'] = irradiation_vector['irradiation_vector']

            # calculate angle of incidence
            face_normals = np.array([x.normal for x in self.geo_model.faces])
            face_ids = [x.id for x in self.geo_model.faces]
            aois = irradiation_vector.apply(calc_aoi, args=(face_normals, ), axis=1, result_type='expand')
            aois.columns = face_ids
            try:
                self.db_interface.save_dataframe(aois, 'angle of incidence')
            except Exception as e:
                logger.error(f"Error writing 'angle of incidence' to database:\n{e}\n{traceback.format_exc()}\n{sys.exc_info()[2]}")

            logger.info(f'Calculating sun windows')
            sun_window = create_sun_window(self.foi_mesh, np.stack(base_df['irradiation_vector'].values))
            base_df['windows'] = [x for x in sun_window]

            save_df = copy.copy(base_df)
            save_df['windows'] = [x.tolist() for x in sun_window]

            try:
                self.db_interface.save_dataframe(base_df, 'base_df')
            except Exception as e:
                logger.error(f'Error writing base_df to database:\n{e}\n{traceback.format_exc()}\n{sys.exc_info()[2]}')

            num_cells = self.mesh.cells_dict['triangle'].shape[0]

            my_client = Client(ip=f'tcp://localhost:{self.shading_service.port}')

            # my_shading_service.write_compose_file('docker_compose_test.yml')

            logger.info(f'Starting shading service...')
            with self.shading_service:
                logger.info(f'Shading service started')

                # check if database ready:
                logger.info(f'Initializing database')
                tablename = 'f_sh'

                # delete existing table:
                sql.execute('DROP TABLE IF EXISTS %s' % tablename, engine)

                # create f_sh table
                meta = MetaData()

                f_sh_table = Table(
                    'f_sh', meta,
                    Column('date', sqlalchemy.TIMESTAMP(), primary_key=True),
                    Column('irradiation_vector', postgresql.ARRAY(sqlalchemy.types.FLOAT)),
                    Column('f_sh', postgresql.ARRAY(sqlalchemy.types.FLOAT)),
                )

                meta.create_all(engine)
                logger.info(f'Sending mesh to clients')
                my_client.send_mesh(self.mesh)

                logger.info(f'Starting calculation...')

                rt_engine = RTEngine(port=self.shading_service.port,
                                     sample_dist=self.setup_component.simulation_setup.ShadingSetup.RayResolution,
                                     num_cells=num_cells,
                                     tablename=tablename,
                                     user_name=self.user_name,
                                     password=self.password,
                                     db_port=self.db_service.port,
                                     id=str(self.id.GlobalId),
                                     f_sh_table=f_sh_table)

                if int(self.setup_component.run_configuration.NumWorkers) == 1:
                    for row in tqdm(list(base_df.iterrows()),
                                    total=base_df.shape[0],
                                    desc='Running ray casting for timesteps:',
                                    colour='green'):
                        rt_engine(row)
                    logger.info(f'Ray casting finished')
                else:
                    try:
                        logger.info('Creating multiprocessing pool...')
                        max_num_cpu = cpu_count()
                        pool = Pool(min([int(self.setup_component.run_configuration.NumWorkers), max_num_cpu]))
                        # pool.map(rt_engine, list(df.iterrows()))
                        for _ in tqdm(pool.imap_unordered(rt_engine, list(base_df.iterrows())),
                                      total=base_df.shape[0],
                                      desc='Running ray casting for timesteps',
                                      colour='green'):
                            pass
                        logger.info(f'Ray casting finished')
                    finally:  # To make sure processes are closed in the end, even if errors happen
                        pool.close()
                        pool.join()

            self.evaluate_shading_results()

                # logger.info(f'Getting results from database')
                #
                # f_sh = pd.read_sql_query(f"""select * from {'"'}{tablename}{'"'}""", con=engine, index_col='date').sort_values(by='date')
                #
                # # ----------------------------------------------------------------------------------------------------------
                # # create f_sh_for named faces:
                # logger.info(f'Aggregating results')
                # face_f_sh = pd.DataFrame(f_sh.index.values, columns=['date'])
                # face_f_sh.set_index('date', inplace=True)
                # tri_mesh = Trimesh(vertices=self.mesh.points,
                #                    faces=self.mesh.cells_dict['triangle'])
                #
                # areas = tri_mesh.area_faces
                #
                # face_names = dict(zip([x.id for x in self.scene.faces if x.components],
                #                       [x.components[0].name for x in self.scene.faces if x.components]))
                #
                # # dni: direct normal irradiation from weather data:
                # dni = self.location.data['dni']
                # # remove localization of data
                # dni.index = dni.index.tz_localize(None)
                # # replace the year of the required timestamps with 2021 -> the year with which weather data is loaded
                # req_timestamps = pd.Series([x.replace(year=2021) for x in f_sh.index])
                #
                # # dni_req_ts: direct normal irradiation from weather data at requested timesteps:
                # dni_req_ts = df_interpolate_at(dni, req_timestamps, method='linear', axis='index')
                # # write dni to database
                # try:
                #     write_df_in_empty_table(dni_req_ts, 'dni', engine)
                # except Exception as e:
                #     logger.error(f'Error writing dni to database: {e}')
                #
                # # for every face calculate the mean f_sh
                #
                # face_areas = pd.DataFrame(index=[0])
                # # for key, value in self.mesh.cell_sets.items():
                # f_sh_mat = np.vstack(f_sh['f_sh'].values)
                # for key, value in tqdm(self.mesh.cell_sets.items(),
                #                        total=len(self.mesh.cell_sets),
                #                        colour='green',
                #                        desc="Aggregating results for faces"):
                #
                #     f_areas = areas[value[1]]
                #     f_areas_sum = sum(areas[value[1]])
                #     face_areas[key] = f_areas_sum
                #
                #     # def aggregate(x):
                #     #     return np.sum(np.array(x)[value[1]] * f_areas) / f_areas_sum
                #     #
                #     # # face_f_sh[key] = f_sh['f_sh'].apply(lambda x: sum(np.array(x)[value[1]] * areas[value[1]]) / sum(areas[value[1]]))
                #     # face_f_sh[key] = f_sh['f_sh'].apply(aggregate)
                #     face_f_sh[key] = np.sum(f_sh_mat[:, value[1]] * f_areas, axis=1) / f_areas_sum
                #
                # logger.info(f'Writing aggregated results to database')
                # # write to database:
                # try:
                #     write_df_in_empty_table(face_areas, 'face_areas', engine, index=False)
                #     write_df_in_empty_table(face_f_sh, 'face_f_sh', engine)
                # except Exception as e:
                #     logger.error(f'Error writing face_areas, face_f_sh to database: {e}')
                #
                # # calculate specific irradiation:
                # face_q_dot = face_f_sh.multiply(dni_req_ts, axis=0)
                # try:
                #     write_df_in_empty_table(face_q_dot, 'face_q_dot', engine)
                # except Exception as e:
                #     logger.error(f'Error writing face_q_dot to database: {e}')
                #
                # # calculate total irradiation:
                # face_Q_dot = pd.DataFrame(index=face_q_dot.index)
                # for column in face_q_dot.columns:
                #     face_Q_dot[column] = face_q_dot[column].multiply(face_areas[column][0], axis=0)
                # try:
                #     write_df_in_empty_table(face_Q_dot, 'face_q_tot_dot', engine)
                # except Exception as e:
                #     logger.error(f'Error writing face_q_tot_dot to database: {e}')
                #
                # # calculate irradiated amount of heat:
                # from scipy import integrate
                # face_Q = pd.DataFrame(integrate.cumtrapz(face_Q_dot.values,
                #                                          (face_Q_dot.index.asi8 - face_Q_dot.index.asi8[0]) * 1e-9, axis=0),
                #                       index=face_Q_dot.index[1:])
                # try:
                #     write_df_in_empty_table(face_Q, 'face_Q', engine)
                # except Exception as e:
                #     logger.error(f'Error writing face_Q to database: {e}')

                # ---------------------------------------------------------------------------------------------------------
                # write vtk
                # ---------------------------------------------------------------------------------------------------------

                # if bool(self.setup_component.WriteVTK):
                #
                #     vtk_mesh = copy.deepcopy(self.mesh)
                #     vtk_mesh.cell_data = {}
                #
                #     logger.info(f'Writing .vtk files')
                #
                #     # ------------------------------------------------------------------------------------------------------
                #
                #     logger.info(f'Writing vtk raw mesh results')
                #     vtk_path = os.path.join(self.setup_component.ExportDirectory, 'vtk', 'raw')
                #
                #     f_sh_array = np.vstack(f_sh['f_sh'].values)
                #     q_dot = np.multiply(f_sh_array, dni_req_ts.values[:, np.newaxis])
                #     Q_dot = q_dot * areas
                #     Q = np.zeros(q_dot.shape)
                #     Q[1:, :] = integrate.cumtrapz(q_dot, (f_sh.index.asi8 - f_sh.index.asi8[0]) * 1e-9, axis=0)
                #
                #     if not os.path.isdir(vtk_path):
                #         os.makedirs(vtk_path, exist_ok=True)
                #     for i, (index, row) in enumerate(tqdm(f_sh.iterrows(),
                #                                           total=f_sh.shape[0],
                #                                           colour='green',
                #                                           desc="Writing raw mesh results")):
                #     # for i, (index, row) in enumerate(f_sh.iterrows()):
                #         if f_sh.iloc[i]['irradiation_vector'][2] > 0:
                #             continue
                #         vtk_mesh.cell_data['f_sh'] = [f_sh_array[i, :]]
                #         vtk_mesh.cell_data['q_dot'] = [q_dot[i, :]]
                #         vtk_mesh.cell_data['Q_dot'] = [Q_dot[i, :]]
                #         vtk_mesh.cell_data['Q'] = [Q[i, :]]
                #         meshio.vtk.write(os.path.join(vtk_path, f"shading_{index.strftime('%Y%m%d_%H%M%S')}.vtk"),
                #                          vtk_mesh,
                #                          binary=True)
                #
                #     # ------------------------------------------------------------------------------------------------------
                #
                #     logger.info(f'Writing vtk f_sh_mean')
                #     vtk_path = os.path.join(self.setup_component.ExportDirectory, 'vtk')
                #     vtk_mesh.cell_data = {}
                #     vtk_mesh.cell_data['f_sh_mean'] = [np.stack(f_sh['f_sh'].values,
                #                                                 axis=1).sum(axis=1) / f_sh['f_sh'].__len__()]
                #
                #     meshio.vtk.write(os.path.join(vtk_path, f"f_sh_mean.vtk"),
                #                      vtk_mesh,
                #                      binary=True)
                #
                #     # ------------------------------------------------------------------------------------------------------
                #
                #     logger.info(f'Writing vtk f_sh_faces')
                #     vtk_mesh.cell_data = {}
                #     vtk_path = os.path.join(self.setup_component.ExportDirectory, 'vtk', 'face_f_sh')
                #     if not os.path.isdir(vtk_path):
                #         os.makedirs(vtk_path, exist_ok=True)
                #
                #     face_f_sh_vec = np.zeros([face_f_sh.shape[0], self.mesh.cells[0].data.shape[0]])
                #     face_q_dot_vec = np.zeros([face_f_sh.shape[0], self.mesh.cells[0].data.shape[0]])
                #     face_Q_dot_vec = np.zeros([face_f_sh.shape[0], self.mesh.cells[0].data.shape[0]])
                #     face_Q_vec = np.zeros([face_f_sh.shape[0], self.mesh.cells[0].data.shape[0]])
                #     # for key, value in self.mesh.cell_sets.items():
                #     for key, value in tqdm(self.mesh.cell_sets.items(),
                #                            total=f_sh.shape[0],
                #                            colour='green',
                #                            desc="Writing f_sh for faces"):
                #
                #         values = np.array(face_f_sh[key])
                #
                #         cell_ids = self.mesh.cell_sets[key][1]
                #
                #         face_f_sh_vec[:, cell_ids] = np.broadcast_to(values, (cell_ids.shape[0],values.shape[0])).T
                #         face_q_dot_vec[:, cell_ids] = np.multiply(face_f_sh_vec[:, cell_ids],
                #                                                   dni_req_ts.values[:, np.newaxis])
                #         face_Q_dot_vec[:, cell_ids] = face_q_dot_vec[:, cell_ids] * areas[cell_ids]
                #         face_Q_vec[1:, cell_ids] = integrate.cumtrapz(face_q_dot_vec[:, cell_ids],
                #                                                       (f_sh.index.asi8 - f_sh.index.asi8[0]) * 1e-9,
                #                                                       axis=0)
                #
                #     # for i in range(face_f_sh.shape[0]):
                #     for i in trange(face_f_sh.shape[0],
                #                     total=f_sh.shape[0],
                #                     colour='green',
                #                     desc="Writing irradiation vector vtks"):
                #
                #         if f_sh['irradiation_vector'].iloc[i][2] > 0:
                #             continue
                #         vtk_mesh.cell_data['face_f_sh'] = [face_f_sh_vec[i, :]]
                #         vtk_mesh.cell_data['face_q_dot'] = [face_q_dot_vec[i, :]]
                #         vtk_mesh.cell_data['face_Q_dot'] = [face_Q_dot_vec[i, :]]
                #         vtk_mesh.cell_data['face_Q'] = [face_Q_vec[i, :]]
                #         meshio.vtk.write(os.path.join(vtk_path,
                #                                       f"shading_{face_f_sh.index[i].strftime('%Y%m%d_%H%M%S')}.vtk"
                #                                       ),
                #                          vtk_mesh,
                #                          binary=True)
                #
                #     # ------------------------------------------------------------------------------------------------------
                #
                #     logger.info(f'Writing vtk f_sh_faces_mean')
                #     vtk_path = os.path.join(self.setup_component.ExportDirectory, 'vtk')
                #     if not os.path.isdir(vtk_path):
                #         os.makedirs(vtk_path, exist_ok=True)
                #     vtk_mesh.cell_data = {}
                #     vtk_mesh.cell_data['f_sh_mean'] = [np.stack(face_f_sh_vec,
                #                                                 axis=1).sum(axis=1) / face_f_sh_vec.shape[0]]
                #
                #     meshio.vtk.write(os.path.join(vtk_path, f"f_sh_faces_mean.vtk"),
                #                      vtk_mesh,
                #                      binary=True)
                #
                # # ---------------------------------------------------------------------------------------------------------
                # # write xls
                # # ---------------------------------------------------------------------------------------------------------
                #
                # if bool(self.setup_component.WriteXLSX):
                #
                #     xlsx_path = os.path.join(self.setup_component.ExportDirectory, 'xls')
                #     if not os.path.isdir(xlsx_path):
                #         os.makedirs(xlsx_path, exist_ok=True)
                #
                #     with pd.ExcelWriter(os.path.join(xlsx_path, 'output.xlsx')) as writer:
                #
                #         workbook = writer.book
                #
                #         logger.info(f'Writing xlsx summary')
                #
                #         summary_df = pd.DataFrame(data={'Analysis ID': self.id.LocalId,
                #                                         'Analysis Name': self.setup_component.name,
                #                                         'North Angle': self.setup_component.NorthAngle,
                #                                         'Weather File': os.path.basename(self.setup_component.Weather.weather_file_name),
                #                                         'Mesh Size': self.setup_component.MeshSize,
                #                                         'Ray Resolution': self.setup_component.RayResolution,
                #                                         'Start Date': self.setup_component.StartDate,
                #                                         'Number of Timesteps': self.setup_component.NumTimesteps,
                #                                         'Timestep Size': self.setup_component.TimestepSize,
                #                                         'Timestep Unit': self.setup_component.TimestepUnit,
                #                                         'AddTerrain': self.setup_component.AddTerrain,
                #                                         'Terrain Height': self.setup_component.TerrainHeight,
                #                                         'Mesh': '',
                #                                         'Num Triangles': self.mesh.cells[0].data.shape[0]
                #                                         }, index=[0])
                #
                #         summary_df.T.to_excel(writer,
                #                               sheet_name='Summary',
                #                               index=True,
                #                               header=True,
                #                               startrow=1,
                #                               startcol=1)
                #
                #         writer.save()
                #
                #         # -------------------------------------------------------------------------------------------------
                #         # write irradiation vectors
                #         # -------------------------------------------------------------------------------------------------
                #         logger.info(f'Writing xlsx irradiation vectors')
                #
                #         irradiation_vectors_df = pd.DataFrame(data={'x': base_df['irradiation_vector'].apply(lambda x: x[0]),
                #                                                     'y': base_df['irradiation_vector'].apply(lambda x: x[1]),
                #                                                     'z': base_df['irradiation_vector'].apply(lambda x: x[2])})
                #
                #         irradiation_vectors_df.index = irradiation_vectors_df.index.tz_localize(None)
                #         irradiation_vectors_df.to_excel(writer,
                #                                         sheet_name='Irradiation Vectors',
                #                                         index=True,
                #                                         startrow=0,
                #                                         startcol=0)
                #
                #         writer.save()
                #
                #         # -------------------------------------------------------------------------------------------------
                #         # write face_f_sh
                #         # -------------------------------------------------------------------------------------------------
                #         logger.info(f'Writing xlsx Shading Factors')
                #
                #         face_f_sh.to_excel(writer,
                #                            sheet_name='Shading Factors',
                #                            index=True,
                #                            startrow=1,
                #                            startcol=0
                #                            )
                #
                #         worksheet = workbook['Shading Factors']
                #
                #         for i in range(self.scene.faces.__len__()):
                #             c1 = worksheet.cell(row=1, column=i+2)
                #             if self.scene.faces[i].components:
                #                 c1.value = self.scene.faces[i].components[0].name
                #             else:
                #                 c1.value = ''
                #
                #         writer.save()
                #
                #         # -------------------------------------------------------------------------------------------------
                #         # write face_f_sh_mean
                #         # -------------------------------------------------------------------------------------------------
                #         logger.info(f'Writing xlsx Mean Shading Factors')
                #
                #         face_f_sh.mean(axis=0).T.to_excel(writer,
                #                                           sheet_name='Mean Shading Factors',
                #                                           index=True,
                #                                           startrow=1,
                #                                           startcol=0
                #                                           )
                #         worksheet = workbook['Mean Shading Factors']
                #
                #         for i in range(self.scene.faces.__len__()):
                #             c1 = worksheet.cell(row=1, column=i + 2)
                #             if self.scene.faces[i].components:
                #                 c1.value = self.scene.faces[i].components[0].name
                #             else:
                #                 c1.value = ''
                #
                #         writer.save()
                #
                #         # -------------------------------------------------------------------------------------------------
                #         # write face_solar_irradiation q_dot
                #         # -------------------------------------------------------------------------------------------------
                #         logger.info(f'Writing xlsx Specific Irradiation')
                #
                #         face_q_dot.to_excel(writer,
                #                             sheet_name='Specific Irradiation',
                #                             index=True,
                #                             startrow=1,
                #                             startcol=0
                #                             )
                #
                #         worksheet = workbook['Specific Irradiation']
                #
                #         for i in range(self.scene.faces.__len__()):
                #             c1 = worksheet.cell(row=1, column=i + 2)
                #             if self.scene.faces[i].components:
                #                 c1.value = self.scene.faces[i].components[0].name
                #             else:
                #                 c1.value = ''
                #
                #         writer.save()
                #
                #         # -------------------------------------------------------------------------------------------------
                #         # write face_solar_irradiation Q_dot
                #         # -------------------------------------------------------------------------------------------------
                #         logger.info(f'Writing xlsx Absolute irradiation')
                #
                #         face_Q_dot.to_excel(writer,
                #                             sheet_name='Absolute irradiation',
                #                             index=True,
                #                             startrow=1,
                #                             startcol=0
                #                             )
                #
                #         worksheet = workbook['Absolute irradiation']
                #
                #         for i in range(self.scene.faces.__len__()):
                #             c1 = worksheet.cell(row=1, column=i + 2)
                #             if self.scene.faces[i].components:
                #                 c1.value = self.scene.faces[i].components[0].name
                #             else:
                #                 c1.value = ''
                #
                #         writer.save()
                #
                #         # -------------------------------------------------------------------------------------------------
                #         # write face_solar_irradiation Q
                #         # -------------------------------------------------------------------------------------------------
                #         logger.info(f'Writing xlsx Specific amount of heat')
                #
                #         face_Q.to_excel(writer,
                #                         sheet_name='Specific amount of heat',
                #                         index=True,
                #                         startrow=1,
                #                         startcol=0
                #                         )
                #
                #         worksheet = workbook['Specific amount of heat']
                #
                #         for i in range(self.scene.faces.__len__()):
                #             c1 = worksheet.cell(row=1, column=i + 2)
                #             if self.scene.faces[i].components:
                #                 c1.value = self.scene.faces[i].components[0].name
                #             else:
                #                 c1.value = ''
                #
                #         writer.save()

    def write_mesh_to_db(self, mesh):

        logger.info(f'Writing mesh to database...')

        return self.db_interface.save_object(mesh, 'mesh')
        #
        # with TemporaryFile() as t:
        #     mesh.write(t, file_format='vtk')
        #     t.seek(0, 0)
        #     blob = t.read()
        #
        # with self.db_service:
        #
        #     engine = create_engine(
        #         f'postgresql://{self.user_name}:{self.password}@localhost:{self.db_service.port}/{str(self.id.GlobalId)}')
        #     engine.dispose()
        #     # delete existing table:
        #     sql.execute('DROP TABLE IF EXISTS %s' % 'mesh', engine)
        #
        #     meta = MetaData(engine)
        #
        #     mesh_table = Table('mesh',
        #                        meta,
        #                        Column('id', UUID(as_uuid=True), primary_key=True),
        #                        Column('mesh', LargeBinary)
        #                        )
        #     meta.create_all(engine)
        #
        #     ins = mesh_table.insert().values(id=uuid.UUID(str(self.id.GlobalId)),
        #                                      mesh=blob)
        #     with engine.connect() as conn:
        #         result = conn.execute(ins)
        #
        #     logger.info(f'Mesh written to database')
        #
        #     return result

    def read_mesh_from_db(self):

        logger.info(f'Loading mesh from database...')

        with self.db_service:
            return self.db_interface.load_object('mesh')

        # with self.db_service:
        #
        #     engine = create_engine(
        #         f'postgresql://{self.user_name}:{self.password}@localhost:{self.db_service.port}/{str(self.id.GlobalId)}')
        #     meta = MetaData(engine)
        #     meta.reflect()
        #     if 'mesh' not in meta.tables.keys():
        #         logger.info(f'Could not find mesh in database')
        #         return
        #     mesh_table = meta.tables['mesh']
        #     sel = mesh_table.select().where(mesh_table.c.id == self.id.GlobalId.ToString())
        #     with engine.connect() as conn:
        #         result = conn.execute(sel).fetchone()
        #
        #     with NamedTemporaryFile(delete=False) as t:
        #         t.write(result.mesh)
        #     return meshio.read(t.name, file_format='vtk')

    def export_results(self):

        if not any([self.setup_component.run_configuration.WriteVTK, self.setup_component.run_configuration.WriteXLSX]):
            return

        self.db_service.keep_running = False
        export_directory = self.setup_component.run_configuration.ExportDirectory

        # if not any([self.setup_component.run_configuration.WriteVTK, self.setup_component.run_configuration.WriteXLSX]):
        #     return
        #
        # with self.db_service:
        #
        #     logger.info(f'Getting results from database')
        #
        #     engine = create_engine(
        #         f'postgresql://{self.user_name}:{self.password}@localhost:{self.db_service.port}/{str(self.id.GlobalId)}')
        #     engine.dispose()
        #
        #     base_df = self.db_interface.get_dataframe('base_df')
        #     # f_sh = self.db_interface.get_dataframe('f_sh')
        #
        #     # base_df = pd.read_sql_query(f"""select * from {'"'}{'base_df'}{'"'}""", con=engine, index_col='index').sort_values(
        #     #     by='index')
        #     #
        #     f_sh = pd.read_sql_query(f"""select * from {'"'}{'f_sh'}{'"'}""", con=engine, index_col='date').sort_values(
        #         by='date')
        #
        #     # ----------------------------------------------------------------------------------------------------------
        #     # create f_sh_for named faces:
        #     logger.info(f'Aggregating results')
        #     face_f_sh = pd.DataFrame(f_sh.index.values, columns=['date'])
        #     face_f_sh.set_index('date', inplace=True)
        #     tri_mesh = Trimesh(vertices=self.mesh.points,
        #                        faces=self.mesh.cells_dict['triangle'])
        #
        #     areas = tri_mesh.area_faces
        #
        #     # face_names = dict(zip([x.id for x in self.scene.faces if x.components],
        #     #                       [x.components[0].name for x in self.scene.faces if x.components]))
        #
        #     # dni: direct normal irradiation from weather data:
        #     dni = self.location.data['dni']
        #     # remove localization of data
        #     dni.index = dni.index.tz_localize(None)
        #     # replace the year of the required timestamps with 2021 -> the year with which weather data is loaded
        #     req_timestamps = pd.Series([x.replace(year=2021) for x in f_sh.index])
        #
        #     # dni_req_ts: direct normal irradiation from weather data at requested timesteps:
        #     dni_req_ts = df_interpolate_at(dni, req_timestamps, method='linear', axis='index')
        #     # write dni to database
        #     try:
        #         # write_df_in_empty_table(dni_req_ts, 'dni', engine)
        #         self.db_interface.save_dataframe(dni_req_ts, 'dni')
        #     except Exception as e:
        #         logger.error(f'Error writing dni to database:\n{e}\n{traceback.format_exc()}\n{sys.exc_info()[2]}')
        #
        #     # for every face calculate the mean f_sh
        #
        #     face_areas = pd.DataFrame(index=[0])
        #     # for key, value in self.mesh.cell_sets.items():
        #     f_sh_mat = np.vstack(f_sh['f_sh'].values)
        #     for key, value in tqdm(self.mesh.cell_sets.items(),
        #                            total=len(self.mesh.cell_sets),
        #                            colour='green',
        #                            desc="Aggregating results for faces"):
        #         f_areas = areas[value[1]]
        #         f_areas_sum = sum(areas[value[1]])
        #         face_areas[key] = f_areas_sum
        #
        #         face_f_sh[key] = np.sum(f_sh_mat[:, value[1]] * f_areas, axis=1) / f_areas_sum
        #
        #     logger.info(f'Writing aggregated results to database')
        #     # write to database:
        #     try:
        #         self.db_interface.save_dataframe(face_areas, 'face_areas')
        #         self.db_interface.save_dataframe(face_f_sh, 'face_f_sh')
        #         # write_df_in_empty_table(face_areas, 'face_areas', engine, index=False)
        #         # write_df_in_empty_table(face_f_sh, 'face_f_sh', engine)
        #     except Exception as e:
        #         logger.error(f'Error writing face_areas, face_f_sh to database:\n{e}\n{traceback.format_exc()}\n{sys.exc_info()[2]}')
        #
        #     # calculate specific irradiation:
        #     face_q_dot = face_f_sh.multiply(dni_req_ts, axis=0)
        #     try:
        #         self.db_interface.save_dataframe(face_q_dot, 'face_q_dot')
        #         # write_df_in_empty_table(face_q_dot, 'face_q_dot', engine)
        #     except Exception as e:
        #         logger.error(f'Error writing face_q_dot to database:\n{e}\n{traceback.format_exc()}\n{sys.exc_info()[2]}')
        #
        #     # calculate total irradiation:
        #     face_Q_dot = pd.DataFrame(index=face_q_dot.index)
        #     for column in face_q_dot.columns:
        #         face_Q_dot[column] = face_q_dot[column].multiply(face_areas[column][0], axis=0)
        #     try:
        #         self.db_interface.save_dataframe(face_Q_dot, 'face_q_tot_dot')
        #         # write_df_in_empty_table(face_Q_dot, 'face_q_tot_dot', engine)
        #     except Exception as e:
        #         logger.error(f'Error writing face_q_tot_dot to database:\n{e}\n{traceback.format_exc()}\n{sys.exc_info()[2]}')
        #
        #     # calculate irradiated amount of heat:
        #     from scipy import integrate
        #     face_Q = pd.DataFrame(integrate.cumtrapz(face_Q_dot.values,
        #                                              (face_Q_dot.index.asi8 - face_Q_dot.index.asi8[0]) * 1e-9, axis=0),
        #                           index=face_Q_dot.index[1:],
        #                           columns=face_Q_dot.columns)
        #     try:
        #         self.db_interface.save_dataframe(face_Q, 'face_Q')
        #         # write_df_in_empty_table(face_Q, 'face_Q', engine)
        #     except Exception as e:
        #         logger.error(f'Error writing face_Q to database:\n{e}\n{traceback.format_exc()}\n{sys.exc_info()[2]}')

        face_f_sh = None
        face_q_dot = None
        face_q_tot_dot = None
        face_Q = None
        base_df = None
        dni_req_ts = None
        aois = None

        if bool(self.setup_component.run_configuration.WriteVTK):

            vtk_mesh = copy.deepcopy(self.mesh)
            vtk_mesh.cell_data = {}

            logger.info(f'Writing .vtk files')

            # ------------------------------------------------------------------------------------------------------

            logger.info(f'Writing vtk raw mesh results')
            vtk_path = os.path.join(export_directory, 'vtk', 'raw')

            # load required results
            f_sh = pd.read_sql_query(f"""select * from {'"'}{'f_sh'}{'"'}""",
                                     con=self.db_interface.engine,
                                     index_col='date').sort_values(
                by='date')

            with self.db_service:
                if dni_req_ts is None:
                    dni_req_ts = self.db_interface.get_dataframe('dni')
                if face_f_sh is None:
                    face_f_sh = self.db_interface.get_dataframe('face_f_sh')
                if base_df is None:
                    base_df = self.db_interface.get_dataframe('base_df')

            tri_mesh = Trimesh(vertices=self.mesh.points,
                               faces=self.mesh.cells_dict['triangle'])

            areas = tri_mesh.area_faces

            f_sh_array = np.vstack(f_sh['f_sh'].values)
            q_dot = np.multiply(f_sh_array, dni_req_ts.values[:, np.newaxis])
            Q_dot = q_dot * areas
            Q = np.zeros(q_dot.shape)
            Q[1:, :] = integrate.cumtrapz(q_dot, (f_sh.index.asi8 - f_sh.index.asi8[0]) * 1e-9, axis=0)

            if not os.path.isdir(vtk_path):
                os.makedirs(vtk_path, exist_ok=True)
            for i, (index, row) in enumerate(tqdm(f_sh.iterrows(),
                                                  total=f_sh.shape[0],
                                                  colour='green',
                                                  desc="Writing raw mesh results")):
                # for i, (index, row) in enumerate(f_sh.iterrows()):
                if f_sh.iloc[i]['irradiation_vector'][2] > 0:
                    continue
                vtk_mesh.cell_data['f_sh'] = [f_sh_array[i, :]]
                vtk_mesh.cell_data['q_dot'] = [q_dot[i, :]]
                vtk_mesh.cell_data['Q_dot'] = [Q_dot[i, :]]
                vtk_mesh.cell_data['Q'] = [Q[i, :]]
                meshio.vtk.write(os.path.join(vtk_path, f"shading_{index.strftime('%Y%m%d_%H%M%S')}.vtk"),
                                 vtk_mesh,
                                 binary=True)

            # ------------------------------------------------------------------------------------------------------

            logger.info(f'Writing vtk f_sh_mean')
            vtk_path = os.path.join(export_directory, 'vtk')
            vtk_mesh.cell_data = {}
            vtk_mesh.cell_data['f_sh_mean'] = [np.stack(f_sh['f_sh'].values,
                                                        axis=1).sum(axis=1) / f_sh['f_sh'].__len__()]

            meshio.vtk.write(os.path.join(vtk_path, f"f_sh_mean.vtk"),
                             vtk_mesh,
                             binary=True)

            # ------------------------------------------------------------------------------------------------------

            logger.info(f'Writing vtk f_sh_faces')
            vtk_mesh.cell_data = {}
            vtk_path = os.path.join(export_directory, 'vtk', 'face_f_sh')
            if not os.path.isdir(vtk_path):
                os.makedirs(vtk_path, exist_ok=True)

            face_f_sh_vec = np.zeros([face_f_sh.shape[0], self.mesh.cells[0].data.shape[0]])
            face_q_dot_vec = np.zeros([face_f_sh.shape[0], self.mesh.cells[0].data.shape[0]])
            face_Q_dot_vec = np.zeros([face_f_sh.shape[0], self.mesh.cells[0].data.shape[0]])
            face_Q_vec = np.zeros([face_f_sh.shape[0], self.mesh.cells[0].data.shape[0]])
            # for key, value in self.mesh.cell_sets.items():
            for key, value in tqdm(self.mesh.cell_sets.items(),
                                   total=f_sh.shape[0],
                                   colour='green',
                                   desc="Writing f_sh for faces"):
                values = np.array(face_f_sh[key])

                cell_ids = self.mesh.cell_sets[key][1]

                face_f_sh_vec[:, cell_ids] = np.broadcast_to(values, (cell_ids.shape[0], values.shape[0])).T
                face_q_dot_vec[:, cell_ids] = np.multiply(face_f_sh_vec[:, cell_ids],
                                                          dni_req_ts.values[:, np.newaxis])
                face_Q_dot_vec[:, cell_ids] = face_q_dot_vec[:, cell_ids] * areas[cell_ids]
                face_Q_vec[1:, cell_ids] = integrate.cumtrapz(face_q_dot_vec[:, cell_ids],
                                                              (f_sh.index.asi8 - f_sh.index.asi8[0]) * 1e-9,
                                                              axis=0)

            # for i in range(face_f_sh.shape[0]):
            for i in trange(face_f_sh.shape[0],
                            total=f_sh.shape[0],
                            colour='green',
                            desc="Writing irradiation vector vtks"):

                if f_sh['irradiation_vector'].iloc[i][2] > 0:
                    continue
                vtk_mesh.cell_data['face_f_sh'] = [face_f_sh_vec[i, :]]
                vtk_mesh.cell_data['face_q_dot'] = [face_q_dot_vec[i, :]]
                vtk_mesh.cell_data['face_Q_dot'] = [face_Q_dot_vec[i, :]]
                vtk_mesh.cell_data['face_Q'] = [face_Q_vec[i, :]]
                meshio.vtk.write(os.path.join(vtk_path,
                                              f"shading_{face_f_sh.index[i].strftime('%Y%m%d_%H%M%S')}.vtk"
                                              ),
                                 vtk_mesh,
                                 binary=True)

            # ------------------------------------------------------------------------------------------------------

            logger.info(f'Writing vtk f_sh_faces_mean')
            vtk_path = os.path.join(export_directory, 'vtk')
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

        if bool(self.setup_component.run_configuration.WriteXLSX):

            xlsx_path = os.path.join(export_directory, 'xls')
            if not os.path.isdir(xlsx_path):
                os.makedirs(xlsx_path, exist_ok=True)

            result_export = self.setup_component.simulation_setup.ShadingSetup.ResultExport

            with self.db_service:

                # if self.setup_component.simulation_setup.ShadingSetup.WriteAoI:
                #     aois = self.db_interface.get_dataframe('angle of incidence')

                with pd.ExcelWriter(os.path.join(xlsx_path, 'output.xlsx')) as writer:

                    workbook = writer.book

                    if result_export.WriteSummary:

                        logger.info(f'Writing xlsx summary')

                        summary_df = pd.DataFrame(data={'Analysis ID': self.id.LocalId,
                                                        'Analysis Name': self.setup_component.name,
                                                        'North Angle': self.setup_component.location.NorthAngle,
                                                        'Weather File': os.path.basename(
                                                            self.setup_component.location.Weather.weather_file_name),
                                                        'Mesh Size': self.setup_component.simulation_setup.ShadingSetup.MeshSize,
                                                        'Ray Resolution': self.setup_component.simulation_setup.ShadingSetup.RayResolution,
                                                        'Start Date': self.setup_component.simulation_setup.ShadingSetup.StartDate,
                                                        'Number of Timesteps': self.setup_component.simulation_setup.ShadingSetup.NumTimesteps,
                                                        'Timestep Size': self.setup_component.simulation_setup.ShadingSetup.TimestepSize,
                                                        'Timestep Unit': self.setup_component.simulation_setup.ShadingSetup.TimestepUnit,
                                                        'AddTerrain': self.setup_component.geometry.AddTerrain,
                                                        'Terrain Height': self.setup_component.geometry.TerrainHeight,
                                                        'Mesh': '',
                                                        'Num Triangles': self.mesh.cells[0].data.shape[0]
                                                        }, index=[0])

                        summary_df.T.to_excel(writer,
                                              sheet_name='Summary',
                                              index=True,
                                              header=True,
                                              startrow=1,
                                              startcol=1)

                        writer.save()

                    # write irradiation vectors
                    if result_export.WriteIrradiationVectors:
                        logger.info(f'Writing xlsx irradiation vectors')
                        if base_df is None:
                            base_df = self.db_interface.get_dataframe('base_df')

                        irradiation_vectors_df = pd.DataFrame(data={'x': base_df['irradiation_vector'].apply(lambda x: x[0]),
                                                                    'y': base_df['irradiation_vector'].apply(lambda x: x[1]),
                                                                    'z': base_df['irradiation_vector'].apply(lambda x: x[2])})

                        irradiation_vectors_df.index = irradiation_vectors_df.index.tz_localize(None)
                        irradiation_vectors_df.to_excel(writer,
                                                        sheet_name='Irradiation Vectors',
                                                        index=True,
                                                        startrow=0,
                                                        startcol=0)

                        writer.save()

                    # angle of incidence
                    if result_export.WriteAngleOfIncidence:
                        logger.info(f'Writing xlsx AngleOfIncidence')
                        if aois is None:
                            aois = self.db_interface.get_dataframe('AngleOfIncidence')
                        write_face_results(aois,
                                           'AngleOfIncidence',
                                           writer,
                                           workbook,
                                           self.geo_model.FaceCls)

                    # write face_f_sh
                    if result_export.WriteShadingFactors:
                        logger.info(f'Writing xlsx Shading Factors')
                        if face_f_sh is None:
                            face_f_sh = self.db_interface.get_dataframe('face_f_sh')
                        write_face_results(face_f_sh,
                                           'ShadingFactors',
                                           writer,
                                           workbook,
                                           self.geo_model.FaceCls)

                    # write face_f_sh_mean
                    if result_export.WriteMeanShadingFactors:
                        logger.info(f'Writing xlsx MeanShadingFactors')
                        write_face_results(face_f_sh.mean(axis=0),
                                           'MeanShadingFactors',
                                           writer,
                                           workbook,
                                           self.geo_model.FaceCls)

                    # write face_solar_irradiation q_dot
                    if result_export.WriteMeanShadingFactors:

                        logger.info(f'Writing xlsx Specific Irradiation')
                        if face_q_dot is None:
                            face_q_dot = self.db_interface.get_dataframe('face_q_dot')
                        write_face_results(face_q_dot,
                                           'SpecificIrradiation',
                                           writer,
                                           workbook,
                                           self.geo_model.FaceCls)

                    # write face_solar_irradiation Q_dot
                    if result_export.WriteAbsoluteIrradiation:
                        logger.info(f'Writing xlsx AbsoluteIrradiation')
                        if face_q_tot_dot is None:
                            face_q_tot_dot = self.db_interface.get_dataframe('face_q_tot_dot')
                        write_face_results(face_q_tot_dot,
                                           'AbsoluteIrradiation',
                                           writer,
                                           workbook,
                                           self.geo_model.FaceCls)

                    # write face_solar_irradiation Q
                    if result_export.WriteIrradiatedAmountOfHeat:
                        logger.info(f'Writing xlsx IrradiatedAmountOfHeat')
                        if face_Q is None:
                            face_Q = self.db_interface.get_dataframe('face_Q')
                        write_face_results(face_Q,
                                           'IrradiatedAmountOfHeat',
                                           writer,
                                           workbook,
                                           self.geo_model.FaceCls)

    def generate_hull_mesh(self):

        hull_mesh = MioMesh(points=self.mesh.points,
                            cells=[("triangle",
                                    self.mesh.cells_dict['triangle'][np.where(self.mesh.cell_data['hull_face'][0]), :][0])
                                   ])
        return hull_mesh

    def evaluate_shading_results(self):

        with self.db_service:

            logger.info(f'Getting results from database')

            engine = create_engine(
                f'postgresql://{self.user_name}:{self.password}@localhost:{self.db_service.port}/{str(self.id.GlobalId)}')
            engine.dispose()

            f_sh = pd.read_sql_query(f"""select * from {'"'}{'f_sh'}{'"'}""", con=engine, index_col='date').sort_values(
                by='date')

            # ----------------------------------------------------------------------------------------------------------
            # create f_sh_for named faces:
            # ----------------------------------------------------------------------------------------------------------
            logger.info(f'Aggregating results')
            face_f_sh = pd.DataFrame(f_sh.index.values, columns=['date'])
            face_f_sh.set_index('date', inplace=True)
            tri_mesh = Trimesh(vertices=self.mesh.points,
                               faces=self.mesh.cells_dict['triangle'])

            areas = tri_mesh.area_faces

            # dni: direct normal irradiation from weather data:
            dni = self.location.data['dni']
            # remove localization of data
            dni.index = dni.index.tz_localize(None)
            # replace the year of the required timestamps with 2021 -> the year with which weather data is loaded
            req_timestamps = pd.Series([x.replace(year=2021) for x in f_sh.index])

            # dni_req_ts: direct normal irradiation from weather data at requested timesteps:
            dni_req_ts = df_interpolate_at(dni, req_timestamps, method='linear', axis='index')
            # write dni to database
            try:
                # write_df_in_empty_table(dni_req_ts, 'dni', engine)
                self.db_interface.save_dataframe(dni_req_ts, 'dni')
                # self.db_interface.get_dataframe('dni')
            except Exception as e:
                logger.error(f'Error writing dni to database:\n{e}\n{traceback.format_exc()}\n{sys.exc_info()[2]}')

            # for every face calculate the mean f_sh

            face_areas = pd.DataFrame(index=[0])
            # for key, value in self.mesh.cell_sets.items():
            f_sh_mat = np.vstack(f_sh['f_sh'].values)
            for key, value in tqdm(self.mesh.cell_sets.items(),
                                   total=len(self.mesh.cell_sets),
                                   colour='green',
                                   desc="Aggregating results for faces"):
                f_areas = areas[value[1]]
                f_areas_sum = sum(areas[value[1]])
                face_areas[key] = f_areas_sum

                face_f_sh[key] = np.sum(f_sh_mat[:, value[1]] * f_areas, axis=1) / f_areas_sum

            logger.info(f'Writing aggregated results to database')
            # write to database:
            try:
                self.db_interface.save_dataframe(face_areas, 'face_areas')
                self.db_interface.save_dataframe(face_f_sh, 'face_f_sh')
            except Exception as e:
                logger.error(
                    f'Error writing face_areas, face_f_sh to database:\n{e}\n{traceback.format_exc()}\n{sys.exc_info()[2]}')

            # calculate specific irradiation:
            face_q_dot = face_f_sh.multiply(dni_req_ts, axis=0)
            try:
                self.db_interface.save_dataframe(face_q_dot, 'face_q_dot')
            except Exception as e:
                logger.error(
                    f'Error writing face_q_dot to database:\n{e}\n{traceback.format_exc()}\n{sys.exc_info()[2]}')

            # calculate total irradiation:
            face_Q_dot = pd.DataFrame(index=face_q_dot.index)
            for column in face_q_dot.columns:
                face_Q_dot[column] = face_q_dot[column].multiply(face_areas[column][0], axis=0)
            try:
                self.db_interface.save_dataframe(face_Q_dot, 'face_q_tot_dot')
            except Exception as e:
                logger.error(
                    f'Error writing face_q_tot_dot to database:\n{e}\n{traceback.format_exc()}\n{sys.exc_info()[2]}')

            # calculate irradiated amount of heat:
            face_Q = pd.DataFrame(integrate.cumtrapz(face_Q_dot.values,
                                                     (face_Q_dot.index.asi8 - face_Q_dot.index.asi8[0]) * 1e-9, axis=0),
                                  index=face_Q_dot.index[1:],
                                  columns=face_Q_dot.columns)
            try:
                self.db_interface.save_dataframe(face_Q, 'face_Q')
            except Exception as e:
                logger.error(f'Error writing face_Q to database:\n{e}\n{traceback.format_exc()}\n{sys.exc_info()[2]}')


class RTEngine(object):

    def __init__(self, *args, **kwargs):

        self.client = None
        self.db_engine = None
        self.conn = None

        self.port = kwargs.get('port')
        # self.db_engine = kwargs.get('db_engine')
        self.sample_dist = kwargs.get('sample_dist')
        self.num_cells = kwargs.get('num_cells')
        self.tablename = kwargs.get('tablename')

        self.user_name = kwargs.get('user_name')
        self.password = kwargs.get('password')
        self.db_port = kwargs.get('db_port')

        self.id = kwargs.get('id')  # str(self.id.GlobalId)
        self.f_sh_table = kwargs.get('f_sh_table')

    def __call__(self, *args, **kwargs):

        date = args[0][0]
        df_row = args[0][1]

        if self.db_engine is None:
            self.engine = create_engine(
                f'postgresql://{self.user_name}:{self.password}@localhost:{self.db_port}/{self.id}')
        if self.client is None:
            self.client = Client(ip=f'tcp://localhost:{self.port}')

        if self.conn is None:
            self.conn = self.engine.connect()

        irradiation_vector = df_row['irradiation_vector']
        sun_window = df_row['windows']

        f_sh = np.zeros([self.num_cells])

        if irradiation_vector[2] < 0:
            rt_start_time = time.time()
            try:
                count = self.client.rt_sun_window(scene='hull',
                                                  sun_window=sun_window,
                                                  sample_dist=self.sample_dist,
                                                  irradiation_vector=irradiation_vector)
            except Exception as e:
                logger.error(f'Error calling shading service:\n{e}\n{traceback.format_exc()}\n{sys.exc_info()[2]}')

            f_sh[0:count.shape[0]] = count
            rt_end_time = time.time()
            # logger.info(f'RayTracing needed: {rt_end_time - rt_start_time}')

        # write to database
        # df0 = pd.DataFrame({'irradiation_vector': [irradiation_vector.tolist()],
        #                     'f_sh': [f_sh.tolist()]},
        #                    index=[date])
        ins = self.f_sh_table.insert().values(date=date,
                                              irradiation_vector=irradiation_vector.tolist(),
                                              f_sh=f_sh.tolist())
        result = self.conn.execute(ins)

        # # logger.info(f'writing results for timestep: {date}')
        # df0.to_sql(self.tablename,
        #            self.db_engine,
        #            if_exists='append',
        #            index=True,
        #            dtype={'date': sqlalchemy.TIMESTAMP(),
        #                   'irradiation_vector': postgresql.ARRAY(sqlalchemy.types.FLOAT),
        #                   'f_sh': postgresql.ARRAY(sqlalchemy.types.FLOAT)
        #                   }
        #            )


def calc_timestep_async(port=None,
                        db_engine=None,
                        sun_window=None,
                        sample_dist=None,
                        irradiation_vector=None,
                        num_cells=None,
                        date=None,
                        tablename=None,
                        pbar=None,
                        process=None):

    # start_time = time.time()
    client = Client(ip=f'tcp://localhost:{port}')

    # logger.info(f'processing timestep: {date}')

    f_sh = np.zeros([num_cells])

    if irradiation_vector[2] < 0:
        # rt_start_time = time.time()
        count = client.rt_sun_window(scene='hull',
                                     sun_window=sun_window,
                                     sample_dist=sample_dist,
                                     irradiation_vector=irradiation_vector)
        f_sh[0:count.shape[0]] = count
        # rt_end_time = time.time()
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

    # end_time = time.time()
    # logger.info(f'processing needed: {end_time - start_time}')
