import uuid

from . import logger
import tempfile
from ..config import config
import subprocess
import time
from appdirs import user_data_dir
from pathlib import Path
import os
import psycopg2
import docker
from sqlalchemy import create_engine

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 importlib_resources.
    import importlib_resources as pkg_resources


from . import resources

with pkg_resources.path(resources, 'docker-compose_shading.yml') as r_path:
    compose_template_filename = str(r_path)

with pkg_resources.path(resources, 'docker-compose_db.yml') as r_path:
    db_compose_template_filename = str(r_path)


class DatabaseService(object):

    def __init__(self, *args, **kwargs):

        self._shared_dir = None
        self._engine = None
        # self._workdir = kwargs.get('workdir', None)

        self.port = kwargs.get('port', 9006)
        self.user = kwargs.get('user', 'admin')
        self.password = kwargs.get('password', 'admin')
        self.db_name = kwargs.get('db_name', 'shading')

        self.log_dir = kwargs.get('log_dir', 'logs')
        self.logging_mode = kwargs.get('logging_mode', 'DEBUG')

        self._db_compose_file_name = None

        self.running = False

        self.keep_running = kwargs.get('keep_running', False)

        self._volume = None
        self.docker_client = docker.from_env()
        self.persist_volume = kwargs.get('persist_volume', False)

    @property
    def engine(self):
        if self._engine is None:
            engine = create_engine(
                f'postgresql://{self.user}:{self.password}@localhost:{self.port}/{self.db_name}',
                pool_pre_ping=True)
            engine.dispose()
            self._engine = engine
        return self._engine

    @property
    def volume(self):
        if self._volume is None:
            volume = next((x for x in self.docker_client.volumes.list() if x.name == self.user), None)
            if volume is None:
                logger.info(f'Creating database volume...')
                self._volume = self.docker_client.volumes.create(name=self.user, driver='local')
                logger.info(f'Created database volume {self._volume.name}')
            else:
                self._volume = volume
        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume = value

    # @property
    # def workdir(self):
    #
    #     if self._workdir is None:
    #         # check if directory exist, otherwise create
    #         # data_path = user_data_dir('PySimultanRadiation', 'TU_Wien')
    #         data_path = db_path
    #         workdir = Path(data_path, 'db', str(self.user))
    #         workdir.mkdir(parents=True, exist_ok=True)
    #         self._workdir = str(workdir)
    #     return self._workdir

    @property
    def shared_dir(self):
        if self._shared_dir is None:
            self._shared_dir = tempfile.TemporaryDirectory()
            time.sleep(1)
        return self._shared_dir

    @shared_dir.setter
    def shared_dir(self, value):
        self._shared_dir = value

    @property
    def db_compose_file_name(self):
        if self._db_compose_file_name is None:
            self._db_compose_file_name = os.path.join(self.shared_dir.name, 'db_docker_compose.yml')
        return self._db_compose_file_name

    @db_compose_file_name.setter
    def db_compose_file_name(self, value):
        self._db_compose_file_name = value

    def write_db_compose_file(self, filename):

        with open(db_compose_template_filename, 'r') as f:
            compose_template = f.read()

        compose_template = compose_template.replace('<DB_BIND_VOLUME>', self.volume.name)
        compose_template = compose_template.replace('<DB_PORT>', str(self.port))
        compose_template = compose_template.replace('<UserDBName>', str(self.user))
        compose_template = compose_template.replace('<UserDBPassword>', str(self.password))
        compose_template = compose_template.replace('<DBName>', self.db_name)

        with open(filename, 'wt') as f_out:
            f_out.write(compose_template)

    def shut_down_db_service(self):

        folder_name = os.path.basename(os.path.normpath(self.shared_dir.name))
        logger.info(f'Shutting down db compose {folder_name} on port {self.port}...')

        cmd = config.docker_path + ' compose ' + f" -f {self.db_compose_file_name} -p {folder_name} down --remove-orphans"
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.returncode:
            logger.error(f"Command Result: {result.stderr}")
        else:
            logger.debug(f"Command Result: {result.stderr}")
            logger.info(f"Database service successfully shut down")

            self.shared_dir.cleanup()
            self.shared_dir = None
            self.db_compose_file_name = None
            self.running = False

        if not self.persist_volume:
            if self.volume is not None:
                logger.info(f'Removing volume {self.volume.name}')
                self.volume.remove()
                self.volume = None

    def start_database_service(self):
        logger.info(f"Starting database container on port: {self.port}...")
        self.write_db_compose_file(self.db_compose_file_name)

        annot_mark = '\"'
        cmd = config.docker_path + ' compose ' + "-f " + f"{annot_mark}{self.db_compose_file_name}{annot_mark}" + " up" + " -d"
        result = subprocess.run(cmd,
                                capture_output=True,
                                text=True,
                                shell=True)
        if result.returncode:
            logger.error(f"Command Result: {result.stderr}\nfor \n\n{cmd}")
            self.shut_down_db_service()
        else:
            logger.debug(f"Command Result: {result.stderr}")
            logger.info(f"Database service successfully started")
            self.running = True

        conn_ok = self.test_connection()
        start_time = time.time()
        while (not conn_ok) and ((time.time() - start_time) < 10):
            conn_ok = self.test_connection()
        if not conn_ok and ((time.time() - start_time) > 10):
            raise Exception(f'Could not connect to database')

    def __enter__(self):
        if not self.running:
            self.start_database_service()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.running and not self.keep_running:
            self.shut_down_db_service()

    def __del__(self):
        if self.running:
            self.shut_down_db_service()

    def test_connection(self):

        try:
            conn = self.get_connection({'connect_timeout': 1})
            # conn = psycopg2.connect(dbname=self.db_name,
            #                         user=self.user,
            #                         host='localhost',
            #                         port=self.port,
            #                         password=self.password,
            #                         connect_timeout=1)
            conn.close()
            return True
        except:
            return False

    def get_connection(self, kwargs):
        conn = psycopg2.connect(dbname=self.db_name,
                                user=self.user,
                                host='localhost',
                                port=self.port,
                                password=self.password, **kwargs)
        return conn


class ShadingService(object):

    def __init__(self, *args, **kwargs):

        self.port = kwargs.get('port')
        # self.workdir = kwargs.get('workdir')
        self.num_workers = kwargs.get('num_workers')

        self.frontend_port = kwargs.get('frontend_port', 8006)
        self.backend_port = kwargs.get('backend_port', 9006)

        self.log_dir = kwargs.get('log_dir', 'logs')
        self.logging_mode = kwargs.get('logging_mode', 'DEBUG')

        self._shared_dir = None
        self._compose_file_name = None

        self.running = False

        self._volume = None
        self.docker_client = docker.from_env()

    @property
    def volume(self):
        if self._volume is None:
            logger.info(f'Creating client volume...')
            self._volume = self.docker_client.volumes.create(name=str(uuid.uuid4()), driver='local')
            logger.info(f'Created client volume {self._volume.name}')
        return self._volume

    @volume.setter
    def volume(self, value):
        self._volume = value

    @property
    def shared_dir(self):
        if self._shared_dir is None:
            self._shared_dir = tempfile.TemporaryDirectory()
            time.sleep(1)
        return self._shared_dir

    @shared_dir.setter
    def shared_dir(self, value):
        self._shared_dir = value

    @property
    def compose_file_name(self):
        if self._compose_file_name is None:
            self._compose_file_name = os.path.join(self.shared_dir.name, 'docker_compose.yml')
        return self._compose_file_name

    @compose_file_name.setter
    def compose_file_name(self, value):
        self._compose_file_name = value

    def write_compose_file(self, filename):

        with open(compose_template_filename, 'r') as f:
            compose_template = f.read()

        compose_template = compose_template.replace('<port>', str(self.port))
        compose_template = compose_template.replace('<FRONTEND_PORT>', str(self.frontend_port))
        compose_template = compose_template.replace('<BACKEND_PORT>', str(self.backend_port))

        compose_template = compose_template.replace('<LOG_DIR>', self.log_dir)
        compose_template = compose_template.replace('<SERVER_LOG_MODE>', self.logging_mode)

        compose_template = compose_template.replace('<BIND_VOLUME>', self.volume.name)

        compose_template = compose_template.replace('<WORKER_LOG_MODE>', self.logging_mode)

        compose_template = compose_template.replace('<NUM_WORKERS>', str(self.num_workers))

        # database
        # compose_template = compose_template.replace('<DB_BIND_VOLUME>', str(self.db_dir))
        # compose_template = compose_template.replace('<DB_PORT>', str(self.db_port))
        # compose_template = compose_template.replace('<UserDBName>', str(self.user))
        # compose_template = compose_template.replace('<UserDBPassword>', str(self.password))
        # compose_template = compose_template.replace('<DBName>', self.db_name)

        with open(filename, 'wt') as f_out:
            f_out.write(compose_template)

    def start_service(self):

        logger.info(f"Starting shading service with {self.num_workers} workers on port: {self.frontend_port}...")
        self.write_compose_file(self.compose_file_name)

        cmd = config.docker_path + ' compose ' + '-f ' + self.compose_file_name + ' up -d'
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.returncode:
            logger.error(f"Command Result: {result.stderr}\nfor \n\n{cmd}")
            self.shut_down_service()
        else:
            logger.debug(f"Command Result: {result.stderr}")
            logger.info(f"Shading service successfully started")
            self.running = True

    def shut_down_service(self):

        folder_name = os.path.basename(os.path.normpath(self.shared_dir.name))
        logger.info(f'Shutting down shading compose {folder_name} on port {self.frontend_port}...')

        cmd = config.docker_path + ' compose ' + f'-f {self.compose_file_name} -p {folder_name} down --remove-orphans'
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.returncode:
            logger.error(f"Command Result: {result.stderr}")
        else:
            logger.debug(f"Command Result: {result.stderr}")
            logger.info(f"Shading service successfully shut down")
            self.running = False

            self.shared_dir.cleanup()
            self.shared_dir = None
            self.compose_file_name = None

        if self.volume is not None:
            logger.info(f'Removing volume {self.volume.name}')
            try:
                self.volume.remove()
                self.volume = None
                logger.info(f'Volume removed')
            except Exception as e:
                logger.error(f'Error removing volume {self.volume.name}:\n{e}')

    def __enter__(self):

        if not self.running:
            self.start_service()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.running:
            self.shut_down_service()

    def __del__(self):
        if self.running:
            self.shut_down_service()

#
# class Worker(object):
#
#     def __init__(self, *args, **kwargs):
#
#         self.frontend_port = kwargs.get('frontend_port', 8006)
#         self.backend_port = kwargs.get('backend_port', 9006)
#
#         self.name = kwargs.get('name', 'unnamed_worker')                #
#         self.image = kwargs.get('image')                                # example: 'maxxiking/shading_worker:1.0.1'
#         self.bind_port = kwargs.get('bind_port')                        # example: 9006
#
#         self.bind_to_ip = kwargs.get('bind_to_ip', None)                # ip-address of the server to connect to
#         self.log_dir = kwargs.get('log_dir', None)                      # directory where logs are 'logs'
#         self.logging_mode = kwargs.get('logging_mode: ', 'INFO')        # logging mode of the worker
#         self.env_vars = kwargs.get('env_vars: ', {})                    # environment variables of the worker; {'WORKER_ID': '1'}
#         self.container_name = kwargs.get('container_name: ', None)      # container_name
#         self.network = kwargs.get('network', None)
#
#         self.mounts = kwargs.get('mounts', [])                          # volumes mounted to the container
#
#         self.server = kwargs.get('server', None)
#
#
# class Server(object):
#
#     def __init__(self, *args, **kwargs):
#
#         self.frontend_port = kwargs.get('frontend_port', 8006)
#         self.backend_port = kwargs.get('backend_port', 9006)
#
#         self.name = kwargs.get('name', 'unnamed_worker')  #
#         self.image = kwargs.get('image')  # example: 'maxxiking/shading_worker:1.0.1'
#         self.bind_port = kwargs.get('bind_port')  # example: 9006
#
#         self.bind_to_ip = kwargs.get('bind_to_ip', None)  # ip-address of the server to connect to
#         self.log_dir = kwargs.get('log_dir', None)  # directory where logs are 'logs'
#         self.logging_mode = kwargs.get('logging_mode: ', 'INFO')  # logging mode of the worker
#         self.env_vars = kwargs.get('env_vars: ', {})  # environment variables of the worker; {'WORKER_ID': '1'}
#         self.container_name = kwargs.get('container_name: ', None)  # container_name
#         self.network = kwargs.get('network', None)
#
#         self.mounts = kwargs.get('mounts', [])  # volumes mounted to the container
#
#
# class Service(object):
#
#     def __init__(self, *args, **kwargs):
#
#         self.worker = kwargs.get('worker', None)
#         self.server = kwargs.get('server', None)
#
#         self.num_workers = kwargs.get('num_workers', None)
#         self.network_name = kwargs.get('network_name',
#                                        'shading_net' + str(uuid.uuid4().hex[:8])
#                                        )
#
#     @property
#     def needed_images(self):
#         return [self.server_image, self.worker_image]
#
#
# class DockerManager(object):
#
#     def __init__(self, *args, **kwargs):
#         self.client = docker.from_env()
#         self.shared_dir = tempfile.TemporaryDirectory()
#
#         self.own_containers = set()
#         self.own_networks = set()
#
#         self.worker = kwargs.get('worker')
#         self.server = kwargs.get('server')
#
#     @property
#     def networks(self):
#         return self.client.networks.list(greedy=True)
#
#     @property
#     def network_names(self):
#         return [x.name for x in self.networks]
#
#     @property
#     def images(self):
#         return self.client.images.list(all=True)
#
#     @property
#     def image_tags(self):
#         tags = set()
#         for image in self.images:
#             i_tags = image.tags
#             tags.update(i_tags)
#
#         return tags
#
#     @property
#     def containers(self):
#         return self.client.containers.list(all=True)
#
#     @property
#     def container_names(self):
#         return [x.name for x in self.containers]
#
#     def create_network(self, name, driver="bridge"):
#
#         network_name = self.get_unique_name(name, name_type='network')
#         network = self.client.networks.create(network_name, driver=driver)
#         self.own_networks.add(network)
#
#         return network
#
#     def create_worker_container(self, *args, **kwargs):
#
#         name = self.get_unique_name(kwargs.get('name', self.worker.name),
#                                     name_type='container')
#         # check if container_name is available:
#
#         network = kwargs.get('network', self.worker.network)
#         mounts = kwargs.get('mounts', self.worker.mounts)
#         env_variables = kwargs.get('env_variables', self.worker.env_vars)
#
#         env_variables['MY_CONTAINER_NAME'] = name
#
#         container = self.client.containers.create(self.worker.image,
#                                                   mounts=mounts,
#                                                   name=name,
#                                                   network=network,
#                                                   environment=env_variables,
#                                                   detach=True)
#
#         self.own_containers.add(container)
#
#         return container
#
#     # def check_needed_images(self):
#     #     for image in self.needed_images:
#     #         if image not in self.image_tags:
#     #             logger.info(f'DockerManager: image {image} missing. pulling image...')
#     #             image = self.client.images.pull(image)
#     #             logger.info(f'DockerManager: image {image} successfully pulled')
#
#     def __del__(self):
#
#         self.shared_dir.cleanup()
#
#         for network in self.own_networks:
#             network.remove()
#
#     def get_unique_name(self, name, name_type='container'):
#
#         if name_type=='container':
#             searchable = self.container_names
#         elif name_type=='network':
#             searchable = self.network_names
#
#         if name not in searchable:
#             return name
#
#         i = 0
#         new_name = name
#         while new_name in searchable:
#             i += 1
#             new_name = name + f'_{i}'
#
#         return new_name
