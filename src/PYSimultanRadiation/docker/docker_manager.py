import docker
from . import logger
import tempfile
import uuid


class Worker(object):

    def __init__(self, *args, **kwargs):

        self.name = kwargs.get('name', 'unnamed_worker')                #
        self.image = kwargs.get('image')                                # example: 'maxxiking/shading_worker:1.0.1'
        self.bind_port = kwargs.get('bind_port')                        # example: 9006
        self.container_bind_mount = kwargs.get('container_bind_mount')  # directories to append; dict: {'workdir': '/workdir'}
        self.bind_to_ip = kwargs.get('bind_to_ip', None)                # ip-address of the server to connect to
        self.log_dir = kwargs.get('log_dir', None)                      # directory where logs are 'logs'
        self.logging_mode = kwargs.get('logging_mode: ', 'INFO')        # logging mode of the worker
        self.env_vars = kwargs.get('env_vars: ', {})                    # environment variables of the worker
        self.container_name = kwargs.get('container_name: ', None)      # environment variables of the worker

    def start(self, *args, **kwargs):

        client = kwargs.get('client', None)
        if client is None:
            client = docker.from_env()







class Service(object):

    def __init__(self, *args, **kwargs):

        self.worker = kwargs.get('worker', None)
        self.server = kwargs.get('server', None)

        self.num_workers = kwargs.get('num_workers', None)
        self.network_name = kwargs.get('network_name',
                                       'shading_net' + str(uuid.uuid4().hex[:8])
                                       )

    @property
    def needed_images(self):
        return [self.server_image, self.worker_image]


class DockerManager(object):

    def __init__(self, *args, **kwargs):
        self.client = docker.from_env()
        self.shared_dir = tempfile.TemporaryDirectory()
        self.containers = kwargs.get('containers', [])

        self.check_needed_images()

    @property
    def images(self):
        return self.client.images.list()

    @property
    def image_tags(self):
        tags = set()
        for image in self.images:
            i_tags = image.tags
            tags.update(i_tags)

        return tags

    def check_needed_images(self):
        for image in self.needed_images:
            if image not in self.image_tags:
                logger.info(f'DockerManager: image {image} missing. pulling image...')
                image = self.client.images.pull(image)
                logger.info(f'DockerManager: image {image} successfully pulled')

    def run_container(self, image, dest='/mnt/vol1'):

        container = self.client.containers.run(
            image=image.tags[0],
            detach=True,
            stdin_open=True,
            tty=True,
            volumes=[self.host_file_dir]
        )

        return container

    def create_server_config(self, config_dir):
        pass

    def run_service(self):

        self.network = self.client.networks.create(self.network_name, driver="bridge")

    def __del__(self):

        self.shared_dir.cleanup()
        self.network.remove()
