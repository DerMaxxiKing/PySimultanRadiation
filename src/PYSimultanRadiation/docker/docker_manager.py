import docker
from . import logger


class DockerManager(object):

    needed_images = ['']
    needed_containers = ['maxxiking/embree3:1.0.2']

    def __init__(self, *args, **kwargs):
        self.client = docker.from_env()
        self.host_file_dir = kwargs.get('host_file_dir')

    def __del__(self):
        for container in self.client.containers.list():
            container.stop()

    @property
    def images(self):
        return self.client.images.list()

    @property
    def image_tags(self):
        return [x.attrs.tags[0] for x in self.images]

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



if __name__ == '__main__':

    docker_manager = DockerManager(host_file_dir=r'K://docker_test')


    docker_manager.client.containers.run("maxxiking/embree3", detach=True)
