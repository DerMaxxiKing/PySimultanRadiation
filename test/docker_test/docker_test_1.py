from src.PYSimultanRadiation.docker.docker_manager import DockerManager


docker_manager = DockerManager(host_file_dir='/k://docker_test/')

container = docker_manager.run_container(docker_manager.images[0])

print('done')
