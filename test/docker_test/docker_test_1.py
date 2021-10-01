from src.PYSimultanRadiation.docker.docker_manager import DockerManager, Service, Worker


# create a worker:

shading_worker = Worker(name='shading_worker',
                        image='maxxiking/shading_worker:1.0.1',
                        bind_port=9006,
                        container_bind_mount={'workdir': '/workdir'},
                        bind_to_ip=None,
                        log_dir='/workdir/logs',
                        logging_mode='DEBUG',
                        container_name='shading_worker')


docker_manager = DockerManager()

# container = docker_manager.run_container(docker_manager.images[0])

print('done')
