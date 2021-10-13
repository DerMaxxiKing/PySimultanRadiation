import sys
import time

from src.PYSimultanRadiation.docker.docker_manager import ShadingService
# from src.PYSimultanRadiation.docker.docker_manager import DockerManager, Service, Worker, Mount
#
#
# docker_manager = DockerManager()
# print(f'Container names: {docker_manager.container_names}')
# print(f'Network names: {docker_manager.network_names}')
#
# shading_network = docker_manager.create_network('shading_network')
#
# worker_mount = Mount(target='/workdir',
#                      source=r'K:\docker_test\zmq_server',
#                      type='bind',
#                      read_only=False)
#
# # create a worker:
# shading_worker = Worker(name='shading_worker',
#                         image='maxxiking/shading_worker:1.0.1',
#                         bind_port=9006,
#                         mounts=[worker_mount],
#                         bind_to_ip=None,
#                         log_dir='/workdir/logs',
#                         logging_mode='DEBUG',
#                         container_name='shading_worker',
#                         network=shading_network.name)
#
# docker_manager.worker = shading_worker
# container = docker_manager.create_worker_container()
#
# # container = shading_worker.create_container(docker_manager.client)
#
#
# # container = docker_manager.run_container(docker_manager.images[0])
#
# print('done')

my_shading_service = ShadingService(workdir=r'K:\docker_test\zmq_server',
                                    port=8006,
                                    num_workers=2)

my_shading_service.write_compose_file('docker_compose_test.yml')

with my_shading_service:

    time.sleep(5)

with my_shading_service:

    time.sleep(5)

print('done')
sys.exit()
