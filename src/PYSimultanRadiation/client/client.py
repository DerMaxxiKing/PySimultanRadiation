import zmq
# import os
import socket
import colorlog
from service_tools.message import Message
import socketserver

logger = colorlog.getLogger('PySimultanRadiation')

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources


class Client(object):

    def __init__(self, *args, **kwargs):

        self.ip = kwargs.get('ip', 'tcp://localhost:8006')
        self.ctx = zmq.Context.instance()
        self.client = self.ctx.socket(zmq.REQ)
        self.client.connect(self.ip)
        self.logger = logger

    def send_mesh(self, mesh):

        self.logger.info('Sending mesh to worker...')
        message = Message(method='receive_mesh', kwargs={'mesh': mesh})
        self.client.send_pyobj(message)
        return_value = self.client.recv_pyobj()

        if return_value is True:
            self.logger.info('Mesh successfully send to worker...')
        else:
            self.logger.error(f'Error while sending mesh to worker:\n {return_value}')

    def rt_sun_window(self, *args, **kwargs):
        """

        :param args:

        Keyword Arguments
        -----------------
        * *scene* (``str``) --
          Define which parts of the mesh to raytrace: 'all', 'hull', 'internal'; default 'hull
        * *sun_window* (``4x3 np.array``) --
          rectangle to sample
        * *sample_dist* (``float``) --
          distance of sampled points
        * *method* (``str``) --
          use 'length_dependent'
        * *irradiation_vector* (``3x0 np.array``) --
          vector of the irradiation, normalized
        """

        self.logger.debug('Ray tracing sun window...')

        message = Message(method='rt_sun_window',
                          kwargs=kwargs)

        self.client.send_pyobj(message)

        return_value = self.client.recv_pyobj()

        if isinstance(return_value, Exception):
            self.logger.info('Mesh successfully send to worker...')
        else:
            return return_value


def get_free_port():
    with socketserver.TCPServer(("localhost", 0), None) as s:
        port = s.server_address[1]
    return port


def next_free_port(port=1024, max_port=65535):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port <= max_port:
        try:
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise IOError('no free ports')
