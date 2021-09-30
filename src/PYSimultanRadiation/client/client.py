import zmq
import os
import colorlog
from service_tools.message import Message

logger = colorlog.getLogger('PySimultanRadiation')


try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources


from .resources import private_keys
from .resources import public_keys


# class Message(object):
#
#     def __init__(self, *args, **kwargs):
#
#         self.method = kwargs.get('method', None)
#         self.args = kwargs.get('args', list())
#         self.kwargs = kwargs.get('args', dict())


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

        self.logger.info('Ray tracing sun window...')

        message = Message(method='rt_sun_window',
                          kwargs=kwargs)

        self.client.send_pyobj(message)

        return_value = self.client.recv_pyobj()

        if isinstance(return_value, Exception):
            self.logger.info('Mesh successfully send to worker...')
        else:
            return return_value
