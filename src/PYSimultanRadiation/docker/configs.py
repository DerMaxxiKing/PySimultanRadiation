import uuid
from ruamel.yaml import YAML, yaml_object, add_representer


def represent_none(self, _):
    return self.represent_scalar('tag:yaml.org,2002:null', '')


add_representer(type(None), represent_none)
yaml = YAML()
yaml.indent(sequence=4, offset=2)


@yaml_object(yaml)
class ServerConfig(object):

    yaml_tag = u'!ServerConfig'

    @classmethod
    def create_new(cls, *args, **kwargs):
        """
        Creates a new server config file at config_path.

        :param args:
        :param kwargs:
        :keyword id:                id of the server; uuid.UUID4
        :keyword name:              name of the server; str
        :keyword ip:                ip address of the server; str; examples: 'localhost', '127.0.0.1'
        :keyword port:              port of the server; str; examples: 8005
        :keyword backend_port:      port used for communication with the workers; int; examples: 8006; optional; as default a free port between 9001 and 9050 is choosen
        :keyword public_keys_dir:   directory where public_keys are stored; str
        :keyword secret_keys_dir:   directory where secret_keys are stored; str
        :keyword log_dir:           directory of log; str
        :return:
        """

        server_config = cls(*args, **kwargs)
        server_config.config_path = kwargs.get('config_path')

        return server_config

    def __init__(self, *args, **kwargs):

        self.id = kwargs.get('id', str(uuid.uuid4()))
        self.name = kwargs.get('name', 'unnamed_server')
        self.secure = kwargs.get('secure', False)
        self.public_keys_dir = kwargs.get('public_keys_dir', None)
        self.secret_keys_dir = kwargs.get('secret_keys_dir', None)

        self.ip = kwargs.get('ip', None)

        self.port = kwargs.get('port', 8006)
        self.backend_port = kwargs.get('backend_port', 9006)

        # logging
        self.log_dir = kwargs.get('log_dir', '')
        self.logging_mode = kwargs.get('logging_mode', 'INFO')

    def write(self, filepath=None):

        with open(filepath, mode='w') as f_obj:
            yaml.dump(self, f_obj)


@yaml_object(yaml)
class WorkerConfig(object):

    yaml_tag = u'!WorkerConfig'

    def __init__(self, *args, **kwargs):

        self.id = kwargs.get('id', str(uuid.uuid4()))
        self.name = kwargs.get('name', str(uuid.uuid4()))

        self.ip = kwargs.get('ip', '0.0.0.0')
        self.port = kwargs.get('port', 9006)

        # logging
        self.log_dir = kwargs.get('log_dir', '')
        self.logging_mode = kwargs.get('logging_mode', 'INFO')

    def write(self, filepath):

        with open(filepath, mode='w') as f_obj:
            yaml.dump(self, f_obj)


if __name__ == '__main__':

    config = ServerConfig(name='zmq_server',
                          secure=False,
                          ip='0.0.0.0',
                          port=8006,
                          backend_port=9006,
                          log_dir='logs',
                          logging_mode='INFO')

    config.write('server_config.yml')

    worker_config = WorkerConfig(name='zmq_server',
                                 ip='0.0.0.0',
                                 port=9006,
                                 log_dir='logs',
                                 logging_mode='INFO'
                                 )

    worker_config.write('worker_config.yml')
