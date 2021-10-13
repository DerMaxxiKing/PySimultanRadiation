
class ConfigCls(object):

    def __init__(self, *args, **kwargs):
        self.default_mesh_size = 10
        self.docker_path = r"docker-compose"

config = ConfigCls()

# "C:\Program Files\Docker\Docker\resources\bin\docker-compose.exe" -f F:\OneDrive\PythonProjects\SmartCampusRadiation\docker-compose.yml up -d
