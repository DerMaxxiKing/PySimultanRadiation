from tkinter import filedialog as fd
import subprocess
import os
from .logger import logger


class ConfigCls(object):

    def __init__(self, *args, **kwargs):
        self.default_mesh_size = 10
        self._docker_path = None

    @property
    def docker_path(self):
        if self._docker_path is None:
            process = subprocess.run('where docker-compose', capture_output=True, text=True, universal_newlines=True)
            for line in process.stdout.splitlines():
                if os.path.basename(line) == 'docker-compose.exe':
                    self._docker_path = line
                    break
                if self._docker_path is None:
                    logger.warn(f'Could not find docker-compose path. Please select manually:')
                    self._docker_path = fd.askopenfilename(title='Select docker-compose.exe',
                                                           filetypes=[("docker-compose.exe", ".exe")]
                                                           )
                    if not self._docker_path:
                        logger.error(f'No docker-compose path selected')
        return self._docker_path


config = ConfigCls()

# "C:\Program Files\Docker\Docker\resources\bin\docker-compose.exe" -f F:\OneDrive\PythonProjects\SmartCampusRadiation\docker-compose.yml up -d


if __name__ == '__main__':

    print(config.docker_path)
