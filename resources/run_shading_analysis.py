import sys
from tkinter import Tk
from tkinter import filedialog as fd
import logging

from PYSimultanRadiation.shading_analysis import ProjectLoader

if __name__ == '__main__':

    Tk().withdraw()
    project_filename = fd.askopenfilename(title='Select a SIMULTAN Project...',
                                          filetypes=[("SIMULTAN", ".simultan")]
                                          )
    if project_filename in [None, '']:
        logging.error('No SIMULTAN Project selected')
        sys.exit()
    print(f'selected {project_filename}')

    project_loader = ProjectLoader(project_filename=project_filename,
                                   user_name='admin',
                                   password='admin')

    project_loader.load_project()
    project_loader.run()

    sys.exit()
