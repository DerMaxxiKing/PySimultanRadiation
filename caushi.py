from tkinter import Tk
from tkinter import filedialog as fd
import logging
import os

from PySimultan import DataModel
from PYSimultanRadiation import TemplateParser
from PYSimultanRadiation.geometry.scene import Scene

print('hello mr. caushi... I hope now it works :-)')


if __name__ == '__main__':

    Tk().withdraw()
    project_filename = fd.askopenfilename(title='Select a SIMULTAN Project...')
    if project_filename is None:
        logging.error('No SIMULTAN Project selected')
    print(f'selected {project_filename}')

    template_filename = fd.askopenfilename(title='Select a Template-File...')
    if template_filename is None:
        logging.error('No Template-File selected')
    print(f'selected {template_filename}')

    template_parser = TemplateParser(template_filepath=template_filename)
    data_model = DataModel(project_path=project_filename)
    typed_data = data_model.get_typed_data(template_parser=template_parser, create_all=True)

    geo_model = template_parser.typed_geo_models[0]

    my_scene = Scene(vertices=geo_model.vertices,
                     edges=geo_model.edges,
                     edge_loops=geo_model.edge_loops,
                     faces=geo_model.faces,
                     volumes=geo_model.volumes,
                     terrain_height=-10)

    mesh = my_scene.generate_shading_analysis_mesh()
    mesh.write(os.path.join(output_dir, 'mesh.vtk'))

    print('done')
