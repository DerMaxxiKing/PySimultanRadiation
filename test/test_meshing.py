from PySimultan import DataModel, Template, yaml
from src.PYSimultanRadiation import TemplateParser
from src.PYSimultanRadiation.geometry.scene import Scene
from src.PYSimultanRadiation.geometry.utils import mesh_planar_face

import os


def create_geometry_templates():
    building_template = Template(template_name='Building',
                                 template_id='102',
                                 content=['Zones', 'Usage', 'Constructions'],
                                 documentation='',
                                 units={},
                                 types={},
                                 slots={'Zones': 'Liste_00',
                                        'Aufbauzuweisungen': 'Liste_01',
                                        'TGA': 'Liste_03',
                                        'Nutzung': 'Liste_04'}
                                 )

    return [building_template]


def run_example():
    templates = create_geometry_templates()

    with open(r'F:\OneDrive\PythonProjects\SmartCampusRadiation\test\output\smart_campus_template.yml',
              mode='w',
              encoding="utf-8") as f_obj:
        yaml.dump(templates, f_obj)

    template_file = r'F:\OneDrive\PythonProjects\SmartCampusRadiation\test\output\smart_campus_template.yml'
    project_file = r'F:\OneDrive\PythonProjects\SmartCampusRadiation\resources\SMART_CAMPUS_TU_WIEN_BIBLIOTHEK_2020.03.22_richtig_RAUMMODELL.simultan'

    if not os.path.isfile(project_file):
        raise FileExistsError(f'File {project_file} does not exist')

    template_parser = TemplateParser(template_filepath=template_file)
    data_model = DataModel(project_path=project_file)
    typed_data = data_model.get_typed_data(template_parser=template_parser, create_all=True)

    geo_model = template_parser.typed_geo_models[123]

    my_scene = Scene(vertices=geo_model.vertices,
                     edges=geo_model.edges,
                     edge_loops=geo_model.edge_loops,
                     faces=geo_model.faces,
                     volumes=geo_model.volumes)

    # print(my_scene.mesh)

    for volume in my_scene.volumes:
        print(volume.mesh)

    print(my_scene.windows)

    scene_mesh = my_scene.mesh
    my_scene.export_mesh('test.vtk')

    # for face in my_scene.faces:
    #     filename = "F:\\OneDrive\\PythonProjects\\SmartCampusRadiation\\test\\output\\" + face.name + '.vtk'
    #     my_scene.export_face_mesh_vtk(face, filename)

    # for face in my_scene.faces:
    #     mesh = face.mesh

    print('done')


if __name__ == '__main__':
    run_example()
