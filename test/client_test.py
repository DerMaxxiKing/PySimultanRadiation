import os
import sys
import asyncio
from time import time

import trimesh
import numpy as np
import pandas as pd

from tkinter import Tk
from tkinter import filedialog as fd

from PySimultan import DataModel

from src.PYSimultanRadiation.client.client import Client
from src.PYSimultanRadiation.radiation.location import Location

from src.PYSimultanRadiation import TemplateParser
from src.PYSimultanRadiation.geometry.scene import Scene
from src.PYSimultanRadiation.radiation.utils import create_sun_window, npyAppendableFile, calc_timestamp

import logging
import multiprocessing
from functools import partial

# from service_tools import Message
import meshio

try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 importlib_resources.
    import importlib_resources as pkg_resources


import resources
import results


def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)

    return wrapped


def create_scene(project_filename=None,
                 template_filename=None,
                 output_dir=None):

    if project_filename is None:
        Tk().withdraw()
        project_filename = fd.askopenfilename(title='Select a SIMULTAN Project...')
        if project_filename is None:
            logging.error('No SIMULTAN Project selected')
        print(f'selected {project_filename}')

    if template_filename is None:
        template_filename = fd.askopenfilename(title='Select a Template-File...')
        if template_filename is None:
            logging.error('No Template-File selected')
        print(f'selected {template_filename}')

    if output_dir is None:
        output_dir = fd.askdirectory(title='Select output directory...')
        if output_dir is None:
            logging.error('No output directory selected')
        print(f'selected {output_dir}')

    template_parser = TemplateParser(template_filepath=template_filename)
    data_model = DataModel(project_path=project_filename)
    typed_data = data_model.get_typed_data(template_parser=template_parser, create_all=True)

    geo_model = template_parser.typed_geo_models[123]

    my_scene = Scene(vertices=geo_model.vertices,
                     edges=geo_model.edges,
                     edge_loops=geo_model.edge_loops,
                     faces=geo_model.faces,
                     volumes=geo_model.volumes,
                     terrain_height=14.2)

    return my_scene


def run_shading_analysis(scene,
                         weather_filename=None,
                         output_dir=None,
                         vtk_res_path=None,
                         write_vtk=False,
                         binary=False
                         ):

    print('creating shading analysis mesh')
    mesh = scene.generate_shading_analysis_mesh(mesh_size=5)
    print('writing scene')
    mesh.write(os.path.join(output_dir, 'mesh.vtk'))

    print('starting shading analysis')
    foi_mesh = trimesh.Trimesh(vertices=mesh.points,
                               faces=mesh.cells_dict['triangle'][np.where(mesh.cell_data['foi'][0]), :][0])

    location = Location(file_name=weather_filename,
                        north_angle=100)

    dti = pd.Series(pd.date_range("2020-06-06 00:00:00", periods=96, freq="15min"))
    df = pd.DataFrame(index=dti, columns=[])
    irradiation_vector = location.generate_irradiation_vector(dti)
    df['irradiation_vector'] = irradiation_vector['irradiation_vector']

    sun_window = create_sun_window(foi_mesh, np.stack(df['irradiation_vector'].values))
    df['windows'] = [x for x in sun_window]

    # irrad_vec = np.vstack(irradiation_vector.irradiation_vector.values)
    # df_to_calc = df.loc[np.stack(df['irradiation_vector'].values)[:, 2] < 0]

    num_cells = [1, mesh.cells_dict['triangle'][np.where(mesh.cell_data['hull_face'][0]), :][0].shape[0]]
    cell_ids = np.where(mesh.cell_data['hull_face'][0])

    my_client = Client(ip='tcp://localhost:8006')
    my_client.send_mesh(mesh)

    if write_vtk:
        f_sh = np.zeros(num_cells)
        hull_mesh = meshio.Mesh(points=mesh.points,
                                cells=[
                                    ("triangle", mesh.cells_dict['triangle'][np.where(mesh.cell_data['hull_face'][0]), :][0])
                                ],
                                cell_data={"f_sh": [f_sh[0, :]]})

        meshio.vtk.write(os.path.join(vtk_res_path, f'test0.vtk'), hull_mesh, '4.2', binary=binary)

    numpy_file = npyAppendableFile(os.path.join(output_dir, 'f_sh.npy'), True)

    # @background

    ################################
    #
    #       Calculate Timesteps
    #
    #################################

    nb_cpus = 4
    pool = multiprocessing.Pool(processes=nb_cpus)

    part_fcn = partial(calc_timestamp,
                       sample_dist=1,
                       num_cells=num_cells,
                       numpy_file=numpy_file,
                       write_vtk=write_vtk,
                       hull_mesh=hull_mesh,
                       dti=dti,
                       binary=binary,
                       vtk_res_path=vtk_res_path,
                       )

    t_start = time()
    results = pool.map(part_fcn, zip(range(df.shape[0]),
                                           df['windows'],
                                           df['irradiation_vector']))
    t_end = time()
    print(f'elapsed Time: {t_end- t_start}')

    #
    # t_start = time()
    # for i, (index, row) in enumerate(df.iterrows()):
    #     if row['irradiation_vector'][2] > 0:
    #         continue
    #     calc_timestamp(i, row['windows'], row['irradiation_vector'], 0.5)
    #
    # t_end = time()
    # print(f'elapsed Time: {t_end- t_start}')

    # for i, (index, row) in enumerate(df.iterrows()):
    #     print(f"{i}: Irradiation vector {row['irradiation_vector']}")
    #
    #     if row['irradiation_vector'][2] > 0:
    #         continue
    #
    #     count = my_client.rt_sun_window(scene='hull',
    #                                     sun_window=row['windows'],
    #                                     sample_dist=0.025,
    #                                     irradiation_vector=row['irradiation_vector'])
    #     f_sh = np.zeros(num_cells)
    #     f_sh[0, 0:count.shape[0]] = count
    #
    #     if write_vtk:
    #         # write_vtk
    #         hull_mesh.cell_data['f_sh'] = [f_sh[0, :]]
    #         meshio.vtk.write(os.path.join(vtk_res_path, f'shading_{dti[i].strftime("%Y%m%d_%H%M%S")}.vtk'), hull_mesh, '4.2', binary=binary)
    #
    #     print('done')


if __name__ == '__main__':

    with pkg_resources.path(resources, 'SMART_CAMPUS_TU_WIEN_BIBLIOTHEK_2020.03.22_richtig_RAUMMODELL.simultan') as r_path:
        project_file = str(r_path)

    with pkg_resources.path(resources, 'shading_template.yml') as r_path:
        template_filename = str(r_path)

    with pkg_resources.path(results, '') as r_path:
        output_dir = str(r_path)

    with pkg_resources.path(results, 'vtk') as r_path:
        vtk_res_path = str(r_path)

    with pkg_resources.path(resources, 'AUT_Vienna.Schwechat.110360_IWEC.epw') as r_path:
        weather_filename = str(r_path)

    my_scene = create_scene(project_filename=project_file,
                            template_filename=template_filename,
                            output_dir=output_dir)
    run_shading_analysis(my_scene,
                         weather_filename=weather_filename,
                         vtk_res_path=vtk_res_path,
                         output_dir=output_dir,
                         write_vtk=True)

    sys.exit()
