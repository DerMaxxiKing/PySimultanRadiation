import pandas as pd
import numpy as np
import pvlib
import trimesh
import meshio

from copy import copy
import pickle

from pathlib import Path
import os


def generate_irradiation_vector(time, north_angle=0):

    data, metadata = pvlib.iotools.read_epw(r'resources/AUT_Vienna.Schwechat.110360_IWEC.epw')
    location = pvlib.location.Location.from_epw(metadata)
    solar_position = location.get_solarposition(time)

    phi = np.deg2rad(- (solar_position.azimuth.values + north_angle))
    theta = np.deg2rad(solar_position.elevation.values)

    cos_theta = np.cos(theta)

    irradiation_vector = np.zeros([time.shape[0], 3], dtype=np.float32)

    irradiation_vector[:, 0] = - cos_theta * np.cos(phi)
    irradiation_vector[:, 1] = - cos_theta * np.sin(phi)
    irradiation_vector[:, 2] = - np.sin(theta)

    df = pd.DataFrame(index=time,
                      columns=['irradiation_vector'])
    df['irradiation_vector'] = [x for x in irradiation_vector]

    return df


def create_sun_window(mesh, irradiation_vector):

    n = irradiation_vector.shape[0]

    u, v = two_orthogonal_vectors(irradiation_vector)

    sun_cs = np.empty((n, 3, 3))
    sun_cs[:, 0, :] = u
    sun_cs[:, 1, :] = v
    sun_cs[:, 2, :] = irradiation_vector

    rectangles = np.empty((n, 4, 3))

    oriented_mesh_corners = trimesh.bounds.corners(mesh.bounding_box_oriented.bounds).T

    for i in range(n):
        rot = sun_cs[i, :, :]
        rectangle = calc_local_rect(rot, oriented_mesh_corners)
        rectangles[i, :, :] = np.linalg.inv(rot).dot(rectangle.T).T

    return rectangles


def two_orthogonal_vectors(vector):

    if isinstance(vector, pd.DataFrame):
        vector = vector.values

    if vector.shape.__len__() == 1:
        vector = np.array([vector])

    x, y = generate_basis_vectorized(vector)

    return x, y


def generate_basis_vectorized(z):
    """
    from: trimesh.util.generate_basis
    Generate an arbitrary basis (also known as a coordinate frame)
    from a given z-axis vector.

    Parameters
    ------------
    z : (3,) float
      A vector along the positive z-axis.
    epsilon : float
      Numbers smaller than this considered zero.

    Returns
    ---------
    x : (3,) float
      Vector along x axis.
    y : (3,) float
      Vector along y axis.
    z : (3,) float
      Vector along z axis.
    """
    epsilon = 1e-12

    # X as arbitrary perpendicular vector
    x = np.zeros((z.shape[0], 3))
    x[:, 0] = -z[:, 1]
    x[:, 1] = z[:, 0]
    # avoid degenerate case
    x_norm = trimesh.util.row_norm(x)

    ind1 = x_norm < epsilon
    x[ind1, 0] = -z[ind1, 2]
    x[ind1, 1] = z[ind1, 1]
    x[ind1, 2] = z[ind1, 0]

    x[ind1, :] /= trimesh.util.row_norm(x[ind1, :])[:, None]

    x[np.logical_not(ind1), :] /= trimesh.util.row_norm(x[np.logical_not(ind1), :])[:, None]

    # get perpendicular Y with cross product
    y = np.cross(z, x)

    return x, y


def calc_local_rect(rot_mat, oriented_mesh_corners):
    import cv2

    local_oriented_corners = rot_mat.dot(oriented_mesh_corners).T
    z_translation = -abs(min(local_oriented_corners[:, 2]) * 1.1)

    rec = np.zeros((4, 3))

    local_points = local_oriented_corners[:, 0:2]

    c_hull = cv2.convexHull(local_points.astype('float32'))
    local_rectangle = cv2.boxPoints(cv2.minAreaRect(c_hull))

    rec[:, 0:2] = local_rectangle
    rec[:, 2] = z_translation

    return rec


def sample_rectangle(rect, n, scale=1.01, method='length_dependent'):
    """
    create equally distant points in rectangle

    :param rect: edge points of the rectangle (4 * 3)
    :param n: number of points
    :param scale: scale rectangle
    :param method: how to sample points:
    constant: n points along x' and y';
    constant_adapted: n^2 = n1 * n2 points in total. n1 and n2 according to length of x' and y'; l
    length_dependent: number of points  = (x' / n) * (y' / n) ; n is distance between points
    :return: sampled points (n * 3)
    """

    # scale rectangle
    if scale != 1:
        center = rect[0, :] + (rect[2, :] - rect[0, :]) * 0.5
        rect = (rect - center) * scale + center

    origin = (rect[0, :]).astype(np.float32)
    vec0 = (rect[1, :] - origin).astype(np.float32)
    vec1 = (rect[-1, :] - origin).astype(np.float32)

    area = np.linalg.norm(vec0) * np.linalg.norm(vec1)

    if method == 'constant':

        spaces = np.linspace(0, 1, n, dtype=np.float32)

        n_spaces = spaces.shape[0]

        points = ((np.transpose(np.broadcast_to(vec1 * spaces[:, None], (n_spaces, n_spaces, 3)), axes=(1, 0, 2)) + \
                   np.broadcast_to(vec0 * spaces[:, None], (n_spaces, n_spaces, 3)) + \
                   origin[None, None, :])).reshape(n_spaces**2, 3)

    else:
        if method == 'constant_adapted':
            frac = np.linalg.norm(vec0) / np.linalg.norm(vec1)

            n1 = (n * np.sqrt(frac)).astype(int)
            n2 = (n / np.sqrt(frac)).astype(int)

        elif method == 'length_dependent':

            n1 = np.ceil(np.linalg.norm(vec0) / n).astype(int)
            n2 = np.ceil(np.linalg.norm(vec1) / n).astype(int)

        if n1 < 20:
            n1 = 20

        if n2 < 20:
            n2 = 20

        spaces0 = np.linspace(0, 1, n1, endpoint=False, dtype=np.float32)[1:]
        spaces1 = np.linspace(0, 1, n2, endpoint=False, dtype=np.float32)[1:]

        # x, y = np.meshgrid(spaces0, spaces1, copy=False, dtype=np.float32)
        # points = (x[:, :, None] * vec0 + y[:, :, None] * vec1).reshape(spaces0.shape[0] * spaces1.shape[0], 3) + origin

        points = (np.transpose(np.broadcast_to(vec0 * spaces0[:, None],
                                               (spaces1.shape[0], spaces0.shape[0], 3)),
                               axes=(1, 0, 2)) + \
                  np.broadcast_to(vec1 * spaces1[:, None],
                                  (spaces0.shape[0], spaces1.shape[0], 3)
                                  )).reshape(spaces0.shape[0] * spaces1.shape[0], 3) + origin

    return points, area


def export_rays_vtk(orig, norm, filename='rays.vtk'):

    pt1 = orig
    pt2 = orig + norm
    points = np.concatenate((pt1, pt2), axis=0)

    pt_range = np.arange(0, pt1.shape[0], 1)

    lines = np.vstack((pt_range, pt_range[-1] + pt_range + 1)).T

    cells = [
        ("line", lines)
    ]

    mesh = meshio.Mesh(
        points,
        cells,
    )
    mesh.write(
        filename,  # str, os.PathLike, or buffer/open file
        # file_format="vtk",  # optional if first argument is a path; inferred from extension
    )


def export_vtk(mesh, count, filename):
    cells = [
        ("triangle", mesh.faces)
    ]

    cell_data = np.zeros(mesh.faces.shape[0])
    cell_data[0:count.shape[0]] = count

    mesh = meshio.Mesh(
        mesh.vertices,
        cells,
        # Optionally provide extra data on points, cells, etc.
        # Each item in cell data must match the cells array
        cell_data={"count": [cell_data]},
    )
    mesh.write(
        filename,  # str, os.PathLike, or buffer/open file
        # file_format="vtk",  # optional if first argument is a path; inferred from extension
    )


def export_sun_window_vtk(window, filename):

    cells = [
        ("quad", np.array([[0, 1, 2, 3]])),
    ]

    mesh = meshio.Mesh(
        window,
        cells,
    )
    mesh.write(
        filename,  # str, os.PathLike, or buffer/open file
        # file_format="vtk",  # optional if first argument is a path; inferred from extension
    )


def export_points_vtk(points, filename):

    from pyntcloud import PyntCloud

    cloud = PyntCloud(pd.DataFrame(
        # same arguments that you are passing to visualize_pcl
        data=points,
        columns=["x", "y", "z"]))

    cloud.to_file(filename)


class npyAppendableFile():
    """
    https://stackoverflow.com/questions/30376581/save-numpy-array-in-append-mode
    :param

    """

    def __init__(self, fname, newfile=True):
        '''
        Creates a new instance of the appendable filetype
        If newfile is True, recreate the file even if already exists
        '''
        self.fname = Path(fname)
        if newfile:
            with open(self.fname, "wb") as fh:
                fh.close()

    def write(self, data):
        '''
        append a new array to the file
        note that this will not change the header
        '''
        with open(self.fname, "ab") as fh:
            np.save(fh, data)

    def load(self, axis=2):
        '''
        Load the whole file, returning all the arrays that were consecutively
        saved on top of each other
        axis defines how the arrays should be concatenated
        '''

        with open(self.fname, "rb") as fh:
            fsz = os.fstat(fh.fileno()).st_size
            out = np.load(fh)
            while fh.tell() < fsz:
                out = np.concatenate((out, np.load(fh)), axis=axis)

        return out

    def update_content(self):
        '''
        '''
        content = self.load()
        with open(self.fname, "wb") as fh:
            np.save(fh, content)

    @property
    def _dtype(self):
        return self.load().dtype

    @property
    def _actual_shape(self):
        return self.load().shape

    @property
    def header(self):
        '''
        Reads the header of the npy file
        '''
        with open(self.fname, "rb") as fh:
            version = np.lib.format.read_magic(fh)
            shape, fortran, dtype = np.lib.format._read_array_header(fh, version)

        return version, {'descr': dtype,
                         'fortran_order': fortran,
                         'shape': shape}


if __name__ == '__main__':
    arr_a = np.random.rand(5, 40, 10)
    arr_b = np.random.rand(5, 40, 7)
    arr_c = np.random.rand(5, 40, 3)

    f = npyAppendableFile("testfile.npy", True)

    f.write(arr_a)
    f.write(arr_b)
    f.write(arr_c)

    out = f.load()

    print(f.header)
    print(f._actual_shape)

    # after update we can load with regular np.load()
    f.update_content()

    new_content = np.load('testfile.npy')
    print(new_content.shape)
