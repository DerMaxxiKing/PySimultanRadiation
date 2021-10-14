import pvlib
import numpy as np
import pandas as pd


class Location():

    def __init__(self, *args, **kwargs):

        file_name = kwargs.get('file_name')
        self.north_angle = kwargs.get('north_angle', 0)

        self.data, self.metadata = pvlib.iotools.read_epw(file_name,
                                                          coerce_year=2021)

        self.location = pvlib.location.Location.from_epw(self.metadata)

    def generate_irradiation_vector(self, time):

        solar_position = self.location.get_solarposition(time)

        phi = np.deg2rad(- (solar_position.azimuth.values + self.north_angle))
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
