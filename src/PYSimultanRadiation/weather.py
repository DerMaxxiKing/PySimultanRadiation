from PySimultan.default_types import SimultanObject

class Weather(SimultanObject):

    def __init__(self, *args, **kwargs):

        SimultanObject.__init__(self, *args, **kwargs)
        self._location = kwargs.get('location', None)

    @property
    def weather_file_name(self):
        return self._wrapped_obj.ReferencedAssets.Items.Items[0].Resource.CurrentFullPath
