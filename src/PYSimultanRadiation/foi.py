from collections import UserList


class ReferenceList(UserList):

    def __init__(self, *args, **kwargs):
        self._wrapped_obj = kwargs.get('wrapped_obj', None)
        if self._wrapped_obj is not None:
            self.data = [x.Reference for x in self._wrapped_obj.ReferencedComponents.Items]
        self._contained_components = kwargs.get('contained_components', None)
        self._contained_parameters = kwargs.get('contained_parameters', None)
        self._template_parser = kwargs.get('template_parser', None)
        self._data_model_id = kwargs.get('data_model_id', None)

    def __getitem__(self, i):
        if isinstance(i, slice):
            if self._template_parser is None:
                return self.__class__(self.data[i])
            return [self._template_parser.create_python_object(x) for x in self.__class__(self.data[i])]
        else:
            if self._template_parser is None:
                return self.data[i]
            return self._template_parser.create_python_object(self.data[i])

    def __repr__(self):
        return f'{self.name}: ' + repr(list(self.data))

    @property
    def contained_components(self):
        if self._contained_components is None:
            if self._wrapped_obj is not None:
                self._contained_components = [self._template_parser.create_python_object(x) for x in
                                              self._wrapped_obj.ContainedComponentsAsList]
        return self._contained_components

    @property
    def contained_parameters(self):
        if self._contained_parameters is None:
            if self._wrapped_obj is not None:
                self._contained_parameters = {x.Name: x.get_ValueCurrent() for x in
                                              self._wrapped_obj.ContainedParameters.Items}
        return self._contained_parameters

    @property
    def name(self):
        if self._wrapped_obj is not None:
            if hasattr(self._wrapped_obj, 'Name'):
                return self._wrapped_obj.Name

    @name.setter
    def name(self, value):
        if self._wrapped_obj is not None:
            self._wrapped_obj.Name = value
