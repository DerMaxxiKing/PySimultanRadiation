import numpy as np
from uuid import uuid1

from .extended_geometry import VertexExt, EdgeExt, EdgeLoopExt, FaceExt, VolumeExt


class GeoBaseClass(object):

    def __init__(self, *args, **kwargs):

        self.id = int.from_bytes(uuid1().bytes, byteorder='big', signed=True) >> 64
        self.name = kwargs.get('name', 'unnamed geometry')


class Vertex(VertexExt, GeoBaseClass):

    def __init__(self, *args, **kwargs):
        GeoBaseClass.__init__(self, *args, **kwargs)
        self.position = kwargs.get('position', np.array([0, 0, 0]))
        VertexExt.__init__(self, *args, **kwargs)


class Edge(EdgeExt, GeoBaseClass):

    def __init__(self, *args, **kwargs):
        GeoBaseClass.__init__(self, *args, **kwargs)
        self.vertices = kwargs.get('vertices', None)
        EdgeExt.__init__(self, *args, **kwargs)


class EdgeLoop(EdgeLoopExt, GeoBaseClass):

    def __init__(self, *args, **kwargs):
        GeoBaseClass.__init__(self, *args, **kwargs)

        self.edges = kwargs.get('edges', None)
        self.edge_orientations = kwargs.get('edge_orientations', None)

        EdgeLoopExt.__init__(self, *args, **kwargs)

    @property
    def points(self):

        points = np.zeros([self.edges.__len__(), 3])

        if self.edge_orientations[0] == 1:
            points[0, :] = self.edges[0].vertices[0].position
        else:
            points[0, :] = self.edges[0].vertices[1].position

        for i, edge in enumerate(self.edges):
            if self.edge_orientations[i] == 1:
                points[i, :] = edge.vertices[1].position
            else:
                points[i, :] = edge.vertices[0].position

        return points


class Face(FaceExt, GeoBaseClass):

    def __init__(self, *args, **kwargs):
        GeoBaseClass.__init__(self, *args, **kwargs)

        self.boundary = kwargs.get('boundary', None)
        self.holes = kwargs.get('holes', [])
        self.hole_faces = kwargs.get('hole_faces', [])
        self.construction = kwargs.get('construction', None)

        FaceExt.__init__(self, *args, **kwargs)

    @property
    def points(self):
        return self.boundary.points


class Volume(VolumeExt, GeoBaseClass):

    def __init__(self, *args, **kwargs):
        GeoBaseClass.__init__(self, *args, **kwargs)

        self.faces = kwargs.get('faces', [])

        VolumeExt.__init__(self, *args, **kwargs)


class Terrain(object):

    def __init__(self, *args, **kwargs):

        self.vertices = kwargs.get('vertices')
        self.edges = kwargs.get('edges')
        self.edge_loops = kwargs.get('edge_loops')
        self.faces = kwargs.get('faces')
