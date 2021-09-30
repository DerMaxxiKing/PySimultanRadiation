import numpy as np
from pyemblite.mesh_construction import TriangleMesh
from pyemblite.rtcore_scene import EmbreeScene


class Scene(object):

    def __init__(self, *args, **kwargs):

        self.points = kwargs.get('points')
        self.faces = kwargs.get('faces')

        self.scene = EmbreeScene()
        self.emesh = TriangleMesh(self.scene, self.points, self.faces)

        self.scene.commit()

    def intersect_with_rays(self, n, p):

        primID = self.scene.run(p, n, query='INTERSECT')
        count = np.bincount(primID[primID != -1])

        return count

    def __del__(self):
        pass
