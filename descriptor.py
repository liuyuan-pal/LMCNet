import numpy as np

class DummyDescriptor:
    def __init__(self,cfg):
        pass

    def __call__(self, img, kps):
        return np.zeros([kps.shape[0],0])

name2desc = {
    'none': DummyDescriptor,
}