import cv2 as cv
import numpy as np
from torchvision import transforms
from quadtree import FixedQuadTree

class SequenceImageTransform:
    def __init__(self, to_size=(8, 8, 3), fixed_length=196):
        self.to_size = to_size
        self.fixed_length = fixed_length

    def __call__(self, img):
        img = np.array(img) 
        img = img[..., ::-1]

        edges = cv.Canny(img, 80, 100)

        qdt = FixedQuadTree(domain=edges, fixed_length=self.fixed_length)

        seq_img = qdt.serialize(img, size=self.to_size)
        seq_img = np.asarray(seq_img)
        seq_img = np.reshape(seq_img, [self.to_size[0], -1, self.to_size[2]])

        return seq_img