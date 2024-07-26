import os
import cv2 as cv
import numpy as np
from ..QdTree.quadtree import FixedQuadTree

def seqence_image(image_path, to_size = (8, 8, 3), fixed_length = 1024):
    img = cv.imread(image_path)
    img = cv.resize(img, (512, 512))

    grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_img = cv.GaussianBlur(grey_img, (3, 3), 0)

    edges = cv.Canny(gray_img, 80, 100)

    qdt = FixedQuadTree(domain = edges, fixed_length = fixed_length)

    seq_img = qdt.serialize(img, size = to_size)
    seq_img = np.asarray(seq_img)
    seq_img = np.reshape(seq_img, [to_size[0], -1, to_size[2]])

    return seq_img, qdt

def preprocess_image(root_dir, out_dir, to_size, fixed_length):
    if not os.path.exists(out_dir):
        print(f"Out dir created at {out_dir}.")
        os.makedirs(out_dir)

    for class_id, class_name in enumerate(os.listdir(root_dir)):
        class_dir = os.path.join(root_dir, class_name)
        out_class_dir = os.path.join(out_dir, class_name)

        print(f"Propress image for {class_name}.")
        
        if not os.path.exists(out_class_dir):
            os.makedirs(out_class_dir)

        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)

                seq_img, _ = seqence_image(img_path, to_size, fixed_length)
                out_img_path = os.path.join(out_class_dir, img_name)
                cv.imwrite(out_img_path, seq_img)


if __name__ == '__main__':
    to_size = (8, 8, 3)
    fixed_length = 1024

    root_dir = "/lustre/orion/bif146/world-shared/enzhi/imagenet2012/train/"
    out_dir = f"/lustre/orion/bif146/world-shared/enzhi/qdt_imagenet/preprocess_data/{fixed_length}_{to_size}/train/"

    preprocess_image(root_dir, out_dir)


