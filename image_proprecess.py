import os
import cv2 as cv
import numpy as np
from quadtree import FixedQuadTree

root_dir = "/lustre/orion/bif146/world-shared/enzhi/imagenet2012/train/"
out_dir = "/lustre/orion/bif146/world-shared/enzhi/qdt_imagenet/preorocess_data/train/"

to_size = (8, 8, 3)
fixed_length = 1024

root_dir = "/lustre/orion/bif146/world-shared/enzhi/imagenet2012/train/"
out_dir = f"/lustre/orion/bif146/world-shared/enzhi/qdt_imagenet/preorocess_data/{fixed_length}_{to_size}/train/"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

for class_id, class_name in enumerate(os.listdir(root_dir)):
    class_dir = os.path.join(root_dir, class_name)
    out_class_dir = os.path.join(out_dir, class_name)
    
    if not os.path.exists(out_class_dir):
        os.makedirs(out_class_dir)

    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)

            img = cv.imread(img_path)

            grey_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            gray_img = cv.GaussianBlur(img, (3, 3), 0)

            edges = cv.Canny(gray_img, 80, 100)
            resized_image = cv.resize(img, (512,512))
            qdt = FixedQuadTree(domain = edges, fixed_length = fixed_length)

            seq_img = qdt.serialize(resized_image, size = to_size)
            seq_img = np.asarray(seq_img)
            seq_img = np.reshape(seq_img, [to_size[0], -1, to_size[2]])

            out_img_path = os.path.join(out_class_dir, img_name)
            cv.imwrite(out_img_path, seq_img)

            break
    break




