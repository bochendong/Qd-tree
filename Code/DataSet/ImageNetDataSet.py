import os
import cv2 as cv
import logging
import torch
from Code.DataSet.Preprocess import seqence_image
from torch.utils.data import Dataset


class ImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, img_size=224, to_size=(8, 8, 3), num_patches=196, preprocess_local=False, use_qdt=True):
        self.root_dir = root_dir
        self.transform = transform
        self.img_size = img_size
        self.to_size = to_size
        self.num_patches = num_patches
        self.preprocess_local = preprocess_local
        self.use_qdt = use_qdt
        self.img_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))  # Sort classes for consistent labeling

        for class_id, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.img_paths.append(img_path)
                self.labels.append(class_id)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        img = cv.imread(img_path)
        if img is None:
            # Handle case where image is not loaded properly
            logging.warning(f"Failed to load image {img_path}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

        if self.preprocess_local:
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        if self.use_qdt:
            if self.preprocess_local:
                img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            else:
                img, _ = seqence_image(img_path, self.img_size, self.to_size, self.num_patches)

        if self.transform:
            img = self.transform(img)
        
        return img, label
