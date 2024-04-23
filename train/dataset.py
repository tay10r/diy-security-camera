#!venv/bin/python3

import torch
import torch.utils.data
import torchvision
import os
import torchvision.transforms.v2 as transforms
import json
from typing import Union

def load_labels(path: str) -> dict:
    labels = {}
    with open(path, 'r') as labels_file:
        labels = json.load(labels_file)
    return labels

class Dataset(torch.utils.data.Dataset):
    """
    This dataset implementation will read images stored in this repository and the associated labels from global label
    dictionary (also stored in this repository).
    """

    def __init__(self, root: str, label_dict: dict, transform: Union[transforms.Transform, None]):
        """
        Constructs a new dataset implementation.

        :param root: The root directory containing the different scenes of training data.

        :param label_dict: The global label dictionary.

        :param transform: The transform to apply to each image.
        """
        self.__transform = transform
        self.__frame_paths = []
        self.__labels = []
        for scene in os.listdir(root):
            scene_path = os.path.join(root, scene)
            counter = 0
            for frame in os.listdir(scene_path):
                frame_path = os.path.join(scene_path, frame)
                self.__frame_paths.append(frame_path)
                self.__labels.append(label_dict[frame])
                counter += 1
                #if counter == 16:
                #    break
            #return

    def __len__(self):
        return len(self.__labels)

    def __getitem__(self, idx):
        frame_path = self.__frame_paths[idx]
        label = self.__labels[idx]
        img = torchvision.io.read_image(frame_path, torchvision.io.ImageReadMode.GRAY)
        if self.__transform is not None:
            img = self.__transform(img)
        label = torch.tensor([float(label)])
        label.requires_grad = True
        img.requires_grad = True
        return img, label
