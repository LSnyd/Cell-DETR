from typing import Callable, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import numpy as np

import misc
import augmentation

from typing import Callable, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import albumentations as A
import misc
import augmentation
import cv2
from sklearn.preprocessing import OneHotEncoder
import copy


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (x1, y1, x2, y2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([x1, y1, x2, y2])
    return boxes.astype(np.int32)


# classes for data loading and preprocessing
class Dataset:
    """
    This Dataset implements the instance segmentation dataset for the DETR model.
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = ["11", "12", "13", "14", "15", "16", "17", "18", "21", "22", "23", "24", "25", "26", "27", "28", "31",
               "32", "33", "34", "35", "36", "37", "38", "41", "42", "43", "44", "45", "46", "47", "48"]

    def __init__(
            self,
            images_dir,
            masks_dir,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id[:-4]) for image_id in self.ids]

        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing
        self.size = 256

    def __getitem__(self, i):

        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (self.size, self.size))
        # image = misc.normalize(image)
        input = torch.Tensor(np.expand_dims(image, 0))

        mask_paths = self.masks_fps[i]
        mask_collection = []
        background = np.ones((self.size, self.size, 1)).astype('float')

        labels = []
        bbs = []

        for label_idx, label in enumerate(self.CLASSES):

            mask_path = '{}/{}.jpg'.format(mask_paths, label)

            if os.path.isfile(mask_path) is True:
                train_mask = np.array(Image.open(mask_path))
                train_mask = cv2.resize(train_mask, (self.size, self.size))
                train_mask = train_mask.clip(0, 1)
                train_mask = train_mask.reshape(self.size, self.size, 1)
                # bbs.append(extract_bboxes() )
                x, y = np.nonzero(np.squeeze(train_mask))
                background[x, y] = 0
                one_hot_label = np.zeros(((len(self.CLASSES) + 1)))
                one_hot_label[1 + int(label_idx)] = 1
                labels.append(one_hot_label)

                mask_collection.append(np.array(train_mask))

        mask = np.squeeze(np.array(mask_collection))
        noaug_mask = copy.deepcopy(mask)
        bb = extract_bboxes(np.concatenate(mask_collection, axis=-1).astype('float'))

        if self.augmentation is not None:
            sample = self.augmentation(image=np.moveaxis(input.numpy(), 0, 2).astype(np.uint8), masks=list(mask))

            input = torch.Tensor(np.moveaxis(sample['image'], 2, 0))
            mask = np.array(sample['masks'])

            if np.array_equal(mask, noaug_mask) is False:

                image_center = (input.shape[2] // 2, input.shape[1] // 2)

                bb[:, [0, 2]] += 2 * (image_center - bb[:, [0, 2]])
                bounding_boxes_w = np.abs(bb[:, 0] - bb[:, 2])
                bb[:, 0] -= bounding_boxes_w
                bb[:, 2] += bounding_boxes_w

                flip_label_dict = {
                    "1": "2",
                    "2": "1",
                    "3": "4",
                    "4": "3",
                }

                flipped_labels = []
                for label in labels:

                    old_label = np.argmax(label) - 1
                    old_label = int(self.CLASSES[old_label])

                    new_firstdigit = flip_label_dict[str(old_label)[0]]

                    if len(str(old_label)) == 2:
                        new_label = int(new_firstdigit + str(old_label)[1])

                    else:
                        new_label = int(new_firstdigit)

                    new_label = self.CLASSES.index(str(new_label))
                    one_hot_label = np.zeros(((len(self.CLASSES) + 1)))
                    one_hot_label[1 + int(new_label)] = 1
                    flipped_labels.append(one_hot_label)

                    labels = flipped_labels

        bbs = misc.absolute_bounding_box_to_relative(bounding_boxes=torch.Tensor(bb), height=input.shape[1],
                                                     width=input.shape[2])

        bbs_formated = misc.bounding_box_x0y0x1y1_to_xcycwh(bbs)

        return input, torch.Tensor(mask), bbs_formated, torch.Tensor(labels)

    def __len__(self):
        return len(self.ids)

def collate_function_cell_instance_segmentation(
        batch: List[Tuple[torch.Tensor]]) -> \
        Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Collate function of instance segmentation dataset.
    :param batch: (Tuple[Iterable[torch.Tensor], Iterable[torch.Tensor], Iterable[torch.Tensor], Iterable[torch.Tensor]])
    Batch of input data, instances maps, bounding boxes and class labels
    :return: (Tuple[torch.Tensor, Iterable[torch.Tensor], Iterable[torch.Tensor], Iterable[torch.Tensor]]) Batched input
    data, instances, bounding boxes and class labels are stored in a list due to the different instances.
    """
    return torch.stack([input_samples[0] for input_samples in batch], dim=0), \
           [input_samples[1] for input_samples in batch], \
           [input_samples[2] for input_samples in batch], \
           [input_samples[3] for input_samples in batch]


