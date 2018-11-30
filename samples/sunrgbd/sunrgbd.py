"""
Mask R-CNN
Train on the SUNRGBD dataset for object segmentation with depth images.

Licensed under the MIT License (see LICENSE for details)
By Cecilia Mauceri
Modified from code by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 sunrgbd.py train --dataset=/path/to/sunrgbd/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 sunrgbd.py train --dataset=/path/to/sunrgbd/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 sunrgbd.py train --dataset=/path/to/sunrgbd/dataset --weights=imagenet
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class SUNRGBDConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "sunrgbd"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 37  # Background + sunrgbd

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class SUNRGBDDataset(utils.Dataset):

    def load_sunrgbd(self, dataset_dir, subset):
        """Load a subset of the SUNRGBD dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """

        # Load annotations
        # Uses the cocoapi json format
        annotations = COCO("{}/SUNRGBD/instances_{}.json".format(dataset_dir, subset))

        # Add classes.
        class_ids = sorted(annotations.getCatIds())
        for cat_idx in class_ids:
            cat = annotations.loadCats(cat_idx)[0]["name"]
            self.add_class("sunrgbd", cat_idx, cat)

        # Add images
        image_ids = list(annotations.imgs.keys())
        for i in image_ids:
            self.add_image(
                "sunrgbd", image_id=i,
                path=os.path.join(dataset_dir, annotations.imgs[i]['file_name']),
                width=annotations.imgs[i]["width"],
                height=annotations.imgs[i]["height"],
                annotations=annotations.loadAnns(annotations.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # If not a SUNRGBD image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "sunrgbd":
            return super(SUNRGBDDataset, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id(
                "sunrgbd.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation['segmentation'])

                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue

                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(SUNRGBDDataset, self).load_mask(image_id)

    def image_reference(self, image_id):
        """Return path to the image """
        info = self.image_info[image_id]
        if info["path"]:
            return info["path"]
        else:
            super(SUNRGBDDataset, self).image_reference(image_id)

    def annToMask(self, seg):
        """
        Convert annotation which is RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        m = maskUtils.decode(seg)
        return m


def train(model, dataset, config=SUNRGBDConfig()):
    """Train the model."""
    # Training dataset.
    dataset_train = SUNRGBDDataset()
    dataset_train.load_sunrgbd(dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SUNRGBDDataset()
    dataset_val.load_sunrgbd(dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect object segments with depth images.')
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/sunrgbd/dataset/",
                        help='Directory of the SUNRGBD dataset')
    parser.add_argument('--weights', required=False,
                        default='coco',
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    args = parser.parse_args()

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    config = SUNRGBDConfig()
    config.display()

    # Create model
    model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train
    train(model, args.dataset)

