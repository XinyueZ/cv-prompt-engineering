import numpy as np
import random
import torch
import cv2

def seed_everything(**args):
    """Seed everything for better reproducibility.
    NOTE that some pytorch operation is non-deterministic like the backprop of grid_samples
    """
    seed = args.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def draw_red_point_bbox_based_on_center(image_np):
    """Draw a red point in the center of the image"""
    height, width = image_np.shape[:2]
    center = (
        width // 2,
        height // 2,
    )
    bbox = (
        center[0] - width // 2,
        center[1] - height // 2,
        center[0] + width // 2,
        center[1] + height // 2,
    ) 
    # cv2 to draw bbox
    image_np = cv2.rectangle(image_np, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

    image_np[center[1], center[0]] = [255, 0, 0]
    return image_np

def draw_center_bbox(image_np, center, bbox):
    """Draw a red point in the center of the image""" 
    image_np[center[0], center[1]] = [255, 0, 0]
    # cv2 to draw bbox
    image_np = cv2.rectangle(image_np, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    return image_np


