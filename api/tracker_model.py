# %% #########################################################################################################
# Tracker model
#
import os
import sys
from tqdm import tqdm
import requests
import numpy as np
import torch

pkg_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(pkg_root) 
sys.path.append(os.path.join(pkg_root, "tracker"))
sys.path.append(os.path.join(pkg_root, "tracker", "config"))
sys.path.append(os.path.join(pkg_root, "tracker", "utils"))
sys.path.append(os.path.join(pkg_root, "tracker", "model"))
sys.path.append(os.path.join(pkg_root, "tracker", "inference"))
sys.path.append(os.path.join(pkg_root, "tracker", "model", "modules"))

from api.tracker import base_tracker

from rich import print


class TrackerModel:
    def __init__(self, model, device) -> None:
        self.model = model
        self.device = device

    def setup(self):
        self.tracker = base_tracker.BaseTracker(self.model, self.device)
        return self

    def __call__(self, images: list, template_mask: np.ndarray):
        masks = []
        logits = []
        painted_images = []
        for i in tqdm(range(len(images)), desc="Tracking image"):
            if i == 0:
                mask, logit, painted_image = self.tracker.track(
                    images[i], template_mask
                )
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)

            else:
                mask, logit, painted_image = self.tracker.track(images[i])
                masks.append(mask)
                logits.append(logit)
                painted_images.append(painted_image)
        return masks, logits, painted_images

    def clear_memory(self):
        torch.cuda.empty_cache()
        self.tracker.clear_memory()

    @staticmethod
    def create_instance(device):
        def download_checkpoint(url, folder, filename):
            os.makedirs(folder, exist_ok=True)
            filepath = os.path.join(folder, filename)

            if not os.path.exists(filepath):
                print("Download checkpoints ......")
                response = requests.get(url, stream=True)
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                print("download successfully!")

            return filepath

        return TrackerModel(
            download_checkpoint(
                "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth",
                "./models",
                "XMem-s012.pth",
            ),
            device,
        )
