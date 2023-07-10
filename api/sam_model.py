import os

import gdown
import numpy as np
import torch
from rich import print
from segment_anything import (
    SamPredictor,
    sam_model_registry,
    sam_model_registry_baseline,
)


class SamModel:
    def __init__(self, path, download_url, name, device, is_hq=False):
        self.path = path
        self.download_url = download_url
        self.name = name
        self.device = device
        self.is_hq = is_hq

    def setup(self):
        if not self.is_hq:
            os.system(f"wget -nc -P models {self.download_url}")
        else:
            SamModel._download_checkpoint_from_google_drive(
                self.download_url,
                self.path,
            )

        sam_reg = sam_model_registry_baseline if not self.is_hq else sam_model_registry
        sam = sam_reg[self.name](checkpoint=self.path).to(device=self.device)
        self.predictor = SamPredictor(sam)
        print(f"SAM model loaded, type: {self.name}")

        return self

    def __call__(self, **kwargs):
        image = kwargs.get("image", None)
        xyxy = kwargs.get("xyxy", None)
        point_coords = kwargs.get("point_coords", None)
        point_labels = kwargs.get("point_labels", None)
        logits = kwargs.get("logits", None)

        self.predictor.reset_image()
        self.predictor.set_image(image)

        boxes = torch.tensor(xyxy).to(self.device) if xyxy is not None else None
        transformed_boxes = (
            self.predictor.transform.apply_boxes_torch(boxes, image.shape[:2])
            if boxes is not None
            else None
        )

        masks, scores, logits = self.predictor.predict_torch(
            point_coords=point_coords,
            point_labels=point_labels,
            boxes=transformed_boxes,
            multimask_output=True,
            mask_input=logits,
        )
        if len(masks) == 0:
            return None

        # Find highst score-index from each mask
        argmax_list = torch.argmax(scores, dim=1)
        mask, logits = (
            masks[np.arange(len(masks)), argmax_list][:, np.newaxis],
            logits[np.arange(len(logits)), argmax_list][:, np.newaxis],
        )
        return (mask, logits)

    @staticmethod
    def create_instance(device, sam_type, is_hq=False):
        sam_models = {
            "vit_h": SamModel(
                "./models/sam_vit_h_4b8939.pth"
                if not is_hq
                else "./models/sam_hq_vit_h.pth ",
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
                if not is_hq
                else "1BhhjSZB3HgZw6A6ATVBwuRMJhV6RPy_e",
                "vit_h",
                device=device,
                is_hq=is_hq,
            ),
            "vit_l": SamModel(
                "./models/sam_vit_l_0b3195.pth"
                if not is_hq
                else "./models/sam_hq_vit_l.pth",
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
                if not is_hq
                else "1UhUr9B18caw61MJZ4DK9KA1ti-rNgpz9",
                "vit_l",
                device=device,
                is_hq=is_hq,
            ),
            "vit_b": SamModel(
                "./models/sam_vit_b_01ec64.pth"
                if not is_hq
                else "./models/sam_hq_vit_b.pth",
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
                if not is_hq
                else "1XT6PL45V-aLP-ZQXkKQDwAzGUin2EfFu",
                "vit_b",
                device=device,
                is_hq=is_hq,
            ),
        }

        return sam_models.get(sam_type, None)

    @staticmethod
    def _download_checkpoint_from_google_drive(file_id, filepath):
        folder = os.path.dirname(filepath)
        os.makedirs(folder, exist_ok=True)
        if not os.path.exists(filepath):
            print(
                "Downloading HQ-SAM checkpoints from Google Drive... tips: If you cannot see the progress bar, please try to download it manuall \
                and put it in the checkpointes directory. Issue: https://github.com/SysCV/sam-hq/issues/5#issuecomment-1587379827)"
            )
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, filepath, quiet=False)
            print("Downloaded successfully!")

        return filepath
