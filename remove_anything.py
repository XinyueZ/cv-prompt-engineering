from typing import Any
import PIL

import cv2
import numpy as np
from regex import F
import streamlit as st
import torch
from PIL import Image
from rich import print
from streamlit_image_coordinates import streamlit_image_coordinates

from api.sam_model import SamModel
from utils import seed_everything, VideoWriter

seed_everything(seed=42)


import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))

import supervision as sv
from diffusers import StableDiffusionInpaintPipeline
from supervision.detection.core import Detections
from supervision.detection.utils import mask_to_xyxy
from tqdm import tqdm

from api.tracker_model import TrackerModel


class App:
    _mask_selection_start_idx: int

    def __init__(self, device="cuda"):
        st.set_page_config(layout="wide", initial_sidebar_state="collapsed")

        self.device = device

        st.session_state["app"] = (
            st.session_state["app"] if "app" in st.session_state else dict()
        )
        st.session_state["model"] = (
            st.session_state["model"] if "model" in st.session_state else dict()
        )

        # SAM
        if "sam" not in st.session_state["model"].keys():
            self.sam = SamModel.create_instance(
                self.device + ":1", "vit_h", is_hq=True
            ).setup()
            st.session_state["model"]["sam"] = self.sam
        else:
            self.sam = st.session_state["model"]["sam"]

        if "tracker" not in st.session_state["model"].keys():
            self.tracker = TrackerModel.create_instance(self.device + ":0").setup()
            st.session_state["model"]["tracker"] = self.tracker
        else:
            self.tracker = st.session_state["model"]["tracker"]

    def _gen_mask(self):
        mask_nps, applied_point_coords = list(), list()
        image_np = np.asarray(st.session_state["app"]["image"])

        logits = None
        runner = tqdm(st.session_state["app"]["coords"])
        for coords in runner:
            # coords of positive: [[x1, y1], [x2, y2], ...]
            # coords of negative: [[-x1, -y1], [-x2, -y2], ...]
            if len(coords) == 0:
                continue

            applied_point_coords.extend(coords)

            point_labels = [
                list(map(lambda coord: 0 if coord[0] < 0 else 1, coords))
            ]  # Assign labels associated with coords (positive or negative), one batch, so (1, n)
            coords = list(
                map(
                    lambda coord: [-coord[0], -coord[1]] if coord[0] < 0 else coord,
                    coords,
                )
            )  # SAM cannot understand negative coords, so we need to convert them to positive

            point_coords = [coords]  # one batch, so (1, n, 2)
            # print(
            #     f"point_coords: {point_coords}",
            #     f"point_labels: {point_labels}",
            #     sep="\n",
            # )
            assert len(point_coords[0]) == len(
                point_labels[0]
            ), "count of coords and labels must be equal"

            point_coords_tensor = torch.tensor(point_coords).to(self.device + ":1")
            point_labels_tensor = torch.tensor(point_labels).to(self.device + ":1")

            mask, logits = self.sam(
                image=image_np,
                point_coords=point_coords_tensor,
                point_labels=point_labels_tensor,
                logits=logits,
                return_logits=False,
            )
            mask = mask.to(torch.uint8)  # (1, 1, W, H)
            mask_np = mask[0][0].cpu().numpy()
            # print(
            #     f"mask_np: {mask_np}", f"mask_np unique: {np.unique(mask_np)}", sep="\n"
            # )
            mask_nps.append(mask_np)

        # print("mask_nps--->", len(mask_nps))
        return image_np, applied_point_coords, np.stack(mask_nps)  # (n, W, H)

    def _clear(self):
        if "coords" in st.session_state["app"].keys():
            del st.session_state["app"]["coords"]
        if "sam_output" in st.session_state["app"].keys():
            del st.session_state["app"]["sam_output"]
        if "applied_point_coords" in st.session_state["app"].keys():
            del st.session_state["app"]["applied_point_coords"]
        if "mask_nps" in st.session_state["app"].keys():
            del st.session_state["app"]["mask_nps"]
        if "frames" in st.session_state["app"].keys():
            del st.session_state["app"]["frames"]
        if "template_mask" in st.session_state["app"].keys():
            del st.session_state["app"]["template_mask"]

    def __call__(self) -> Any:
        # print("".join(["§"] * 180))
        # print(f"app: {st.session_state['app']}")
        # print("".join(["§"] * 180))

        def on_file_uploader_change():
            self._clear()

        uploaded = st.file_uploader(
            "Upload a video, use first frame to select tracked object",
            on_change=on_file_uploader_change,
        )
        if uploaded is not None and "frames" not in st.session_state["app"].keys():
            temp_file = "./temp_video.mp4"
            with open(temp_file, "wb") as file:
                file.write(uploaded.getvalue())
            cap = cv2.VideoCapture(temp_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frames = []
            runner = tqdm(range(total_frames), desc="Reading video")
            for _ in runner:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = self.sam.apply_image(frame)  # Important for SAM
                frames.append(frame)
            os.remove(temp_file)
            uploaded = None
            st.session_state["app"]["frames"] = frames
            st.experimental_rerun()

        if "image" in st.session_state["app"].keys():
            st.write(f"_frame shape: {st.session_state['app']['image'] .shape}_")

        if "frames" in st.session_state["app"].keys():
            self._mask_selection_start_idx = st.slider(
                "Select frame",
                min_value=0,
                max_value=len(st.session_state["app"]["frames"]) - 1,
                value=0,
                step=1,
            )
            st.session_state["app"]["image"] = st.session_state["app"]["frames"][
                self._mask_selection_start_idx
            ]
            st.session_state["app"]["image"] = cv2.cvtColor(
                st.session_state["app"]["image"], cv2.COLOR_BGR2RGB
            )
            
        if "image" in st.session_state["app"].keys():
            col1, col2, col3 = st.columns([1, 1, 6])
            with col1:
                #
                # Label selection
                #
                with st.container():
                    st.radio(
                        "Labels, click `New mask` to start a new mask otherwise the multi-clicking is for ONLY-ONE mask, `Clear` to clear all.",
                        ["Positive", "Negative"],
                        index=0,
                        key="label_radios",
                        horizontal=True,
                    )
            with col2:
                st.write("")
                if st.button("New mask"):
                    if "coords" not in st.session_state["app"].keys():
                        st.session_state["app"]["coords"] = []
                    st.session_state["app"]["coords"].append([])
            with col3:
                #
                # Clean every thing
                #
                st.write("")
                if st.button("Clear"):
                    self._clear()
            #
            # Clickable image, draw the image
            #
            new_coord = None
            if "sam_output" not in st.session_state["app"].keys():
                print("sam_output not in app")
                new_coord = streamlit_image_coordinates(
                    st.session_state["app"]["image"],
                    # key="image_view",
                )
            else:
                print("sam_output in app")
                res_img = st.session_state["app"]["sam_output"]
                applied_point_coords = st.session_state["app"]["applied_point_coords"]
                for coord in applied_point_coords:
                    cv2.circle(
                        res_img,
                        list(map(lambda xy: xy * -1, coord))
                        if coord[0] < 0
                        else coord,  # If negative, convert to positive to draw
                        3,
                        (225, 225, 225)
                        if coord[0] > 0
                        else (
                            0,
                            0,
                            0,
                        ),  # Different color for positive and negative prompts
                        thickness=-1,
                    )

                new_coord = streamlit_image_coordinates(res_img)

                n_remove = st.slider(
                    "Number of frames to remove target",
                    min_value=300,
                    max_value=len(
                        st.session_state["app"]["frames"],
                    ),
                )
                if (
                    st.button("Track")
                    and "template_mask" in st.session_state["app"].keys()
                ):
                    st.info("Tracking...")
                    self.tracker.clear_memory()
                    masks, _, painted_images = self.tracker(
                        st.session_state["app"]["frames"][self._mask_selection_start_idx :],
                        st.session_state["app"]["template_mask"],
                    )

                    tracker_output_painted_img_dir = "./api/ProPainter/inputs/object_removal/sam_tracker_output/tracker"
                    tracker_output_mask_dir = "./api/ProPainter/inputs/object_removal/sam_tracker_output/tracker_mask"
                    os.system(f"rm -rf {tracker_output_painted_img_dir}")
                    os.system(f"rm -rf {tracker_output_mask_dir}")
                    os.makedirs(tracker_output_painted_img_dir)
                    os.makedirs(tracker_output_mask_dir)
                    runner = tqdm(enumerate(zip(masks, painted_images)))
                    for i, (mask, painted_image) in runner:
                        if i > n_remove:
                            break

                        cv2.imwrite(
                            os.path.join(
                                tracker_output_painted_img_dir,
                                f"{str(i).zfill(6)}.png",
                            ),
                            painted_image,
                        )
                        cv2.imwrite(
                            os.path.join(
                                tracker_output_mask_dir, f"{str(i).zfill(6)}.png"
                            ),
                            mask,
                        )
                    self.tracker.clear_memory()
                    st.info("Tracked")

                    st.info("Saving video...")
                    painted_images = (
                        st.session_state["app"]["frames"][
                            : self._mask_selection_start_idx
                        ]
                        + painted_images
                    )
                    video_wr = VideoWriter(out_dir="./", name="sam_tracker")
                    runner = tqdm(
                        enumerate(
                            zip(st.session_state["app"]["frames"], painted_images)
                        )
                    )
                    for i, (frame, painted_image) in runner:
                        if i > n_remove:
                            break
                        # Horizontally concatenate images
                        output_height = frame.shape[0]
                        output_width = int(
                            output_height
                            * painted_image.shape[1]
                            / painted_image.shape[0]
                        )
                        img1 = cv2.resize(frame, (output_width, output_height))
                        img2 = cv2.resize(painted_image, (output_width, output_height))
                        image_postprocessed = cv2.hconcat([img1, img2])
                        video_wr(image_postprocessed)

                    video_wr.release()
                    st.video(video_wr.out)
                    # os.remove(video_name)
                    st.info(f"Video ({video_wr.out}) saved")
                    with open(video_wr.out, "rb") as file:
                        st.download_button(
                            label="Download", data=file, file_name=video_wr.fn
                        )

            if new_coord is not None:
                new_coord = [new_coord["x"], new_coord["y"]]
                # print(f"new_coord: {new_coord}")

                if "coords" not in st.session_state["app"].keys():
                    st.session_state["app"]["coords"] = [[]]
                # Push the new coordinate to the last label group
                if st.session_state["label_radios"] == "Positive":
                    st.session_state["app"]["coords"][-1].append(new_coord)
                else:
                    st.session_state["app"]["coords"][-1].append(
                        list(map(lambda xy: xy * -1, new_coord))
                    )
                # Gen mask
                image_np, applied_point_coords, mask_nps = self._gen_mask()
                # print(f"mask_nps.shape", mask_nps.shape)
                detections = Detections(xyxy=mask_to_xyxy(mask_nps), mask=mask_nps)
                mask_annotator = sv.MaskAnnotator()
                st.session_state["app"]["sam_output"] = mask_annotator.annotate(
                    scene=image_np, detections=detections
                )
                template_mask = mask_nps[0]
                for i in range(1, len(mask_nps)):
                    template_mask = np.clip(
                        template_mask + mask_nps[i] * (i + 1), 0, i + 1
                    )
                st.session_state["app"]["mask_nps"] = mask_nps
                st.session_state["app"]["template_mask"] = template_mask
                st.session_state["app"]["applied_point_coords"] = applied_point_coords
                st.experimental_rerun()

        # print("".join(["#"] * 180))
        # print(f"app: {st.session_state['app']}")
        # print("".join(["#"] * 180))


if __name__ == "__main__":
    App()()
