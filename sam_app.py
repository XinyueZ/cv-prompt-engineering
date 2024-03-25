from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import streamlit as st
from sympy import det
import torch
from PIL import Image
from regex import W
from rich import print
from streamlit_cropper import st_cropper
from streamlit_drawable_canvas import st_canvas
from streamlit_image_coordinates import streamlit_image_coordinates

from api.sam_model import SamModel
from utils import seed_everything

seed_everything(seed=42)

import os
from collections import defaultdict

import supervision as sv
from supervision.detection.core import Detections
from supervision.detection.utils import mask_to_xyxy

from tqdm import tqdm


class App:
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
            print(
                f"point_coords: {point_coords}",
                f"point_labels: {point_labels}",
                sep="\n",
            )
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
            mask = mask.to(torch.uint8) # (1, 1, W, H)
            mask_np = mask[0][0].cpu().numpy()
            print(
                f"mask_np: {mask_np}", f"mask_np unique: {np.unique(mask_np)}", sep="\n"
            )
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

    def __call__(self) -> Any:
        print("".join(["§"] * 180))
        print(f"app: {st.session_state['app']}")
        print("".join(["§"] * 180))

        def on_file_uploader_change():
            self._clear()

        uploaded = st.file_uploader("Upload an image", on_change=on_file_uploader_change)
        if uploaded is not None:
            image = Image.open(uploaded)
            image = image.resize((image.size[0], image.size[1]))
            # In order to make it work with SAM
            st.session_state["app"]["image"] = self.sam.apply_image(np.asarray(image))

        if "image" in st.session_state["app"].keys():
            col1, col2, col3 = st.columns([1, 1, 6])
            with col1:
                #
                # Label selection
                #
                with st.container():

                    def on_label_radios_change():
                        print(f"label changed: {st.session_state['label_radios']}")
                        if "coords" not in st.session_state["app"].keys():
                            st.session_state["app"]["coords"] = [[]]
                            return
                        # Every time the label is changed, add a new list to the coords list
                        # for the new label
                        st.session_state["app"]["coords"].append(list())

                    st.radio(
                        "Labels, click `New mask` to start a new mask otherwise the multi-clicking is for ONLY-ONE mask, `Clear` to clear all.",
                        ["Positive", "Negative"],
                        index=0,
                        # on_change=on_label_radios_change,
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

            if new_coord is not None:
                # print(f"new_coord:::: {new_coord}")
                new_coord = [new_coord["x"], new_coord["y"]]
                print(f"new_coord: {new_coord}")

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
                st.session_state["app"]["applied_point_coords"] = applied_point_coords
                st.experimental_rerun()

        print("".join(["#"] * 180))
        print(f"app: {st.session_state['app']}")
        print("".join(["#"] * 180))


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    App(device)()
