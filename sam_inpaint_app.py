from typing import Any
import PIL

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from rich import print
from streamlit_image_coordinates import streamlit_image_coordinates

from api.sam_model import SamModel
from utils import seed_everything

seed_everything(seed=42)


import supervision as sv
from diffusers import StableDiffusionInpaintPipeline
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

        if "inpaint_pipe" not in st.session_state["model"].keys():
            self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16,
            )
            self.inpaint_pipe = self.inpaint_pipe.to(self.device + ":0")
            st.session_state["model"]["inpaint_pipe"] = self.inpaint_pipe
        else:
            self.inpaint_pipe = st.session_state["model"]["inpaint_pipe"]

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
            mask = mask.to(torch.uint8)  # (1, 1, W, H)
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
        if "mask_nps" in st.session_state["app"].keys():
            del st.session_state["app"]["mask_nps"]
        if "inpaint" in st.session_state["app"].keys():
            del st.session_state["app"]["inpaint"]

    def __call__(self) -> Any:
        print("".join(["§"] * 180))
        print(f"app: {st.session_state['app']}")
        print("".join(["§"] * 180))

        def on_file_uploader_change():
            self._clear()

        uploaded = st.file_uploader("Upload a video", on_change=on_file_uploader_change)
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
                src_col, inpaint_col = st.columns([1, 1])
                with src_col:
                    new_coord = streamlit_image_coordinates(res_img)
                with inpaint_col:
                    if "inpaint" in st.session_state["app"].keys():
                        st.image(
                            st.session_state["app"]["inpaint"], use_column_width=False
                        )
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
                st.session_state["app"]["mask_nps"] = mask_nps
                st.session_state["app"]["applied_point_coords"] = applied_point_coords
                st.experimental_rerun()

        print("".join(["#"] * 180))
        print(f"app: {st.session_state['app']}")
        print("".join(["#"] * 180))

        # Inpaint via prompt, also inpainted by the last mask.
        if "sam_output" in st.session_state["app"].keys():
            prompt_col, inpaint_col = st.columns(2)
            with prompt_col:
                prompt = st.text_input("Prompt")
            with inpaint_col:
                st.write("")
                st.write("")
                if st.button("Magic"):
                    if prompt != "":
                        mask_np = st.session_state["app"]["mask_nps"][-1] # Use the last mask to inpaint.
                        mask_np[mask_np == 1] = 255
                        mask_image = Image.fromarray(mask_np)
                        # mask_image.save("inpaint-mask.png")

                        image = (
                            Image.fromarray(st.session_state["app"]["image"])
                            if "inpaint" not in st.session_state["app"].keys()
                            else st.session_state["app"]["inpaint"] # Continue to inpaint from the last inpaint.
                        )
                        ori_w, ori_h = image.size
                        w, h = (ori_w // 8) * 8, (ori_h // 8) * 8
                        resized_image, mask_image = image.resize(
                            (w, h)
                        ), mask_image.resize((w, h))
                        inpaint = self.inpaint_pipe(
                            prompt=prompt,
                            image=resized_image,
                            mask_image=mask_image,
                            width=w,
                            height=h,
                        ).images[0]
                        inpaint = inpaint.resize((ori_w, ori_h))
                        st.session_state["app"]["inpaint"] = inpaint
                        # inpaint.save("inpaint.png")
                        st.experimental_rerun()


if __name__ == "__main__":
    App()()
