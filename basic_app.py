from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from regex import W
import streamlit as st
import torch
from PIL import Image
from rich import print
from streamlit_cropper import st_cropper
from streamlit_drawable_canvas import st_canvas

from api.sam_model import SamModel
from utils import seed_everything, draw_red_point_bbox_based_on_center, draw_center_bbox

seed_everything(seed=42)

import os
from collections import defaultdict
import torchvision.transforms as transforms
from diffusers import StableDiffusionInpaintPipeline, DiffusionPipeline


@dataclass
class Size:
    width: int = 800
    height: int = 250

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return (self.width, self.height)


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

        # diffusion model
        if "pipe" not in st.session_state["model"].keys():
            self.pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-1",
                torch_dtype=torch.float16,
            )
            self.pipe = self.pipe.to(self.device + ":0")
            st.session_state["model"]["pipe"] = self.pipe
        else:
            self.pipe = st.session_state["model"]["pipe"]

        if "inpaint_pipe" not in st.session_state["model"].keys():
            self.inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                "stabilityai/stable-diffusion-2-inpainting",
                torch_dtype=torch.float16,
            )
            self.inpaint_pipe = self.inpaint_pipe.to(self.device + ":0")
            st.session_state["model"]["inpaint_pipe"] = self.inpaint_pipe
        else:
            self.inpaint_pipe = st.session_state["model"]["inpaint_pipe"]

        # SAM
        if "sam" not in st.session_state["model"].keys():
            self.sam = SamModel.create_instance(
                self.device + ":1", "vit_h", is_hq=True
            ).setup()
            st.session_state["model"]["sam"] = self.sam
        else:
            self.sam = st.session_state["model"]["sam"]

    def run(self):
        main_cols = st.columns([2, 1])
        #
        # Input and buttons
        #
        with main_cols[0]:
            prompt = st.text_input("Prompt", placeholder="Prompt....")
            print("prompt", prompt)

            btn_cols = st.columns(4)
            with btn_cols[0]:
                if st.button("Generate") and prompt is not None and prompt != "":
                    st.session_state["app"].clear()
                    st.session_state["app"]["image"] = self.pipe(prompt).images[0]
                    st.session_state["app"]["image"].save("gen.png")
            #
            # Cropper
            #
            with btn_cols[1]:
                if st.button("Crop"):
                    image_np = np.array(st.session_state["app"]["image"])
                    mask, _ = self.sam(
                        image=image_np,
                        xyxy=np.array([st.session_state["app"]["xyxy"]]),
                    )
                    mask = mask.to(torch.int8)
                    mask_np = mask[0][0].cpu().numpy()

                    image_np[mask_np == 0] = 0
                    st.session_state["app"]["image"] = Image.fromarray(image_np)
            #
            # Inpaint
            #
            with btn_cols[2]:
                if st.button("Inpaint"):
                    width, height = (
                        st.session_state["app"]["image"].width,
                        st.session_state["app"]["image"].height,
                    )
                    #
                    # SAM to find inpaint area
                    #
                    image_np = np.array(st.session_state["app"]["image"])
                    inpaint_bbox = st.session_state["app"]["xyxy"]
                    mask, _ = self.sam(
                        image=image_np,
                        xyxy=np.array([inpaint_bbox]),
                    )
                    mask = mask.to(torch.int8)
                    mask_np = mask[0][0].cpu().numpy()
                    mask_np[mask_np == 1] = 255
                    mask_image = Image.fromarray(mask_np)
                    mask_image.save("inpaint-mask.png")  
                    image_source_for_inpaint = st.session_state["app"]["image"]
                    image_mask_for_inpaint = mask_image
                    inpaint = self.inpaint_pipe(
                        prompt=prompt,
                        image=image_source_for_inpaint,
                        mask_image=image_mask_for_inpaint,
                        width=width,
                        height=height,
                    ).images[0]
                    inpaint = inpaint.resize(
                        (width, height)
                    )  
                    inpaint.save("inpaint.png")
                    st.session_state["app"]["image"] = inpaint
            #
            # paint
            #
            with btn_cols[3]:
                if st.button("Paint"):
                    width, height = (
                        st.session_state["app"]["image"].width,
                        st.session_state["app"]["image"].height,
                    )
                    paint = self.pipe(prompt, width=width, height=height).images[0]
                    #
                    # Get center of paint
                    #
                    paint_width, paint_height = (width, height)
                    bbox_width, bbox_height = (
                        st.session_state["app"]["xywh"]["width"],
                        st.session_state["app"]["xywh"]["height"],
                    )
                    bbox_left, bbox_top = (  # left, top
                        st.session_state["app"]["xywh"]["left"],
                        st.session_state["app"]["xywh"]["top"],
                    )
                    center = (paint_width // 2, paint_height // 2)
                    point_coords = np.array([[center]])
                    point_labels = np.ones(point_coords.shape[1])[None, :]
                    point_coords = torch.tensor(point_coords).to(self.device + ":1")
                    point_labels = torch.tensor(point_labels).to(self.device + ":1")
                    #
                    # Get bbox based on center
                    #
                    center_bbox = (
                        center[0] - bbox_width // 2,
                        center[1] - bbox_height // 2,
                        center[0] + bbox_width // 2,
                        center[1] + bbox_height // 2,
                    )
                    #
                    # SAM
                    #
                    paint_np = np.array(paint)
                    # draw_center_bbox(paint_np, center, bbox)
                    Image.fromarray(paint_np).save("paint.png")

                    mask, _ = self.sam(
                        image=paint_np,
                        point_coords=point_coords,
                        point_labels=point_labels,
                        xyxy=np.array([center_bbox]),
                    )
                    mask = mask.to(torch.int8)
                    mask_np = mask[0][0].cpu().numpy()

                    paint_np[mask_np == 0] = 0
                    # draw_red_point_bbox_based_on_center(paint_np)
                    paint = Image.fromarray(paint_np)
                    paint = paint.convert("RGBA")
                    paint_data = paint.getdata()
                    new_data = []
                    for item in paint_data:
                        if item[0] == 0 and item[1] == 0 and item[2] == 0:
                            new_data.append((255, 255, 255, 0))
                        else:
                            new_data.append(item)
                    paint.putdata(new_data)
                    paint = paint.resize((bbox_width, bbox_height))
                    paint.save("paint-masked.png")

                    alpha = Image.new("RGBA", paint.size, (255, 255, 255, 0))
                    alpha.paste(paint, (0, 0), paint)
                    st.session_state["app"]["image"].paste(
                        alpha, (bbox_left, bbox_top), alpha
                    )
        #
        # Show image
        #
        with main_cols[1]:
            #
            # Show current image
            #
            if (
                "image" in st.session_state["app"].keys()
                and st.session_state["app"]["image"] is not None
            ):
                st.session_state["app"]["xywh"] = st_cropper(
                    st.session_state["app"]["image"],
                    realtime_update=True,
                    box_color="#0000FF",
                    return_type="box",
                )

                st.session_state["app"]["xyxy"] = (
                    st.session_state["app"]["xywh"]["left"],
                    st.session_state["app"]["xywh"]["top"],
                    st.session_state["app"]["xywh"]["width"]
                    + st.session_state["app"]["xywh"]["left"],
                    st.session_state["app"]["xywh"]["height"]
                    + st.session_state["app"]["xywh"]["top"],
                )

        print("app: ", st.session_state["app"])
        print("=====================================================================")


if __name__ == "__main__":
    app = App(device="cuda" if torch.cuda.is_available() else "cpu")
    app.run()
