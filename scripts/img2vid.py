import os
import random
import time
from typing import Literal, Optional

import gradio as gr
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from PIL import Image

MAX_64_BIT_INT = 2**63 - 1
LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS

svd_pipeline: Optional["SVDPipeline"] = None
out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "outputs")
os.makedirs(out_dir, exist_ok=True)


class SVDPipeline:
    def __init__(
        self,
        model_type: Literal["svd", "svd_xt"],
    ):
        self.model_type = model_type
        if self.model_type == "svd":
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid",
                torch_dtype=torch.float16,
                variant="fp16",
            )
        elif self.model_type == "svd_xt":
            self.pipe = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                torch_dtype=torch.float16,
                variant="fp16",
            )
        else:
            raise ValueError(f"invalid model type {self.model_type}")
        self.pipe.enable_model_cpu_offload()
        self.pipe.to("cuda")


def resize_image(resize_mode, im, width, height):
    """
    Resizes an image with the specified resize_mode, width, and height.

    Args:
        resize_mode: The mode to use when resizing the image.
            0: Resize the image to the specified width and height.
            1: Resize the image to fill the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, cropping the excess.
            2: Resize the image to fit within the specified width and height, maintaining the aspect ratio, and then center the image within the dimensions, filling empty with data from image.
        im: The image to resize.
        width: The width to resize the image to.
        height: The height to resize the image to.
    """

    def resize(im, w, h):
        if im.mode == "L":
            return im.resize((w, h), resample=LANCZOS)

        im = im.resize((w, h), resample=LANCZOS)
        return im

    if resize_mode == 0:  # Just resize
        res = resize(im, width, height)

    elif resize_mode == 1:  # Crop and resize
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

    else:  # Resize and fill
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = resize(im, src_w, src_h)
        res = Image.new("RGB", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            if fill_height > 0:
                res.paste(
                    resized.resize((width, fill_height), box=(0, 0, width, 0)),
                    box=(0, 0),
                )
                res.paste(
                    resized.resize(
                        (width, fill_height),
                        box=(0, resized.height, width, resized.height),
                    ),
                    box=(0, fill_height + src_h),
                )
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            if fill_width > 0:
                res.paste(
                    resized.resize((fill_width, height), box=(0, 0, 0, height)),
                    box=(0, 0),
                )
                res.paste(
                    resized.resize(
                        (fill_width, height),
                        box=(resized.width, 0, resized.width, height),
                    ),
                    box=(fill_width + src_w, 0),
                )

    return res


def generate_vid(
    img: Image.Image,
    model_type: Literal["svd", "svd_xt"],
    width: int,
    height: int,
    motion_bucket_id: int,
    decode_chunk_size: int,
    num_frames: int,
    fps: int,
    num_inference_steps: int,
    noise_aug_strength: float,
    seed: int,
) -> str:
    global svd_pipeline, out_dir
    if img is None:
        raise ValueError("img cannot be None")
    img = img.resize((width, height))

    if seed < 0:
        seed = random.randint(0, MAX_64_BIT_INT)
    if svd_pipeline is None or svd_pipeline.model_type != model_type:
        del svd_pipeline
        svd_pipeline = SVDPipeline(model_type)

    generator = torch.manual_seed(seed)
    frames = svd_pipeline.pipe(
        img,
        fps=fps,
        width=width,
        height=height,
        num_frames=num_frames,
        motion_bucket_id=motion_bucket_id,
        decode_chunk_size=decode_chunk_size,
        noise_aug_strength=noise_aug_strength,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).frames[0]

    file_name = int(time.time())
    vid_path = os.path.join(out_dir, f"{file_name}.mp4")
    export_to_video(frames, vid_path, fps)
    return vid_path


def on_ui_tabs():
    # TODO: resize_mode
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Row(variant="compact"):
            with gr.Column():
                input_img = gr.Image(type="pil")
                run_btn = gr.Button(value="Generate", variant="primary")
                model_type_radio = gr.Radio(
                    ["svd", "svd-xt"],
                    value="svd",
                    label="Model type",
                    info="Model type to use",
                )
                with gr.Row():
                    resize_mode = gr.Radio(
                        label="Resize mode",
                        choices=[
                            "Just resize",
                            "Crop and resize",
                            "Resize and fill",
                            "Just resize (latent upscale)",
                        ],
                        type="index",
                        value="Just resize",
                    )
                with gr.Row():
                    width = gr.Slider(
                        label="Width", minimum=64, maximum=1024, step=64, value=1024
                    )
                    height = gr.Slider(
                        label="Height", minimum=64, maximum=1024, step=64, value=576
                    )

                with gr.Row():
                    num_frames = gr.Slider(
                        label="Number of frames",
                        step=1,
                        value=25,
                        minimum=1,
                        maximum=150,
                    )
                    fps = gr.Slider(
                        label="Frames per second",
                        step=1,
                        value=7,
                        minimum=5,
                        maximum=30,
                    )
                seed = gr.Number(
                    label="Seed",
                    value=-1,
                    precision=0,
                )
                with gr.Accordion("Advanced options", open=False):
                    motion_bucket_id = gr.Slider(
                        label="Motion bucket id",
                        info="Controls how much motion to add/remove from the image. The higher the number the more motion will be in the video.",
                        step=1,
                        value=127,
                        minimum=1,
                        maximum=255,
                    )
                    decode_chunk_size = gr.Slider(
                        label="Decoding chunk size",
                        info="The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency between frames, but also the higher the memory consumption. Set to a lower if you are getting OOMs.",
                        step=1,
                        value=4,
                        minimum=1,
                        maximum=150,
                    )
                    num_inference_steps = gr.Slider(
                        label="Number of inference steps chunk size",
                        info="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.",
                        step=1,
                        value=25,
                        minimum=1,
                        maximum=150,
                    )
                    noise_aug_strength = gr.Slider(
                        label="Noise strength",
                        info="The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.",
                        step=0.01,
                        value=0.02,
                        minimum=0.01,
                        maximum=1.0,
                    )

            output_vid = gr.Video(interactive=False)
            run_btn.click(
                generate_vid,
                inputs=[
                    input_img,
                    model_type_radio,
                    width,
                    height,
                    motion_bucket_id,
                    decode_chunk_size,
                    num_frames,
                    fps,
                    num_inference_steps,
                    noise_aug_strength,
                    seed,
                ],
                outputs=[output_vid],
            )

    return [(ui_component, "img2vid", "svd_img2vid_tab")]


if __name__ == "__main__":
    ((demo, _, _),) = on_ui_tabs()
    demo.launch()
else:
    from modules import script_callbacks

    script_callbacks.on_ui_tabs(on_ui_tabs)
