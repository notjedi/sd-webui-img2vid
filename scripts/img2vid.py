import os
from typing import Literal

import gradio as gr
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
from modules import script_callbacks, shared
from PIL import Image


def get_pipeline(model_type: Literal["svd", "svd_xt"]):
    if model_type == "svd":
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid",
            torch_dtype=torch.float16,
            variant="fp16",
        )
    elif model_type == "svd_xt":
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid-xt",
            torch_dtype=torch.float16,
            variant="fp16",
        )
    else:
        raise ValueError(f"invalid model type {model_type}")
    pipe.enable_model_cpu_offload()
    pipe.to("cuda")
    return pipe


def generate_vid(
    img: Image.Image,
    model_type: Literal["svd", "svd_xt"],
    width: int,
    height: int,
    motion_bucket_id: int,
    decode_chunk_size: int,
    num_frames: int,
    fps: int,
    seed: int,
):
    # num_inference_steps: int = 25,
    # noise_aug_strength: int = 0.02,
    # num_videos_per_prompt: Optional[int] = 1,
    # min_guidance_scale: float = 1.0,
    # max_guidance_scale: float = 3.0,
    # resize_mode

    if img is None:
        print("img is None, please choose an image")
        return None

    pipe = get_pipeline(model_type)
    img = img.resize((width, height))
    # TODO: clamp decode_chunk_size

    generator = torch.manual_seed(seed)
    frames = pipe(
        img,
        fps=fps,
        width=width,
        height=height,
        num_frames=num_frames,
        motion_bucket_id=motion_bucket_id,
        decode_chunk_size=decode_chunk_size,
        generator=generator,
    ).frames[0]

    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "outputs")
    vid_path = os.path.join(out_dir, "generated.mp4")
    export_to_video(frames, vid_path, fps=fps)
    return vid_path


def on_ui_tabs():
    # TODO: have a text box for seed
    # TODO: have a slider for width and height
    # TODO: have a text box for decode_chunk_size
    # TODO: have a resize checkbox to resize the image

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
                    width = gr.Slider(
                        label="Width", minimum=64, maximum=1024, step=32, value=512
                    )
                    height = gr.Slider(
                        label="Height", minimum=64, maximum=1024, step=32, value=288
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
                    min_width=100,
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
                    seed,
                ],
                outputs=[output_vid],
            )

    return [(ui_component, "img2vid", "svd_img2vid_tab")]


def on_ui_settings():
    section = ("svd_img2vid_tab", "img2vid")
    shared.opts.add_option(
        "svd_models_dir",
        shared.OptionInfo(
            default="",
            label="Stable Video Diffusion models directory; If empty, defaults to [img2vid extension folder]/models",
            component=gr.Textbox,
            component_args={"interactive": True},
            section=section,
        ),
    )


script_callbacks.on_ui_tabs(on_ui_tabs)
script_callbacks.on_ui_settings(on_ui_settings)
