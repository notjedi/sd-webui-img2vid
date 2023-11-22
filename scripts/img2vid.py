import os
from typing import Literal

import gradio as gr
from modules import script_callbacks
from torch.hub import download_url_to_file


def download_model(model_type: Literal["svd", "svd_xt"]):
    """Download SVD model.

    Args:
        model_type (str): type of model (vanilla or xt)
    """

    if model_type == "svd":
        model_url = "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd.safetensors"
        decoder_url = "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid/resolve/main/svd_image_decoder.safetensors"
    elif model_type == "svd_xt":
        model_url = "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt.safetensors"
        decoder_url = "https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt/resolve/main/svd_xt_image_decoder.safetensors"
    else:
        raise ValueError(f"invalid model type {model_type}")

    # model_checkpoint = os.path.join(ia_file_manager.models_dir, sam_model_id)
    model_checkpoint = ""
    if not os.path.isfile(model_checkpoint):
        try:
            download_url_to_file(model_url, model_checkpoint)
        except Exception as e:
            print(e)

    decoder_checkpoint = ""
    if not os.path.isfile(decoder_checkpoint):
        try:
            download_url_to_file(decoder_url, decoder_checkpoint)
        except Exception as e:
            print(e)


def generate_vid(model_type: str, img):
    if img is None:
        print("img is None, please choose an image")
    return None


def on_ui_tabs():
    with gr.Blocks(analytics_enabled=False) as ui_component:
        with gr.Row(variant="compact"):
            with gr.Column():
                model_type_radio = gr.Radio(
                    ["svd", "svd-xt"],
                    value="svd",
                    label="Model type",
                    info="Model type to use",
                )
                input_img = gr.Image()
                run_btn = gr.Button(value="Generate", variant="primary")
            output_vid = gr.Video(interactive=False)

            run_btn.click(
                generate_vid,
                inputs=[model_type_radio, input_img],
                outputs=[output_vid],
            )

    return [(ui_component, "img2vid", "svd_img2vid_tab")]


script_callbacks.on_ui_tabs(on_ui_tabs)
