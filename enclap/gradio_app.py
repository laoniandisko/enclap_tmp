import os
from typing import Tuple

import gradio as gr
import numpy as np
import torch
from transformers import AutoProcessor

from inference import EnClap


def input_toggle(choice: str):
    if choice == "file":
        return gr.update(visible=True), gr.update(visible=False)
    return gr.update(visible=False), gr.update(visible=True)


if __name__ == "__main__":
    import logging

    logging.getLogger().setLevel(logging.INFO)
    ckpt_path = "./ckpt"  # os.getenv("ckpt_path")
    device = "cpu"  # os.getenv("device")

    enclap = EnClap(ckpt_path=ckpt_path, device=device)

    def run_enclap(
        input_type: str,
        file_input: Tuple[int, np.ndarray],
        mic_input: Tuple[int, np.ndarray],
        seed: int,
    ) -> str:
        print(input_type, file_input, mic_input)
        input = file_input if input_type == "file" else mic_input
        if input is None:
            raise gr.Error("Input audio was not provided.")
        res, audio = input
        torch.manual_seed(seed)
        return enclap.infer_from_audio(torch.from_numpy(audio), res)[0]

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                radio = gr.Radio(
                    ["file", "mic"],
                    value="file",
                    label="Choose the input method of the audio.",
                )
                file = gr.Audio(label="Input", visible=True)
                mic = gr.Mic(label="Input", visible=False)
                slider = gr.Slider(minimum=0, maximum=100, label="Seed")
                radio.change(fn=input_toggle, inputs=radio, outputs=[file, mic])
                button = gr.Button("Run", label="run")
            with gr.Column():
                output = gr.Text(label="Output")
            button.click(
                fn=run_enclap, inputs=[radio, file, mic, slider], outputs=output
            )

    demo.launch()
