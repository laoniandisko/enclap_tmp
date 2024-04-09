import argparse
from math import ceil
from pathlib import Path

import torch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", "-c", type=str)
    args = parser.parse_args()

    weight_name_map = {
        "model.encodec_embeddings": None,
        "encodec_embeddings": "embed_encodec",
        "encodec_mlm_head": "mcm_heads",
    }

    ckpt_path = Path(args.ckpt_path)
    weight_file = ckpt_path / "pytorch_model.bin"
    state_dict = torch.load(weight_file, map_location="cpu")
    new_state_dict = {}
    for key in state_dict:
        new_key = key
        for orig, repl in weight_name_map.items():
            if repl is None:
                if orig in new_key:
                    new_key = None
                    break
                continue
            new_key = new_key.replace(orig, repl)
        if new_key:
            new_state_dict[new_key] = state_dict[key]
    for key in new_state_dict:
        if "model.encoder.embed_encodec" in key:
            dim = new_state_dict[key].shape[0]
            new_weight = torch.normal(
                0, 1, (ceil(dim / 64) * 64, new_state_dict[key].shape[1])
            )
            new_weight[:dim] = new_state_dict[key]
            new_state_dict[key] = new_weight
    weight_file.rename(weight_file.with_suffix(".bin.bak"))
    torch.save(new_state_dict, weight_file)
