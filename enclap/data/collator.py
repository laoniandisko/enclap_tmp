from dataclasses import dataclass

import torch
from transformers import BatchEncoding, DataCollatorForSeq2Seq


@dataclass
class DataCollatorForEnClapBart(DataCollatorForSeq2Seq):
    input_pad_token_id: int = 1024
    num_rvq: int = 16

    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors

        batch_size = len(features)
        # stacked_features = {k: [f[k] for f in features] for k in features[0]}
        clap_embedding = torch.Tensor(
            [feature["clap_embedding"] for feature in features]
        )

        pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.pad_token_id = self.input_pad_token_id
        keys = ["input_ids", "mcm_labels"]
        tmp_key_map = {"input_ids": "input_ids", "mcm_labels": "labels"}
        input_features = super().__call__(
            [
                {tmp_key_map[key]: feature[key][:, i] for key in keys}
                for feature in features
                for i in range(feature[keys[0]].shape[-1])
            ],
            return_tensors,
        )

        self.tokenizer.pad_token_id = 1
        keys = ["encodec_mask", "attention_mask", "labels"]
        tmp_key_map = {
            "encodec_mask": "input_ids",
            "attention_mask": "attention_mask",
            "labels": "labels",
        }
        other_features = super().__call__(
            [{tmp_key_map[key]: feature[key] for key in keys} for feature in features],
            return_tensors,
        )
        self.tokenizer.pad_token_id = pad_token_id

        return BatchEncoding(
            {
                "input_ids": input_features["input_ids"]
                .reshape(batch_size, self.num_rvq, -1)
                .transpose(1, 2),
                "mcm_labels": input_features["labels"]
                .reshape(batch_size, self.num_rvq, -1)
                .transpose(1, 2),
                "attention_mask": other_features["attention_mask"],
                "encodec_mask": other_features["input_ids"],
                "labels": other_features["labels"],
                "clap_embedding": clap_embedding,
            }
        )
