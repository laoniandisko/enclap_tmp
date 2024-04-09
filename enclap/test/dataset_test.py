import sys
sys.path.append(".")
sys.path.append("..")

from datasets import load_dataset
from transformers import AutoTokenizer
from modeling.audiobart import AudioBartForConditionalGeneration
from torch.utils.data import DataLoader
from data.collator import EncodecCollator

import numpy as np
import torch
import os

if __name__=="__main__":
    model = AudioBartForConditionalGeneration.from_pretrained('bart/model')
    base_path = "/data/jyk/aac_dataset/AudioCaps/encodec_16/"
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
    data_files = {"train": "csv/AudioCaps/train.csv"}
    max_encodec_length = 1021
    clap_base_path = "/data/jyk/aac_dataset/AudioCaps/clap"
    
    raw_dataset = load_dataset("csv", data_files=data_files)

    def preprocess_function(example):
        path = example['file_path']
        encodec = np.load(os.path.join(base_path, path))
        if encodec.shape[0]>max_encodec_length:
            encodec = encodec[:max_encodec_length, :]
        clap = np.load(os.path.join(clap_base_path, path))
        attention_mask = np.ones(encodec.shape[0]+3).astype(np.int64)
        target_text = tokenizer(text_target=example['caption'])
 
        return {'input_ids': encodec, 'clap': clap, 'attention_mask': attention_mask, 'labels': target_text['input_ids'], 'decoder_attention_mask': target_text['attention_mask']}
   
    train_dataset = raw_dataset['train'].map(preprocess_function)
    train_dataset.set_format("pt", columns=['input_ids', 'attention_mask', 'clap', 'labels', 'decoder_attention_mask'])

    train_data_collator = EncodecCollator(
        tokenizer=tokenizer, 
        model=model, 
        return_tensors="pt",
        random_sampling=False,
        max_length=max_encodec_length, 
        num_subsampling=0,
        clap_masking_prob=-1,
        encodec_masking_prob=0.15,
        encodec_masking_length=10
    )
 
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=train_data_collator, batch_size=16)
    
    for idx, batch in enumerate(train_dataloader):
        # output = model.generate(**batch, max_length=100)
        output = model(**batch)
        print(output)
