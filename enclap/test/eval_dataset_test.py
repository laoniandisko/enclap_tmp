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
    basepath = "/data/jyk/aac_dataset/clotho/encodec/"
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
    data_files = {"validation": "csv/valid_allcaps.csv"}
    num_captions = 5
    
    raw_dataset = load_dataset("csv", data_files=data_files)

    def preprocess_eval(example):
        path = example['file_path']
        encodec = np.load(os.path.join(basepath, path))
        if encodec.shape[0]>1022:
            encodec = encodec[:1022, :]
        attention_mask = np.ones(encodec.shape[0]+2).astype(np.int64)
        captions = []
        for i in range(1, num_captions+1):
            captions.append(example['caption_'+str(i)])

        return {'input_ids': encodec, 'attention_mask': attention_mask, 'captions': captions}

    train_dataset = raw_dataset['validation'].map(preprocess_eval)
    train_dataset.set_format('pt', columns=['input_ids', 'attention_mask'], output_all_columns=True)
    # train_dataset.remove_columns('file_path', 'caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5')
    data_collator = EncodecCollator(tokenizer=tokenizer, model=model, return_tensors="pt")
 
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=16)
    
    for idx, batch in enumerate(train_dataloader):
        output = model.generate(**batch, max_length=100)
        print(output)
