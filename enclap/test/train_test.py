from datasets import load_dataset
from transformers import AutoTokenizer
from modeling.audiobart import AudioBartForConditionalGeneration
from data.collator import EncodecCollator
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

import numpy as np
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

if __name__=="__main__":
    model = AudioBartForConditionalGeneration.from_pretrained('bart/model')
    basepath = "/data/jyk/aac_dataset/clotho/encodec/"
    tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large')
    data_files = {"train": "csv/train_short.csv", "validation": "csv/valid_short.csv"}
    
    raw_dataset = load_dataset("csv", data_files=data_files)

    def preprocessing(example):
        path = example['file_path']
        encodec = np.load(os.path.join(basepath, path))
        if encodec.shape[0]>1022:
            encodec = encodec[:1022, :]
        attention_mask = np.ones(encodec.shape[0]+2)
        target_text = tokenizer(text_target=example['caption'])
 
        return {'input_ids': encodec , 'attention_mask': attention_mask, 'labels': target_text['input_ids'], 'decoder_attention_mask': target_text['attention_mask']}
    
    train_dataset = raw_dataset['validation'].map(preprocessing)
    train_dataset.set_format("pt", columns=['input_ids', 'attention_mask', 'labels', 'decoder_attention_mask'])

    data_collator = EncodecCollator(tokenizer=tokenizer, model=model, return_tensors="pt")

    training_args = Seq2SeqTrainingArguments('summary_test', per_gpu_train_batch_size=20)

    trainer = Seq2SeqTrainer(
        model, training_args, train_dataset=train_dataset, eval_dataset=train_dataset, data_collator=data_collator, tokenizer=tokenizer
    )

    trainer.train()