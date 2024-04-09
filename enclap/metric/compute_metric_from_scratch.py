import sys 
sys.path.append('..')
sys.path.append('.')

from aac_metrics import evaluate
from inference import AudioBartInference
from tqdm import tqdm
import os
import pandas as pd 

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
metric_list = ["bleu_1", "bleu_4", "rouge_l", "meteor", "spider_fl"]

if __name__ == "__main__":
    dataset = "AudioCaps"
    # dataset = "clotho"
    ckpt_path = "/data/jyk/aac_results/bart_base/audiocaps_35e5_2000/checkpoints/epoch_8"

    # ckpt_path = "/data/jyk/aac_results/masking/linear_scalinEg/checkpoints/epoch_14"
    max_encodec_length = 1022
    infer_module = AudioBartInference(ckpt_path, max_encodec_length)
    from_encodec = True
    csv_path = f"/workspace/audiobart/csv/{dataset}/test.csv"
    base_path = f"/data/jyk/aac_dataset/{dataset}/encodec_16"
    clap_name = "clap_audio_fused"
    df = pd.read_csv(csv_path)

    generation_config = {
        "_from_model_config": True,
        "bos_token_id": 0,
        "decoder_start_token_id": 2,
        "early_stopping": True,
        "eos_token_id": 2,
        "forced_bos_token_id": 0,
        "forced_eos_token_id": 2,
        "no_repeat_ngram_size": 3,
        "num_beams": 4,
        "pad_token_id": 1,
        "max_length": 50
    }

    print(f"> Making Predictions for model {ckpt_path}...")
    predictions = []
    references = []
    for idx in tqdm(range(len(df)), dynamic_ncols=True, colour="BLUE"):
        if not from_encodec:
            wav_path = df.loc[idx]['file_name']
        else:
            wav_path = df.loc[idx]['file_path']
        wav_path = os.path.join(base_path,wav_path)
        if not os.path.exists(wav_path):
            pass
        
        if not from_encodec:
            prediction = infer_module.infer(wav_path)
        else:
            prediction = infer_module.infer_from_encodec(wav_path, clap_name, generation_config)

        predictions.append(prediction[0])
        reference = [df.loc[idx]['caption_1'],df.loc[idx]['caption_2'],df.loc[idx]['caption_3'],df.loc[idx]['caption_4'],df.loc[idx]['caption_5'] ]
        references.append(reference)

    print("> Evaluating predictions...")
    result = evaluate(predictions, references, metrics=metric_list)
    result = {k: round(v.item(),4) for k, v in result[0].items()}
    keys = list(result.keys())
    for key in keys:
        if "fluerr" in key:
            del result[key]
    print(result)