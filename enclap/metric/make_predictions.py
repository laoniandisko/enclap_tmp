import sys 
sys.path.append('..')

from inference import AudioBartInference
from tqdm import tqdm
import os
import pandas as pd 
import csv
os.environ["CUDA_VISIBLE_DEVICES"] = "5"


if __name__ == "__main__":
    ckpt_path = "/data/jyk/aac_results/clap/clap/checkpoints/epoch_12"
    infer_module = AudioBartInference(ckpt_path)
    from_encodec = True
    csv_path = "/workspace/audiobart/csv/test.csv"
    base_path = "/data/jyk/aac_dataset/clotho/encodec"
    df = pd.read_csv(csv_path)
    save_path = "/workspace/audiobart/csv/predictions/prediction_clap.csv"
    f = open(save_path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['file_path', 'prediction', 'caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5'])

    print(f"> Making Predictions for model {ckpt_path}...")
    for idx in tqdm(range(len(df)), dynamic_ncols=True, colour="red"):
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
            prediction = infer_module.infer_from_encodec(wav_path)
        line = [wav_path, prediction[0], df.loc[idx]['caption_1'], df.loc[idx]['caption_2'],df.loc[idx]['caption_3'],df.loc[idx]['caption_4'],df.loc[idx]['caption_5']]
        writer.writerow(line)

    f.close()