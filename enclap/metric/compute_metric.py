import pandas as pd
from aac_metrics import evaluate
import copy
metric_list = ["bleu_1", "bleu_4", "rouge_l", "meteor", "spider_fl"]

if __name__=='__main__':
    csv_path = "/workspace/audiobart/csv/predictions/prediction_clap.csv"
    df = pd.read_csv(csv_path)

    predictions = []
    references = []
    for idx in range(len(df)):
        predictions.append(df.loc[idx]['prediction'])
        reference = [df.loc[idx]['caption_1'],df.loc[idx]['caption_2'],df.loc[idx]['caption_3'],df.loc[idx]['caption_4'],df.loc[idx]['caption_5'] ]
        references.append(reference)

    print("> Evaluating predictions...")
    result = evaluate(predictions, references, metrics=metric_list)
    result = {k: v.item() for k, v in result[0].items()}
    keys = list(result.keys())
    for key in keys:
        if "fluerr" in key:
            del result[key]
    print(result)