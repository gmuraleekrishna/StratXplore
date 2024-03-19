import json
import os.path
from pathlib import Path

directory = Path('/home/krishna/Projects/VLN-BEVBert/test')

for dir in directory.iterdir():

    for file in ['detail_val_seen', 'detail_val_unseen']:
        avg_scores = {
            "CLS": 0.0,
            "DTW": 0.0,
            "SDTW": 0.0,
            "action_steps": 0.0,
            "nDTW": 0.0,
            "nav_error": 0.0,
            "oracle_error": 0.0,
            "oracle_success": 0.0,
            "spl": 0.0,
            "success": 0.0,
            "trajectory_lengths": 0.0,
            "trajectory_steps": 0.0
        }
        if os.path.exists(dir / 'preds' / f"{file}.json"):
            with open(dir / 'preds' / f"{file}.json") as f_:
                jsn = json.load(f_)
                for item in jsn:
                    avg_scores["CLS"] += item["CLS"]
                    avg_scores["DTW"] += item["DTW"]
                    avg_scores["SDTW"] += item["SDTW"]
                    avg_scores["action_steps"] += item["action_steps"]
                    avg_scores["nDTW"] += item["nDTW"]
                    avg_scores["nav_error"] += item["nav_error"]
                    avg_scores["oracle_error"] += item["oracle_error"]
                    avg_scores["oracle_success"] += item["oracle_success"]
                    avg_scores["spl"] += item["spl"]
                    avg_scores["success"] += item["success"]
                    avg_scores["trajectory_lengths"] += item["trajectory_lengths"]
                    avg_scores["trajectory_steps"] += item["trajectory_steps"]
            avg_scores = [f"{v/len(jsn):0.5f}" for k, v in avg_scores.items()]
            with open(f'consolidated_results_{file}.txt', 'a') as f_:
                f_.write(dir.name + ',' + ",".join(avg_scores) + '\n')
        else:
            print(','.join(dir.name.split('_')))
