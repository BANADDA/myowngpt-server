import json
import sys

import pandas as pd

import wandb


def fetch_wandb_data(entity_name, project_name, run_id):
    # Initialize the API and fetch the run
    api = wandb.Api()
    run = api.run(f"{entity_name}/{project_name}/{run_id}")

    # Fetch the history of the run (logged metrics)
    history = run.history()
    
    # print(f"Wandb training history: {history}")

    # Extract relevant data
    data = {
        'step': history['_step'].tolist(),
        'train_loss': history['train/loss'].tolist(),
        'train_grad_norm': history['train/grad_norm'].tolist(),
        'train_learning_rate': history['train/learning_rate'].tolist()
    }
    
    # print(f"Fetched Wandb data: {data}")

    return data

if __name__ == "__main__":
    entity_name = sys.argv[1]
    project_name = sys.argv[2]
    run_id = sys.argv[3]

    data = fetch_wandb_data(entity_name, project_name, run_id)
    print(json.dumps(data))
