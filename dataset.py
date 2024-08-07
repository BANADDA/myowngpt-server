import argparse
import json
import os
import tempfile

import pandas as pd
from huggingface_hub import HfApi, create_repo

# Predefine the Hugging Face API token
HF_TOKEN = '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'

class DatasetCreator:
    def __init__(self, dataframe, question_col, response_col):
        self.dataframe = dataframe
        self.question_col = question_col
        self.response_col = response_col
        self.processed_dataframe = None

    def create_dataset(self, model_type):
        creation_methods = {
            'gpt2': self._create_gpt2_dataset,
            'llama2': self._create_llama2_dataset,
            'openelm': self._create_openelm_dataset
        }
        if model_type not in creation_methods:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types are: {', '.join(creation_methods.keys())}")
        self.processed_dataframe = creation_methods[model_type]()

    def _create_gpt2_dataset(self):
        return pd.DataFrame({
            'text': self.dataframe.apply(lambda row: f"Question: {row[self.question_col]}\nAnswer: {row[self.response_col]}", axis=1)
        })

    def _create_llama2_dataset(self):
        return pd.DataFrame({
            'text': self.dataframe.apply(lambda row: f"<s> {row[self.question_col]} | {row[self.response_col]} </s>", axis=1)
        })

    def _create_openelm_dataset(self):
        return pd.DataFrame({
            'messages': self.dataframe.apply(lambda row: [
                {"content": row[self.question_col], "role": "user"},
                {"content": row[self.response_col], "role": "assistant"}
            ], axis=1)
        })

    def upload_to_huggingface(self, repo_name):
        api = HfApi()
        user = api.whoami(token=HF_TOKEN)["name"]
        repo_id = f"{user}/{repo_name}"
        create_repo(repo_id, repo_type="dataset", token=HF_TOKEN, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_dir = os.path.join(tmp_dir, "data")
            os.makedirs(data_dir)
            if 'messages' in self.processed_dataframe.columns:
                json_path = os.path.join(data_dir, "data.json")
                self.processed_dataframe['messages'].to_json(json_path, orient='records', lines=True)
            else:
                parquet_path = os.path.join(data_dir, "data.parquet")
                self.processed_dataframe.to_parquet(parquet_path, index=False)
            api.upload_folder(folder_path=data_dir, repo_id=repo_id, repo_type="dataset", path_in_repo="", token=HF_TOKEN)
        return repo_id

def main(file_path, model_type, repo_name):
    dataframe = pd.read_csv(file_path) if file_path.endswith('.csv') else pd.read_excel(file_path)
    creator = DatasetCreator(dataframe, 'question', 'answer')
    creator.create_dataset(model_type)
    repo_id = creator.upload_to_huggingface(repo_name)
    print(f"Dataset uploaded successfully. Repository ID: {repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and upload dataset to Hugging Face.")
    parser.add_argument('file_path', type=str, help='Path to the input file (CSV, XLS, or XLSX)')
    parser.add_argument('model_type', choices=['gpt2', 'llama2', 'openelm'], help='Model type for the dataset')
    parser.add_argument('repo_name', help='Name of the repository to create/update on Hugging Face')
    args = parser.parse_args()
    main(args.file_path, args.model_type, args.repo_name)
