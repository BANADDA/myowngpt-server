import argparse
import json
import os
import tempfile
from typing import Literal

import pandas as pd
from huggingface_hub import HfApi, HfFolder, create_repo


class DatasetCreator:
    def __init__(self, dataframe: pd.DataFrame, question_col: str, response_col: str):
        self.dataframe = dataframe
        self.question_col = question_col
        self.response_col = response_col
        self.processed_dataframe = None
        self.dataset_type = None

    def create_dataset(self, model_type: Literal['gpt2', 'llama2', 'openelm']) -> str:
        creation_methods = {
            'gpt2': self._create_gpt2_dataset,
            'llama2': self._create_llama2_dataset,
            'openelm': self._create_openelm_dataset
        }

        if model_type not in creation_methods:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types are: {', '.join(creation_methods.keys())}")

        try:
            creation_methods[model_type]()
            self.dataset_type = model_type
            return self.dataset_type
        except Exception as e:
            print(f"Error during dataset creation: {e}")
            raise

    def _create_gpt2_dataset(self):
        self.processed_dataframe = pd.DataFrame({
            'text': self.dataframe.apply(
                lambda row: f"={row[self.question_col]} =\n {row[self.response_col]}", axis=1
            )
        })

    def _create_llama2_dataset(self):
        self.processed_dataframe = pd.DataFrame({
            'text': self.dataframe.apply(
                lambda row: f"<s> [INST] {row[self.question_col]} [/INST] {row[self.response_col]} </s>",
                axis=1
            )
        })

    def _create_openelm_dataset(self):
        self.processed_dataframe = pd.DataFrame({
            'messages': self.dataframe.apply(
                lambda row: [
                    {"content": row[self.question_col], "role": "user"},
                    {"content": row[self.response_col], "role": "assistant"}
                ],
                axis=1
            )
        })

    def upload_to_huggingface(self, token: str, repo_name: str) -> str:
        if self.processed_dataframe is None:
            raise ValueError("Dataset not created. Call create_dataset() first.")
        
        try:
            # Save the token for the session
            HfFolder.save_token(token)
            
            api = HfApi()
            user = api.whoami(token=token)["name"]
            repo_id = f"{user}/{repo_name}"

            create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True)

            with tempfile.TemporaryDirectory() as tmp_dir:
                data_dir = os.path.join(tmp_dir, "data")
                os.makedirs(data_dir)
                
                parquet_path = os.path.join(data_dir, "train.parquet")
                self.processed_dataframe.to_parquet(parquet_path, index=False)
                
                if self.dataset_type == 'openelm':
                    jsonl_path = os.path.join(data_dir, "train.jsonl")
                    with open(jsonl_path, 'w') as f:
                        for _, row in self.processed_dataframe.iterrows():
                            json.dump({'messages': row['messages']}, f)
                            f.write('\n')
                
                api.upload_folder(
                    folder_path=data_dir,
                    repo_id=repo_id,
                    repo_type="dataset",
                    path_in_repo="data",
                    token=token
                )
            
            return repo_id
        except Exception as e:
            print(f"Error uploading to Hugging Face: {e}")
            raise


def create_and_upload_dataset(
    file_path: str,
    model_type: Literal['gpt2', 'llama2', 'openelm'],
    repo_name: str,
) -> str:
    try:
        if file_path.endswith('.csv'):
            dataframe = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            dataframe = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file type. Please provide a CSV, XLS, or XLSX file.")
    except Exception as e:
        print(f"Error reading file: {e}")
        raise
    
    question_col = None
    response_col = None
    
    try:
        for col in dataframe.columns:
            lower_col = col.lower()
            if 'question' in lower_col:
                question_col = col
            elif 'answer' in lower_col or 'response' in lower_col:
                response_col = col
        
        if not question_col or not response_col:
            raise ValueError("Could not automatically detect question and response columns. Please ensure the file contains columns with names including 'question' and 'answer' or 'response'.")
    except Exception as e:
        print(f"Error detecting question and response columns: {e}")
        raise
    
    try:
        creator = DatasetCreator(dataframe, question_col, response_col)
        creator.create_dataset(model_type)
    except Exception as e:
        print(f"Error creating dataset: {e}")
        raise

    try:
        hf_token = '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'
        repo_id = creator.upload_to_huggingface(hf_token, repo_name)
        return repo_id
    except Exception as e:
        print(f"Error uploading dataset to Hugging Face: {e}")
        raise


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Create and upload dataset to Hugging Face")
        parser.add_argument('file_path', type=str, help='Path to the input file (CSV, XLS, or XLSX)')
        parser.add_argument('model_type', type=str, choices=['gpt2', 'llama2', 'openelm'], help='Model type')
        parser.add_argument('repo_name', type=str, help='Name of the repository to create/update on Hugging Face')

        args = parser.parse_args()
        repo_id = create_and_upload_dataset(args.file_path, args.model_type, args.repo_name)
        print(repo_id)
    except Exception as e:
        print(f"Error in main execution: {e}")
