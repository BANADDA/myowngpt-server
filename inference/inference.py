# ./inference/inference.py
import json

import requests
import runpod

# Runpod API Key
RUNPOD_API_KEY = "UVM02ULWTIDBLM6URRHCGXJ2DIWP2H0THNPKH1L8"


def get_existing_pod(model_id, model_name):
    runpod.api_key = RUNPOD_API_KEY
    pods = runpod.get_pods()
    
    for pod in pods:
        if pod.get('name') == model_name and pod.get('machine', {}).get('gpuDisplayName') == "RTX 4090" and pod.get('desiredStatus') == 'RUNNING':
            return pod
    return None

def create_pod(model_id, model_name):
    runpod.api_key = RUNPOD_API_KEY
    gpu_count = 1
    pod = runpod.create_pod(
        name=model_name,
        image_name="ghcr.io/huggingface/text-generation-inference:0.9.4",
        gpu_type_id="NVIDIA GeForce RTX 4090",
        data_center_id="EU-RO-1",
        cloud_type="SECURE",
        docker_args=f"--model-id {model_id}",
        gpu_count=gpu_count,
        volume_in_gb=50,
        container_disk_in_gb=5,
        ports="80/http,29500/http",
        volume_mount_path="/data",
    )
    return pod

def inference_model(model_id, model_name):
    pod = get_existing_pod(model_id, model_name)
    if not pod:
        pod = create_pod(model_id, model_name)
    return pod

def generate_text(endpoint_url, prompt):
    # Define the payload
    payload = {
        "inputs": prompt,
        "parameters": {
            "best_of": 1,
            "decoder_input_details": True,
            "details": True,
            "do_sample": True,
            "max_new_tokens": 512,
            "repetition_penalty": 1.03,
            "return_full_text": False,
            "seed": None,
            "stop": ["photographer"],
            "temperature": 0.5,
            "top_k": 10,
            "top_p": 0.95,
            "truncate": None,
            "typical_p": 0.95,
            "watermark": True
        }
    }
    # Define headers
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    # Send the POST request
    response = requests.post(endpoint_url, headers=headers, data=json.dumps(payload))

    # Check if the request was successful
    if response.status_code == 200:
        return response.json()
    else:
        response.raise_for_status()

if __name__ == "__main__":
    import sys
    model_id = "nvidia/Llama3-ChatQA-1.5-8B"  # Example model ID
    model_name = "Agricultural-llama"  # Example model name
    user_prompt = sys.argv[1]  # Get the prompt from the command-line arguments

    pod = inference_model(model_id, model_name)
    server_url = f'https://{pod["id"]}-80.proxy.runpod.net/generate'
    chat = generate_text(server_url,user_prompt)
    print(chat["generated_text"])
    
