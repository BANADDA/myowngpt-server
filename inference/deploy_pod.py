import json
import sys

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

if __name__ == "__main__":
    model_id = sys.argv[1]  # Get the model ID from the command-line arguments
    model_name = sys.argv[2]  # Get the model name from the command-line arguments

    pod = inference_model(model_id, model_name)
    print(json.dumps(pod))  # Print pod details as JSON
