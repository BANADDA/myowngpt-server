import requests
from transformers import AutoModelForCausalLM, AutoTokenizer

# Runpod API Key
RUNPOD_API_KEY = "UVM02ULWTIDBLM6URRHCGXJ2DIWP2H0THNPKH1L8"
DEFAULT_SYSTEM_PROMPT = """
You are a helpful assistant. Always answer as concisely as possible while being safe. Your answers should not be harmful, unethical, or offensive.
""".strip()

def get_existing_pod(model_id, model_name):
    import runpod
    runpod.api_key = RUNPOD_API_KEY
    pods = runpod.get_pods()
    
    for pod in pods:
        if pod.get('name') == model_name and pod.get('machine', {}).get('gpuDisplayName') == "RTX 4090" and pod.get('desiredStatus') == 'RUNNING':
            return pod
    return None

def create_pod(model_id, model_name):
    import runpod
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

def generate_prompt(prompt, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return f"""
    [INST] <<SYS>>
    {system_prompt}
    <</SYS>>
    {prompt} [/INST]
    """.strip()

def make_request(prompt, server_url):
    data = {
        "inputs": prompt,
        "parameters": {"best_of": 1, "temperature": 0.01, "max_new_tokens": 512}
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(f"{server_url}/generate", json=data, headers=headers)
    return response

def model_chat(prompt, server_url):
    prompt = generate_prompt(prompt)
    print(f"Generated prompt: {prompt}")  # Debug print

    response = make_request(prompt, server_url)
    print(f"Response status code: {response.status_code}")  # Debug print
    print(f"Full response from the server: {response.text}")  # Debug print
    
    result = response.json().get("generated_text", "")
    print(f"Generated text: {result}")  # Debug print

    # Extract the generated text part
    start_index = result.find("[/INST]") + len("[/INST]")
    end_index = result.rfind("<<SYS>>")
    if start_index != -1 and end_index != -1:
        generated_text = result[start_index:end_index].strip()
    else:
        generated_text = result.strip()
    
    return generated_text

if __name__ == "__main__":
    import sys
    model_id = "nvidia/Llama3-ChatQA-1.5-8B"  # Example model ID
    model_name = "Agricultural-llama"  # Example model name
    user_prompt = sys.argv[1]  # Get the prompt from the command-line arguments

    pod = inference_model(model_id, model_name)
    server_url = f'https://{pod["id"]}-80.proxy.runpod.net'

    result = model_chat(user_prompt, server_url)
    print(server_url)
