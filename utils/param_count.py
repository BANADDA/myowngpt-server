from transformers import AutoModel, AutoConfig
import sys

def get_parameters_count(model_id, token):
    # Load model with trust_remote_code set to True and using the new token parameter
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True, token=token)
    model = AutoModel.from_pretrained(model_id, config=config, trust_remote_code=True, token=token)
    return sum(p.numel() for p in model.parameters())

if __name__ == "__main__":
    model_id = sys.argv[1]
    token = sys.argv[2]  # Assuming the second argument is the token
    try:
        params_count = get_parameters_count(model_id, token)
        print(params_count)  # Output the number of parameters
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)
