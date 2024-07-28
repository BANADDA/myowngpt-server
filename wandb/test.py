import json
import os

import pandas as pd

import wandb

# Set your wandb API key
os.environ["WANDB_API_KEY"] = "efa7d98857a922cbe11e78fa1ac22b62a414fbf3"

# Set your entity name, project name, and run ID
entity_name = "ai-research-lab"
project_name = "gpt_v4"
run_id = "15v9e834"  # replace with your actual run ID

# Initialize the API and fetch the run
api = wandb.Api()

try:
    run = api.run(f"{entity_name}/{project_name}/{run_id}")

    # Fetch the history of the run (logged metrics)
    history = run.history()

    # Convert the DataFrame to JSON
    history_json = history.to_json(orient='records')

    # Print the JSON data
    print(history_json)

except wandb.errors.CommError as e:
    error_message = {"error": str(e)}
    print(json.dumps(error_message))

except Exception as e:
    error_message = {"error": "An unexpected error occurred: " + str(e)}
    print(json.dumps(error_message))


# import os

# import pandas as pd

# import wandb

# # Set your wandb API key
# os.environ["WANDB_API_KEY"] = "efa7d98857a922cbe11e78fa1ac22b62a414fbf3"

# # Set your entity name, project name, and run ID
# entity_name = "ai-research-lab"
# project_name = "gpt_v4"
# run_id = "15v9e834"  # replace with your actual run ID

# # Initialize the API and fetch the run
# api = wandb.Api()
# run = api.run(f"{entity_name}/{project_name}/{run_id}")

# # Fetch the history of the run (logged metrics)
# history = run.history()

# # Print the data in the run
# pd.set_option('display.max_columns', None)  # Display all columns
# print(history)

# # Optionally display the first few rows in a more readable format
# print(history.head())


# Create subplots
# fig = make_subplots(rows=1, cols=3, subplot_titles=('Train Loss Over Steps', 'Train Grad Norm Over Steps', 'Train Learning Rate Over Steps'))

# # Add Train Loss plot
# fig.add_trace(
#     go.Scatter(x=history['_step'], y=history['train/loss'], mode='lines', name='Train Loss'),
#     row=1, col=1
# )

# # Add Train Grad Norm plot
# fig.add_trace(
#     go.Scatter(x=history['_step'], y=history['train/grad_norm'], mode='lines', name='yyTrain Grad Norm'),
#     row=1, col=2
# )

# # Add Train Learning Rate plot
# fig.add_trace(
#     go.Scatter(x=history['_step'], y=history['train/learning_rate'], mode='lines', name='Train Learning Rate'),
#     row=1, col=3
# )

# # Update layout
# fig.update_layout(
#     title='Training Metrics Over Steps',
#     showlegend=True,
#     width=1400,
#     height=500
# )

# # Display the plot
# fig.show()
