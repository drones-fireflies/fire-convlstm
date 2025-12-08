import torch
import numpy as np

from models.ConvLSTM import ConvLSTM
from training.train import load_config, device
from autoregressive import autoregressive_predict

"""Wildfire propagation prediction demo"""

if __name__ == "__main__":

    # Load main configuration
    config_path = "config_files/main_config.yaml"    
    config = load_config(config_path)

    history_length = config["history_length"]
    prediction_steps = config["prediction_steps"]

    # Load test scenario
    scenarios = np.load(config["dataset_path"])['X']
    scenario_number = config["scenario_number"]

    # Initialize input sequence for the model
    initial_frame = scenarios[scenario_number, 0:history_length]  # first sequence
    input_sequence = torch.from_numpy(initial_frame) # add batch dimension

    # Build ground truth sequence for evaluation
    ground_truth_sequence = torch.from_numpy(scenarios[scenario_number, history_length:, 0:2, :, :]).float()

    # Load model
    model_config = load_config(config["model_config_path"])
    model = ConvLSTM(input_sequence.shape[1], model_config).to(device)
    
    # Load model weights
    model_weights = torch.load(config["model_path"], map_location=device)
    model.load_state_dict(model_weights)
    model.eval()

    # Run autoregressive prediction
    autoregressive_predict(model, input_sequence, ground_truth_sequence, prediction_steps, history_length, visualize=config["visualization_enabled"])

