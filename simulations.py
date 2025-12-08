import torch
import numpy as np

from models.ConvLSTM import ConvLSTM
from training.train import load_config, device
from autoregressive import autoregressive_predict

from display_utils import SimulationDisplay


def run_evaluation(num_sim, num_scenarios = 10):
    """

    Returns
    -------
    pa_scores, jsc_scores, hd_scores, time_scores : np.ndarray
        Arrays of shape (num_scenarios, num_datasets) with metrics.
    """

    if num_sim == 1:
        config_path = "config_files/simulations_config/simulation1_config.yaml"
    elif num_sim == 2:
        config_path = "config_files/simulations_config/simulation2_config.yaml"
    elif num_sim == 3:
        config_path = "config_files/simulations_config/simulation3_config.yaml"
    else:
        raise ValueError(f"Unsupported num_sim={num_sim}, expected 1, 2 or 3.")

    config = load_config(config_path)
    num_datasets = config["num_datasets"]

    # Pre-allocate metric arrays
    pa_scores = np.zeros((num_scenarios, num_datasets), dtype=np.float32)
    jsc_scores = np.zeros((num_scenarios, num_datasets), dtype=np.float32)
    hd_scores = np.zeros((num_scenarios, num_datasets), dtype=np.float32)
    time_scores = np.zeros((num_scenarios, num_datasets), dtype=np.float32)

    # Whether to visualize during prediction
    visualize = config.get("visualization_enabled", False)

    for dataset_idx in range(num_datasets):

        # ---------------------- Simulation 1 ------------------------------
        if num_sim == 1:
            history_length = config["history_length"]
            model_config = load_config(config["model_config_path"])

            prediction_steps = config[f"prediction_steps_{dataset_idx}"]
            scenario_path = config[f"dataset_path_{dataset_idx}"]

            scenario = np.load(scenario_path)["X"]  # shape: (num_scenarios, T, C, H, W)
            input_channels = scenario.shape[2]

            model = ConvLSTM(input_channels, model_config).to(device)
            model_weights = torch.load(config["model_path"], map_location=device)
            model.load_state_dict(model_weights)
            model.eval()

        # ---------------------- Simulation 2 ------------------------------
        elif num_sim == 2:
            history_length = config["history_length"]
            model_config = load_config(config[f"model_config_path_{dataset_idx}"])

            prediction_steps = config["prediction_steps"]
            scenario_path = config["dataset_path"]

            scenario = np.load(scenario_path)["X"]
            input_channels = scenario.shape[2]

            model = ConvLSTM(input_channels, model_config).to(device)
            model_weights = torch.load(
                config[f"model_path_{dataset_idx}"], map_location=device
            )
            model.load_state_dict(model_weights)
            model.eval()

        # ---------------------- Simulation 3 ------------------------------
        elif num_sim == 3:
            # simulation 3 can have dataset-specific history length
            history_length = config[f"history_length_{dataset_idx}"]
            model_config = load_config(config["model_config_path"])

            prediction_steps = config["prediction_steps"]
            scenario_path = config["dataset_path"]

            scenario = np.load(scenario_path)["X"]
            input_channels = scenario.shape[2]

            model = ConvLSTM(input_channels, model_config).to(device)
            model_weights = torch.load(
                config[f"model_path_{dataset_idx}"], map_location=device
            )
            model.load_state_dict(model_weights)
            model.eval()

        # -----------------------------------------------------------------
        # Loop over scenarios for this dataset
        # -----------------------------------------------------------------
        for scenario_idx in range(num_scenarios):

            initial_frame = scenario[scenario_idx, 0:history_length]  # (T_hist, C, H, W)
            input_sequence = torch.from_numpy(initial_frame).float().to(device)

            gt_sequence_np = scenario[scenario_idx, history_length:, 0:2, :, :]
            ground_truth_sequence = torch.from_numpy(gt_sequence_np).float().to(device)

            _, metrics, times = autoregressive_predict(model, input_sequence, ground_truth_sequence, prediction_steps, history_length, visualize=config["visualization_enabled"])

            jsc_scores[scenario_idx, dataset_idx] = float(metrics["jaccard_mean"])
            hd_scores[scenario_idx, dataset_idx] = float(metrics["hausdorff_mean"])
            pa_scores[scenario_idx, dataset_idx] = float(metrics["pixel_accuracy_mean"])
            time_scores[scenario_idx, dataset_idx] = float(times)

    return pa_scores, jsc_scores, hd_scores, time_scores


def print_latex_rows(j_scores: np.ndarray, d_scores: np.ndarray, h_scores: np.ndarray):

    num_scenarios, num_datasets = j_scores.shape

    print("\n===== LaTeX table: =====")
    for s in range(num_scenarios):
        row = [str(s + 1)]
        for d in range(num_datasets):
            row.append(f"{j_scores[s, d]:.3f}")
            row.append(f"{d_scores[s, d]:.3f}")
            row.append(f"{h_scores[s, d]:.3f}")
        print(" & ".join(row) + r" \\")
    print("================================\n")


if __name__ == "__main__":

    # ---------------------------- Simulation 1 ----------------------------
    print("Running first simulation...")

    pa_scores, jsc_scores, hd_scores, time_scores = run_evaluation(1)

    jsc_means = np.nanmean(jsc_scores, axis=0).tolist()
    hd_means = np.nanmean(hd_scores, axis=0).tolist()

    print_latex_rows(pa_scores, jsc_scores, hd_scores)

    config_labels = [
        r"$100\times100$",
        r"$150\times150$",
        r"$200\times200$",
        r"$250\times250$",
        r"$300\times300$",
    ]

    viewer = SimulationDisplay(jsc_means, hd_means, [], config_labels)
    viewer.draw_jsc()
    viewer.draw_hd()

    print(jsc_means)
    print(hd_means)

    # ---------------------------- Simulation 2 ----------------------------
    print("Running second simulation...")

    pa_scores, jsc_scores, hd_scores, time_scores = run_evaluation(2)

    jsc_means = np.nanmean(jsc_scores, axis=0).tolist()
    hd_means = np.nanmean(hd_scores, axis=0).tolist()
    time_means = np.nanmean(time_scores, axis=0).tolist()

    config_labels = [
        r"A",
        r"B",
        r"C",
        r"D",
        r"E",
        r"F",
        r"G",
        r"H",
        r"I",
    ]

    viewer = SimulationDisplay(jsc_means, hd_means, time_means, config_labels)
    viewer.draw_jsc()
    viewer.draw_hd()
    viewer.draw_time(baseline_time=35.7)

    print(jsc_means)
    print(hd_means)
    print(time_means)

    # ---------------------------- Simulation 3 ----------------------------
    print("Running third simulation...")

    pa_scores, jsc_scores, hd_scores, time_scores = run_evaluation(3)

    jsc_means = np.nanmean(jsc_scores, axis=0).tolist()
    hd_means = np.nanmean(hd_scores, axis=0).tolist()

    config_labels = [
        r"l=1",
        r"l=2",
        r"l=3",
        r"l=4",
    ]

    viewer = SimulationDisplay(jsc_means, hd_means, [], config_labels)
    viewer.draw_jsc()
    viewer.draw_hd()

    print(jsc_means)
    print(hd_means)