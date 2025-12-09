import time
import torch
import numpy as np
from typing import List

from training.train import device
from display_utils import FireDisplay, MetricsDisplay
from metrics import pixel_accuracy, jaccard_similarity, hausdorff_distance


def autoregressive_predict(model, initial_sequence, true_sequence, steps, history_length, visualize = False, save_predictions = True):

    """
    Args:
        model (torch.nn.Module): Trained ConvLSTM model
        initial_sequence (torch.Tensor): Input sequence of shape (T, C, H, W).
        true_sequence (torch.Tensor): Ground-truth frames for comparison and visualization.
        steps (int): Number of prediction steps.
        visualize (bool): Whether to visualize predictions in real time.
        save_predictions (bool): Whether to keep all predicted frames in memory.

    Returns:
        predictions: All predicted frames if "save_predictions" is enabled
        mean_jaccard: Mean Jaccard similarity over predicted steps.
        mean_dice: Mean pixel accuracy (here used as Dice-like metric) over steps.
        mean_hausdorff: Mean Hausdorff distance over steps.
        elapsed_time: Total rollout time in seconds.
    """

    model.eval()

    # Extract infos
    wind_direction = float(initial_sequence[0, 2, 0, 0].item())
    elevation = initial_sequence[0, 3, :, :]

    # Move sequence to device
    sequence = initial_sequence.clone().to(device)

    # Get spatial dimensions
    _, _, height, width = sequence.shape

    # Storage for predicted fire maps (Optional)
    predicted_fire_frames: List[np.ndarray] = []

    if save_predictions:
        # Store initial fire frames
        # shape: (T, H, W) in numpy
        initial_fire = sequence[:, 0, :, :].detach().cpu().numpy()
        predicted_fire_frames.extend(initial_fire)

    # Visualization
    if visualize:
        FireVisualizer = FireDisplay(height, width, wind_direction, elevation)
        _, visualizer_elements = FireVisualizer._initialize_visualization()

    pixel_accuracy_scores: List[float] = []
    jaccard_scores: List[float] = []
    hausdorff_scores: List[float] = []

    start_time = time.time()

    with torch.no_grad():

        for step in range(steps - history_length):

            # Prepare input for the model
            model_input = sequence.unsqueeze(0)  # shape: (1, T, C, H, W)
            predicted_frame = model(model_input)[0]  # shape: (C, H, W)

            true_fire_map = true_sequence[step][0].cpu().numpy()
            true_fuel_map = true_sequence[step][1].cpu().numpy()
            pred_fire_map = predicted_frame[0].cpu().numpy()
            pred_fuel_map = predicted_frame[1].cpu().numpy()

            if save_predictions:
                predicted_fire_frames.append(predicted_frame[0].cpu().numpy())

            # Build next input sequence
            #   - Take last frame
            #   - Replace fire & fuel channels with predicted ones
            last_frame = sequence[-1].clone()
            last_frame[0:2] = predicted_frame[0:2]  # fire & fuel channels

            # New sequence: drop oldest frame, append updated last frame
            sequence = torch.cat(
                (sequence[1:], last_frame.unsqueeze(0)),
                dim=0,
            )

            if visualize and step < len(true_sequence):

                FireVisualizer._update(
                    visualizer_elements, pred_fire_map, pred_fuel_map, true_fire_map, true_fuel_map
                )

            # Compute metrics
            if step < len(true_sequence):

                pixel_accuracy_scores.append(pixel_accuracy(pred_fire_map, true_fire_map))

                jaccard_scores.append(jaccard_similarity(pred_fire_map, true_fire_map))

                hausdorff_scores.append(hausdorff_distance(pred_fire_map, true_fire_map))

    # Runtime
    elapsed_time = time.time() - start_time
    print(f"Simulation complete | Time: {elapsed_time:.2f}s ")

    # Post-run visualization of full fire spread
    if visualize and save_predictions:

        pred_stack = np.stack(predicted_fire_frames, axis=0)  # (T_pred, H, W)

        # Show fire spread
        FireVisualizer.show(pred_stack, true_sequence[:, 0, :, :].cpu().numpy())

        # Show metrics roll-out
        MetricsVisualizer = MetricsDisplay(jaccard_scores, hausdorff_scores)
        MetricsVisualizer.plot()

    # Compute averages
    mean_pixel_accuracy = np.mean(pixel_accuracy_scores)
    mean_jaccard = np.mean(jaccard_scores)
    mean_hausdorff = np.nanmean(hausdorff_scores)

    
    metrics_summary = {
        "pixel_accuracy_mean": mean_pixel_accuracy,
        "jaccard_mean": mean_jaccard,
        "hausdorff_mean": mean_hausdorff,
    }

    return predicted_fire_frames, metrics_summary, elapsed_time