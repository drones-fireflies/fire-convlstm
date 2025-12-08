import os
import numpy as np
import random
import yaml
from tqdm import tqdm

from firemodel.propagation import Environment, FirePropagation

# ----------------------- Load config file -----------------------
def load_config(config_path: str):
    """
    Load parameters from a configuration file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# ----------------------- Data generation -----------------------
def generate_wildfire_scenarios(
    output_file: str,
    num_scenarios: int,
    sequence_length: int,
    grid_size_x: int,
    grid_size_y: int,
    wind_directions: tuple,
    vegetation_density_range: tuple
):
    """
    Generates wildfire scenarios and saves them as a compressed .npz dataset.

    Each scenario simulates the spread of fire over time in a 2D grid. 

    The dataset includes four channels:
        - Fire presence (binary)
        - Fuel levels (between 20-255, normalized here)
        - Wind direction (scaled value)
        - Burn rate (scaled value)
    """

    # Dataset shape: (scenarios, timesteps, X, Y, channels = 4)
    dataset = np.zeros(
        (num_scenarios, sequence_length, 5, grid_size_x, grid_size_y),
        dtype=np.float32,
    )

    # Generate multiple wildfire scenarios
    for scenario_idx in tqdm(range(num_scenarios), desc="Generating wildfire scenarios"):

        # Randomize simulation parameters within given ranges
        wind_dir = random.uniform(*wind_directions)
        veg_density = random.uniform(*vegetation_density_range)

        # Select a random ignition point
        ignition_x = grid_size_x // 2 + random.randint(-grid_size_x // 6, grid_size_x // 6)
        ignition_y = random.randint(grid_size_y // 8, grid_size_y - grid_size_y // 8)
        ignition_points = [(ignition_x, ignition_y)]

        env = Environment(
            grid_size=grid_size_x,
            ignition_points=ignition_points,
            wind_direction=wind_dir,
            vegetation_density=veg_density
        )

        sim = FirePropagation(timesteps=sequence_length,visualize=False)

        fire_history, fuel_history, elevation, moisture = sim.propagate_from_points(env)

        # Process results for each timestep
        for t in range(sequence_length):

            # Fire presence
            fire_layer = (np.array(fire_history[t], dtype=np.float32) > 0).astype(np.float32)

            # Fuel level normalized to [0,1]
            fuel_layer = np.array(fuel_history[t], dtype=np.float32)
            if fuel_layer.max() > 1.0:
                fuel_layer = fuel_layer / (fuel_layer.max() + 1e-8) 

            # Wind direction normalized to [0,1]
            wind_layer = np.full((grid_size_x, grid_size_y), wind_dir / 6.28, dtype=np.float32)

            # Store layers into dataset
            dataset[scenario_idx, t, 0, :, :] = fire_layer
            dataset[scenario_idx, t, 1, :, :] = fuel_layer
            dataset[scenario_idx, t, 2, :, :] = wind_layer
            dataset[scenario_idx, t, 3, :, :] = elevation
            dataset[scenario_idx, t, 4, :, :] = moisture

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save dataset as compressed NumPy file
    np.savez_compressed(output_file, X=dataset)


if __name__ == "__main__":

    # Load parameters from YAML config
    config = load_config("../config_files/dataloader_config.yaml")
        
    grid_size_x = config["grid_size_x"]
    grid_size_y = config["grid_size_y"]
    
    num_scenarios = config["num_scenarios"]
    sequence_length = config["sequence_length"]
    wind_directions = config["wind_directions"]
    vegetation_density_range = config["vegetation_density_range"]

    # The output file
    output_file = f"datasets/trainData_{num_scenarios}_{grid_size_x}x{grid_size_y}.npz" 

    # Run dataset generator
    generate_wildfire_scenarios(
        output_file=output_file,
        num_scenarios=num_scenarios,
        sequence_length=sequence_length,
        grid_size_x=grid_size_x,
        grid_size_y=grid_size_y,
        wind_directions=wind_directions,
        vegetation_density_range=vegetation_density_range
    )