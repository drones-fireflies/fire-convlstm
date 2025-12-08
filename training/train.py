import os
import sys
import yaml
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy

sys.path.append(os.path.abspath(""))

from dataloader.dataloader import PyTorchWildfireDataset
from models.ConvLSTM import ConvLSTM

# ----------------------- Load YAML configuration -----------------------
def load_config(config_path: str):
    """
    Load parameters from a configuration file.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# ----------------------- Device ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------- Training and evaluation -----------------------
def train_model(config: dict) -> torch.nn.Module:
    """
    Args:
        config (dict): Dictionary containing all training, model, and dataset parameters.
    """

    # Dataset setup
    dataset_path = config["dataset_path"]
    history_length = config["history_length"]

    dataset = PyTorchWildfireDataset(dataset_path, history_length)

    # Split dataset
    train_split_ratio = config["train_split_ratio"]
    num_train_samples = int(len(dataset) * train_split_ratio)
    num_val_samples = len(dataset) - num_train_samples
    train_dataset, val_dataset = random_split(dataset, [num_train_samples, num_val_samples])

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    # Model setup
    input_channels = dataset.data.shape[2]
    model_config = load_config(config["model_config_path"])
    model = ConvLSTM(input_channels, model_config).to(device)
    
    # Training setup
    learning_rate = config["learning_rate"]
    patience = config["patience"]
    epochs = config["epochs"]
    model_save_path = config["model_save_path"]

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.BCELoss()
    accuracy_metric = Accuracy(task="multilabel", num_labels=2).to(device)
    
    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("\n Starting training...\n")
    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = 0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            assert xb.device == model.decoder.weight.device
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * xb.size(0)
            train_acc += accuracy_metric(preds, yb.int()).item() * xb.size(0)

        train_loss /= len(train_dataset)
        train_acc /= len(train_dataset)

        # ------------------ Validation ------------------
        model.eval()
        val_loss, val_acc = 0, 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += criterion(preds, yb).item() * xb.size(0)
                val_acc += accuracy_metric(preds, yb.int()).item() * xb.size(0)

        val_loss /= len(val_dataset)
        val_acc /= len(val_dataset)
        
        # ------------------ Logging ------------------
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | "
            f"Train Accuracy: {train_acc:.4f} | Validation Accuracy: {val_acc:.4f}"
        )

        # ------------------ Scheduler + early stop ------------------
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
            print("Saved best model")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    return model

if __name__ == "__main__":
    config_path = "config_files/training_config.yaml"
    config = load_config(config_path)
    train_model(config)