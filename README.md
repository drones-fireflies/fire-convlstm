# 🔥 Fire-ConvLSTM

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Deep Learning](https://img.shields.io/badge/DeepLearning-ConvLSTM-orange)
![Status](https://img.shields.io/badge/Status-Research-green)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## 🚀 Overview

**Fire-ConvLSTM** is a deep learning framework for **fast wildfire spread prediction** using **Convolutional LSTM (ConvLSTM)** networks.

It learns to approximate a wildfire simulator and enables:

- ⚡ **Up to ~180× faster simulation**
- 🔁 **Auto-regressive multi-step prediction**
- 🤖 Integration into **reinforcement learning (RL)** loops
- 🚁 Deployment for **UAV wildfire monitoring**

---

## 🌍 Real Wildfire Example

<p align="center">
  <img src="assets/wildfire.png" width="600"/>
</p>

---

## 🧠 Model Architecture

The model combines spatial and temporal modeling:

<p align="center">
  <img src="assets/architecture.png" width="700"/>
</p>

### Key Features

- 📍 Spatial modeling via **convolutions**
- ⏳ Temporal modeling via **LSTM memory**
- 🔁 Supports **sequential rollout prediction**

---

## 🔁 Auto-Regressive Prediction

The model predicts wildfire spread iteratively:

<p align="center">
  <img src="assets/autoregressive.png" width="700"/>
</p>

**Process:**
1. Input initial fire frames  
2. Predict next frame  
3. Feed prediction back  
4. Repeat for long-term simulation  

---

## 📊 Prediction Results

<p align="center">
  <img src="assets/prediction.png" width="700"/>
</p>

✔ Captures:
- Fire front shape  
- Direction of spread  
- Interaction with terrain  

⚠️ Errors mainly:
- Near boundaries  
- Long-horizon predictions  

---

## ⚡ Performance

<p align="center">
  <img src="assets/runtime.png" width="600"/>
</p>

### 🔥 Speed

| Model Type | Speed-up |
|----------|--------|
| Small | ~180× |
| Medium | ~80–150× |
| Large | ~50× |

---

## 📈 Metrics

| Metric | Description |
|------|------------|
| **PA** | Pixel Accuracy |
| **JSC (IoU)** | Spatial overlap |
| **HD** | Boundary accuracy |

Typical performance:

- **JSC ≈ 0.75–0.83**
- **HD ≈ low (good boundary alignment)**

---

## 🧪 Dataset

Generated using a **Cellular Automata wildfire simulator** with:

- 🌬 Wind (direction & strength)
- 🌱 Fuel density
- ⛰ Elevation & slope
- 💧 Moisture

Each sample:
- Grid: `130 × 130`
- Time: `150 steps`
- Multiple randomized scenarios

📦 See: `dataloader/README.md`

---

## 🎯 Applications

- 🚁 UAV wildfire monitoring  
- 🤖 Reinforcement Learning environments  
- 🔥 Real-time wildfire simulation  
- 🧭 Decision support systems  

---

## ⚠️ Limitations

- Trained on **synthetic data**
- **Error accumulation** over time
- Simplified physics vs real fires

---

## 🔮 Future Work

- 🌍 Real-world wildfire datasets  
- 🧠 Attention / Transformer models  
- ⚡ TensorRT / ONNX optimization  
- 🤖 RL-based autonomous systems  

---

## 💻 Installation

```bash
git clone https://github.com/drones-fireflies/fire-convlstm.git
cd fire-convlstm
pip install -r requirements.txt

## Citation

