# 🔥 Fire-ConvLSTM

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![Deep Learning](https://img.shields.io/badge/DeepLearning-ConvLSTM-orange)

---

## 🚀 Overview

**Fire-ConvLSTM** is a deep learning framework for **fast wildfire spread prediction** using **Convolutional LSTM (ConvLSTM)** networks.

It learns to approximate a wildfire simulator and enables:

- ⚡ **Up to ~180× gpu-accelerated simulations**
- 🔁 **Auto-regressive multi-step prediction**
- 🤖 Integration into **Reinforcement Learning (RL)** training frameworks
- 🚁 Deployment for **UAV wildfire monitoring**

---

## 🧪 Dataset

Generated using a **Cellular Automata wildfire simulator** with:

- 🌬 Wind (direction & strength)
- 🌱 Fuel density
- ⛰ Elevation & slope
- 💧 Moisture

📦 See: `dataloader/README.md`

---

## 🔁 Auto-Regressive Prediction

The model predicts wildfire spread iteratively:

1. Input initial fire frames  
2. Predict next frame  
3. Feed prediction back  
4. Repeat for long-term simulation  

---

## 🌍 Fire propagation example: prediction with convLSTM vs. ground truth

<p align="center">
  <img src="assets/fire_prediction.pdf" width="600"/>
</p>

---


## 📊 Performance

✔ Computational time required to generate a full propagation for different convLSTM configurations, compared with the time required by the original simulator.

<p align="center">
  <img src="assets/acceleration.pdf" width="700"/>
</p>

---
---

## 💻 Installation

```bash
git clone https://github.com/drones-fireflies/fire-convlstm.git
cd fire-convlstm
pip install -r requirements.txt
```

## 📚 Citation

@inproceedings{chakraa2026accelerating,
  title={Accelerating Wildfire Spread Prediction for Guidance: A ConvLSTM Deep Learning Approach},
  author={Chakraa, Hamza and Bronz, Murat},
  booktitle={AIAA SCITECH 2026 Forum},
  year={2026},
  publisher={American Institute of Aeronautics and Astronautics (AIAA)},
  doi={10.2514/6.2026-0423},
}

