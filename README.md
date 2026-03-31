# Multi-frame Spatio-Temporal Attention-enhanced Dual-Stream Network for BPPV Diagnosis

**Enhanced Dual-stream Network for Accurate Benign Paroxysmal Positional Vertigo Diagnosis**  
*Chaoyue Tang, Hong Zheng*, Tianyu Wang, Tao Tao, Yuzi Yang, Jianfeng Yang*  
School of Electronic Information, Wuhan University, Wuhan, China

📄 **Published in *Biomedical Signal Processing and Control***  
[![Paper](https://img.shields.io/badge/Paper-BSPC-blue)](https://www.sciencedirect.com/journal/biomedical-signal-processing-and-control) 
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 🧠 Overview

This repository provides the official implementation of our proposed **BPPV intelligent diagnosis framework**, which integrates:

- **QuickPOS**: Efficient nystagmus localization from long VNG videos  
- **MSTA-DSN**: Multi-frame Spatio-Temporal Attention-enhanced Dual-Stream Network for disease classification  

Our method is specifically designed for **small-scale medical datasets**, addressing key challenges in:

- Temporal modeling of rapid eye movements  
- Robustness under eyelid occlusion  
- Efficient long-video processing  

---

## 🚀 Key Contributions

- 🔹 **QuickPOS Framework**
  - Adaptive eyelid occlusion compensation (trajectory-level, not video-level)
  - Automatic extraction of 9-second key nystagmus segments
  - Efficient processing of long VNG recordings (~5 min → 9 s)

- 🔹 **MSTA-DSN Network**
  - Multi-frame RGB input with temporal attention (Time module)
  - Optical flow stream enhanced with CBAM spatial attention
  - Lightweight + high-performance dual-stream architecture

- 🔹 **State-of-the-art Performance**
  - **Self-created dataset**
    - 100% accuracy (binary classification)
    - ~95% accuracy (subtype classification)
  - **Public dataset**
    - 90.43% (4-class classification)
    - 100% (right horizontal semicircular canal BPPV)

---
## ⚙️ Usage

### 1. Data Preparation

We provide a complete pipeline to convert raw VNG videos into model-ready inputs.

#### Step 1: Generate RGB Frames and Optical Flow

Run the following script:

```bash
python dmk_two_Stream_Network_PyTorch/Gnerate_RGB&FLOW/generate_rgb_and_flow.py
```
# Input:
#   .mp4 video
#
# Output:
#   RGB frames
#   Optical flow frames (x / y directions)
#
# Description:
#   This step converts long VNG videos into spatial and motion representations
#   required by the dual-stream network.

#### Step 2: Prepare Training and Testing Lists

# Directory:
dmk_two_Stream_Network_PyTorch/TrainTestlist

# Description:
#   Define training/testing splits and corresponding labels.
#   Ensure paths correctly point to generated RGB and optical flow data.

### 2. Training
python dmk_two_Stream_Network_PyTorch/train.py

# Description:
#   Load RGB and optical flow data
#   Train the MSTA-DSN model
#   Save checkpoints automatically

### 3. Testing
python dmk_two_Stream_Network_PyTorch/test.py

# Description:
#   Load trained weights
#   Perform inference on the test set
#   Output classification results

📌 Notes
# Ensure RGB and optical flow data are correctly aligned
# Optical flow must be generated before training
# Dataset paths must be consistent across all configuration files

## 🎥 Demo Video

👉<img width="2553" height="1296" alt="image" src="https://github.com/user-attachments/assets/8a6c251f-af2f-45cc-aed2-97d864074256" />

Our team has developed a BPPV diagnostic system that integrates our proposed network. The demonstration video is located in the root directory as BPPVsystem.mp4.
This demo shows:
- Pupil tracking
- Quick positioning
- BPPV classification results

---

## 🏆 Model Zoo

We provide pretrained models for reproducibility:

👉 **Download Pretrained Models:**  
https://pan.baidu.com/s/1443LCmC_4aP6pcmNU_nDGQ 
Extraction code: BSPC
Available models:
- self-created dataset
- public dataset

---

## 📊 Framework Pipeline


