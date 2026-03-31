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

## 🎥 Demo Video

👉 [Watch Demo Video](https://your-demo-link-here)

This demo shows:
- Pupil tracking
- Fast–slow phase extraction
- BPPV classification results

---

## 🏆 Model Zoo

We provide pretrained models for reproducibility:

👉 **Download Pretrained Models:**  
https://your-model-download-link-here

Available models:
- QuickPOS (pupil tracking & localization)
- MSTA-DSN (classification)
- Full pipeline model

---

## 📊 Framework Pipeline
