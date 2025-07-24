# 🌌 TopoMorph-RGNet: Hyperspectral Image Classification on Indian Pines

This repository presents **TopoMorph-RGNet**, a novel architecture for hyperspectral image classification combining **spatial convolution**, **morphological band attention**, and **topological learning**. It is evaluated on the widely-used **Indian Pines 1992** dataset.

---

## 🧠 Architecture Overview

                              ┌──────────────────────────────┐
                              │      Input HSI Cube (X)      │
                              └─────────────┬────────────────┘
                                            ▼
                               ┌──────────────────────────┐
                               │ MBSA (Morphological Band │
                               │     Saliency Attention)  │
                               └───────┬──────────────────┘
                                       ▼
                  ┌──────────────────────────────┐
                  │ RGConv Block (Spatial Path)  │ ◄─────┐
                  └──────────────────────────────┘       │
                                       ▼                 ▼
                  ┌──────────────────────────────┐   ┌───────────────┐
                  │    UNet++ (Spectral Decoder)  │   │ Topological   │
                  │     or Attention UNet         │   │ Feature Path  │
                  └─────────────┬─────────────────┘   └───────────────┘
                                ▼
                      ┌────────────────────┐
                      │ Feature Fusion +   │
                      │ Global Avg Pooling │
                      └──────────┬─────────┘
                                 ▼
                      ┌────────────────────┐
                      │  Final Classifier   │
                      └────────────────────┘


---

## ⚙️ Architectural Innovations

✅ **Morphological Band Saliency Attention (MBSA)**  
→ Learns salient spectral bands via dilation, erosion, opening, and closing  
→ Adaptive, learnable replacement of PCA

✅ **Topological Learning**  
→ Uses Persistent Homology to extract per-pixel spectral entropy  
→ Topological descriptors fused via attention

✅ **Upgraded UNet++ Decoder**  
→ Deep spectral reconstruction with gated residuals

✅ **Dual-Stage Fusion**  
→ Combines spatial, spectral, and topological paths using:
- Attention gating
- Channel attention
- Gated units

---

## 📊 Dataset Info

| Attribute            | Value                      |
|----------------------|----------------------------|
| Dataset              | Indian Pines (1992)        |
| Dimensions           | (145, 145, 200)            |
| Ground Truth Labels  | 16 classes (0 = background)|
| Patch Size           | 13 × 13                    |
| Top-k Bands Selected | 30                         |

---

## 🏗️ Pipeline Steps

1. **Data Normalization**: Per-pixel band standardization  
2. **Patch Extraction**: Around labeled pixels (patch size = 13)  
3. **Band Selection via MBSA**  
4. **Topological Feature Extraction** using Persistent Homology  
5. **Model Training** using Adam + Early Stopping  
6. **Evaluation on Test Set**

---

## 📈 Results

| Metric                | Value          |
|-----------------------|----------------|
| **Overall Accuracy**  | 99.76%         |
| **Average Accuracy**  | 99.59%         |
| **Kappa Coefficient** | 0.9972         |
| **F1 Score (Macro)**  | 0.9971         |

📊 **Confusion Matrix (Simplified)**

All non-zero classes perfectly or near-perfectly classified.

> ✨ *State-of-the-art level performance on Indian Pines.*

---

## 🧪 Evaluation Report

- **Classifier**: Dense Softmax over fused features  
- **Loss Function**: Categorical Crossentropy  
- **Optimizer**: Adam  
- **Batch Size**: 64  
- **Epochs**: 100 (with early stopping)

---

## 📁 Directory Structure

## 🛠 Installation

---
## 🚀 How to Run
- Place dataset .mat files in the input/ folder

- Run topomorph_rgnet.ipynb or the script

- Best model saved as best_topomorph.h5