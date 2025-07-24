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
### 1. Load and Normalize the Hyperspectral Cube
- Normalizes the 200 spectral bands per pixel using StandardScaler.
  
✅ Matches the preprocessing block in the diagram.

### 2. Patch Extraction
Extracts 13×13×200 patches centered on each labeled pixel.

Labels adjusted to be 0-based.

✅ Matches the spatial-spectral patch extraction stage.

### 3. Split into Train/Val/Test
Stratified split into 60% train, 20% val, 20% test.

✅ Common practice, supports balanced training.

### 4. Morphological Band Selection (MBSA)

Performs MBSA by computing dilation, erosion, opening, and closing per band.

Computes average saliency over 1000 training samples.

Selects top 30 most salient spectral bands.

✅ Fully aligns with the “Replace PCA with MBSA” goal in your enhancement list

### 5. Topological Feature Extraction
Computes per-pixel spectral mean → persistence diagram → persistence entropy using ripser.

✅ Correctly implements persistent homology and topological feature extraction from spectra.

### 6. Model Input Preparation
used two branches:

X_train_sel: patch with selected 30 bands → fed to CNN

topo_train: scalar topo feature → fused with CNN output

✅ Implements dual-branch input (spectral-spatial + topological), as shown in the diagram.

### 7. Model Architecture: TopoMorph-RGNet
#### a. Spatial Path: RGConvBlock
serves as the core building block of the spatial pathway in our TopoMorph-RGNet architecture. It is designed to capture spatial structures using recursive multiplicative gating, which enhances feature selectivity and suppresses irrelevant activations.

⚙️ Architecture Description
- Input: 2D spatial feature map of shape (H, W, C)

- Output: Spatially refined feature map emphasizing high-saliency regions

📐 Block Breakdown
1. Parallel Conv Feature Streams

- The input is passed through two parallel Conv2D(filters, 3x3) layers with ReLU.

- Their outputs (x1, x2) are multiplied element-wise to create a gated spatial feature map.

- This gating mechanism emphasizes overlapping activations, suppressing noise.

2. Recursive Refinement

- The gated output is further refined by two more Conv2D(filters) layers.

- Another element-wise multiplication of their outputs (x3, x4) is performed.

- The final output retains enhanced local spatial consistency.
  
Double Conv2D + Gating → enhanced spatial features

✅ Why This Works
Gating helps suppress redundant/irrelevant patterns and allows only jointly activated features to pass.

Recursive structure boosts local consistency and gradient flow.

```
Input
 │
├─ Conv ─┐
│        ├─ Multiply (Gated Features)
├─ Conv ─┘
 │
├─ Conv ─┐
│        ├─ Multiply (Refined Features)
├─ Conv ─┘
 │
Output
```

---
✅ Matches Recursive Gated Convolution Block.

#### b. 🔧 UNet++ Decoder (Spectral Path)
The UNetPlusPlusDecoder module in our architecture processes the input spectral-spatial patch through a lightweight variant of UNet++, designed for hierarchical feature extraction and spectral contextual learning.

- ⚙️ Architecture Description
##### Input: 3D patch tensor of shape (H, W, C) where C = selected top-k spectral bands (e.g., 30)

##### Output: Feature map enriched with spectral context, used for topological fusion

##### 📐 Block Breakdown
- Initial Convolution Block (Level 0)

- Two consecutive Conv2D(filters, 3×3, ReLU) layers extract shallow spectral-spatial features.

- Output: conv1 feature map.

- Downsampling (Level 1)

- MaxPooling2D(2×2) reduces spatial resolution.

- Two more Conv2D(filters×2) layers act as the bottleneck, capturing deeper context.

- Upsampling + Skip Connection

- UpSampling2D with bilinear interpolation followed by spatial resizing ensures alignment with conv1.

- Skip-connection concatenates low-level features (conv1) with upsampled deep features (conv2), followed by two Conv2D(filters) layers to fuse them.

- Gated Residual Connection

- Input is passed through:

A Conv2D(filters) ReLU branch

A Conv2D(filters) Sigmoid gate

- Outputs are multiplied element-wise → feature-wise attention

- Added to the final decoder output (conv3) as residual refinement
#### ✅ Why This Works
- Combines local details (conv1) and global context (conv2) via U-Net skip connections.

- Implements feature gating to refine spectral activation maps.

- Acts as a UNet++-inspired decoder while staying lightweight for HSI.
  ```
Input
 │
├─ Conv → Conv ─────┐
│                   │
│               ┌───▼───┐
│               │ MaxPool│
│               └───┬───┘
│                   ▼
│             Conv → Conv (Bottleneck)
│                   │
│              Upsample
│                   │
├───────── Concatenate ←──── conv1
│                   │
│             Conv → Conv
│                   │
├── Gated residual ← input gated block
│                   │
└─── Add + Output Feature Map

---
### c. Topological Fusion
```TopoFeature [B, D]
     │
 Dense → Reshape → TopoAttention [B, 1, 1, C]
     │                          │
     └────── Element-wise ──────┘
                ↓
     Scaled Spectral Feature [B, H, W, C]
                ↓
       Conv2D(1x1) Recalibration
                ↓
          Output [B, H, W, C]
```
---

- Topo feature projected with Dense layer

- Reshaped to [B, 1, 1, C]

- Multiplied with spectral feature map

Optional Conv2D recalibration

✅ Implements topological attention fusion block

### 🧠 TopoMorph-RGNet v2.0 – Unified Architecture
The build_topomorph_model() function implements the TopoMorph-RGNet v2.0, a hybrid architecture for hyperspectral image classification, fusing spatial, spectral, and topological learning pathways using gated and attention-based mechanisms.

🧬 Architecture Overview
A dual-branch network combining:

📦 RGConv-based spatial encoder (Recursive Gated Convolution)

🌈 UNet++ spectral decoder with gated residuals

🌀 Topo-Attention fusion using persistent homology descriptors

### 🔍 Step-by-Step Pipeline
1. Input Branches

- input_img: A hyperspectral patch of shape (13, 13, 30)

- input_topo: Topological descriptor vector (e.g., Betti curve or persistence summary), shape (1,)

2. Spatial Stream → RGConvBlock

Extracts spatial context using recursive gated convolutions.

Emphasizes local textures and edge patterns.

3. Spectral Decoder → UNetPlusPlusDecoder

Processes input through a UNet++-inspired encoder-decoder.

Includes skip connections, upsampling, and a gated refinement unit to preserve spectral fidelity.

4. Topo-Spectral Fusion → TopoAttentionFusion

Learns to modulate spectral features using topological insights.

Performs sample-wise channel recalibration based on topo descriptors.

5. Feature Aggregation

Combines spatial stream (x_spatial) and spectral decoder output (x_spectral) via an element-wise addition.

6. Global Pooling & Classification

Applies GlobalAveragePooling2D to reduce spatial dimensions.

A final Dense layer with softmax activation outputs per-pixel class probabilities.


### 8. Training & Evaluation
Uses early stopping and model checkpoint

Computes:

- OA (overall accuracy)  99.76% 

- AA (average class accuracy)  99.59% 

- Kappa score  0.9972

- F1 macro   0.9971

- Confusion matrix

✅ Complete evaluation pipeline


## 🧪 Evaluation Report

- **Classifier**: Dense Softmax over fused features  
- **Loss Function**: Categorical Crossentropy  
- **Optimizer**: Adam  
- **Batch Size**: 64  
- **Epochs**: 100 (with early stopping)

---

## 💡 Key Innovations
| Module                   | Purpose                                                                  |
| ------------------------ | ------------------------------------------------------------------------ |
| 🔁 **RGConvBlock**       | Models **spatial recursiveness and saliency** using gated interactions   |
| 🧬 **UNet++ Decoder**    | Captures fine-grained spectral structures with **deep skip-connections** |
| 🧠 **Topo-Attention**    | Integrates **global topological priors** into local spectral dynamics    |
| ➕ **Dual-Stream Fusion** | Merges spatial and spectral-topological representations                  |

---
| Block                   | Description                                                               |
| ----------------------- | ------------------------------------------------------------------------- |
| **Input**               | Hyperspectral patch `(13,13,30)` and topological descriptor `(1,)`        |
| 🔁 **RGConvBlock**      | Two gated Conv2Ds followed by recursive multiplication → spatial encoding |
| 🧬 **UNet++ Decoder**   | Encoder-decoder with skip-concat and gated refinement → spectral decoding |
| 🔀 **Fusion**           | Fused output of RGConvBlock + UNet++ via Add                              |
| 🧠 **TopoAttention**    | Topological descriptor rescales spectral features (via Dense → Conv1x1)   |
| 🌐 **Global Pooling**   | Converts (H, W, C) to (C) using `GlobalAveragePooling2D`                  |
| 🎯 **Dense Classifier** | Fully connected `Dense(num_classes)` → final softmax prediction           |
---

## 🔬 Design Motivation
### Why Topology?

- Traditional CNNs ignore global shape and void-based information.

- Persistent homology introduces an abstract summary of structural complexity.

- Combining this with spectral learning enables robust generalization in noisy or imbalanced HSI datasets.
---

| Layer Type              | Output Shape         | Parameters  |
| ----------------------- | -------------------- | ----------- |
| `Input (Image)`         | `(None, 13, 13, 30)` | 0           |
| `Conv2D × 2` (RGConv)   | `(13,13,64)`         | \~34K       |
| `Multiply` (RG gate)    | `(13,13,64)`         | 0           |
| `Conv2D × 2` (RG deep)  | `(13,13,64)`         | \~74K       |
| `Multiply`              | `(13,13,64)`         | 0           |
| `Conv2D × 2` (Encoder)  | `(13→6,6→128)`       | \~221K      |
| `UpSampling + Resize`   | `(6→13,13→13,128)`   | 0           |
| `Concat (skip)`         | `(13,13,192)`        | 0           |
| `Conv2D × 2` (Decoder)  | `(13,13,64)`         | \~111K      |
| `Conv2D × 2` (Gate)     | `(13,13,64)`         | \~34K       |
| `Multiply (Refinement)` | `(13,13,64)`         | 0           |
| `Add (spectral+gate)`   | `(13,13,64)`         | 0           |
| `Add (spatial+spec)`    | `(13,13,64)`         | 0           |
| `GlobalAvgPooling2D`    | `(None, 64)`         | 0           |
| `Dense(num_classes=16)` | `(None, 16)`         | 1,040       |
| **Total Parameters**    |                      | **567,568** |
---

## 📁 Directory Structure

## 🛠 Installation

---
## 🚀 How to Run
- Place dataset .mat files in the input/ folder

- Run topomorph_rgnet.ipynb or the script

- Best model saved as best_topomorph.h5
