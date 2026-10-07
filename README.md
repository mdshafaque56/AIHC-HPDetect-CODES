<div align="center">

# 🔬 HPDetect AI

### An Explainable Anchor-Free Deep Learning Framework for *Helicobacter pylori* Detection in Histopathology Images

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![Backbone](https://img.shields.io/badge/Backbone-ResNet50-green)
![Attention](https://img.shields.io/badge/Attention-HPA--Net%20(SE)-orange)
![XAI](https://img.shields.io/badge/Explainability-Grad--CAM%2B%2B-blueviolet)
![Detection](https://img.shields.io/badge/Detection-Anchor--Free-red)
![Course](https://img.shields.io/badge/NIT%20Calicut-AI%20for%20Healthcare-informational)

*Detect. Localize. Explain.*

</div>

---

## 📖 Overview

*Helicobacter pylori* (*H. pylori*) is a leading cause of chronic gastritis, peptic ulcer disease, and gastric carcinoma. Diagnosis through histopathology depends on expert pathologists manually inspecting biopsy slides, which is **time-consuming, subjective, and hard to scale**.

**HPDetect AI** is an explainable, anchor-free deep learning framework that automatically **detects** and **localizes** *H. pylori* in gastric histopathology images, and shows **why** it made each prediction. It combines a partially fine-tuned **ResNet50** backbone with a custom attention module, **HPA-Net** (*H. pylori* Attention Network), and an FCOS/CenterNet-inspired detection pipeline that works with **image-level labels only**.

### ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **HPA-Net Attention** | Squeeze-and-Excitation channel attention to amplify diagnostically relevant features and suppress background tissue |
| 🎯 **Anchor-Free Localization** | Directly regresses `(l, t, r, b)` distances from pathological centers, with no predefined anchor boxes |
| 🗺️ **Dense Classification Maps** | Spatial probability heatmaps instead of a single image-level label |
| 📍 **Centerness Estimation** | Suppresses low-quality, off-center detections to reduce false positives |
| 🏷️ **Weak Supervision** | Learns localization from image-level labels, which reduces expensive annotation effort |
| 🔍 **Grad-CAM++ Explainability** | Clinically interpretable heatmaps highlighting regions driving each prediction |
| 💬 **Gemma 4 Clinical Layer** | Converts model outputs into human-readable diagnostic summaries |

---

## 🏗️ Architecture

```
Input Histopathology Image (224 × 224 × 3)
            │
            ▼
   Preprocessing (resize, normalize, augment)
            │
            ▼
   ResNet50 Backbone (ImageNet pre-trained, Layer3 & Layer4 fine-tuned)
            │
            ▼
   Deep Feature Maps  (7 × 7 × 2048)
            │
            ▼
   HPA-Net Attention Module (Squeeze-and-Excitation)
            │
   ┌────────┼─────────────────┐
   ▼        ▼                 ▼
Dense     Regression       Centerness
Classif.  Head             Head
Head      (l, t, r, b)     (confidence)
   │        │                 │
   └────────┼─────────────────┘
            ▼
   Anchor-Free ROI Localization
            │
            ▼
   Weakly Supervised Refinement (Grad-CAM++)
            │
            ▼
   Final Detection + Classification
            │
            ▼
   Explainability Maps + Confidence Scores
            │
            ▼
   Gemma 4 Clinical Explanation Layer
```

### Component Details

**1. ResNet50 Backbone**
Pre-trained on ImageNet with the FC layer removed. Only `Layer3` and `Layer4` are fine-tuned, while earlier layers stay frozen to preserve low-level visual features.

**2. HPA-Net Attention (Squeeze-and-Excitation)**

```
Squeeze:     z_c = (1 / H×W) ΣΣ x_c(i, j)
Excitation:  s   = σ( W₂ · δ( W₁ · z ) )
Recalibrate: x̂_c = s_c · x_c
```

**3. Prediction Heads**

| Head | Architecture | Output | Loss |
|---|---|---|---|
| **Dense Classification** | `Conv(2048→512) → ReLU → Conv(512→1)` | *H. pylori* probability map | Binary Cross Entropy |
| **Regression** | `Conv(2048→512) → ReLU → Conv(512→4)` | `(l, t, r, b)` box distances | Smooth L1 |
| **Centerness** | `Conv(2048→256) → ReLU → Conv(256→1)` | Localization confidence map | Centerness loss |

**4. Centerness**

```
C = √[ (min(l, r) / max(l, r)) × (min(t, b) / max(t, b)) ]
```

**5. Total Loss**

```
L_total = L_cls + λ₁ · L_reg + λ₂ · L_center
```

**6. Grad-CAM++ Explainability**

```
L^c = ReLU( Σ_k α_k^c · A^k )
```

Heatmap interpretation: 🔴🟡 **red/yellow** is high pathological significance, 🟢 **green** is moderate or uncertain, 🔵 **blue** is background or minimal influence.

---

## 🧮 Worked Example

A step-by-step numerical walkthrough from the report:

| Step | Operation | Result |
|---|---|---|
| 1 | Normalize pixel `R = 180` with `μ=128, σ=64` | `0.8125` |
| 2 | ResNet50 feature extraction | `7×7×2048` |
| 3 | SE attention: `z_c = 5`, `s_c = 0.92`, `x_c = 4.2` | `x̂_c = 3.864` |
| 4 | Dense classification (peak probability) | `0.95` |
| 5 | Regression: center `(112,112)`, `l=20, t=15, r=35, b=25` | ROI `(92, 97, 147, 137)` |
| 6 | Centerness | `0.585` |
| 7 | Final confidence = `0.95 × 0.585` | `0.555`, **Borderline / Low Positive** at a 0.6 threshold |
| 8 | Grad-CAM++ weighted activation | `5.88` → 🔴/🟡 region |

> This example shows how **localization quality directly influences final diagnostic confidence**.

---

## 🏋️ Training Strategy

| Setting | Value |
|---|---|
| **Approach** | Transfer learning + partial fine-tuning |
| **Backbone init** | ImageNet pre-trained ResNet50 |
| **Trainable layers** | `Layer3`, `Layer4` |
| **Optimizer** | Adam |
| **Learning rate** | `0.001` |
| **Regularization** | Batch Normalization, Dropout |
| **Classification loss** | Binary Cross Entropy |
| **Regression loss** | Smooth L1 |
| **Input size** | `224 × 224 × 3` |
| **Augmentation** | Random horizontal flip, intensity and color normalization |

---

## 📊 Results & Observations

HPDetect AI was benchmarked against baseline architectures (ResNet50, DenseNet121, EfficientNetV2, ViT). Comparison plots (accuracy, F1, ROC-AUC, training curves) are in the project report.

**Qualitative findings**

- ✅ Grad-CAM++ attention consistently concentrated on **suspicious glandular and epithelial regions**
- ✅ Strong ROI-focused attention for high-confidence predictions
- ✅ Uncertain predictions produced weaker, more diffuse attention, a useful signal of low confidence
- ✅ Background tissue activations were effectively suppressed
- ✅ Stable localization across different samples

| Input | Actual Label | HPDetect AI Prediction |
|---|---|---|
| Sample 1 | HP Negative | ✅ HP Negative |
| Sample 2 | HP Positive | ✅ HP Positive |
| Sample 3 | HP Positive | ✅ HP Positive |
| Sample 4 | HP Positive | ✅ HP Positive |

> 📝 Qualitative samples were drawn from a ResearchGate figure illustrating HP-negative and HP-positive cases (H&E staining, ×200). See the report for details.

---

## 📁 Repository Structure

> Adjust this section to match your actual repo layout.

```
AIHC-HPDetect-CODES/
├── data/               # Dataset (not included)
├── models/             # HPA-Net, detection heads, backbone
├── explainability/     # Grad-CAM++ implementation
├── train.py            # Training script
├── evaluate.py         # Evaluation & metrics
├── inference.py        # Run predictions + heatmaps
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- PyTorch & torchvision
- A CUDA-capable GPU (recommended)

### Installation

```bash
git clone https://github.com/mdshafaque56/AIHC-HPDetect-CODES.git
cd AIHC-HPDetect-CODES

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### Usage

```bash
# Train
python train.py --epochs 5 --lr 0.001

# Evaluate
python evaluate.py --weights path/to/weights.pth

# Inference with Grad-CAM++ heatmap
python inference.py --image path/to/image.png --weights path/to/weights.pth
```

> ⚠️ Script names and arguments above are placeholders. Update them to match your code.

---

## ⚠️ Limitations

- **Limited dataset scale and diversity**, a common challenge for annotated histopathology data
- **Weak supervision** may give less precise boundaries than pixel-level segmentation, especially for tiny bacterial regions
- **Training cost**: deep CNNs with attention and dense heads need substantial GPU resources
- **Needs large-scale clinical validation** across hospitals, scanners, staining protocols, and patient populations before real-world use

> 🩺 **Disclaimer:** HPDetect AI is a **research prototype** built for an academic course. It is **not a certified medical device** and must not be used as a substitute for professional clinical diagnosis.

---

## 🔮 Future Work

- 🔹 Transformer-based pathology attention and multi-scale feature pyramids
- 🔹 Semi-/self-supervised learning to reduce labeling needs
- 🔹 Whole-Slide Image (WSI) processing
- 🔹 Uncertainty-aware predictions and active learning
- 🔹 Federated learning for privacy-preserving training
- 🔹 Cross-domain adaptation across scanners and staining protocols
- 🔹 Edge AI deployment and real-time pathologist assistance
- 🔹 Multi-modal analysis (patient records, genomics, lab reports)
- 🔹 Automated report generation and severity grading

---

## 📚 References

1. K. He et al., *Deep Residual Learning for Image Recognition*, CVPR, 2016.
2. J. Redmon et al., *You Only Look Once: Unified Real-Time Object Detection*, CVPR, 2016.
3. X. Zhou et al., *Objects as Points*, arXiv:1904.07850, 2019.
4. S. Woo et al., *CBAM: Convolutional Block Attention Module*, ECCV, 2018.
5. J. Hu et al., *Squeeze-and-Excitation Networks*, IEEE TPAMI, 2020.
6. R. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks*, ICCV, 2017.
7. A. Chattopadhay et al., *Grad-CAM++*, IEEE WACV, 2018.
8. G. Litjens et al., *A Survey on Deep Learning in Medical Image Analysis*, Medical Image Analysis, 2017.
9. I. Goodfellow et al., *Deep Learning*, MIT Press, 2016.
10. T. Lin et al., *Focal Loss for Dense Object Detection*, ICCV, 2017.

---

## 👥 Authors

| Name | Department |
|---|---|
| **MD Shafaque** | Civil Engineering, NIT Calicut |
| **Arsh Uniyal** | Chemical Engineering, NIT Calicut |

**Under the guidance of** **Dr. Jayaraj PB**, Associate Professor, Department of Computer Science & Engineering, NIT Calicut

*Submitted in partial fulfilment of the **AI for Healthcare (Minor Programme)**, National Institute of Technology Calicut, India, Academic Year 2025–2026.*

---

## 📄 License

This project was developed for academic purposes.

---

<div align="center">

⭐ **If you found this project useful, please consider starring the repo!** ⭐

</div>
