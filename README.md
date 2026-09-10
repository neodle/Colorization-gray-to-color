<div align="center">

# 🎨 Gray → Color

### Image Colorization with pix2pix

Restoring grayscale photographs into natural, lifelike color images<br/>
with a U-Net Generator + PatchGAN Discriminator.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

**English** · [한국어](README.ko.md)

<img src="https://github.com/user-attachments/assets/fc5eb83b-0ff9-4a00-bc1e-e18c08d8fde8" width="900" />

</div>

---

## 📑 Table of Contents

| Section | Description |
|---|---|
| [Motivation](#-motivation) | Why this topic was chosen |
| [Objective](#-objective) | What the project aims to build |
| [Pipeline](#-pipeline) | End-to-end workflow |
| [Dataset](#-dataset) | Data collection, EDA, preprocessing |
| [Model Architecture](#-model-architecture) | Why pix2pix, and how it works |
| [Analysis & Improvement](#-analysis--improvement) | Diagnosing failures and fixing them |
| [Results](#-results) | Quantitative and subjective comparison |
| [Web Demo](#-web-demo-huerevive) | Flask service and how to run it |
| [Project Structure](#-project-structure) | Files in this repository |
| [Limitations & Future Work](#%EF%B8%8F-limitations--future-work) | What is left to solve |
| [Expected Impact](#-expected-impact) | Value of the outcome |

---

## 💡 Motivation

As AI matures, demand for **digitally restoring historical records** keeps growing.

Grayscale photographs are limited in how much information and emotion they can convey. Adding color raises visual comprehension, immersion, and vividness — but doing it by hand is **slow and expensive**.

This project therefore explores a **deep-learning approach to colorization** that is far more efficient in time and cost than manual work. To ensure the restored images look natural, the model is judged not only by pixel-level similarity metrics but also by **subjective human evaluation**.

The result can be applied to historical archives, photo restoration, and educational material — and, more personally, to recovering a cherished fragment of someone's memory.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4c2ef17c-7355-46f5-8ad2-bebae5f75b61" width="700" />
</p>

---

## 🎯 Objective

> Build an AI system that uses a deep-learning model to restore grayscale photographs into **natural, photorealistic color images**.

---

## 🔄 Pipeline

<p align="center">
  <img src="https://github.com/user-attachments/assets/4d278af9-5f86-477f-bb96-d5dcee76fed9" width="700" />
</p>

---

## 📂 Dataset

### 1. Collection

Paired image datasets covering a wide range of subjects — landscapes, people, animals, flowers, food, and more.

### 2. Composition

| Split | Color | Grayscale |
|:---|:---:|:---:|
| **Train** | 1,500 | 1,500 |
| **Validation** | 100 | 100 |
| **Test** | — | 150 |

### 3. EDA

Height / width distribution of the grayscale and color sets:

| Grayscale image EDA | Color image EDA |
|---------|---------|
| <img src="https://github.com/user-attachments/assets/c6f8657a-f4f0-4979-ba07-77073cb719b3" width="550" height="420" /> | <img src="https://github.com/user-attachments/assets/f89caa2f-705a-43a0-98b6-4a7c5fbc8bd9" width="550" height="420" /> |

RGB channel distribution of the color images:

<p align="center">
  <img src="https://github.com/user-attachments/assets/cd87f8b7-a9b7-462c-aece-c277cec31b6b" width="700" />
</p>

**Findings**

- Both color and grayscale images are square, **150 × 150** throughout.
- The **R, G, B** channels are all distributed fairly evenly, so the color balance is healthy.

### 4. Preprocessing

Because every image already shares the same resolution, **no separate resizing or additional preprocessing** was required before training.

---

## 🧠 Model Architecture

### pix2pix — U-Net Generator + PatchGAN Discriminator

<p align="center">
  <img src="https://github.com/user-attachments/assets/92efdf19-051b-421b-9006-aa8322b542e5" width="700" height="400" />
</p>

### Why pix2pix?

| # | Reason |
|:---:|---|
| 1 | Grayscale → color is an **image-to-image translation** problem that needs precise pixel-level mapping. |
| 2 | U-Net places **skip connections** between the encoder (compression) and the decoder (reconstruction), so low-level spatial information is restored accurately at high resolution. |
| 3 | Thanks to that structure, the **edges, textures, and boundaries** that matter most in colorization are preserved. |
| 4 | A conventional discriminator judges real/fake by looking at the **whole image** at once. |
| 5 | In colorization, what matters more is whether the color layout and texture of **local patches** look natural. |
| 6 | PatchGAN instead applies a small shared CNN filter repeatedly across patches → **fewer parameters, faster, and more stable to train**. |

> **Conclusion:** pix2pix, which combines a U-Net Generator with a PatchGAN Discriminator, was selected.

### U-Net Generator

1. In the **encoder**, the input image is progressively downsampled while features are extracted.
2. In the **decoder**, the resolution is progressively restored.
3. The dashed arrows are the **skip connections** at the heart of U-Net. They pass low-level information straight to the higher layers so fine detail survives.

*Implementation: 8 down-blocks (64 → 512) and 8 up-blocks, dropout 0.5 on the first three up-blocks, `Tanh` output, 1-channel input → 3-channel output.*

### PatchGAN Discriminator

1. Takes both the generated image `T(x)` and the ground truth `y` as the basis for its judgment.
2. Being CNN-based, it shrinks the spatial size step by step while extracting features, checking the consistency of the `T(x)` / `y` pair at several stages (**P1–P4**).
3. The final output decides whether the image is real or was produced by the generator.
4. Using that verdict, the generator learns to produce a `T(x)` closer to the ground truth.

---

## 🔬 Analysis & Improvement

### Problems with the baseline model

<p align="center">
  <img src="https://github.com/user-attachments/assets/75ccf1ca-92ff-4786-a7ea-92f09f820bd6" width="700" height="400" />
</p>

1. The result image is **noticeably lower in quality** than the original.
2. The **evaluation metrics** also came out weak.

**Evaluation metrics used**

| Metric | Full name | Direction |
|:---|:---|:---:|
| **PSNR** | Peak Signal-to-Noise Ratio | ↑ higher is better |
| **SSIM** | Structural Similarity Index Measure | ↑ higher is better |
| **LPIPS** | Learned Perceptual Image Patch Similarity | ↓ lower is better |
| **FID** | Fréchet Inception Distance | ↓ lower is better |

### Root causes

1. **Insufficient training.** PSNR and SSIM were both low while LPIPS and FID were both high — the signature of undertraining (only 10 epochs at that point).
2. **Generator loss never converged.** The discriminator loss stayed consistently low, while the generator loss kept oscillating at a high level (**10–12**) no matter how many epochs passed: the discriminator had overpowered the generator.

| Evaluation metrics | Loss graph |
|---------|---------|
| <img src="https://github.com/user-attachments/assets/358dd685-c8a1-47c1-81fa-5069cb44206d" width="550" height="420" /> | <img src="https://github.com/user-attachments/assets/f6f4919b-dbac-46fc-a46d-7c7078365128" width="550" height="420" /> |

### Improvements

1. **Hyperparameter tuning** — 50 epochs, early stopping, and a lowered learning rate.
2. **Delayed discriminator training** — the discriminator is only updated from **epoch 8 onward**, and then only every 3rd step. Holding it back early gives the generator room to stabilize before it has to face a strong critic.

| Improved evaluation metrics | Improved loss graph |
|---------|---------|
| <img src="https://github.com/user-attachments/assets/a6e3338f-b729-4d04-b2ea-1c3ab0f3ae26" width="550" height="420" /> | <img src="https://github.com/user-attachments/assets/31b2f404-0173-4082-9645-940656137a9a" width="550" height="420" /> |

### Configuration: baseline vs. improved

| Setting | Baseline | Improved |
|:---|:---|:---|
| Epochs | 10 | 50 (early stopping, patience 10) |
| Learning rate | 5e-4 (G / D) | 5e-5 (G / D) + `ReduceLROnPlateau` (factor 0.5, patience 3) |
| Batch size | 16 | 16 |
| Optimizer | Adam | Adam (β = 0.5, 0.999) |
| Generator loss | BCE + L1 | BCE + **30 ×** L1 + **5 ×** VGG19 perceptual loss |
| Discriminator update | every step, from epoch 1 | from **epoch 8**, every 3rd step |
| Labels | hard 0 / 1 | one-sided smoothing — real `U(0.8, 1.0)`, fake `U(0.0, 0.2)` |
| Seed | 42 | 42 |

---

## 📊 Results

<p align="center">
  <img src="https://github.com/user-attachments/assets/fc5eb83b-0ff9-4a00-bc1e-e18c08d8fde8" width="900" height="400" />
</p>

### A note on the ill-posed nature of the task

> **Better metrics do not automatically mean a better-looking image.**

Colorization is an **ill-posed problem**: one grayscale input has many plausible color solutions. That is why **subjective human evaluation** deserves as much weight as the numbers do.

In this project we were fortunate — the model with the best metrics also produced the images that scored highest in subjective evaluation.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8ed6cdff-9e64-4666-8568-a73cf7184723" width="900" height="200" />
</p>

---

## 🌐 Web Demo (HueRevive)

<p align="center">
  <img src="https://github.com/user-attachments/assets/9ab66b7d-0d88-4310-9efc-6514ff7f64ef" width="800" height="400" />
</p>

The service was prototyped as a web page built with **Flask**, a Python micro web framework.
A user uploads a grayscale image, **three trained generators** each colorize it, and the user picks whichever result they prefer.

### Quick start

```bash
# 1. Install dependencies
pip install torch torchvision flask pillow

# 2. Place the trained weights
#    HueRevive/saved_models/best_generator_1.pth
#    HueRevive/saved_models/best_generator_2.pth
#    HueRevive/saved_models/best_generator_3.pth

# 3. Run the server
cd HueRevive
python app.py
```

Then open **http://127.0.0.1:5000** in a browser.

### Flow

```
Upload a grayscale image
        ↓
Convert to L (grayscale) → resize to 256×256 → normalize to [-1, 1]
        ↓
Inference with 3 generators
        ↓
Denormalize → save as PNG → present the 3 candidates
        ↓
User selects the final image
```

> ⚠️ **Note** — `app.py` runs with `debug=True`, and the demo login credentials and `secret_key` are hard-coded inside it. Replace both and disable debug mode before deploying anywhere real.
> The `saved_models/*.pth` weights and the `templates/` · `static/` assets are not committed to this repository.

---

## 📁 Project Structure

```
Gray-to-color-colorization-pix2pix-based/
├── README.md                                   # English (this file)
├── README.ko.md                                # Korean
├── 흑백 이미지 컬러 복원 색상화 A 모델.ipynb      # Training / evaluation notebook (Colab)
└── HueRevive/                                  # Flask web service
    ├── app.py                                  # Routes: upload, inference, result selection
    └── colorize_model.py                       # U-Net Generator + inference logic
```

### Notebook contents

| Step | Description |
|:---:|---|
| 1 | Mount Google Drive, fix the random seed (42) |
| 2 | Build the dataset and the `DataLoader` |
| 3 | Define the U-Net Generator and the PatchGAN Discriminator |
| 4 | Baseline training (10 epochs) |
| 5 | Improved training (50 epochs, perceptual loss, delayed discriminator, early stopping) |
| 6 | Inference on the test set, results upscaled to 512×512 |
| 7 | Evaluation with PSNR / SSIM / LPIPS / FID |

---

## ⚠️ Limitations & Future Work

<p align="center">
  <img src="https://github.com/user-attachments/assets/8e719aa3-b714-458c-ace8-d7eac8a301e7" width="800" height="200" />
</p>

**Limitation** — The model fails to colorize kinds of images it never saw during training, because the training set was too small.

**1. Add and augment training data**

Expand the volume and variety of the training set to improve generalization, and collect images spanning more scenes, lighting conditions, and subjects so the model can handle harder colorization cases.

**2. Strengthen evaluation, both metric and subjective**

Accumulate subjective evaluation data so that quality is measured from the perspective of the people actually using the service.

---

## ✨ Expected Impact

**1. Restoring and popularizing historical value**

Turning black-and-white photos and footage into color makes the past feel present, letting people experience history vividly. Drawing empathy for people and places separated from us by time closes the emotional distance to history.

**2. Psychological and emotional value**

Personal memories and emotions are stirred, evoking nostalgia and genuine feeling.

**3. Archival preservation and research material**

The restored color images become useful data for AI research and development across many fields.

---

<div align="center">

**Konyang University** · AI Project

</div>
