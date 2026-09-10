<div align="center">

# 🎨 Gray → Color

### pix2pix를 활용한 이미지 색상화

U-Net Generator + PatchGAN Discriminator 구조로<br/>
흑백 사진을 자연스럽고 실사에 가까운 컬러 이미지로 복원합니다.

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

[English](README.md) · **한국어**

<img src="https://github.com/user-attachments/assets/fc5eb83b-0ff9-4a00-bc1e-e18c08d8fde8" width="900" />

</div>

---

## 📑 목차

| 항목 | 설명 |
|---|---|
| [Motivation](#-motivation-주제-선정-배경) | 프로젝트 주제 선정 및 배경 |
| [프로젝트 목표](#-프로젝트-목표) | 무엇을 만들고자 했는가 |
| [Flowchart](#-flowchart) | 전체 진행 흐름 |
| [Dataset](#-dataset) | 데이터 수집 · EDA · 전처리 |
| [모델 구조](#-모델-구조) | pix2pix 선정 배경과 동작 원리 |
| [모델 분석 및 개선](#-모델-분석-및-개선) | 성능 저하 원인 진단과 해결 |
| [결과 비교](#-결과-비교) | 정량 지표와 주관적 평가 |
| [웹 서비스](#-웹-서비스-huerevive) | Flask 서비스 및 실행 방법 |
| [프로젝트 구조](#-프로젝트-구조) | 저장소 파일 구성 |
| [한계 및 Future Works](#%EF%B8%8F-한계-및-future-works) | 남은 과제 |
| [기대효과](#-기대효과) | 결과물의 가치 |

---

## 💡 Motivation (주제 선정 배경)

최근 AI 기술의 발달로 **과거의 기록물을 디지털로 복원**하려는 수요가 증가하고 있습니다.

흑백 사진은 정보 전달력과 감성 전달 측면에서 한계를 가지며, 이를 색상화하면 시각적 이해도와 몰입도, 생동감을 높일 수 있습니다. 그러나 기존의 수작업 색상화 방식은 **시간과 비용이 많이 듭니다**.

이에 본 프로젝트에서는 기존 방식보다 시간·비용 측면에서 효율적인 **딥러닝 기반 색상화 방식**을 연구했습니다. 자연스러운 복원 결과를 위해 시각적 유사성을 평가하는 성능 지표뿐만 아니라 **사용자 주관적 평가**도 함께 고려했습니다.

이를 통해 역사적 기록물, 사진 복원, 교육용 자료 등 다양한 분야에 활용할 수 있으며, 나아가 개인의 소중한 기억의 한 조각을 되찾는 효과를 기대할 수 있습니다.

<p align="center">
  <img src="https://github.com/user-attachments/assets/4c2ef17c-7355-46f5-8ad2-bebae5f75b61" width="700" />
</p>

---

## 🎯 프로젝트 목표

> 딥러닝 모델을 활용해 흑백 사진을 **자연스럽고 실사에 가까운 컬러 이미지**로 복원하는 AI 시스템 구현

---

## 🔄 Flowchart

<p align="center">
  <img src="https://github.com/user-attachments/assets/4d278af9-5f86-477f-bb96-d5dcee76fed9" width="700" />
</p>

---

## 📂 Dataset

### 1. 데이터셋 수집

다양한 풍경, 인물, 동물, 꽃, 음식 등을 포함한 **쌍(pair) 이미지 기반 데이터셋**을 수집했습니다.

### 2. 데이터셋 구성

| 구분 | Color | Gray-scale |
|:---|:---:|:---:|
| **Train** | 1,500장 | 1,500장 |
| **Validation** | 100장 | 100장 |
| **Test** | — | 150장 |

### 3. EDA (탐색적 데이터 분석)

컬러 이미지와 흑백 이미지의 높이(Height) · 너비(Width) 분포:

| Gray-scale img EDA | Color img EDA |
|---------|---------|
| <img src="https://github.com/user-attachments/assets/c6f8657a-f4f0-4979-ba07-77073cb719b3" width="550" height="420" /> | <img src="https://github.com/user-attachments/assets/f89caa2f-705a-43a0-98b6-4a7c5fbc8bd9" width="550" height="420" /> |

컬러 이미지의 RGB 채널 분포:

<p align="center">
  <img src="https://github.com/user-attachments/assets/cd87f8b7-a9b7-462c-aece-c277cec31b6b" width="700" />
</p>

**분석 결과**

- 컬러 이미지와 흑백 이미지 모두 **150 × 150** 크기의 정사각형 이미지입니다.
- 컬러 이미지의 **R, G, B** 세 채널이 대체로 균일하게 분포되어 색상의 균형이 잘 맞춰져 있습니다.

### 4. 데이터 전처리

모든 이미지가 동일한 크기를 가지므로, 모델 학습에 앞서 **별도의 리사이징이나 추가 전처리 과정이 필요하지 않았습니다**.

---

## 🧠 모델 구조

### pix2pix — U-Net Generator + PatchGAN Discriminator

<p align="center">
  <img src="https://github.com/user-attachments/assets/92efdf19-051b-421b-9006-aa8322b542e5" width="700" height="400" />
</p>

### pix2pix 모델 선정 배경

| # | 근거 |
|:---:|---|
| 1 | 흑백 → 컬러 색상화는 **픽셀 단위의 정밀한 매핑**이 필요한 image-to-image translation 문제입니다. |
| 2 | U-Net 구조는 인코더(압축)와 디코더(복원) 사이에 **skip connection**이 있어, 저수준의 공간 정보를 고해상도로 정확히 복원할 수 있습니다. |
| 3 | 이 구조 덕분에 색상화에서 중요한 **윤곽선 · 질감 · 경계 정보**가 잘 유지됩니다. |
| 4 | 일반적인 Discriminator는 **이미지 전체**를 보고 Real / Fake를 구분합니다. |
| 5 | 색상화 문제에서는 **국소 영역(작은 패치)** 의 색 배치나 질감이 자연스러운지가 더 중요합니다. |
| 6 | PatchGAN은 전체 이미지를 하나로 처리하지 않고 공통된 소형 CNN 필터를 반복 적용해 패치를 평가하므로 → **파라미터 수가 적고 빠르며 학습이 안정적**입니다. |

> **결론:** U-Net 기반 Generator와 PatchGAN Discriminator가 결합된 구조인 pix2pix 모델을 선정했습니다.

### U-Net based Generator

1. **인코더(encoder)** 부분에서 입력 이미지가 점점 축소되며 특징을 추출합니다.
2. **디코더(decoder)** 부분에서 이미지의 해상도를 다시 복원합니다.
3. 점선 화살표는 입력 정보와 출력 정보를 직접 연결하는 U-Net의 핵심인 **스킵 커넥션**으로, 저수준 정보를 고수준 레이어에 직접 전달해 디테일을 유지합니다.

*구현: down-block 8단(64 → 512), up-block 8단(앞 3단에 dropout 0.5), 출력 `Tanh`, 입력 1채널 → 출력 3채널*

### PatchGAN Discriminator

1. Generator가 생성한 결과 이미지 `T(x)`와 정답 이미지 `Ground Truth y` 두 개를 판별에 사용합니다.
2. CNN 기반으로 점점 공간의 크기를 줄여가며 특징을 추출하고, **P1 · P2 · P3 · P4** 여러 단계에서 `T(x)`와 `y` 쌍의 정합성을 확인합니다.
3. 마지막 출력에서 실제 이미지인지, Generator가 만든 가짜 이미지인지 판단합니다.
4. 판별 결과를 바탕으로 Generator는 더 정답에 가까운 `T(x)`를 만들도록 학습됩니다.

---

## 🔬 모델 분석 및 개선

### 초기 모델의 문제점

<p align="center">
  <img src="https://github.com/user-attachments/assets/75ccf1ca-92ff-4786-a7ea-92f09f820bd6" width="700" height="400" />
</p>

1. 결과 이미지(Result Image)가 원본과 비교했을 때 **품질이 좋지 않음**
2. **성능 지표** 또한 수치가 높게 나오지 않음

**성능 지표 (Evaluation Metrics)**

| 지표 | 정식 명칭 | 방향 |
|:---|:---|:---:|
| **PSNR** | Peak Signal-to-Noise Ratio | ↑ 높을수록 좋음 |
| **SSIM** | Structural Similarity Index Measure | ↑ 높을수록 좋음 |
| **LPIPS** | Learned Perceptual Image Patch Similarity | ↓ 낮을수록 좋음 |
| **FID** | Fréchet Inception Distance | ↓ 낮을수록 좋음 |

### 성능 저하의 원인

1. **학습 부족** — PSNR과 SSIM이 모두 낮고 LPIPS와 FID가 모두 높다면 학습이 부족하다는 의미입니다 (당시 epoch 10회).
2. **G_LOSS 미수렴** — 학습 결과 Discriminator의 loss는 일정하게 낮은 반면, Generator의 loss는 epoch이 증가해도 감소하지 않고 **10 ~ 12** 수준에서 진동했습니다. Discriminator가 Generator를 압도한 상태입니다.

| Evaluation Matrix | LOSS Graph |
|---------|---------|
| <img src="https://github.com/user-attachments/assets/358dd685-c8a1-47c1-81fa-5069cb44206d" width="550" height="420" /> | <img src="https://github.com/user-attachments/assets/f6f4919b-dbac-46fc-a46d-7c7078365128" width="550" height="420" /> |

### 개선 방안

1. **하이퍼파라미터 조정** — epoch 50회, early stopping 적용, learning rate 하향 조정
2. **Discriminator 학습 지연** — Generator가 충분히 학습할 시간을 확보하기 위해 **8번째 epoch 이후부터**, 그것도 3 스텝에 1회만 Discriminator를 학습시켰습니다. 초반에 Discriminator 학습을 막아 Generator가 더 안정적으로 학습을 시작할 수 있게 했습니다.

| Improve Evaluation Matrix | Improve LOSS Graph |
|---------|---------|
| <img src="https://github.com/user-attachments/assets/a6e3338f-b729-4d04-b2ea-1c3ab0f3ae26" width="550" height="420" /> | <img src="https://github.com/user-attachments/assets/31b2f404-0173-4082-9645-940656137a9a" width="550" height="420" /> |

### 학습 설정 비교

| 설정 | 초기 모델 | 개선 모델 |
|:---|:---|:---|
| Epochs | 10 | 50 (early stopping, patience 10) |
| Learning rate | 5e-4 (G / D) | 5e-5 (G / D) + `ReduceLROnPlateau` (factor 0.5, patience 3) |
| Batch size | 16 | 16 |
| Optimizer | Adam | Adam (β = 0.5, 0.999) |
| Generator loss | BCE + L1 | BCE + **30 ×** L1 + **5 ×** VGG19 perceptual loss |
| Discriminator 갱신 | 1 epoch부터 매 스텝 | **8 epoch부터**, 3 스텝에 1회 |
| Label | hard 0 / 1 | one-sided smoothing — real `U(0.8, 1.0)`, fake `U(0.0, 0.2)` |
| Seed | 42 | 42 |

---

## 📊 결과 비교

<p align="center">
  <img src="https://github.com/user-attachments/assets/fc5eb83b-0ff9-4a00-bc1e-e18c08d8fde8" width="900" height="400" />
</p>

### 생각할 점 — Ill-posed Problem

> **모델의 성능 지표가 좋다고 해서 결과 이미지의 품질이 반드시 좋아지는 것은 아닙니다.**

색상화는 하나의 흑백 입력에 대해 여러 개의 그럴듯한 정답이 존재하는 **ill-posed problem**입니다. 따라서 정량 지표와 함께 **사용자의 주관적 평가**에도 초점을 둘 필요가 있습니다.

이번 프로젝트에서는 운이 좋게도, 성능 지표가 가장 좋은 모델에서 나온 결과 이미지가 사용자 주관적 평가에서도 가장 좋은 평가를 받았습니다.

<p align="center">
  <img src="https://github.com/user-attachments/assets/8ed6cdff-9e64-4666-8568-a73cf7184723" width="900" height="200" />
</p>

---

## 🌐 웹 서비스 (HueRevive)

<p align="center">
  <img src="https://github.com/user-attachments/assets/9ab66b7d-0d88-4310-9efc-6514ff7f64ef" width="800" height="400" />
</p>

파이썬 기반의 마이크로 웹 프레임워크인 **Flask**를 이용해 웹페이지를 개발하여 서비스 방안을 구상했습니다.
사용자가 흑백 이미지를 업로드하면 **학습된 3개의 Generator**가 각각 색상화한 결과를 제시하고, 사용자가 마음에 드는 결과를 선택합니다.

### 실행 방법

```bash
# 1. 의존성 설치
pip install torch torchvision flask pillow

# 2. 학습된 가중치 배치
#    HueRevive/saved_models/best_generator_1.pth
#    HueRevive/saved_models/best_generator_2.pth
#    HueRevive/saved_models/best_generator_3.pth

# 3. 서버 실행
cd HueRevive
python app.py
```

이후 브라우저에서 **http://127.0.0.1:5000** 으로 접속합니다.

### 처리 흐름

```
흑백 이미지 업로드
        ↓
L(흑백) 변환 → 256×256 리사이즈 → [-1, 1] 정규화
        ↓
3개 Generator로 추론
        ↓
역정규화 → PNG 저장 → 3개 후보 제시
        ↓
사용자가 최종 이미지 선택
```

> ⚠️ **참고** — `app.py`는 `debug=True` 상태로 실행되며, 데모 로그인 계정과 `secret_key`가 코드에 하드코딩되어 있습니다. 실제 배포 전에는 반드시 교체하고 디버그 모드를 해제해야 합니다.
> `saved_models/*.pth` 가중치 파일과 `templates/` · `static/` 리소스는 이 저장소에 포함되어 있지 않습니다.

---

## 📁 프로젝트 구조

```
Gray-to-color-colorization-pix2pix-based/
├── README.md                                   # 영어
├── README.ko.md                                # 한국어 (현재 문서)
├── 흑백 이미지 컬러 복원 색상화 A 모델.ipynb      # 학습 · 평가 노트북 (Colab)
└── HueRevive/                                  # Flask 웹 서비스
    ├── app.py                                  # 라우팅: 업로드 · 추론 · 결과 선택
    └── colorize_model.py                       # U-Net Generator 정의 및 추론 로직
```

### 노트북 구성

| 단계 | 내용 |
|:---:|---|
| 1 | Google Drive 마운트, 시드 고정 (42) |
| 2 | 데이터셋 및 `DataLoader` 구성 |
| 3 | U-Net Generator, PatchGAN Discriminator 정의 |
| 4 | 초기 학습 (10 epochs) |
| 5 | 개선 학습 (50 epochs, perceptual loss, Discriminator 지연, early stopping) |
| 6 | 테스트셋 추론, 결과 512×512 업스케일 |
| 7 | PSNR / SSIM / LPIPS / FID 평가 |

---

## ⚠️ 한계 및 Future Works

<p align="center">
  <img src="https://github.com/user-attachments/assets/8e719aa3-b714-458c-ace8-d7eac8a301e7" width="800" height="200" />
</p>

**한계** — 학습되지 않은 유형의 이미지 데이터에 대해서는 색상화를 하지 못합니다. 학습 이미지 데이터가 부족한 것이 원인입니다.

**1. 학습 데이터 추가 및 증강 시도**

학습 데이터의 양과 다양성을 확장하여 모델의 일반화 성능을 향상시키고, 보다 다양한 장면 · 조명 · 피사체를 포함하는 이미지를 수집하여 복잡한 컬러화 상황에 대한 대응력을 강화합니다.

**2. 평가 지표 및 주관적 평가 보완**

사용자 주관적 평가 데이터를 축적하여, 실제로 서비스를 이용하는 사용자 관점의 품질 평가를 반영합니다.

---

## ✨ 기대효과

**1. 역사적 가치의 복원과 대중화**

흑백 사진·영상을 컬러로 전환함으로써 과거가 더욱 현실감 있게 다가오며, 역사의 생생한 감각을 느낄 수 있습니다. 단절된 시간 속 인물과 공간에 감정을 이입하게 함으로써 공감과 관심을 이끌어내어 사람들의 역사적 거리감을 해소할 수 있습니다.

**2. 심리적 · 정서적 기대효과**

개인의 추억, 기억 또는 감정을 자극하여 향수와 감동을 유도할 수 있습니다.

**3. 기록 보존 및 연구 자료 활용**

복원된 컬러 이미지들이 다양한 분야에서 인공지능 연구·개발에 유용한 자료가 됩니다.

---

<div align="center">

**건양대학교** · AI 프로젝트

</div>
