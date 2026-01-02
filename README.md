# 🧑‍💻 Face Search & Clustering Project

이 프로젝트는 **딥러닝 기반의 얼굴 인식 및 분석 파이프라인**입니다.
대량의 인물 이미지에서 얼굴을 검출하여 전처리하고, 특징 벡터(Embedding)를 추출하여 **특정 인물 식별(Binary Classification)** 및 **자동 군집화(Clustering)**를 수행합니다.

## 📋 Project Overview

* **Goal**: 다수의 인물 사진 중 특정 인물(Target)을 찾아내거나, 라벨링 없이 유사한 얼굴끼리 그룹화.
* **Core Tech**: `MediaPipe` (Detection), `PyTorch` (Embedding), `Scikit-learn` (Classification), `HDBSCAN` (Clustering).

---

## ⚙️ Execution Pipeline & Model Architecture

```mermaid
graph TD
    %% =========================
    %% 1) Execution Pipeline
    %% =========================
    A["Raw Images (famous_picture/)"] -->|"MediaPipe BlazeFace"| B["Face Cutting (resize 256x256)"]
    B --> C["Cropped Faces (faces/)"]
    C -->|"FaceEmbeddingNet"| D["Feature Extraction"]
    D --> E["Embeddings (.npy files)"]

    E --> F{"Analysis Mode"}
    F -->|"Binary Classification"| G["Target Identification (SVM / Cosine Sim)"]
    F -->|"Clustering"| H["Unsupervised Grouping (PCA + HDBSCAN)"]

    %% =========================
    %% 2) Model Architecture
    %% =========================
    D -.->|"uses"| I0

    subgraph ARCH["Model Architecture: FaceEmbeddingNet / FrozenFeatureNet"]
      direction LR

      subgraph FE["Feature Extractor (Backbone CNN)"]
        direction LR
        I0["Input Image (3x112x112)"] --> L1["Conv Block 1 (32 filters)"]
        L1 --> L2["Conv Block 2 (64 filters)"]
        L2 --> L3["Conv Block 3 (128 filters)"]
        L3 --> P0["Adaptive AvgPool"]
      end

      subgraph EH["Embedding Head"]
        direction LR
        P0 --> FL["Flatten"]
        FL --> FC["Linear (Embedding Dim)"]
        FC --> N0["L2 Normalization"]
      end

      subgraph CH["Classifier Head (Training Only)"]
        direction LR
        N0 --> LOG["Linear Classifier (Logits)"]
      end

      N0 -.->|"Inference"| VEC["Feature Vector (Embedding)"]
    end

    %% =========================
    %% 3) Notes INSIDE the diagram (so no extra paste)
    %% =========================
    subgraph NOTES["Diagram Notes (구조 요약)"]
      direction TB
      N1["Feature Extractor: 얼굴 로컬 패턴(눈/코/입/윤곽)을 단계적으로 추출해 고수준 특징으로 압축"]
      N2["Embedding Head: Pooling-Linear로 고정 길이 벡터 생성 + L2 Normalize로 cosine 비교 안정화"]
      N3["Classifier Head: 라벨(타겟/비타겟) 있을 때만 logits 학습. 추론은 embedding만 뽑아도 됨"]
      N4["Analysis: Target=Cosine 템플릿매칭 또는 SVM 분리 / Clustering=PCA로 노이즈 완화 후 HDBSCAN 군집"]
    end

    D -.-> N1
    FC -.-> N2
    LOG -.-> N3
    F -.-> N4


