# 🧑‍💻 Face Search & Clustering Project

이 프로젝트는 **딥러닝 기반의 얼굴 인식 및 분석 파이프라인**입니다.
대량의 인물 이미지에서 얼굴을 검출하여 전처리하고, 특징 벡터(Embedding)를 추출하여 **특정 인물 식별(Binary Classification)** 및 **자동 군집화(Clustering)**를 수행합니다.

## 📋 Project Overview

* **Goal**: 다수의 인물 사진 중 특정 인물(Target)을 찾아내거나, 라벨링 없이 유사한 얼굴끼리 그룹화.
* **Core Tech**: `MediaPipe` (Detection), `PyTorch` (Embedding), `Scikit-learn` (Classification), `HDBSCAN` (Clustering).

---

## ⚙️ Execution Pipeline

전체 시스템은 **데이터 수집 → 얼굴 추출 → 특징 추출 → 분석(분류/군집)**의 4단계로 구성됩니다.

```mermaid
graph TD
    A[📂 Raw Images<br/>(famous_picture/)] -->|MediaPipe BlazeFace| B(✂️ Face Cutting<br/>resizing 256x256)
    B --> C[📂 Cropped Faces<br/>(faces/)]
    C -->|FaceEmbeddingNet<br/>CNN Encoder| D(💎 Feature Extraction)
    D --> E[📂 Embeddings<br/>.npy files]
    
    E --> F{Analysis Mode}
    F -->|Binary Classification| G[🎯 Target Identification<br/>(SVM / Cosine Sim)]
    F -->|Clustering| H[🧩 Unsupervised Grouping<br/>(PCA + HDBSCAN)]

🧠 Model Architecture
얼굴의 특징을 추출하는 모델(FaceEmbeddingNet / FrozenFeatureNet)의 구조입니다. Backbone CNN을 통해 이미지 특징을 압축하고, Embedding Layer를 통해 고차원 벡터로 변환합니다.

graph LR
    subgraph Feature Extractor
    Input[Input Image<br/>(3x112x112)] --> L1[Conv Block 1<br/>32 filters]
    L1 --> L2[Conv Block 2<br/>64 filters]
    L2 --> L3[Conv Block 3<br/>128 filters]
    L3 --> Pool[Adaptive AvgPool]
    end

    subgraph Embedding Head
    Pool --> Flat[Flatten]
    Flat --> Dense1[Linear Layer<br/>(Embedding Dim)]
    Dense1 --> Norm[L2 Normalization]
    end

    subgraph Classifier Head
    Norm --> Out[Linear Classifier<br/>(Logits)]
    end

    Feature Extractor --> Embedding Head
    Embedding Head -.->|Inference| Output(Feature Vector)
    Embedding Head -->|Training| Classifier Head
