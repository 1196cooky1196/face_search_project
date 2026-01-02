# 🧑‍💻 Face Search & Clustering Project

이 프로젝트는 **딥러닝 기반의 얼굴 인식 및 분석 파이프라인**입니다.
대량의 인물 이미지에서 얼굴을 검출하여 전처리하고, 특징 벡터(Embedding)를 추출하여 **특정 인물 식별(Binary Classification)** 및 **자동 군집화(Clustering)**를 수행합니다.

## 📋 Project Overview

* **Goal**: 다수의 인물 사진 중 특정 인물(Target)을 찾아내거나, 라벨링 없이 유사한 얼굴끼리 그룹화.
* **Core Tech**: `MediaPipe` (Detection), `PyTorch` (Embedding), `Scikit-learn` (Classification), `HDBSCAN` (Clustering).

---

## ⚙️ Execution Pipeline & Model Architecture

전체 시스템은 **데이터 수집 → 얼굴 추출 → 특징 추출 → 분석(분류/군집)**의 4단계로 구성됩니다.  
또한 아래 다이어그램에는 **Feature Extraction 단계에서 사용하는 모델 구조**까지 함께 포함했습니다.

```mermaid
graph TD
    %% =========================
    %% 1) Execution Pipeline
    %% =========================
    A["Raw Images<br/>(famous_picture/)"] -->|"MediaPipe BlazeFace"| B["Face Cutting<br/>resize 256x256"]
    B --> C["Cropped Faces<br/>(faces/)"]
    C -->|"FaceEmbeddingNet"| D["Feature Extraction"]
    D --> E["Embeddings<br/>(.npy files)"]

    E --> F{"Analysis Mode"}
    F -->|"Binary Classification"| G["Target Identification<br/>(SVM / Cosine Sim)"]
    F -->|"Clustering"| H["Unsupervised Grouping<br/>(PCA + HDBSCAN)"]

    %% =========================
    %% 2) Model Architecture (same block)
    %% =========================
    D -.->|"uses"| Input

    subgraph ARCH["Model Architecture: FaceEmbeddingNet / FrozenFeatureNet"]
      direction LR

      subgraph FE["Feature Extractor"]
        direction LR
        Input["Input Image<br/>(3x112x112)"] --> L1["Conv Block 1<br/>32 filters"]
        L1 --> L2["Conv Block 2<br/>64 filters"]
        L2 --> L3["Conv Block 3<br/>128 filters"]
        L3 --> Pool["Adaptive AvgPool"]
      end

      subgraph EH["Embedding Head"]
        direction LR
        Pool --> Flat["Flatten"]
        Flat --> Dense1["Linear Layer<br/>(Embedding Dim)"]
        Dense1 --> Norm["L2 Normalization"]
      end

      subgraph CH["Classifier Head"]
        direction LR
        Norm --> Out["Linear Classifier<br/>(Logits)"]
      end

      Norm -.->|"Inference"| Vec["Feature Vector"]
    end

🧠 Diagram Notes (구조 요약)

Feature Extractor(Backbone CNN): 얼굴 이미지에서 로컬 패턴(눈/코/입/윤곽 등)을 단계적으로 추출해 고수준 특징으로 압축합니다.

Embedding Head: Backbone 출력(feature map)을 Pooling → Flatten → Linear로 고정 길이 벡터로 만들고, 마지막에 L2 Normalization을 적용해
벡터 크기(스케일) 영향 없이 cosine similarity 기반 비교가 안정적으로 동작하게 합니다.

Classifier Head(학습용): 라벨(타겟/비타겟)이 있을 때만 사용하며, 임베딩 위에 선형 분류기를 붙여 결정 경계를 학습합니다.
추론/검색 단계에서는 보통 Classifier 없이 Embedding만 뽑아 코사인 유사도 또는 SVM 등으로 판별합니다.

Analysis Mode

Target Identification: (1) 코사인 유사도(템플릿 매칭) 또는 (2) SVM(임베딩 공간에서 초평면 분리)로 타겟 여부를 결정합니다.

Clustering: PCA로 차원을 줄여 노이즈를 완화한 뒤, HDBSCAN으로 클러스터 수를 자동 추정하며 유사 인물군을 묶습니다.
