
# model.py

"""
model.py
──────────────────────────────────────────────────────────────────────────────
FrozenFeatureNet: 고정된 CNN 인코더 + 학습 가능한 임베딩 + 이진 분류 헤드.

- 출력은 **logit**(시그모이드 미적용) → 학습 시 `nn.BCEWithLogitsLoss` 사용
- 추론 시 `torch.sigmoid(output)` 로 0~1 확률 변환
"""

import time
start_time = time.perf_counter()

import torch
import torch.nn as nn


class FrozenFeatureNet(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()

        # 🔒 1. 고정(Frozen) CNN 특징 추출기
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # → [B, 32, 64, 64]
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), # → [B, 64, 32, 32]
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))                                 # → [B, 128, 1, 1]
        )
        for p in self.encoder.parameters():
            p.requires_grad = False  # 백본은 업데이트 X

        # 2. 학습 가능한 임베딩 변환
        self.embedding = nn.Sequential(
            nn.Flatten(),                        # → [B, 128]
            nn.Linear(128, embedding_dim), nn.ReLU()
        )

        # 3. 이진 분류 헤드 (logit 1개)
        self.classifier = nn.Linear(embedding_dim, 1)

    # ──────────────────────────────────────────────────────────────────────
    # 기본 forward: logit 반환 ─ 학습·추론 공용
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.embedding(x)
        return self.classifier(x)               # [B, 1] logit

    # 임베딩만 뽑고 싶을 때
    def forward_embed(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return self.embedding(x)                # [B, embedding_dim]


# 빠른 동작 확인용
if __name__ == "__main__":
    model = FrozenFeatureNet()
    dummy = torch.randn(4, 3, 128, 128)        # 4장의 128×128 RGB 이미지
    logits = model(dummy)
    print("Output shape:", logits.shape)        # [4, 1]
    trainable = sum(p.numel() for p in model.parameters()
                    if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    end_time = time.perf_counter()
    print(f"Loaded in {end_time - start_time:.2f} s")



'''
# model.py (ResNet18 기반 feature extractor + 개선된 classifier + triplet 지원 + 강제 초기화)

import time
start_time = time.perf_counter()

import torch
import torch.nn as nn
import torchvision.models as models

class FrozenFeatureNet(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()

        # ✅ ResNet18 사전학습 모델 불러오기
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # [B, 512, 1, 1]

        # 🔓 encoder 학습 허용 (fine-tuning)
        for param in self.encoder.parameters():
            param.requires_grad = True

        # ✅ embedding projection
        self.embedding = nn.Sequential(
            nn.Flatten(),                    # [B, 512]
            nn.Linear(512, embedding_dim),   # [B, embedding_dim]
            nn.ReLU()
        )

        # ✅ 개선된 이진 분류 헤드
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # ✅ classifier weight 강제 초기화
        self.classifier.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=0.1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        x = self.embedding(x)
        return self.classifier(x)  # [B, 1]

    def forward_embed(self, x):
        x = self.encoder(x)
        return self.embedding(x)  # [B, embedding_dim]

    def forward_triplet(self, anchor, positive, negative):
        a = self.forward_embed(anchor)
        p = self.forward_embed(positive)
        n = self.forward_embed(negative)
        return a, p, n

end_time = time.perf_counter()
print(f"Total time: {end_time - start_time:.2f} s")
'''
