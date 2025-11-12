
# train.py

# train.py ────────────────────────────────────────────────────────────────
"""
얼굴 이진 분류 학습 스크립트
- 모델  : FrozenFeatureNet (모델 참고: model.py)
- 손실  : BCEWithLogitsLoss
- 증강  : RandomHorizontalFlip, RandomRotation, ColorJitter, RandomResizedCrop
- 스케줄러 : ReduceLROnPlateau
- EarlyStopping : Val Loss 기준
"""

import time
start_time = time.perf_counter()

import os
from glob import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split

from model import FrozenFeatureNet   # 같은 디렉토리에 model.py

# ────────────────────────────────────────────────────────────────────────
# 1. 데이터셋 클래스
class FaceDataset(Dataset):
    """paths, labels 리스트를 받아 (Tensor, label) 반환"""

    def __init__(self, paths, labels, train: bool = True):
        self.paths = paths
        self.labels = labels

        # 공통 전처리
        base_tf = [
            T.Resize((80, 80)),            # 먼저 살짝 크게 맞춘 뒤
            T.CenterCrop((64, 64)),        # 64×64 고정
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ]

        # Train 전용 증강
        if train:
            aug = [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=8),
                T.ColorJitter(0.2, 0.2, 0.2, 0.05),
                T.RandomResizedCrop(64, scale=(0.9, 1.1),
                                    ratio=(0.9, 1.1))
            ]
            base_tf = aug + base_tf

        self.transform = T.Compose(base_tf)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transform(img)
        label = torch.tensor([self.labels[idx]], dtype=torch.float32)
        return img, label


# ────────────────────────────────────────────────────────────────────────
# 2. Trainer 클래스
class Trainer:
    def __init__(
        self,
        twice_dir: str = "twice_faces",      # 1 클래스(나연)
        stuff_dir: str = "stuff_faces",      # 0 클래스(기타)
        batch_size: int = 32,
        epochs: int = 200,
        patience: int = 10,
        save_path: str = "model_best.pth"
    ):
        # ─── 초기화
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FrozenFeatureNet(embedding_dim=128).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=1e-3
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )

        self.epochs = epochs
        self.patience = patience
        self.save_path = save_path

        # ─── 데이터 로드
        twice_paths = sorted(glob(os.path.join(twice_dir, "*.jpg")))
        stuff_paths = sorted(glob(os.path.join(stuff_dir, "*.jpg")))

        all_paths = twice_paths + stuff_paths
        all_labels = [1] * len(twice_paths) + [0] * len(stuff_paths)

        train_p, val_p, train_l, val_l = train_test_split(
            all_paths, all_labels,
            test_size=0.3, stratify=all_labels, random_state=42
        )

        self.train_loader = DataLoader(
            FaceDataset(train_p, train_l, train=True),
            batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

        self.val_loader = DataLoader(
            FaceDataset(val_p, val_l, train=False),
            batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # ───────────────────────────────────────────
    def evaluate(self):
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0

        with torch.no_grad():
            for x, y in self.val_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                correct += (preds == y).sum().item()
                total += y.size(0)
                loss_sum += loss.item() * y.size(0)

        val_acc = correct / total * 100
        val_loss = loss_sum / total
        return val_acc, val_loss

    # ───────────────────────────────────────────
    def train(self):
        print(f"🚀 Start Training | Epochs={self.epochs} | Patience={self.patience}")
        best_val_loss = float('inf')
        patience_cnt = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss, train_correct, train_total = 0.0, 0, 0

            for x, y in self.train_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                loss = self.criterion(logits, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                train_correct += (preds == y).sum().item()
                train_loss += loss.item() * y.size(0)
                train_total += y.size(0)

            train_acc = train_correct / train_total * 100
            train_loss /= train_total

            val_acc, val_loss = self.evaluate()
            self.scheduler.step(val_loss)

            print(f"Epoch {epoch:03}/{self.epochs} | "
                  f"Train Loss {train_loss:.4f}, Acc {train_acc:.2f}% || "
                  f"Val Loss {val_loss:.4f}, Acc {val_acc:.2f}%")

            # EarlyStopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_cnt = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"  ✅ Saved new best model (Val Loss {best_val_loss:.4f})")
            else:
                patience_cnt += 1
                print(f"  ⏳ No improvement ({patience_cnt}/{self.patience})")

            if patience_cnt >= self.patience:
                print("🛑 Early stopping triggered.")
                break


# ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    trainer = Trainer(
        twice_dir="twice_faces",   # ← 경로 확인
        stuff_dir="stuff_faces",   # ← 경로 확인
        batch_size=32,
        epochs=200,
        patience=10,
        save_path="frozen_feature_best.pth"
    )
    trainer.train()

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} s")


'''
# train.py (BCE + TripletLoss 동시 학습 + EarlyStopping)

# train.py (디버깅용: BCE + TripletLoss + 출력값/손실값 출력 포함)

import time
start_time = time.perf_counter()

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from glob import glob
from model import FrozenFeatureNet

class NayeonTripletDataset(Dataset):
    def __init__(self, nayeon_dir, others_dir):
        self.nayeon_paths = sorted(glob(os.path.join(nayeon_dir, "*.jpg")))
        self.others_paths = sorted(glob(os.path.join(others_dir, "*.jpg")))
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return min(len(self.nayeon_paths), len(self.others_paths))

    def __getitem__(self, idx):
        anchor_path = self.nayeon_paths[idx % len(self.nayeon_paths)]
        positive_path = self.nayeon_paths[(idx + 1) % len(self.nayeon_paths)]
        negative_path = self.others_paths[idx % len(self.others_paths)]

        a = self.transform(Image.open(anchor_path).convert("RGB"))
        p = self.transform(Image.open(positive_path).convert("RGB"))
        n = self.transform(Image.open(negative_path).convert("RGB"))

        label = torch.tensor(1.0)

        return a, p, n, a, label


class Trainer:
    def __init__(self, nayeon_dir="twice_faces", others_dir="stuff_faces",
                 batch_size=16, epochs=100, alpha=0.5, patience=10, save_path="model.pth"):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🖥️ 사용 디바이스: {self.device}")

        self.model = FrozenFeatureNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.bce = nn.BCEWithLogitsLoss()
        self.triplet = nn.TripletMarginLoss(margin=1.0, p=2)
        self.alpha = alpha
        self.epochs = epochs
        self.patience = patience
        self.save_path = save_path

        dataset = NayeonTripletDataset(nayeon_dir, others_dir)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train(self):
        print("\n🚀 학습 시작 (BCE + TripletLoss 조합 + EarlyStopping + 디버깅 출력)")
        self.model.train()

        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, self.epochs + 1):
            total_loss, bce_loss_sum, triplet_loss_sum = 0, 0, 0
            for i, (a, p, n, x_cls, y_cls) in enumerate(self.loader):
                a, p, n = a.to(self.device), p.to(self.device), n.to(self.device)
                x_cls = x_cls.to(self.device)
                y_cls = y_cls.to(self.device).view(-1, 1)

                out_cls = self.model(x_cls)  # [B, 1]
                prob = torch.sigmoid(out_cls)
                if epoch == 1 and i == 0:
                    print("📊 sigmoid 출력 예시:", prob[:5].view(-1).tolist())

                a_emb, p_emb, n_emb = self.model.forward_triplet(a, p, n)

                loss_bce = self.bce(out_cls, y_cls)
                if epoch == 1 and i == 0:
                    print("🧪 BCE Loss 예시:", loss_bce.item())

                loss_triplet = self.triplet(a_emb, p_emb, n_emb)
                loss = loss_bce + self.alpha * loss_triplet

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                bce_loss_sum += loss_bce.item()
                triplet_loss_sum += loss_triplet.item()

            print(f"Epoch {epoch}/{self.epochs} | Total: {total_loss:.4f} | BCE: {bce_loss_sum:.4f} | Triplet: {triplet_loss_sum:.4f}")

            if total_loss < best_loss:
                best_loss = total_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"✅ 모델 저장됨 (Best Loss: {best_loss:.4f})")
            else:
                patience_counter += 1
                print(f"⏳ 개선 없음 ({patience_counter}/{self.patience})")

            if patience_counter >= self.patience:
                print("🛑 Early stopping triggered.")
                break

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()

end_time = time.perf_counter()
print(f"Total time: {end_time - start_time:.2f} s")
'''



