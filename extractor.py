'''
#face_net 사용함
import time
stat_time = time.time()

import os
import glob
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

try:
    from facenet_pytorch import InceptionResnetV1
except ImportError as e:
    raise ImportError("facenet_pytorch 가 설치되어 있지 않습니다. `pip install facenet-pytorch` 로 설치 후 다시 실행하세요.") from e


class FaceDataset(torch.utils.data.Dataset):
    """faces 폴더에 있는 jpg/png 이미지를 로드하여 전처리한 뒤 반환합니다."""

    def __init__(self, faces_dir: str, transform: transforms.Compose):
        self.image_paths = sorted(
            glob.glob(os.path.join(faces_dir, "**", "*.jpg"), recursive=True)
            + glob.glob(os.path.join(faces_dir, "**", "*.png"), recursive=True)
        )
        if not self.image_paths:
            raise FileNotFoundError(f"{faces_dir} 안에 .jpg 또는 .png 파일이 존재하지 않습니다.")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), path


class FaceFeatureExtractor:
    """얼굴 임베딩을 추출하여 .npy 파일로 저장하는 클래스"""

    def __init__(
        self,
        faces_dir: str = "faces",
        output_dir: str = "features",
        batch_size: int = 32,
        num_workers: int = 0,
        device: str | torch.device | None = None,
    ):
        self.faces_dir = faces_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 디바이스 설정
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 모델 로드 (FaceNet, 512‑d 임베딩)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

        # 전처리 파이프라인
        self.transform = transforms.Compose([
            transforms.Resize((160, 160)),  # FaceNet 입력 사이즈
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # 데이터로더
        dataset = FaceDataset(self.faces_dir, self.transform)
        self.data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    @torch.no_grad()
    def extract(self) -> List[str]:
        """모든 얼굴 이미지에 대한 특징 벡터를 추출하고 저장한 후, 저장된 파일 경로 리스트를 반환합니다."""
        saved_paths: List[str] = []
        for imgs, paths in self.data_loader:
            imgs = imgs.to(self.device)
            embeddings = self.model(imgs)  # (B, 512)
            embeddings = embeddings.cpu().numpy()
            for emb, path in zip(embeddings, paths):
                filename = Path(path).stem + ".npy"
                save_path = os.path.join(self.output_dir, filename)
                np.save(save_path, emb)
                saved_paths.append(save_path)
        return saved_paths


if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Face feature extraction script")
    parser.add_argument("--faces_dir", type=str, default="faces", help="잘린 얼굴 이미지 폴더 경로")
    parser.add_argument("--output_dir", type=str, default="features", help="임베딩 저장 폴더 경로")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="cuda 또는 cpu 지정. 미지정 시 자동선택")
    args = parser.parse_args()

    extractor = FaceFeatureExtractor(
        faces_dir=args.faces_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )

    paths = extractor.extract()
    print(f"총 {len(paths)}개의 임베딩이 {args.output_dir}에 저장되었습니다.")


end_time = time.time()
print(f"total time is {end_time - stat_time}")
'''

'''
#TripletNet
import time
stat_time = time.time()

import os
import glob
import numpy as np
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# =============================
# 1. TripletNet 기반 특징 추출용 CNN
# =============================
class TripletNet(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512), nn.ReLU(),
            nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.head(x)
        return x


# =============================
# 2. faces 폴더 이미지 로딩
# =============================
class FaceDataset(Dataset):
    def __init__(self, img_dir: str, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, '*')))
        self.transform = transform or transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img_path, img


# =============================
# 3. 특징 추출기
# =============================
class FeatureExtractor:
    def __init__(self, model_ckpt: str, embedding_dim=512, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TripletNet(embedding_dim).to(self.device)
        # 주석 해제 후 학습된 모델 불러올 수 있음
        #self.model.load_state_dict(torch.load(model_ckpt, map_location=self.device))
        self.model.eval()

    def extract_all(self, img_dir: str, output_dir: str, batch_size=32):
        os.makedirs(output_dir, exist_ok=True)
        dataset = FaceDataset(img_dir)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        with torch.no_grad():
            for paths, imgs in loader:
                imgs = imgs.to(self.device)
                embeddings = self.model(imgs).cpu().numpy()

                for path, emb in zip(paths, embeddings):
                    name = Path(path).stem
                    np.save(os.path.join(output_dir, f"{name}.npy"), emb)
                    #print(f"{name}: [", ', '.join(f"{v:.4f}" for v in emb[:10]), "...", ', '.join(f"{v:.4f}" for v in emb[-10:]), "]")


# =============================
# 4. main 실행 예시
# =============================

if __name__ == '__main__':
    extractor = FeatureExtractor(
        model_ckpt='tripletnet.pth',
        embedding_dim=512
    )

    # 나연 (positive 클래스)
    extractor.extract_all(
        img_dir='twice_faces',
        output_dir='features_twice',
        batch_size=32
    )

    # 기타 인물들 (negative 클래스)
    extractor.extract_all(
        img_dir='stuff_faces',
        output_dir='features_stuff',
        batch_size=32
    )

end_time = time.time()
print(f"total time is {end_time - stat_time:.2f} s")
'''

#with 준혁이형 보강

# extractor.py

import time
start_time = time.perf_counter()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

class FaceEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim=128, dropout_prob=0.3):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Linear(256, embedding_dim)
        #self.fc = nn.Sequential( nn.Linear(256, 128),nn.ReLU())   



    def forward(self, x):
        x = self.conv_block(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x


class FaceEmbedder:
    def __init__(self, model_path: str = None, device=None, dropout_prob=0.3):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FaceEmbeddingNet(dropout_prob=dropout_prob).to(self.device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor()
        ])

    def extract(self, image_path: str) -> np.ndarray:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor).squeeze(0).cpu().numpy()
        return embedding
    
end_time = time.perf_counter()
print(f"Total time: {end_time - start_time:.2f} s")
