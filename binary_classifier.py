'''
import time
start_time = time.perf_counter()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import BCEWithLogitsLoss

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

import numpy as np
from pathlib import Path
from PIL import Image

from extractor import FaceEmbeddingNet

def get_transform(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor()
        ])

def visualize_svm_decision(X, y, clf, title="SVM Decision Boundary"):
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    clf.fit(X_reduced, y)

    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    for label in np.unique(y):
        idx = y == label
        plt.scatter(X_reduced[idx, 0], X_reduced[idx, 1], label=f"Class {label}", alpha=0.6)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)

class BinaryClassificationTrainer:
    def __init__(self, train_dir, val_dir, test_dir, batch_size=32, lr=0.001, patience=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = FaceEmbeddingNet().to(self.device)
        self.classifier = BinaryClassifier().to(self.device)

        self.criterion = BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            list(self.embedding_model.parameters()) + list(self.classifier.parameters()),
            lr=lr, weight_decay=1e-4
        )

        self.train_loader = DataLoader(datasets.ImageFolder(train_dir, transform=get_transform(True)), batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(datasets.ImageFolder(val_dir, transform=get_transform(False)), batch_size=batch_size)
        self.test_loader = DataLoader(datasets.ImageFolder(test_dir, transform=get_transform(False)), batch_size=batch_size)

        self.best_val_acc = 0
        self.early_stop_counter = 0
        self.patience = patience

    def train(self, epochs):
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        for epoch in range(epochs):
            self.embedding_model.train()
            self.classifier.train()
            total_loss = 0
            for imgs, labels in self.train_loader:
                imgs = imgs.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                embed = self.embedding_model(imgs)
                logits = self.classifier(embed)
                loss = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(self.train_loader):.4f}")

            val_acc = self.validate()
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    print("[Early Stopping] Validation accuracy not improving.")
                    break

    def validate(self):
        self.embedding_model.eval()
        self.classifier.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                embed = self.embedding_model(imgs)
                logits = self.classifier(embed)
                preds = torch.sigmoid(logits) > 0.5
                correct += (preds.squeeze().long() == labels).sum().item()
                total += labels.size(0)
        acc = correct / total * 100
        print(f"[Val Accuracy] {acc:.2f}%")
        return acc

    def test(self):
        self.embedding_model.eval()
        self.classifier.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                embed = self.embedding_model(imgs)
                logits = self.classifier(embed)
                preds = torch.sigmoid(logits) > 0.5
                correct += (preds.squeeze().long() == labels).sum().item()
                total += labels.size(0)
        acc = correct / total * 100
        print(f"[CNN + Linear] Test Accuracy: {acc:.2f}%")
        return acc

    def save_embedder(self, path="embedder.pt"):
        torch.save(self.embedding_model.state_dict(), path)

class SklearnClassifierRunner:
    def __init__(self, model_path, train_dir, val_dir, test_dir, pca_dim=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = FaceEmbeddingNet().to(self.device)
        self.embedder.load_state_dict(torch.load(model_path, map_location=self.device))
        self.embedder.eval()
        self.pca = PCA(n_components=pca_dim)

        self.train_loader = DataLoader(datasets.ImageFolder(train_dir, transform=get_transform(False)), batch_size=32)
        self.val_loader = DataLoader(datasets.ImageFolder(val_dir, transform=get_transform(False)), batch_size=32)
        self.test_loader = DataLoader(datasets.ImageFolder(test_dir, transform=get_transform(False)), batch_size=32)

    def _extract(self, loader):
        X, y = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                feats = self.embedder(imgs).cpu().numpy()
                X.append(feats)
                y.append(labels.numpy())
        return np.vstack(X), np.hstack(y)

    def run(self, model_name="logistic"):
        X_train, y_train = self._extract(self.train_loader)
        X_val, y_val = self._extract(self.val_loader)
        X_test, y_test = self._extract(self.test_loader)

        self.pca.fit(np.vstack([X_train, X_val]))
        X_train = self.pca.transform(X_train)
        X_val = self.pca.transform(X_val)
        X_test = self.pca.transform(X_test)

        best_val_acc = 0
        best_clf = None
        best_name = ""

        if model_name == "logistic":
            for C in [0.01, 0.1, 1, 10, 100]:
                clf = LogisticRegression(C=C, max_iter=1000)
                clf.fit(X_train, y_train)
                val_acc = accuracy_score(y_val, clf.predict(X_val)) * 100
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_clf = clf
                    best_name = f"LOGISTIC (C={C})"
        elif model_name == "svm":
            for C in [0.01, 0.1, 1, 10]:
                clf = SVC(kernel="rbf", probability=True, C=C)
                clf.fit(X_train, y_train)
                val_acc = accuracy_score(y_val, clf.predict(X_val)) * 100
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_clf = clf
                    best_name = f"SVM (C={C})"
        else:
            raise ValueError("지원되지 않는 모델")

        test_acc = accuracy_score(y_test, best_clf.predict(X_test)) * 100
        print(f"[{best_name} + PCA] Val Accuracy: {best_val_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")

        if model_name == "svm":
            visualize_svm_decision(np.vstack([X_train, X_val]), np.hstack([y_train, y_val]), best_clf)

        return test_acc

class SklearnRunner:
    def __init__(self, model_path, train_dir, val_dir, test_dir, pca_dim=32):
        self.runner = SklearnClassifierRunner(model_path, train_dir, val_dir, test_dir, pca_dim)

    def run_all(self):
        acc_log = self.runner.run("logistic")
        acc_svm = self.runner.run("svm")
        return acc_log, acc_svm

if __name__ == "__main__":
    TRAIN = "training_data"
    VAL = "val_data"
    TEST = "test_data"

    trainer = BinaryClassificationTrainer(TRAIN, VAL, TEST, patience=5)
    trainer.train(epochs=45)
    acc1 = trainer.test()
    trainer.save_embedder("embedder.pt")

    acc2, acc3 = SklearnRunner("embedder.pt", TRAIN, VAL, TEST, pca_dim=32).run_all()

    print("\n=== Summary ===")
    print(f"CNN + Linear        : {acc1:.2f}%")
    print(f"Logistic + PCA      : {acc2:.2f}%")
    print(f"SVM (RBF) + PCA     : {acc3:.2f}%")
    print(f"Total time: {time.perf_counter() - start_time:.2f}s")
    '''

# binary_classifier.py

import time
start_time = time.perf_counter()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.nn import BCEWithLogitsLoss

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from pathlib import Path
from PIL import Image

from extractor import FaceEmbeddingNet


def get_transform(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])
    else:
        return transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor()
        ])


class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


class BinaryClassificationTrainer:
    def __init__(self, train_dir, val_dir, test_dir, batch_size=32, lr=0.001, patience=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = FaceEmbeddingNet().to(self.device)
        self.classifier = BinaryClassifier().to(self.device)

        self.criterion = BCEWithLogitsLoss()
        self.optimizer = torch.optim.AdamW(
            list(self.embedding_model.parameters()) + list(self.classifier.parameters()),
            lr=lr, weight_decay=1e-4
        )

        self.train_loader = DataLoader(datasets.ImageFolder(train_dir, transform=get_transform(True)), batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(datasets.ImageFolder(val_dir, transform=get_transform(False)), batch_size=batch_size)
        self.test_loader = DataLoader(datasets.ImageFolder(test_dir, transform=get_transform(False)), batch_size=batch_size)

        self.best_val_acc = 0
        self.early_stop_counter = 0
        self.patience = patience

    def train(self, epochs):
        scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        for epoch in range(epochs):
            self.embedding_model.train()
            self.classifier.train()
            total_loss = 0
            for imgs, labels in self.train_loader:
                imgs = imgs.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                embed = self.embedding_model(imgs)
                logits = self.classifier(embed)
                loss = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            scheduler.step()
            print(f"[Epoch {epoch+1}] Train Loss: {total_loss / len(self.train_loader):.4f}")

            val_acc = self.validate()
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.early_stop_counter = 0
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    print("[Early Stopping] Validation accuracy not improving.")
                    break

    def validate(self):
        self.embedding_model.eval()
        self.classifier.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in self.val_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                embed = self.embedding_model(imgs)
                logits = self.classifier(embed)
                preds = torch.sigmoid(logits) > 0.5
                correct += (preds.squeeze().long() == labels).sum().item()
                total += labels.size(0)
        acc = correct / total * 100
        print(f"[Val Accuracy] {acc:.2f}%")
        return acc

    def test(self):
        self.embedding_model.eval()
        self.classifier.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                embed = self.embedding_model(imgs)
                logits = self.classifier(embed)
                preds = torch.sigmoid(logits) > 0.5
                correct += (preds.squeeze().long() == labels).sum().item()
                total += labels.size(0)
        acc = correct / total * 100
        print(f"[CNN + Linear] Test Accuracy: {acc:.2f}%")
        return acc

    def save_embedder(self, path="embedder.pt"):
        torch.save(self.embedding_model.state_dict(), path)


class SklearnClassifierRunner:
    def __init__(self, model_path, train_dir, val_dir, test_dir, pca_dim=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedder = FaceEmbeddingNet().to(self.device)
        self.embedder.load_state_dict(torch.load(model_path, map_location=self.device))
        self.embedder.eval()
        self.pca = PCA(n_components=pca_dim)

        self.train_loader = DataLoader(datasets.ImageFolder(train_dir, transform=get_transform(False)), batch_size=32)
        self.val_loader = DataLoader(datasets.ImageFolder(val_dir, transform=get_transform(False)), batch_size=32)
        self.test_loader = DataLoader(datasets.ImageFolder(test_dir, transform=get_transform(False)), batch_size=32)

    def _extract(self, loader):
        X, y = [], []
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                feats = self.embedder(imgs).cpu().numpy()
                X.append(feats)
                y.append(labels.numpy())
        return np.vstack(X), np.hstack(y)

    def run(self, model_name="logistic"):
        X_train, y_train = self._extract(self.train_loader)
        X_val, y_val = self._extract(self.val_loader)
        X_test, y_test = self._extract(self.test_loader)

        self.pca.fit(np.vstack([X_train, X_val]))
        X_train = self.pca.transform(X_train)
        X_val = self.pca.transform(X_val)
        X_test = self.pca.transform(X_test)

        best_val_acc = 0
        best_clf = None
        best_name = ""

        if model_name == "logistic":
            for C in [0.01, 0.1, 1, 10, 100]:
                clf = LogisticRegression(C=C, max_iter=1000)
                clf.fit(X_train, y_train)
                val_acc = accuracy_score(y_val, clf.predict(X_val)) * 100
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_clf = clf
                    best_name = f"LOGISTIC (C={C})"
        elif model_name == "svm":
            for C in [0.01, 0.1, 1, 10]:
                clf = SVC(kernel="rbf", probability=True, C=C)
                clf.fit(X_train, y_train)
                val_acc = accuracy_score(y_val, clf.predict(X_val)) * 100
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_clf = clf
                    best_name = f"SVM (C={C})"
        else:
            raise ValueError("지원되지 않는 모델")

        test_acc = accuracy_score(y_test, best_clf.predict(X_test)) * 100
        print(f"[{best_name} + PCA] Val Accuracy: {best_val_acc:.2f}%, Test Accuracy: {test_acc:.2f}%")
        return test_acc


class SklearnRunner:
    def __init__(self, model_path, train_dir, val_dir, test_dir, pca_dim=32):
        self.runner = SklearnClassifierRunner(model_path, train_dir, val_dir, test_dir, pca_dim)

    def run_all(self):
        acc_log = self.runner.run("logistic")
        acc_svm = self.runner.run("svm")
        return acc_log, acc_svm

# --- 핵심: DistanceBasedClassifier 복원 버전 ---

class DistanceBasedClassifier:
    def __init__(self, model, transform, device, threshold=0.99):
        self.model = model
        self.model.eval()
        self.transform = transform
        self.device = device
        self.threshold = threshold

    def _embed(self, img_path):
        image = Image.open(img_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vec = self.model(tensor).squeeze(0).cpu().numpy()
        return vec / np.linalg.norm(vec)

    def build_anchor(self, image_folder):
    # 🔧 오직 nayeon_train 폴더만 anchor 생성에 사용
        paths = sorted(Path(image_folder).joinpath("나연_train").glob("*.jpg"))
        vectors = [self._embed(p) for p in paths]
        anchor = np.mean(vectors, axis=0)
        return anchor / np.linalg.norm(anchor)


    def predict(self, anchor, img_path):
        vec = self._embed(img_path)
        sim = cosine_similarity([anchor], [vec])[0][0]
        return sim >= self.threshold, sim

    def test_on_dataset(self, anchor, test_folder, target_class="나연_vali"):
        correct = total = 0
        for path in Path(test_folder).glob("*/*.jpg"):
            label = 1 if path.parent.name == target_class else 0
            pred, score = self.predict(anchor, path)
            if pred == bool(label):
                correct += 1
            total += 1
            result = "✔" if pred == bool(label) else "✘"
            print(f"[{result}] {path.name} | GT: {label} | Score: {score:.4f}")
        acc = correct / total * 100
        return acc

class SklearnRunner:
    def __init__(self, model_path, train_dir, val_dir, test_dir, pca_dim=32):
        self.runner = SklearnClassifierRunner(model_path, train_dir, val_dir, test_dir, pca_dim=pca_dim)

    def run_all(self):
        acc_log = self.runner.run("logistic")
        acc_svm = self.runner.run("svm")
        return acc_log, acc_svm

class DistanceRunner:
    def __init__(self, model_path, train_dir, test_dir, threshold=0.76):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FaceEmbeddingNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.transform = get_transform(train=False)
        self.classifier = DistanceBasedClassifier(self.model, self.transform, self.device, threshold)
        self.anchor = self.classifier.build_anchor(train_dir)
        self.test_dir = test_dir

    def run(self):
        print("\n[Distance-Based Classification]")
        acc = self.classifier.test_on_dataset(self.anchor, self.test_dir)
        print(f"[Cosine Similarity] Test Accuracy: {acc:.2f}%")
        return acc

if __name__ == "__main__":
    TRAIN = "training_data"
    VAL = "val_data"
    TEST = "test_data"

    trainer = BinaryClassificationTrainer(TRAIN, VAL, TEST, patience=5)
    trainer.train(epochs=45)
    acc1 = trainer.test()
    trainer.save_embedder("embedder.pt")

    acc2, acc3 = SklearnRunner("embedder.pt", TRAIN, VAL, TEST, pca_dim=32).run_all()
    acc4 = DistanceRunner("embedder.pt", TRAIN, TEST, threshold=0.76).run()

    print("\n=== Summary ===")
    print(f"CNN + Linear        : {acc1:.2f}%")
    print(f"Logistic + PCA      : {acc2:.2f}%")
    print(f"SVM (RBF) + PCA     : {acc3:.2f}%")
    print(f"Cosine Similarity   : {acc4:.2f}%")
    print(f"Total time: {time.perf_counter() - start_time:.2f}s")



