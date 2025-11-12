# cluster.py

import numpy as np
import cv2
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import hdbscan

from extractor import FaceEmbedder


class EmbeddingExtractor:
    def __init__(self, model_path: str, face_dir: str):
        self.face_dir = Path(face_dir)
        self.embedder = FaceEmbedder(model_path)
        self.paths = sorted(self.face_dir.glob("*.jpg"))

    def extract(self):
        features, names, imgs = [], [], []
        for path in self.paths:
            try:
                vec = self.embedder.extract(str(path))
                img = cv2.imread(str(path))
                features.append(vec)
                names.append(path.name)
                imgs.append(img)
            except Exception as e:
                print(f"❌ {path.name} 처리 실패: {e}")
        print(f"✅ 임베딩 완료: {len(features)}개")
        return np.stack(features), names, imgs


class FaceClustering:
    def __init__(self, embeddings, names, pca_dim=32,
                 min_cluster_size=5, min_samples=3, epsilon=0.1):
        self.original_embeddings = embeddings
        self.names = names
        self.pca_dim = pca_dim
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.epsilon = epsilon
        self.labels = None
        self.num_clusters = 0

    def run_hdbscan(self):
        print("📉 PCA 차원 축소 중...")
        pca = PCA(n_components=self.pca_dim)
        reduced = pca.fit_transform(self.original_embeddings.astype(np.float64))

        print("🔍 HDBSCAN 클러스터링 중...")
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_epsilon=self.epsilon
        )
        raw_labels = hdb.fit_predict(reduced)

        # -1 (outlier) → 고유 클러스터로 재배정
        labels = raw_labels.copy()
        next_label = labels.max() + 1 if labels.max() >= 0 else 0
        for i, label in enumerate(labels):
            if label == -1:
                labels[i] = next_label
                next_label += 1

        self.labels = labels
        self.num_clusters = len(set(labels))
        self.reduced_embeddings = reduced
        print(f"\n✅ 최종 클러스터 수: {self.num_clusters}")

    def get_cluster_dict(self):
        cluster_dict = {}
        for label, name in zip(self.labels, self.names):
            cluster_dict.setdefault(label, []).append(name)
        return cluster_dict

    def print_cluster_info(self):
        cluster_dict = self.get_cluster_dict()
        print(f"\n📂 클러스터 목록:")
        for k in sorted(cluster_dict.keys()):
            print(f"클러스터 {k}: {', '.join(cluster_dict[k])}")

    def print_cluster_centers(self):
        unique_labels = sorted(set(self.labels))
        for k in unique_labels:
            indices = [i for i, label in enumerate(self.labels) if label == k]
            cluster_vecs = self.reduced_embeddings[indices]
            center = cluster_vecs.mean(axis=0)
            dists = np.linalg.norm(cluster_vecs - center, axis=1)
            best_idx = indices[np.argmin(dists)]
            print(f"📌 클러스터 {k} 대표 이미지: {self.names[best_idx]}")


class ClusterVisualizer:
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def plot(self):
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(self.embeddings)
        plt.figure(figsize=(8, 6))
        for i, label in enumerate(self.labels):
            color = f"C{label % 10}"
            plt.scatter(reduced[i, 0], reduced[i, 1], c=color, alpha=0.7)
        plt.title("HDBSCAN Clustering (PCA 2D)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    model_path = "embedder.pt"
    face_dir = "stuff_faces"

    extractor = EmbeddingExtractor(model_path, face_dir)
    embeddings, names, _ = extractor.extract()

    clustering = FaceClustering(
        embeddings, names,
        pca_dim=32,                  # PCA 차원 축소
        min_cluster_size=5,          # 최소 클러스터 크기
        min_samples=1,               # 핵심 샘플 수
        epsilon=0.3                  # 병합 허용 범위
    )
    clustering.run_hdbscan()
    clustering.print_cluster_info()
    clustering.print_cluster_centers()

    visualizer = ClusterVisualizer(clustering.reduced_embeddings, clustering.labels)
    visualizer.plot()
