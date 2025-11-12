import time
start_time = time.perf_counter() 

from pathlib import Path
from cutting import FaceDetector, FaceCropPipeline
from extractor import FeatureExtractor
from binary_classifier import load_embeddings, train_svm_with_pca
from cluster import FaceClusterer
import time


def run_face_cutting():
    print("\n[1] 얼굴 자르기 시작...")
    ROOT = Path(__file__).resolve().parent
    detector = FaceDetector(ROOT / "models" / "blaze_face_short_range.tflite")
    pipeline = FaceCropPipeline(
        src_dir = ROOT / "famous_picture",
        out_dir = ROOT / "faces",
        detector= detector,
        margin  = 0.25,
    )
    pipeline.run()


def run_feature_extraction():
    print("\n[2] 특징 벡터 추출 중...")
    extractor = FeatureExtractor(
        model_ckpt='tripletnet.pth',
        embedding_dim=512
    )
    extractor.extract_all(
        img_dir='faces',
        output_dir='features',
        batch_size=32
    )


def run_binary_classification():
    print("\n[3] 안유진 여부 이진 분류 중...")
    X, y = load_embeddings("features")
    if len(set(y)) < 2:
        print("⚠️ 라벨이 하나뿐이라 분류 불가. 데이터를 확인하세요.")
    else:
        train_svm_with_pca(X, y, n_components=64)


def run_cluster_search():
    print("\n[4] 군집화 및 쿼리 탐색 중...")
    clusterer = FaceClusterer("features")
    clusterer.fit_gmm(max_components=10)
    clusterer.predict_cluster("query/query1.npy")


if __name__ == '__main__':
    start = time.time()
    run_face_cutting()
    run_feature_extraction()
    run_binary_classification()
    run_cluster_search()
    print(f"\n총 소요 시간: {time.time() - start:.2f}초")


end_time = time.perf_counter()
print(f"Total time: {end_time - start_time:.2f} s")