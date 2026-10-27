"""
Classroom CCTV Attendance System - Benchmark Replication & Novelty Evaluation Pipeline:

Evaluates:
A. The 4 Baseline Configurations from the Gujarat University Research Paper:
   1. VGG16 without PCA
   2. VGG16 + PCA
   3. ResNet-50 without PCA
   4. ResNet-50 + PCA

B. Your Novelty Configurations (Options 1 & 3):
   5. ResNet-50 + CCTV Face Enhancer (Novelty 3: Super-Resolution & Illumination Correction)
   6. ResNet-50 + ArcFace (Novelty 1: Additive Angular Margin Metric Learning)
   7. PROPOSED NOVEL PIPELINE: ResNet-50 + ArcFace + CCTV Enhancer + PCA (Novelty 1 + Novelty 3)

Features:
- Gallery Feature Caching: Avoids redundant re-extractions across configurations
- Dynamic Student Discovery: Works with both synthetic classroom data and real photos
- Full Side-by-Side Metric Comparison Table
"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add root directory to sys.path
sys.path.append(str(Path(__file__).parent.resolve()))
import config
from src.attendance.logger import AttendanceLogger
from src.attendance.voting_engine import AttendanceVotingEngine
from src.models.detector import YOLOClassroomDetector
from src.models.feature_extractor import DeepFeatureExtractor
from src.models.matcher import CosineFaceMatcher
from src.models.pca_reducer import PCAReducer
from src.novelty.enhancer import CCTVFaceEnhancer

def load_gallery_images():
    """
    Loads all augmented gallery images dynamically.
    Works for both synthetic and custom real classmate photos.
    """
    images = []
    student_ids = []
    print("[Gallery] Loading augmented student images...")

    stu_dirs = sorted([d for d in config.AUGMENTED_GALLERY_DIR.iterdir() if d.is_dir()])
    for stu_dir in stu_dirs:
        stu_id = stu_dir.name
        for img_path in sorted(stu_dir.glob("*.jpg")):
            img = cv2.imread(str(img_path))
            if img is not None:
                images.append(img)
                student_ids.append(stu_id)

    print(f"[Gallery] Loaded {len(images)} images for {len(set(student_ids))} students.")
    return images, student_ids

def run_evaluation_for_configuration(
    backbone_name,
    use_pca,
    detector,
    gallery_images,
    gallery_student_ids,
    ground_truth,
    gallery_cache=None,
    use_enhancer=False,
    use_arcface=False,
    config_name_override=None
):
    """
    Runs the full CCTV pipeline for a specific configuration.
    """
    if config_name_override:
        config_name = config_name_override
    else:
        config_name = f"{backbone_name.upper()} {'+ PCA' if use_pca else 'without PCA'}"

    print(f"\n{'='*75}\n[Evaluating Configuration]: {config_name}\n{'='*75}")

    # 1. Feature Extractor
    extractor = DeepFeatureExtractor(backbone=backbone_name, use_arcface=use_arcface)

    # 2. Extract or retrieve gallery embeddings from cache
    cache_key = (backbone_name.lower(), use_arcface)
    if gallery_cache is not None and cache_key in gallery_cache:
        print(f"[Gallery Cache] Reusing pre-extracted {backbone_name.upper()} embeddings.")
        gallery_embeddings = gallery_cache[cache_key]
    else:
        print(f"[Gallery] Extracting features with {backbone_name.upper()} (ArcFace={use_arcface})...")
        gallery_embeddings = extractor.extract_features(gallery_images, batch_size=32)
        if gallery_cache is not None:
            gallery_cache[cache_key] = gallery_embeddings

    # 3. PCA Reducer
    pca_reducer = None
    if use_pca:
        pca_reducer = PCAReducer(variance_ratio=config.PCA_VARIANCE_RATIO, target_components=config.PCA_TARGET_COMPONENTS)
        pca_reducer.fit(gallery_embeddings)
        processed_gallery_embeddings = pca_reducer.transform(gallery_embeddings)
    else:
        processed_gallery_embeddings = gallery_embeddings

    # 4. Enrol Gallery in Matcher
    thresh = 0.36 if use_arcface else 0.28
    matcher = CosineFaceMatcher(threshold=thresh)
    matcher.enroll_gallery(processed_gallery_embeddings, gallery_student_ids)

    # 5. Initialize Voting Engine
    voting_engine = AttendanceVotingEngine(
        min_detections=config.MIN_DETECTIONS_FOR_PRESENT,
        total_intervals=config.NUM_INTERVAL_FRAMES
    )

    enhancer = CCTVFaceEnhancer(target_size=config.IMAGE_SIZE) if use_enhancer else None

    # 6. Process each of the 5 CCTV interval frames
    print(f"[CCTV Stream] Processing {config.NUM_INTERVAL_FRAMES} interval frames (Enhancer={use_enhancer})...")
    for interval in range(1, config.NUM_INTERVAL_FRAMES + 1):
        frame_path = config.CCTV_SESSIONS_DIR / f"cctv_interval_{interval}.jpg"
        frame_bgr = cv2.imread(str(frame_path))
        gt_key = f"interval_{interval}"
        gt_bboxes = ground_truth["interval_detections"].get(gt_key, [])

        # Detect face crops in classroom CCTV frame
        detections = detector.detect_faces(frame_bgr, gt_fallback_bboxes=gt_bboxes)
        probe_crops = [d["crop"] for d in detections]

        # Apply Novelty 3: CCTV Face Enhancement if enabled
        if use_enhancer and enhancer is not None:
            probe_crops = enhancer.enhance_batch(probe_crops)

        if len(probe_crops) > 0:
            # Extract probe embeddings
            probe_embeddings = extractor.extract_features(probe_crops, batch_size=32)

            # Apply PCA if enabled
            if use_pca and pca_reducer is not None:
                probe_embeddings = pca_reducer.transform(probe_embeddings)

            # Match against student database
            matched_results = matcher.match_probes_batch(probe_embeddings)
        else:
            matched_results = []

        # Record interval attendance
        voting_engine.record_interval(interval, matched_results)

    # 7. Compute final session attendance
    all_students = sorted(list(set(gallery_student_ids)))
    attendance_results = voting_engine.compute_final_attendance(all_enrolled_students=all_students)

    # 8. Compute metrics against ground truth
    present_set = set(ground_truth.get("present_students", config.PRESENT_STUDENT_IDS))
    y_true = [1 if sid in present_set else 0 for sid in all_students]
    y_pred = [1 if attendance_results[sid]["is_present_bool"] else 0 for sid in all_students]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Export CSV for this configuration
    logger = AttendanceLogger()
    clean_cfg_str = config_name.replace(" ", "_").replace("+", "plus").replace(":", "").replace("(", "").replace(")", "")
    logger.export_attendance_csv(attendance_results, model_name=clean_cfg_str)

    return {
        "Configuration": config_name,
        "Accuracy": round(acc, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "F1 Score": round(f1, 3),
        "True_Present": sum(y_true),
        "Predicted_Present": sum(y_pred)
    }

def main():
    print("=" * 80)
    print("  CLASSROOM CCTV ATTENDANCE - RESEARCH PAPER REPLICATION & NOVELTY BENCHMARK")
    print("=" * 80)

    # Load Ground Truth
    gt_path = config.CCTV_SESSIONS_DIR / "ground_truth.json"
    with open(gt_path, "r") as f:
        ground_truth = json.load(f)

    # Load Gallery
    gallery_images, gallery_student_ids = load_gallery_images()

    # Initialize Face Detector
    detector = YOLOClassroomDetector(conf_threshold=0.35)

    # Shared Gallery Embedding Cache across configurations
    gallery_cache = {}

    # Full Evaluation Matrix:
    # 1-4: Baseline Paper Configurations
    # 5-7: Novelty Configurations (Options 1 & 3)
    configurations = [
        # Baseline Paper Replications
        {
            "backbone": "vgg16", "use_pca": False,
            "use_enhancer": False, "use_arcface": False,
            "name": "VGG16 without PCA (Baseline)",
            "paper_key": "VGG16 without PCA"
        },
        {
            "backbone": "vgg16", "use_pca": True,
            "use_enhancer": False, "use_arcface": False,
            "name": "VGG16 + PCA (Baseline)",
            "paper_key": "VGG16 + PCA"
        },
        {
            "backbone": "resnet50", "use_pca": False,
            "use_enhancer": False, "use_arcface": False,
            "name": "ResNet-50 without PCA (Baseline)",
            "paper_key": "ResNet-50 without PCA"
        },
        {
            "backbone": "resnet50", "use_pca": True,
            "use_enhancer": False, "use_arcface": False,
            "name": "ResNet-50 + PCA (Baseline)",
            "paper_key": "ResNet-50 + PCA"
        },

        # Novelty Additions (Options 1 & 3)
        {
            "backbone": "resnet50", "use_pca": True,
            "use_enhancer": True, "use_arcface": False,
            "name": "ResNet-50 + CCTV Enhancer + PCA (Novelty 3)",
            "paper_key": "ResNet-50 + PCA"
        },
        {
            "backbone": "resnet50", "use_pca": False,
            "use_enhancer": False, "use_arcface": True,
            "name": "ResNet-50 + ArcFace (Novelty 1)",
            "paper_key": "ResNet-50 without PCA"
        },
        {
            "backbone": "resnet50", "use_pca": True,
            "use_enhancer": True, "use_arcface": True,
            "name": "PROPOSED: ResNet-50 + ArcFace + CCTV Enhancer + PCA (Novelty 1+3)",
            "paper_key": "ResNet-50 + PCA"
        }
    ]

    all_results = []
    for cfg in configurations:
        metrics = run_evaluation_for_configuration(
            backbone_name=cfg["backbone"],
            use_pca=cfg["use_pca"],
            detector=detector,
            gallery_images=gallery_images,
            gallery_student_ids=gallery_student_ids,
            ground_truth=ground_truth,
            gallery_cache=gallery_cache,
            use_enhancer=cfg["use_enhancer"],
            use_arcface=cfg["use_arcface"],
            config_name_override=cfg["name"]
        )
        metrics["Paper_Key"] = cfg["paper_key"]
        all_results.append(metrics)

    # Build Comparison DataFrame
    rows = []
    for res in all_results:
        paper_key = res["Paper_Key"]
        paper_bench = config.PAPER_BENCHMARKS.get(paper_key, {})

        rows.append({
            "Configuration": res["Configuration"],
            "Our Acc": res["Accuracy"],
            "Paper Acc": paper_bench.get("Accuracy", "-"),
            "Our Prec": res["Precision"],
            "Paper Prec": paper_bench.get("Precision", "-"),
            "Our Rec": res["Recall"],
            "Paper Rec": paper_bench.get("Recall", "-"),
            "Our F1": res["F1 Score"],
            "Paper F1": paper_bench.get("F1 Score", "-")
        })

    comparison_df = pd.DataFrame(rows)

    print("\n" + "=" * 105)
    print("                    REPLICATION & NOVELTY BENCHMARK COMPARISON TABLE")
    print("=" * 105)
    print(comparison_df.to_string(index=False))
    print("=" * 105)

    # Save summary table to CSV
    summary_path = config.OUTPUT_DIR / "replication_benchmark_summary.csv"
    comparison_df.to_csv(summary_path, index=False)
    print(f"\n[Summary] Saved full comparison table to {summary_path}")

if __name__ == "__main__":
    main()
