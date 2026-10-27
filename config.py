"""
Configuration parameters for Classroom CCTV Attendance System Replication.
Matches the Gujarat University classroom research paper specifications.
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_GALLERY_DIR = DATA_DIR / "raw_gallery"
AUGMENTED_GALLERY_DIR = DATA_DIR / "augmented_gallery"
MY_CLASS_DIR = DATA_DIR / "my_class"
MY_CLASS_RAW_DIR = MY_CLASS_DIR / "raw_gallery"
MY_CLASS_AUG_DIR = MY_CLASS_DIR / "augmented_gallery"
CCTV_SESSIONS_DIR = DATA_DIR / "test_cctv_sessions"
OUTPUT_DIR = BASE_DIR / "outputs"
MODELS_DIR = BASE_DIR / "saved_models"

for directory in [DATA_DIR, RAW_GALLERY_DIR, AUGMENTED_GALLERY_DIR, MY_CLASS_DIR, MY_CLASS_RAW_DIR, MY_CLASS_AUG_DIR, CCTV_SESSIONS_DIR, OUTPUT_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Specs (from paper)
NUM_TOTAL_STUDENTS = 41
NUM_PRESENT_STUDENTS = 28
NUM_ABSENT_STUDENTS = NUM_TOTAL_STUDENTS - NUM_PRESENT_STUDENTS

STUDENT_IDS = [f"STU_{i:03d}" for i in range(1, NUM_TOTAL_STUDENTS + 1)]
PRESENT_STUDENT_IDS = STUDENT_IDS[:NUM_PRESENT_STUDENTS]
ABSENT_STUDENT_IDS = STUDENT_IDS[NUM_PRESENT_STUDENTS:]

ORIGINAL_IMAGES_PER_STUDENT = 14
AUGMENTED_IMAGES_PER_STUDENT = 45

# CCTV & Attendance Specs
NUM_INTERVAL_FRAMES = 5         # 5 equally spaced intervals during 60-min session
MIN_DETECTIONS_FOR_PRESENT = 3  # Mark present if detected in at least 3 out of 5 frames
COSINE_THRESHOLD = 0.4          # Cosine similarity threshold from paper

# Feature Extraction & Model Specs
IMAGE_SIZE = (224, 224)
PCA_VARIANCE_RATIO = 0.95       # Preserves 95% variance
PCA_TARGET_COMPONENTS = 1953    # Paper target components

# Ground Truth Baseline Metrics (from paper table to benchmark against)
PAPER_BENCHMARKS = {
    "VGG16 without PCA": {
        "Accuracy": 0.829,
        "Precision": 0.893,
        "Recall": 0.862,
        "F1 Score": 0.877
    },
    "VGG16 + PCA": {
        "Accuracy": 0.878,
        "Precision": 0.929,
        "Recall": 0.897,
        "F1 Score": 0.913
    },
    "ResNet-50 without PCA": {
        "Accuracy": 0.927,
        "Precision": 0.929,
        "Recall": 0.929,
        "F1 Score": 0.929
    },
    "ResNet-50 + PCA": {
        "Accuracy": 0.927,
        "Precision": 0.964,
        "Recall": 0.931,
        "F1 Score": 0.947
    }
}
