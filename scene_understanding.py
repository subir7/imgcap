import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


SPLIT = "train"  # choose: "train" or "val"

COCO_IMG_ROOT = "/kaggle/input/datasets/jeffaudi/coco-2014-dataset-for-yolov3/coco2014/images"
COCO_ANNOT_ROOT = "/kaggle/input/datasets/huthtrnh/coco-2014-dataset-train-val-annotations/data/mscoco_reduced"
TRAIN_IMAGE_DIR = os.path.join(COCO_IMG_ROOT, "train2014")
VAL_IMAGE_DIR = os.path.join(COCO_IMG_ROOT, "val2014")
TRAIN_ANNOTATION_FILE = os.path.join(COCO_ANNOT_ROOT, "annotations", "captions_train2014.json")
VAL_ANNOTATION_FILE = os.path.join(COCO_ANNOT_ROOT, "annotations", "captions_val2014.json")


OUTPUT_DIR = "/kaggle/working/context/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MERGED_CSV = os.path.join(OUTPUT_DIR, f"context.csv")

# Number of images to process. Use small number first for testing, e.g. 100.
# Set None to process full split.
MAX_IMAGES = None

# Detection thresholds
OBJECT_CONFIDENCE_THRESHOLD = 0.30
STUFF_COVERAGE_THRESHOLD = 0.01

# SegFormer settings
USE_SEGFORMER = True
SEGFORMER_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# COCO CLASS LISTS
# ============================================================
MSCOCO_OBJECTS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

MSCOCO_STUFF = [
    "banner", "blanket", "branch", "bridge", "building-other", "bush", "cabinet",
    "cage", "cardboard", "carpet", "ceiling-other", "ceiling-tile", "cloth",
    "clothes", "clouds", "counter", "cupboard", "curtain", "desk-stuff", "dirt",
    "door-stuff", "fence", "floor-marble", "floor-other", "floor-stone",
    "floor-tile", "floor-wood", "flower", "fog", "food-other", "fruit", "furniture-other",
    "grass", "gravel", "ground-other", "hill", "house", "leaves", "light", "mat",
    "metal", "mirror-stuff", "moss", "mountain", "mud", "napkin", "net", "paper",
    "pavement", "pillow", "plant-other", "plastic", "platform", "playingfield",
    "railing", "railroad", "river", "road", "rock", "roof", "rug", "salad", "sand",
    "sea", "shelf", "sky-other", "skyscraper", "snow", "solid-other", "stairs",
    "stone", "straw", "structural-other", "table", "tent", "textile-other", "towel",
    "tree", "vegetable", "wall-brick", "wall-concrete", "wall-other", "wall-panel",
    "wall-stone", "wall-tile", "wall-wood", "water-other", "waterdrops", "window-blind",
    "window-other", "wood"
]

assert len(MSCOCO_OBJECTS) == 80
assert len(MSCOCO_STUFF) == 91

# ============================================================
# LOAD YOLO
# ============================================================
print("Loading YOLOv8...")
yolo_model = YOLO("yolov8n.pt")
print("✓ YOLO loaded")

# ============================================================
# LOAD SEGFORMER
# ============================================================
segformer_processor = None
segformer_model = None
ADE_ID2LABEL = {}

if USE_SEGFORMER:
    print(f"Loading SegFormer: {SEGFORMER_MODEL_NAME}")
    segformer_processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL_NAME)
    segformer_model = SegformerForSemanticSegmentation.from_pretrained(SEGFORMER_MODEL_NAME).to(DEVICE)
    segformer_model.eval()
    ADE_ID2LABEL = segformer_model.config.id2label
    print("✓ SegFormer loaded")
else:
    print("SegFormer disabled. Stuff vector will be zeros.")

# ============================================================
# ADE20K -> COCO-STUFF(91D) APPROXIMATE MAPPING
# ============================================================
ADE_TO_COCO_STUFF = {
    "sky": "sky-other", "cloud": "clouds", "tree": "tree", "grass": "grass",
    "road": "road", "earth": "dirt", "mountain": "mountain", "plant": "plant-other",
    "water": "water-other", "sea": "sea", "river": "river", "rock": "rock",
    "sand": "sand", "snow": "snow", "field": "playingfield", "house": "house",
    "building": "building-other", "skyscraper": "skyscraper", "wall": "wall-other",
    "fence": "fence", "railing": "railing", "bridge": "bridge", "floor": "floor-other",
    "floor-wood": "floor-wood", "carpet": "carpet", "rug": "rug", "door": "door-stuff",
    "window": "window-other", "curtain": "curtain", "ceiling": "ceiling-other",
    "stairs": "stairs", "shelf": "shelf", "cabinet": "cabinet", "cupboard": "cupboard",
    "counter": "counter", "table": "table", "desk": "desk-stuff", "pillow": "pillow",
    "blanket": "blanket", "mirror": "mirror-stuff", "flower": "flower", "fruit": "fruit",
    "food": "food-other", "tent": "tent", "rocky": "stone", "stone": "stone",
    "wood": "wood", "pane": "window-other",
}
ADE_TO_COCO_STUFF = {k.lower(): v for k, v in ADE_TO_COCO_STUFF.items()}

# ============================================================
# HELPERS
# ============================================================
def load_bgr_image(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def get_split_paths(split: str):
    split = split.lower().strip()
    if split == "train":
        return TRAIN_IMAGE_DIR, TRAIN_ANNOTATION_FILE
    if split == "val":
        return VAL_IMAGE_DIR, VAL_ANNOTATION_FILE
    raise ValueError("SPLIT must be 'train' or 'val'.")

# ============================================================
# 80D OBJECT VECTOR
# ============================================================
def detect_objects(image_path: str, confidence_threshold: float = 0.3) -> np.ndarray:
    image_bgr = load_bgr_image(image_path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    results = yolo_model(image_rgb, verbose=False)[0]
    object_features = np.zeros(80, dtype=np.float32)

    if results.boxes is None or len(results.boxes) == 0:
        return object_features

    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    scores = results.boxes.conf.cpu().numpy()

    for cls_id, score in zip(cls_ids, scores):
        if score >= confidence_threshold and 0 <= cls_id < 80:
            object_features[cls_id] += 1.0

    max_count = max(object_features.max(), 1.0)
    object_features = object_features / max_count
    return object_features

# ============================================================
# 91D STUFF VECTOR USING SEGFORMER
# ============================================================
def detect_stuff_segformer(image_path: str, coverage_threshold: float = 0.01) -> np.ndarray:
    stuff_features = np.zeros(91, dtype=np.float32)

    if segformer_model is None or segformer_processor is None:
        return stuff_features

    image = Image.open(image_path).convert("RGB")
    inputs = segformer_processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = segformer_model(**inputs)

    upsampled_logits = torch.nn.functional.interpolate(
        outputs.logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0].cpu().numpy()
    total_pixels = pred_seg.size
    unique_ids, counts = np.unique(pred_seg, return_counts=True)

    for ade_id, count in zip(unique_ids, counts):
        ade_label = ADE_ID2LABEL.get(int(ade_id), "").lower().strip()
        if ade_label in ADE_TO_COCO_STUFF:
            coco_label = ADE_TO_COCO_STUFF[ade_label]
            coco_idx = MSCOCO_STUFF.index(coco_label)
            coverage = count / total_pixels
            if coverage >= coverage_threshold:
                stuff_features[coco_idx] += coverage

    stuff_features = np.clip(stuff_features, 0.0, 1.0)

    top_k = 6
    nonzero_idx = np.where(stuff_features > 0)[0]
    if len(nonzero_idx) > top_k:
        top_idx = np.argsort(stuff_features)[-top_k:]
        filtered = np.zeros_like(stuff_features)
        filtered[top_idx] = stuff_features[top_idx]
        stuff_features = filtered

    return stuff_features


def detect_stuff(image_path: str, coverage_threshold: float = 0.01) -> np.ndarray:
    return detect_stuff_segformer(image_path, coverage_threshold)

# ============================================================
# 9D SCENE STATS + 180D FEATURE VECTOR
# ============================================================
def compute_scene_statistics(object_features: np.ndarray, stuff_features: np.ndarray) -> np.ndarray:
    return np.array([
        np.sum(object_features > 0),
        np.sum(stuff_features > 0),
        np.mean(object_features),
        np.std(object_features),
        np.mean(stuff_features),
        np.std(stuff_features),
        np.max(object_features),
        np.max(stuff_features),
        (np.sum(object_features > 0) + np.sum(stuff_features > 0)) / 171.0
    ], dtype=np.float32)


def extract_complete_features(
    image_path: str,
    object_confidence_threshold: float = 0.3,
    stuff_coverage_threshold: float = 0.01
) -> np.ndarray:
    object_features = detect_objects(
        image_path,
        confidence_threshold=object_confidence_threshold
    )
    stuff_features = detect_stuff(
        image_path,
        coverage_threshold=stuff_coverage_threshold
    )
    scene_stats = compute_scene_statistics(object_features, stuff_features)

    complete_features = np.concatenate([
        object_features,
        stuff_features,
        scene_stats
    ]).astype(np.float32)

    return complete_features

# ============================================================
# MSCOCO 2014 CAPTION LOADER
# ============================================================
def coco_image_filename(split: str, image_id: int) -> str:
    split = split.lower().strip()
    if split == "train":
        return f"COCO_train2014_{int(image_id):012d}.jpg"
    if split == "val":
        return f"COCO_val2014_{int(image_id):012d}.jpg"
    raise ValueError("split must be 'train' or 'val'.")


def load_mscoco2014_captions(annotation_file: str, split: str) -> pd.DataFrame:
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_id_to_filename = {}
    for img in data.get("images", []):
        image_id = int(img["id"])
        file_name = img.get("file_name") or coco_image_filename(split, image_id)
        image_id_to_filename[image_id] = file_name

    records = []
    for ann in data.get("annotations", []):
        image_id = int(ann["image_id"])
        caption = str(ann["caption"]).strip()
        image_name = image_id_to_filename.get(image_id, coco_image_filename(split, image_id))

        records.append({
            "image_id": image_id,
            "image_name": image_name,
            "filename": image_name,
            "caption": caption
        })

    captions_df = pd.DataFrame(records)
    captions_df = captions_df.dropna(subset=["image_name", "caption"]).reset_index(drop=True)
    return captions_df

# ============================================================
# BUILD IMAGE-LEVEL FEATURE TABLE FOR MSCOCO 2014
# ============================================================
def build_mscoco2014_feature_table(
    image_dir: str,
    annotation_file: str,
    split: str,
    confidence_threshold: float = 0.3,
    stuff_coverage_threshold: float = 0.01,
    max_images=None
):
    captions_df = load_mscoco2014_captions(annotation_file, split)

    image_table = captions_df[["image_id", "image_name", "filename"]].drop_duplicates("image_id")
    image_table = image_table.sort_values("image_id").reset_index(drop=True)

    if max_images is not None:
        image_table = image_table.iloc[:max_images].reset_index(drop=True)

    feature_rows = []

    for i, row_in in image_table.iterrows():
        image_id = int(row_in["image_id"])
        image_name = str(row_in["image_name"])
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"Skipping missing image: {image_path}")
            continue

        try:
            features = extract_complete_features(
                image_path,
                object_confidence_threshold=confidence_threshold,
                stuff_coverage_threshold=stuff_coverage_threshold
            )

            row = {
                "image_id": image_id,
                "image_name": image_name,
                "filename": image_name,
                "image_path": image_path
            }
            for j in range(180):
                row[f"f_{j}"] = float(features[j])

            feature_rows.append(row)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(image_table)} images")

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    features_df = pd.DataFrame(feature_rows)
    return features_df, captions_df

# ============================================================
# MAIN
# ============================================================
def main():
    image_dir, annotation_file = get_split_paths(SPLIT)

    print("SPLIT:", SPLIT)
    print("IMAGE_DIR:", image_dir)
    print("ANNOTATION_FILE:", annotation_file)
    print("USE_SEGFORMER:", USE_SEGFORMER)
    print("SEGFORMER_MODEL_NAME:", SEGFORMER_MODEL_NAME)
    print("DEVICE:", DEVICE)
    print("MAX_IMAGES:", MAX_IMAGES)

    captions_df = load_mscoco2014_captions(annotation_file, SPLIT)
    print("Caption rows:", len(captions_df))
    print("Unique images in annotations:", captions_df["image_id"].nunique())
    print("Example row:")
    print(captions_df.head(1))

    features_df, captions_df = build_mscoco2014_feature_table(
        image_dir=image_dir,
        annotation_file=annotation_file,
        split=SPLIT,
        confidence_threshold=OBJECT_CONFIDENCE_THRESHOLD,
        stuff_coverage_threshold=STUFF_COVERAGE_THRESHOLD,
        max_images=MAX_IMAGES
    )

    print("Feature table shape:", features_df.shape)

    if features_df.empty:
        raise ValueError("No features were extracted. Check image paths and annotation paths.")

    # Each MSCOCO image has multiple captions. This merge creates one row per caption.
    # Keep one row per image only.
    features_only = features_df[["image_name"] + [f"f_{i}" for i in range(180)]].copy()

    print("Unique images:", len(features_only))
    print("Output shape:", features_only.shape)

    features_only.to_csv(MERGED_CSV, index=False)

    print("Saved feature CSV:", MERGED_CSV)


if __name__ == "__main__":
    main()
