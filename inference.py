import importlib.util
import os
import pickle
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from ultralytics import YOLO


# ============================================================
# USER CONFIGURATION — CHANGE THESE PATHS ONLY
# ============================================================

IMAGE_PATH = "/kaggle/input/datasets/subirdas07/ridingdog/zooandpeople.JPG"

# Folder produced by main.py. It must contain tokenizer.pkl, config.pkl,
# context_pca.pkl, context_scaler.pkl, emotion_to_index.pkl and four
# experiment subfolders.
CAPTION_MODEL_DIR = "/kaggle/working/imgcap/caption_models"

# Use the model.py file that defines your custom ViT layers.
MODEL_PY_PATH = "/kaggle/working/imgcap/model.py"

HUMAN_EMOTION_MODEL_PATH = (
    "/kaggle/working/imgcap/caption_models/image_emotion.keras"
)
PET_EMOTION_MODEL_PATH = (
    "/kaggle/working/imgcap/caption_models/pet_emotion.keras"
)

# Caption model filenames inside each experiment folder.
KERAS_MODEL_NAME = "final_model.keras"
WEIGHT_NAME = "best.weights.h5"
USE_COMPLETE_KERAS_MODEL = True

# Context extraction settings — identical to coco_context.py.
CONTEXT_YOLO_MODEL = "yolov8n.pt"
OBJECT_CONFIDENCE_THRESHOLD = 0.30
STUFF_COVERAGE_THRESHOLD = 0.01
SEGFORMER_MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"

# Emotion detection settings — identical to pet_img_emotion.py.
EMOTION_YOLO_MODEL = "yolo26n.pt"
EMOTION_CONFIDENCE_THRESHOLD = 0.20
EMOTION_IMAGE_SIZE = 960
EMOTION_CROP_DIR = "/kaggle/working/random_image_emotion_crops"

BEAM_WIDTH = 3
LENGTH_PENALTY = 0.7
SAVE_RESULTS_CSV = "/kaggle/working/random_image_captions.csv"

EXPERIMENTS = (
    "image_only",
    "image_context",
    "image_emotion",
    "image_context_emotion",
)

EMOTION_CLASSES = (
    "angry", "contempt", "disgust", "fear",
    "happy", "neutral", "sad", "surprise",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# COCO CONTEXT LABELS — SAME ORDER USED DURING TRAINING
# ============================================================

MSCOCO_OBJECTS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]

MSCOCO_STUFF = [
    "banner", "blanket", "branch", "bridge", "building-other", "bush", "cabinet", "cage",
    "cardboard", "carpet", "ceiling-other", "ceiling-tile", "cloth", "clothes", "clouds",
    "counter", "cupboard", "curtain", "desk-stuff", "dirt", "door-stuff", "fence",
    "floor-marble", "floor-other", "floor-stone", "floor-tile", "floor-wood", "flower", "fog",
    "food-other", "fruit", "furniture-other", "grass", "gravel", "ground-other", "hill",
    "house", "leaves", "light", "mat", "metal", "mirror-stuff", "moss", "mountain", "mud",
    "napkin", "net", "paper", "pavement", "pillow", "plant-other", "plastic", "platform",
    "playingfield", "railing", "railroad", "river", "road", "rock", "roof", "rug", "salad",
    "sand", "sea", "shelf", "sky-other", "skyscraper", "snow", "solid-other", "stairs",
    "stone", "straw", "structural-other", "table", "tent", "textile-other", "towel", "tree",
    "vegetable", "wall-brick", "wall-concrete", "wall-other", "wall-panel", "wall-stone",
    "wall-tile", "wall-wood", "water-other", "waterdrops", "window-blind", "window-other",
    "wood",
]

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

ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
}

assert len(MSCOCO_OBJECTS) == 80
assert len(MSCOCO_STUFF) == 91


# ============================================================
# GENERAL HELPERS
# ============================================================

def require_file(path, description):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{description} not found: {path}")


def load_pickle(path):
    require_file(path, "Pickle file")
    with open(path, "rb") as handle:
        return pickle.load(handle)


def import_caption_model_module(model_py_path):
    require_file(model_py_path, "model.py")
    spec = importlib.util.spec_from_file_location("caption_model_module", model_py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import model module: {model_py_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_caption_image(path, image_size):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, (image_size, image_size))
    return tf.cast(image, tf.float32) / 255.0


# ============================================================
# 180-D CONTEXT EXTRACTION
# ============================================================

class ContextExtractor:
    def __init__(self):
        print(f"[INFO] Loading context YOLO model: {CONTEXT_YOLO_MODEL}")
        self.yolo = YOLO(CONTEXT_YOLO_MODEL)

        print(f"[INFO] Loading SegFormer: {SEGFORMER_MODEL_NAME}")
        self.processor = SegformerImageProcessor.from_pretrained(SEGFORMER_MODEL_NAME)
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            SEGFORMER_MODEL_NAME
        ).to(DEVICE)
        self.segformer.eval()
        self.ade_id_to_label = self.segformer.config.id2label
        self.stuff_to_index = {label: index for index, label in enumerate(MSCOCO_STUFF)}

    def detect_objects(self, image_path):
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        result = self.yolo(image_rgb, verbose=False)[0]
        features = np.zeros(80, dtype=np.float32)

        if result.boxes is not None and len(result.boxes) > 0:
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            scores = result.boxes.conf.cpu().numpy()
            for class_id, score in zip(class_ids, scores):
                if score >= OBJECT_CONFIDENCE_THRESHOLD and 0 <= class_id < 80:
                    features[class_id] += 1.0

        maximum = max(float(features.max()), 1.0)
        return features / maximum

    def detect_stuff(self, image_path):
        features = np.zeros(91, dtype=np.float32)
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            outputs = self.segformer(**inputs)

        logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )
        segmentation = logits.argmax(dim=1)[0].cpu().numpy()
        total_pixels = segmentation.size
        unique_ids, counts = np.unique(segmentation, return_counts=True)

        for ade_id, count in zip(unique_ids, counts):
            ade_label = str(self.ade_id_to_label.get(int(ade_id), "")).lower().strip()
            coco_label = ADE_TO_COCO_STUFF.get(ade_label)
            if coco_label is None:
                continue
            coverage = float(count) / float(total_pixels)
            if coverage >= STUFF_COVERAGE_THRESHOLD:
                features[self.stuff_to_index[coco_label]] += coverage

        features = np.clip(features, 0.0, 1.0)

        # Keep only the six strongest stuff features, as in coco_context.py.
        nonzero_indices = np.where(features > 0)[0]
        if len(nonzero_indices) > 6:
            top_indices = np.argsort(features)[-6:]
            filtered = np.zeros_like(features)
            filtered[top_indices] = features[top_indices]
            features = filtered

        return features

    @staticmethod
    def scene_statistics(object_features, stuff_features):
        return np.array([
            np.sum(object_features > 0),
            np.sum(stuff_features > 0),
            np.mean(object_features),
            np.std(object_features),
            np.mean(stuff_features),
            np.std(stuff_features),
            np.max(object_features),
            np.max(stuff_features),
            (np.sum(object_features > 0) + np.sum(stuff_features > 0)) / 171.0,
        ], dtype=np.float32)

    def extract(self, image_path):
        objects = self.detect_objects(image_path)
        stuff = self.detect_stuff(image_path)
        statistics = self.scene_statistics(objects, stuff)
        raw_context = np.concatenate([objects, stuff, statistics]).astype(np.float32)

        if raw_context.shape != (180,):
            raise ValueError(f"Expected a 180-D context vector, got {raw_context.shape}.")
        if not np.all(np.isfinite(raw_context)):
            raise ValueError("The context vector contains NaN or infinity.")

        detected_objects = [
            MSCOCO_OBJECTS[index]
            for index in np.where(objects > 0)[0]
        ]
        detected_stuff = [
            MSCOCO_STUFF[index]
            for index in np.where(stuff > 0)[0]
        ]
        return raw_context, detected_objects, detected_stuff


# ============================================================
# IMAGE EMOTION EXTRACTION
# ============================================================

class ImageEmotionExtractor:
    def __init__(self):
        require_file(HUMAN_EMOTION_MODEL_PATH, "Human emotion model")
        require_file(PET_EMOTION_MODEL_PATH, "Pet emotion model")

        print("[INFO] Loading human FERPlus model")
        self.human_model = tf.keras.models.load_model(
            HUMAN_EMOTION_MODEL_PATH, compile=False
        )
        print("[INFO] Loading pet emotion model")
        self.pet_model = tf.keras.models.load_model(
            PET_EMOTION_MODEL_PATH, compile=False
        )
        print(f"[INFO] Loading emotion YOLO model: {EMOTION_YOLO_MODEL}")
        self.yolo = YOLO(EMOTION_YOLO_MODEL)

    def predict_human(self, crop_path):
        image = cv2.imread(crop_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read human crop: {crop_path}")
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)
        tensor = image.astype(np.float32) / 255.0
        tensor = tensor[None, ..., None]
        probabilities = self.human_model.predict(tensor, verbose=0)[0]
        return int(np.argmax(probabilities))

    def predict_pet(self, crop_path):
        image = cv2.imread(crop_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read animal crop: {crop_path}")
        image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = image.astype(np.float32) / 255.0
        tensor = tensor[None, ...]
        pet_class = int(np.argmax(self.pet_model.predict(tensor, verbose=0)[0]))

        # Pet dataset classes map to the common FERPlus class indices:
        # angry->0, happy->4, relaxed->7 (surprise), sad->6.
        return {0: 0, 1: 4, 2: 7, 3: 6}.get(pet_class, 5)

    def extract(self, image_path):
        if os.path.isdir(EMOTION_CROP_DIR):
            shutil.rmtree(EMOTION_CROP_DIR)
        os.makedirs(EMOTION_CROP_DIR, exist_ok=True)

        frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Could not read image: {image_path}")

        results = self.yolo.predict(
            source=frame,
            classes=[0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            conf=EMOTION_CONFIDENCE_THRESHOLD,
            imgsz=EMOTION_IMAGE_SIZE,
            agnostic_nms=True,
            save_crop=True,
            project=EMOTION_CROP_DIR,
            name="predict",
            exist_ok=True,
            verbose=False,
        )

        human_scores = []
        pet_scores = []

        if results:
            crop_root = os.path.join(str(results[0].save_dir), "crops")

            person_dir = os.path.join(crop_root, "person")
            if os.path.isdir(person_dir):
                for filename in sorted(os.listdir(person_dir)):
                    path = os.path.join(person_dir, filename)
                    if os.path.isfile(path):
                        human_scores.append(self.predict_human(path))

            for animal in sorted(ANIMAL_CLASSES):
                animal_dir = os.path.join(crop_root, animal)
                if not os.path.isdir(animal_dir):
                    continue
                for filename in sorted(os.listdir(animal_dir)):
                    path = os.path.join(animal_dir, filename)
                    if os.path.isfile(path):
                        pet_scores.append(self.predict_pet(path))

        # Reproduce pet_img_emotion.py fusion behavior: average detected class
        # indices, then round to the nearest common 8-class index.
        if human_scores and pet_scores:
            score = (np.mean(human_scores) + np.mean(pet_scores)) / 2.0
        elif human_scores:
            score = float(np.mean(human_scores))
        elif pet_scores:
            score = float(np.mean(pet_scores))
        else:
            score = 5.0  # neutral when no person/animal is detected

        emotion_index = int(np.floor(score + 0.5))
        emotion_index = int(np.clip(emotion_index, 0, len(EMOTION_CLASSES) - 1))
        emotion_label = EMOTION_CLASSES[emotion_index]
        emotion_vector = np.eye(len(EMOTION_CLASSES), dtype=np.float32)[emotion_index]

        return emotion_vector, emotion_label, human_scores, pet_scores


# ============================================================
# CAPTION MODEL AND BEAM SEARCH
# ============================================================

def model_inputs(experiment, image, context, emotion, sequence):
    """Return model inputs as TensorFlow tensors only.

    Keras 3 does not allow TensorFlow tensors and NumPy arrays to be mixed
    inside a nested input list.
    """
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    if image.shape.rank == 3:
        image = tf.expand_dims(image, axis=0)

    sequence = tf.convert_to_tensor(sequence, dtype=tf.int32)
    if sequence.shape.rank == 1:
        sequence = tf.expand_dims(sequence, axis=0)

    context = tf.convert_to_tensor(context, dtype=tf.float32)
    if context.shape.rank == 1:
        context = tf.expand_dims(context, axis=0)

    emotion = tf.convert_to_tensor(emotion, dtype=tf.float32)
    if emotion.shape.rank == 1:
        emotion = tf.expand_dims(emotion, axis=0)

    if experiment == "image_only":
        return [image, sequence]
    if experiment == "image_context":
        return [image, context, sequence]
    if experiment == "image_emotion":
        return [image, emotion, sequence]
    if experiment == "image_context_emotion":
        return [image, context, emotion, sequence]

    raise ValueError(f"Unknown experiment: {experiment}")


def load_experiment_model(experiment, config, model_module):
    experiment_dir = os.path.join(CAPTION_MODEL_DIR, experiment)
    keras_path = os.path.join(experiment_dir, KERAS_MODEL_NAME)
    weight_path = os.path.join(experiment_dir, WEIGHT_NAME)

    tf.keras.backend.clear_session()

    if USE_COMPLETE_KERAS_MODEL and os.path.isfile(keras_path):
        print(f"[INFO] Loading complete model: {keras_path}")
        return tf.keras.models.load_model(keras_path, compile=False)

    if not os.path.isfile(weight_path):
        raise FileNotFoundError(
            f"No model found for {experiment}. Checked:\n"
            f"  {keras_path}\n  {weight_path}"
        )

    # The uploaded model.py currently may list only one experiment. Temporarily
    # expose all four so its architecture builder can rebuild any saved weights.
    original_experiments = getattr(model_module, "EXPERIMENTS", None)
    model_module.EXPERIMENTS = EXPERIMENTS
    try:
        print(f"[INFO] Rebuilding architecture and loading: {weight_path}")
        model = model_module.build_captioning_model(config, experiment)
        model.load_weights(weight_path)
        return model
    finally:
        if original_experiments is not None:
            model_module.EXPERIMENTS = original_experiments


def beam_search_decode(model, experiment, image, context, emotion, tokenizer,
                       config, beam_width=3, length_penalty=0.7):
    if "startseq" not in tokenizer.word_index or "endseq" not in tokenizer.word_index:
        raise ValueError("Tokenizer must contain startseq and endseq tokens.")

    start_id = tokenizer.word_index["startseq"]
    end_id = tokenizer.word_index["endseq"]
    index_to_word = {index: word for word, index in tokenizer.word_index.items()}
    beams = [([start_id], 0.0)]

    for _ in range(config["MAX_LEN"] - 1):
        candidates = []
        all_finished = True

        for tokens, score in beams:
            if tokens[-1] == end_id:
                candidates.append((tokens, score))
                continue

            all_finished = False
            padded = tf.keras.preprocessing.sequence.pad_sequences(
                [tokens],
                maxlen=config["MAX_LEN"] - 1,
                padding="post",
                truncating="post",
            )
            padded = tf.convert_to_tensor(padded, dtype=tf.int32)
            logits = model(
                model_inputs(experiment, image, context, emotion, padded),
                training=False,
            )[0, len(tokens) - 1]
            log_probabilities = tf.nn.log_softmax(logits).numpy()

            width = min(beam_width, len(log_probabilities))
            top_ids = np.argpartition(log_probabilities, -width)[-width:]
            for token_id in top_ids:
                candidates.append((
                    tokens + [int(token_id)],
                    score + float(log_probabilities[token_id]),
                ))

        if all_finished:
            break

        beams = sorted(
            candidates,
            key=lambda item: item[1] / (len(item[0]) ** length_penalty),
            reverse=True,
        )[:beam_width]

    best_tokens = max(
        beams,
        key=lambda item: item[1] / (len(item[0]) ** length_penalty),
    )[0]

    words = []
    for token_id in best_tokens[1:]:
        if token_id == end_id:
            break
        word = index_to_word.get(token_id)
        if word and word not in ("startseq", "endseq", "<unk>"):
            words.append(word)
    return " ".join(words)


# ============================================================
# MAIN INFERENCE PIPELINE
# ============================================================

def main():
    require_file(IMAGE_PATH, "Input image")
    if BEAM_WIDTH < 1:
        raise ValueError("BEAM_WIDTH must be at least 1.")

    caption_dir = os.path.abspath(CAPTION_MODEL_DIR)
    tokenizer = load_pickle(os.path.join(caption_dir, "tokenizer.pkl"))
    config = load_pickle(os.path.join(caption_dir, "config.pkl"))
    pca = load_pickle(os.path.join(caption_dir, "context_pca.pkl"))
    scaler = load_pickle(os.path.join(caption_dir, "context_scaler.pkl"))
    emotion_to_index = load_pickle(os.path.join(caption_dir, "emotion_to_index.pkl"))

    model_module = import_caption_model_module(MODEL_PY_PATH)

    print("\n" + "=" * 80)
    print("STEP 1: EXTRACTING RAW 180-D CONTEXT")
    print("=" * 80)
    context_extractor = ContextExtractor()
    raw_context, detected_objects, detected_stuff = context_extractor.extract(IMAGE_PATH)

    transformed_context = scaler.transform(
        pca.transform(raw_context.reshape(1, -1))
    ).astype(np.float32)[0]

    expected_context_dim = int(config["CONTEXT_DIM"])
    if transformed_context.shape != (expected_context_dim,):
        raise ValueError(
            f"Expected transformed context shape ({expected_context_dim},), "
            f"got {transformed_context.shape}."
        )

    print(f"[INFO] Raw context shape: {raw_context.shape}")
    print(f"[INFO] PCA/scaled context shape: {transformed_context.shape}")
    print(f"[INFO] Detected objects: {detected_objects or ['none']}")
    print(f"[INFO] Detected stuff: {detected_stuff or ['none']}")

    print("\n" + "=" * 80)
    print("STEP 2: EXTRACTING FULL-IMAGE EMOTION")
    print("=" * 80)
    emotion_extractor = ImageEmotionExtractor()
    emotion_vector, emotion_label, human_scores, pet_scores = emotion_extractor.extract(
        IMAGE_PATH
    )

    # Validate the generated label against the exact mapping saved by main.py.
    if emotion_label not in emotion_to_index:
        raise ValueError(
            f"Generated emotion '{emotion_label}' is absent from emotion_to_index.pkl."
        )
    saved_index = int(emotion_to_index[emotion_label])
    corrected_vector = np.zeros(int(config["EMOTION_DIM"]), dtype=np.float32)
    corrected_vector[saved_index] = 1.0
    emotion_vector = corrected_vector

    print(f"[INFO] Human emotion indices: {human_scores or ['none']}")
    print(f"[INFO] Pet emotion indices: {pet_scores or ['none']}")
    print(f"[INFO] Final image emotion: {emotion_label}")
    print(f"[INFO] Emotion vector: {emotion_vector.tolist()}")

    print("\n" + "=" * 80)
    print("STEP 3: GENERATING FOUR CAPTIONS")
    print("=" * 80)
    caption_image = load_caption_image(IMAGE_PATH, int(config["IMG_SIZE"]))

    results = []
    for experiment in EXPERIMENTS:
        caption_model = load_experiment_model(experiment, config, model_module)
        caption = beam_search_decode(
            model=caption_model,
            experiment=experiment,
            image=caption_image,
            context=transformed_context,
            emotion=emotion_vector,
            tokenizer=tokenizer,
            config=config,
            beam_width=BEAM_WIDTH,
            length_penalty=LENGTH_PENALTY,
        )
        print(f"{experiment:26s}: {caption}")
        results.append({
            "image": os.path.basename(IMAGE_PATH),
            "experiment": experiment,
            "emotion": emotion_label,
            "generated_caption": caption,
        })
        del caption_model

    if SAVE_RESULTS_CSV:
        output_path = os.path.abspath(SAVE_RESULTS_CSV)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pd.DataFrame(results).to_csv(output_path, index=False)
        print(f"\n[INFO] Results saved to: {output_path}")


if __name__ == "__main__":
    main()
