


"""# INFERENCE"""

"""
COMPLETE AUTOMATIC INFERENCE SCRIPT - 180D MSCOCO Features
Input: Image file
Output: Generated caption

Uses 180D MSCOCO features:
- 80D: Object detection (person, car, dog, etc.)
- 91D: Stuff detection (sky, grass, water, etc.)
- 9D: Scene statistics
"""

import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
import pickle
import tensorflow_hub as hub

print("="*80)
print("AUTOMATIC IMAGE CAPTIONING - 180D MSCOCO Features")
print("Loading models and dependencies...")
print("="*80)

# ============================================================================
# MSCOCO CLASS DEFINITIONS
# ============================================================================

MSCOCO_OBJECTS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

MSCOCO_STUFF = [
    'sky', 'grass', 'tree', 'mountain', 'hill', 'rock', 'water', 'sea', 'river', 'lake',
    'sand', 'snow', 'fog', 'clouds', 'bush', 'flower', 'leaves', 'branch', 'dirt', 'mud',
    'building', 'house', 'bridge', 'fence', 'wall', 'roof', 'door', 'window', 'stairs',
    'ceiling', 'floor', 'platform', 'pavement', 'road', 'railroad', 'ground',
    'cabinet', 'shelf', 'table', 'counter', 'carpet', 'rug', 'curtain', 'blanket',
    'pillow', 'towel', 'mirror', 'light', 'paper', 'cardboard', 'wood', 'metal',
    'plastic', 'glass', 'tile', 'brick', 'stone',
    'banner', 'net', 'tent', 'playingfield', 'fruit', 'vegetable', 'food', 'cloth',
    'textile', 'plant', 'gravel', 'moss', 'straw',
    # Pad to 91 classes
    'material', 'surface', 'landscape', 'scenery', 'background', 'foreground',
    'area', 'region', 'space', 'field', 'place', 'location', 'terrain', 'zone',
    'environment', 'setting', 'context', 'atmosphere', 'element', 'component',
    'structure', 'formation'
][:91]  # Ensure exactly 91

# ============================================================================
# LOAD MODEL CONFIGURATION
# ============================================================================

CONFIG_PATH = "/content/drive/MyDrive/MSCOCO/trainingdata/model_config.pkl"
WEIGHTS_PATH = "/content/drive/MyDrive/MSCOCO/trainingdata/final_model.weights.h5"
TOKENIZER_PATH = "/content/drive/MyDrive/MSCOCO/trainingdata/tokenizer.pkl"

print("\n[1/5] Loading configuration...")
with open(CONFIG_PATH, 'rb') as f:
    config = pickle.load(f)

IMG_SIZE = config['IMG_SIZE']
PATCH_SIZE = config['PATCH_SIZE']
NUM_PATCHES = config['NUM_PATCHES']
EMBED_DIM = config['EMBED_DIM']
NUM_HEADS = config['NUM_HEADS']
MLP_DIM = config['MLP_DIM']
ENCODER_LAYERS = config['ENCODER_LAYERS']
DECODER_LAYERS = config['DECODER_LAYERS']
MAX_LEN = config['MAX_LEN']
CONTEXT_DIM = config['CONTEXT_DIM']
VOCAB_SIZE = config['VOCAB_SIZE']

print(f"✓ Configuration loaded")
print(f"  Context dimension: {CONTEXT_DIM}D")

if CONTEXT_DIM != 180:
    print(f"\n  WARNING: Model expects {CONTEXT_DIM}D context, but 180D features will be extracted")
    print(f"  You need to retrain with 180D features if CONTEXT_DIM != 180")

# ============================================================================
# LOAD TOKENIZER
# ============================================================================

print("\n[2/5] Loading tokenizer...")
with open(TOKENIZER_PATH, 'rb') as f:
    tokenizer = pickle.load(f)

idx_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
idx_to_word[0] = '<pad>'

print(f" Tokenizer loaded ({len(tokenizer.word_index)} words)")

# ============================================================================
# OBJECT DETECTOR (80 classes)
# ============================================================================

print("\n[3/5] Loading object detector...")

class ObjectDetector:
    """Detects MSCOCO Objects (80 classes)"""

    def __init__(self, confidence_threshold=0.3, max_objects=20):
        self.confidence_threshold = confidence_threshold
        self.max_objects = max_objects
        self.object_classes = MSCOCO_OBJECTS

        self.model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")
        print("  ✓ Object detector loaded")

    def detect(self, image):
        """Detect objects and return 80D feature vector (class counts)"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run detection
        input_tensor = tf.convert_to_tensor(image)[tf.newaxis, ...]
        detections = self.model(input_tensor)

        boxes = detections['detection_boxes'][0].numpy()
        classes = detections['detection_classes'][0].numpy().astype(int)
        scores = detections['detection_scores'][0].numpy()

        # Count objects by class
        class_counts = np.zeros(len(self.object_classes), dtype=np.float32)

        valid_indices = np.where(scores >= self.confidence_threshold)[0][:self.max_objects]

        detected_objects = []
        for idx in valid_indices:
            class_id = classes[idx] - 1  # COCO is 1-indexed
            if 0 <= class_id < len(self.object_classes):
                class_counts[class_id] += 1
                detected_objects.append({
                    'name': self.object_classes[class_id],
                    'confidence': float(scores[idx])
                })

        # Normalize by max count
        max_count = max(class_counts.max(), 1.0)
        class_counts = class_counts / max_count

        return class_counts, detected_objects

# ============================================================================
# STUFF DETECTOR (91 classes)
# ============================================================================

print("\n[4/5] Loading stuff detector...")

class StuffDetector:
    """Detects MSCOCO Stuff (91 classes) using heuristics"""

    def __init__(self, coverage_threshold=0.01):
        self.coverage_threshold = coverage_threshold
        self.stuff_classes = MSCOCO_STUFF
        print("  ✓ Stuff detector loaded (heuristic-based)")

    def detect(self, image):
        """Detect stuff and return 91D feature vector"""
        if isinstance(image, str):
            image = cv2.imread(image)

        h, w = image.shape[:2]

        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        stuff_features = np.zeros(91, dtype=np.float32)

        # Top region for sky
        top_region = image[:h//4, :]
        blue_ratio = self._detect_color(top_region, 'blue')
        stuff_features[0] = min(blue_ratio * 2, 1.0)  # sky

        # Bottom region for grass
        bottom_region = image[h*3//4:, :]
        green_ratio = self._detect_color(bottom_region, 'green')
        stuff_features[1] = min(green_ratio * 2, 1.0)  # grass

        # Middle region for water
        middle_region = image[h//4:h*3//4, :]
        water_ratio = self._detect_color(middle_region, 'blue')
        stuff_features[6] = min(water_ratio * 1.5, 1.0)  # water

        # Sea (blue in lower half)
        sea_ratio = self._detect_color(bottom_region, 'blue')
        stuff_features[7] = min(sea_ratio * 1.8, 1.0)  # sea

        # Tree (green with texture)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        green_all = self._detect_color(image, 'green')
        stuff_features[2] = min(green_all * edge_density * 5, 1.0)  # tree

        # Building (gray, vertical structures)
        gray_ratio = self._detect_color(image, 'gray')
        stuff_features[20] = min(gray_ratio * 1.5, 1.0)  # building

        # Road (gray in bottom)
        road_ratio = self._detect_color(bottom_region, 'gray')
        stuff_features[33] = min(road_ratio * 1.5, 1.0)  # road

        # Wall
        wall_ratio = self._detect_color(middle_region, 'gray')
        stuff_features[24] = min(wall_ratio * 1.2, 1.0)  # wall

        # Ground (brown/tan)
        brown_ratio = self._detect_color(bottom_region, 'brown')
        stuff_features[35] = min(brown_ratio * 1.5, 1.0)  # ground

        # Sand (light brown/yellow in bottom)
        sand_ratio = self._detect_color(bottom_region, 'yellow')
        stuff_features[10] = min(sand_ratio * 1.2, 1.0)  # sand

        # Snow (white)
        white_ratio = self._detect_color(image, 'white')
        stuff_features[11] = min(white_ratio * 1.5, 1.0)  # snow

        # Mountain (gray/brown in top half)
        mountain_ratio = self._detect_color(top_region, 'gray')
        stuff_features[3] = min(mountain_ratio * 1.3, 1.0)  # mountain

        return stuff_features

    def _detect_color(self, region, color):
        """Detect color presence in image region"""
        hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        color_ranges = {
            'blue': ([100, 50, 50], [130, 255, 255]),
            'green': ([35, 40, 40], [85, 255, 255]),
            'gray': ([0, 0, 50], [180, 50, 200]),
            'brown': ([10, 50, 20], [30, 255, 200]),
            'yellow': ([20, 100, 100], [35, 255, 255]),
            'white': ([0, 0, 200], [180, 50, 255])
        }

        if color not in color_ranges:
            return 0.0

        lower, upper = color_ranges[color]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        ratio = np.sum(mask > 0) / (region.shape[0] * region.shape[1])

        return ratio

# ============================================================================
# COMPLETE FEATURE EXTRACTOR (180D)
# ============================================================================

class CompleteFeatureExtractor:
    """
    Extracts 180D MSCOCO features:
    - 80D: Object counts
    - 91D: Stuff coverage
    - 9D: Scene statistics
    """

    def __init__(self):
        self.object_detector = ObjectDetector()
        self.stuff_detector = StuffDetector()
        print("\n✓ Complete feature extractor initialized (180D)")

    def extract_features(self, image):
        """Extract complete 180D feature vector"""
        # Objects (80D)
        object_features, detected_objects = self.object_detector.detect(image)

        # Stuff (91D)
        stuff_features = self.stuff_detector.detect(image)

        # Scene statistics (9D)
        scene_stats = self._compute_scene_stats(object_features, stuff_features)

        # Combine
        complete_features = np.concatenate([
            object_features,  # 80D
            stuff_features,   # 91D
            scene_stats       # 9D
        ])

        return complete_features.astype(np.float32), detected_objects

    def _compute_scene_stats(self, object_features, stuff_features):
        """Compute 9D scene statistics"""
        stats = [
            np.sum(object_features > 0),           # Number of object types
            np.sum(stuff_features > 0),            # Number of stuff types
            np.mean(object_features),              # Average object presence
            np.std(object_features),               # Object diversity
            np.mean(stuff_features),               # Average stuff coverage
            np.std(stuff_features),                # Stuff diversity
            np.max(object_features),               # Max object count
            np.max(stuff_features),                # Max stuff coverage
            (np.sum(object_features > 0) + np.sum(stuff_features > 0)) / 171
        ]
        return np.array(stats, dtype=np.float32)

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

print("\n[5/5] Building model architecture...")

class PatchExtract(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [tf.shape(images)[0], -1, patch_dims])
        return patches

class PatchEmbedding(layers.Layer):
    def __init__(self, num_patches, embed_dim):
        super().__init__()
        self.proj = layers.Dense(embed_dim)
        self.pos = layers.Embedding(num_patches, embed_dim)

    def call(self, patches):
        pos_ids = tf.range(start=0, limit=NUM_PATCHES, delta=1)
        pos_emb = self.pos(pos_ids)
        pos_emb = tf.expand_dims(pos_emb, axis=0)
        x = self.proj(patches) + pos_emb
        return x

class EncoderBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ff = tf.keras.Sequential([
            layers.Dense(mlp_dim, activation="gelu"),
            layers.Dense(embed_dim)
        ])
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        att_out = self.att(x, x)
        x = self.ln1(x + att_out)
        ff_out = self.ff(x)
        x = self.ln2(x + ff_out)
        return x

class ContextFusion(layers.Layer):
    def __init__(self, embed_dim):
        super().__init__()
        self.context_proj = layers.Dense(embed_dim, name="context_projection")
        self.dropout = layers.Dropout(0.1)
        self.ln = layers.LayerNormalization(epsilon=1e-6)

    def call(self, enc_out, context, training=False):
        context_emb = self.context_proj(context)
        context_emb = self.dropout(context_emb, training=training)
        context_emb = tf.expand_dims(context_emb, 1)

        combined = tf.concat([enc_out, context_emb], axis=1)
        return self.ln(combined)

class DecoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads):
        super().__init__()
        self.self_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.cross_att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ff = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(embed_dim)
        ])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ln3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)

    def _causal_mask(self, seq_len, batch_size):
        mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        mask = tf.cast(mask, tf.bool)
        mask = tf.reshape(mask, (1, seq_len, seq_len))
        return tf.repeat(mask, repeats=batch_size, axis=0)

    def call(self, x, enc_out, training=False):
        batch = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]
        mask = self._causal_mask(seq_len, batch)

        att1 = self.self_att(x, x, attention_mask=mask)
        att1 = self.dropout1(att1, training=training)
        x = self.ln1(x + att1)

        att2 = self.cross_att(x, enc_out)
        att2 = self.dropout2(att2, training=training)
        x = self.ln2(x + att2)

        f = self.ff(x, training=training)
        x = self.ln3(x + f)
        return x

def build_vit_encoder():
    img = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    patches = PatchExtract(PATCH_SIZE)(img)
    x = PatchEmbedding(NUM_PATCHES, EMBED_DIM)(patches)

    for _ in range(ENCODER_LAYERS):
        x = EncoderBlock(EMBED_DIM, NUM_HEADS, MLP_DIM)(x)

    return tf.keras.Model(img, x, name="vit_encoder")

def build_decoder():
    seq_in = layers.Input(shape=(MAX_LEN-1,), dtype=tf.int32)
    enc_out = layers.Input(shape=(NUM_PATCHES + 1, EMBED_DIM))

    tok_emb = layers.Embedding(VOCAB_SIZE, EMBED_DIM)(seq_in)
    pos_ids = tf.range(start=0, limit=MAX_LEN-1, delta=1)
    pos_layer = layers.Embedding(MAX_LEN, EMBED_DIM)
    pos_emb = pos_layer(pos_ids)
    pos_emb = tf.expand_dims(pos_emb, 0)
    x = tok_emb + pos_emb

    for _ in range(DECODER_LAYERS):
        x = DecoderBlock(EMBED_DIM, MLP_DIM, NUM_HEADS)(x, enc_out)

    out = layers.Dense(VOCAB_SIZE)(x)
    return tf.keras.Model([seq_in, enc_out], out, name="decoder")

encoder = build_vit_encoder()
decoder = build_decoder()
context_fusion = ContextFusion(EMBED_DIM)

img_inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_input")
context_inp = layers.Input(shape=(CONTEXT_DIM,), name="context_input")
seq_inp = layers.Input(shape=(MAX_LEN-1,), dtype=tf.int32, name="sequence_input")

enc_out = encoder(img_inp)
fused_out = context_fusion(enc_out, context_inp)
dec_out = decoder([seq_inp, fused_out])

model = tf.keras.Model([img_inp, context_inp, seq_inp], dec_out)

print("✓ Model architecture built")

print("\nLoading trained weights...")
model.load_weights(WEIGHTS_PATH)
print("✓ Weights loaded")

# ============================================================================
# INITIALIZE FEATURE EXTRACTORS
# ============================================================================

feature_extractor = CompleteFeatureExtractor()

print("\n" + "="*80)
print("✓ ALL SYSTEMS READY - 180D MSCOCO FEATURES")
print("="*80)

# ============================================================================
# CAPTION GENERATION FUNCTION
# ============================================================================

def generate_caption(image_path, temperature=1.0, max_length=40, verbose=True):
    """Generate caption using 180D MSCOCO features"""

    if verbose:
        print("\n" + "="*80)
        print(f"GENERATING CAPTION")
        print("="*80)
        print(f"Image: {os.path.basename(image_path)}")

    # Load image
    if verbose:
        print("\n[1/4] Loading image...")

    img_pil = Image.open(image_path).convert('RGB')
    img_array = np.array(img_pil.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    img_cv = cv2.imread(image_path)

    # Extract 180D features
    if verbose:
        print("\n[2/4] Extracting 180D MSCOCO features...")
        print("  - 80D: Object detection")
        print("  - 91D: Stuff detection")
        print("  - 9D: Scene statistics")

    context_features, detected_objects = feature_extractor.extract_features(img_cv)

    if verbose:
        print(f"\n✓ Detected {len(detected_objects)} objects:")
        for obj in detected_objects[:5]:
            print(f"    - {obj['name']}: {obj['confidence']:.2f}")
        if len(detected_objects) > 5:
            print(f"    ... and {len(detected_objects)-5} more")

    # Prepare inputs
    img_batch = np.expand_dims(img_array, axis=0)

    # Pad or truncate context to match CONTEXT_DIM
    if len(context_features) != CONTEXT_DIM:
        if len(context_features) < CONTEXT_DIM:
            # Pad with zeros
            padding = np.zeros(CONTEXT_DIM - len(context_features), dtype=np.float32)
            context_features = np.concatenate([context_features, padding])
        else:
            # Truncate
            context_features = context_features[:CONTEXT_DIM]

    context_batch = np.expand_dims(context_features, axis=0)

    # Encode image
    if verbose:
        print("\n[3/4] Encoding image...")

    enc_output = encoder.predict(img_batch, verbose=0)
    fused_features = context_fusion(enc_output, context_batch, training=False)

    # Generate caption
    if verbose:
        print("\n[4/4] Generating caption...")

    start_token = tokenizer.word_index.get('startseq', 1)
    end_token = tokenizer.word_index.get('endseq', 2)

    sequence = [start_token]

    for _ in range(max_length):
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            [sequence], maxlen=MAX_LEN-1, padding='post'
        )

        preds = decoder.predict([padded, fused_features], verbose=0)
        logits = preds[0, len(sequence)-1, :]
        logits = logits / temperature

        probs = tf.nn.softmax(logits).numpy()
        next_word = np.argmax(probs)

        if next_word == end_token or next_word == 0:
            break

        sequence.append(next_word)

    # Convert to caption
    caption = ' '.join([idx_to_word.get(idx, '') for idx in sequence[1:]])
    caption = caption.replace('startseq', '').replace('endseq', '').strip()

    if caption:
        caption = caption[0].upper() + caption[1:]

    if verbose:
        print("\n" + "="*80)
        print("RESULT:")
        print("="*80)
        print(f"Caption: {caption}")
        print("="*80)

    return caption

# ============================================================================
# MAIN USAGE
# ============================================================================

if __name__ == "__main__":

    print("READY FOR INFERENCE!")

    generate_caption('/content/test_image.jpg')