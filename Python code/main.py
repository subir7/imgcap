from google.colab import drive
drive.mount('/content/drive')

!unzip "/content/drive/MyDrive/MSCOCO/Flicker8k_Dataset.zip" -d "/content"

import os
import re
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
import pickle

# ---------------- HYPERPARAMETERS ----------------
IMG_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2

EMBED_DIM = 256
NUM_HEADS = 4
MLP_DIM = 512

ENCODER_LAYERS = 6
DECODER_LAYERS = 6

MAX_LEN = 40
BATCH_SIZE = 8
EPOCHS_STAGE1 = 5 #20
EPOCHS_STAGE2 = 4 #15

DROPOUT = 0.1

# ============================================================================
# 180D MSCOCO FEATURE DIMENSIONS
# ============================================================================
# Complete MSCOCO features:
# - 80D: Object detection (person, car, dog, etc.)
# - 91D: Stuff detection (sky, grass, water, beach, etc.)
# - 9D: Scene statistics
MSCOCO_OBJECTS_DIM = 80
MSCOCO_STUFF_DIM = 91
SCENE_STATS_DIM = 9

# Total context dimension
CONTEXT_DIM = MSCOCO_OBJECTS_DIM + MSCOCO_STUFF_DIM + SCENE_STATS_DIM  # 180


print("MODEL ARCHITECTURE - 180D MSCOCO FEATURES")

print(f"Input 1: Image ({IMG_SIZE}x{IMG_SIZE}x3)")
print(f"Input 2: Context vector ({CONTEXT_DIM}D)")
print(f"  - Objects: {MSCOCO_OBJECTS_DIM}D (person, car, dog, ...)")
print(f"  - Stuff: {MSCOCO_STUFF_DIM}D (sky, grass, water, beach, ...)")
print(f"  - Scene stats: {SCENE_STATS_DIM}D")

# ---------------- PATHS ----------------
CAPTION_FILE = "/content/Flickr8k_text/Flickr8k.token.txt"
IMG_DIR = "/content/Flicker8k_Dataset/"

# NEW: 180D MSCOCO features CSV
MSCOCO_FEATURES_FILE = "/content/drive/MyDrive/MSCOCO/trainingdata/mscoco_object_stuff_detection.csv"

CHECKPOINT_DIR = "/content/drive/MyDrive/MSCOCO/trainingdata/checkpoints"
STAGE1_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "stage1_latest.weights.h5")
STAGE2_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "stage2_latest.weights.h5")
EPOCH_TRACKER = os.path.join(CHECKPOINT_DIR, "training_progress.pkl")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_training_progress(stage, epoch, history_dict):
    progress = {
        'stage': stage,
        'completed_epochs': epoch,
        'history': history_dict
    }
    with open(EPOCH_TRACKER, 'wb') as f:
        pickle.dump(progress, f)
    print(f"\n Saved progress: Stage {stage}, Epoch {epoch}")

def load_training_progress():
    if os.path.exists(EPOCH_TRACKER):
        with open(EPOCH_TRACKER, 'rb') as f:
            progress = pickle.load(f)
        print(f"\n Found previous training:")
        print(f"  Stage: {progress['stage']}")
        print(f"  Completed epochs: {progress['completed_epochs']}")
        return progress
    return None

def get_latest_checkpoint(stage):
    if stage == 1:
        if os.path.exists(STAGE1_CHECKPOINT):
            return STAGE1_CHECKPOINT
    elif stage == 2:
        if os.path.exists(STAGE2_CHECKPOINT):
            return STAGE2_CHECKPOINT
    return None

# ---------------- LOAD DATA ----------------
print("LOADING DATA")

# Load captions
df = pd.read_csv(CAPTION_FILE, sep='\t', header=None, names=["image_id", "caption"])
df["image_id"] = df["image_id"].apply(lambda x: x.split("#")[0])
df = df.rename(columns={"image_id": "filename"})
print(f"Loaded {len(df)} captions")

# ============================================================================
# LOAD 180D MSCOCO FEATURES
# ============================================================================

print("LOADING 180D MSCOCO FEATURES")

if not os.path.exists(MSCOCO_FEATURES_FILE):
    print(f" ERROR: MSCOCO features file not found!")
    print(f"   Expected: {MSCOCO_FEATURES_FILE}")
    print("\nYou need to run the feature extraction script first:")
    print("   python mscoco_complete_extractor.py")
    print("\nThis will generate the 180D features CSV file.")
    exit(1)

print(f"Loading from: {MSCOCO_FEATURES_FILE}")
mscoco_df = pd.read_csv(MSCOCO_FEATURES_FILE)

print(f"  Loaded MSCOCO features")
print(f"  Rows: {len(mscoco_df)}")
print(f"  Columns: {len(mscoco_df.columns)}")

# Expected format: filename, feat_0, feat_1, ..., feat_179
# Total: 181 columns (1 filename + 180 features)

if len(mscoco_df.columns) < 181:
    print(f"\n WARNING: Expected 181 columns (1 filename + 180 features)")
    print(f"   Found: {len(mscoco_df.columns)} columns")
    print(f"   Will pad with zeros if needed")

# Extract features into dictionary
mscoco_data = {}

for _, row in mscoco_df.iterrows():
    filename = row[0] if isinstance(row[0], str) else row['filename']

    # Extract 180 features (columns 1-180)
    if len(row) >= 181:
        features_180d = row[1:181].values.astype(np.float32).tolist()
    else:
        # Pad if needed
        available_features = row[1:].values.astype(np.float32).tolist()
        padding = [0.0] * (CONTEXT_DIM - len(available_features))
        features_180d = available_features + padding

    mscoco_data[filename] = features_180d

print(f"Processed {len(mscoco_data)} images")

# Sample verification
sample_file = list(mscoco_data.keys())[0]
sample_features = mscoco_data[sample_file]
print(f"\nSample verification:")
print(f"  Filename: {sample_file}")
print(f"  Feature vector length: {len(sample_features)}")
print(f"  Non-zero features: {np.count_nonzero(sample_features)}")
print(f"  Feature range: [{min(sample_features):.3f}, {max(sample_features):.3f}]")

# ============================================================================
# CREATE CONTEXT VECTORS
# ============================================================================

print("CREATING CONTEXT VECTORS")

def create_context_vector(filename):
    """
    Get 180D MSCOCO features as context vector

    Returns: 180D vector
      - [0:80] = Object features (person, car, dog, ...)
      - [80:171] = Stuff features (sky, grass, water, beach, ...)
      - [171:180] = Scene statistics
    """
    return mscoco_data.get(filename, [0.0] * CONTEXT_DIM)

# Add context vector to dataframe
df["context"] = df["filename"].apply(create_context_vector)

# Verify context vectors
print(f"Context vectors created: {len(df)}")
print(f"Context dimension: {len(df.iloc[0]['context'])}")

# Filter complete data
initial_count = len(df)
df = df[df["context"].apply(lambda x: len(x) == CONTEXT_DIM)]
final_count = len(df)

print(f"\nData filtering:")
print(f"  Initial captions: {initial_count}")
print(f"  Complete data: {final_count}")
print(f"  Filtered out: {initial_count - final_count}")

if final_count == 0:
    print("\n ERROR: No valid data found!")
    print("   Check that filenames match between caption file and MSCOCO features CSV")
    exit(1)

# Text preprocessing
print("\n" + "="*80)
print("PREPROCESSING CAPTIONS")
print("="*80)

def clean_caption(c):
    c = c.lower()
    c = re.sub(r"[^a-z0-9 ]", "", c)
    return "startseq " + c.strip() + " endseq"

df["caption"] = df["caption"].apply(clean_caption)

tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<unk>")
tokenizer.fit_on_texts(df["caption"].values)

word_index = tokenizer.word_index
VOCAB_SIZE = len(word_index) + 1

seqs = tokenizer.texts_to_sequences(df["caption"])
seqs = tf.keras.preprocessing.sequence.pad_sequences(seqs, maxlen=MAX_LEN, padding="post")
df["seq"] = seqs.tolist()

print(f"Vocabulary size: {VOCAB_SIZE}")
print(f"Caption sequences created")

# Sample caption
sample_idx = 0
print(f"\nSample caption:")
print(f"  Original: {df.iloc[sample_idx]['caption']}")
print(f"  Sequence length: {len(df.iloc[sample_idx]['seq'])}")

# Image loader
def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        arr = np.array(img).astype(np.float32) / 255.0
        return arr
    except Exception as e:
        return None

# ============================================================================
# DATA GENERATOR - 2 inputs + sequence
# ============================================================================

def data_generator():
    """Yields ((image, context), seq_target)"""
    for _, row in df.iterrows():
        filename = row["filename"]
        img_path = os.path.join(IMG_DIR, filename)

        if not os.path.exists(img_path) and img_path.endswith(".jpg.1"):
            candidate = img_path.replace(".jpg.1", ".jpg")
            if os.path.exists(candidate):
                img_path = candidate

        if not os.path.exists(img_path):
            continue

        img = load_image(img_path)
        if img is None:
            continue

        seq = np.array(row["seq"], dtype=np.int32)
        inp = seq[:-1]
        out = seq[1:]

        if inp.shape[0] != MAX_LEN - 1 or out.shape[0] != MAX_LEN - 1:
            continue

        context = np.array(row["context"], dtype=np.float32)

        yield (img, context, inp), out

output_signature = (
    (tf.TensorSpec(shape=(IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
     tf.TensorSpec(shape=(CONTEXT_DIM,), dtype=tf.float32),
     tf.TensorSpec(shape=(MAX_LEN-1,), dtype=tf.int32)),
    tf.TensorSpec(shape=(MAX_LEN-1,), dtype=tf.int32)
)

dataset = tf.data.Dataset.from_generator(data_generator, output_signature=output_signature)
dataset = dataset.shuffle(2048).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

print("\n Dataset pipeline created")

# ---------------- MODEL COMPONENTS ----------------

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

def build_vit_encoder():
    img = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    patches = PatchExtract(PATCH_SIZE)(img)
    x = PatchEmbedding(NUM_PATCHES, EMBED_DIM)(patches)

    for _ in range(ENCODER_LAYERS):
        x = EncoderBlock(EMBED_DIM, NUM_HEADS, MLP_DIM)(x)

    return tf.keras.Model(img, x, name="vit_encoder")

class ContextFusion(layers.Layer):
    """Fuses 180D MSCOCO context with visual features"""

    def __init__(self, embed_dim):
        super().__init__()
        self.context_proj = layers.Dense(embed_dim, name="context_projection")
        self.dropout = layers.Dropout(DROPOUT)
        self.ln = layers.LayerNormalization(epsilon=1e-6)

    def call(self, enc_out, context, training=False):
        """
        Args:
            enc_out: (batch, NUM_PATCHES, embed_dim) - visual features
            context: (batch, 180) - MSCOCO features (80 objects + 91 stuff + 9 stats)

        Returns:
            (batch, NUM_PATCHES + 1, embed_dim) - fused features
        """
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
            layers.Dropout(DROPOUT),
            layers.Dense(embed_dim)
        ])
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ln3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(DROPOUT)
        self.dropout2 = layers.Dropout(DROPOUT)

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

# ============================================================================
# BUILD MODEL
# ============================================================================

print("BUILDING MODEL")

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

print(f"\n Model created")
print(f"  Encoder: {ENCODER_LAYERS} layers")
print(f"  Decoder: {DECODER_LAYERS} layers")
print(f"  Context: {CONTEXT_DIM}D (180D MSCOCO)")
print(f"  Vocabulary: {VOCAB_SIZE} words")

model.summary()

# ============================================================================
# CHECKPOINT SYSTEM
# ============================================================================

print("CHECKING FOR PREVIOUS TRAINING")

progress = load_training_progress()

if progress is None:
    print("No previous training found. Starting from scratch.")
    current_stage = 1
    stage1_start_epoch = 0
    stage2_start_epoch = 0
    training_history = {'stage1': {}, 'stage2': {}}
else:
    current_stage = progress['stage']
    completed_epochs = progress['completed_epochs']
    training_history = progress.get('history', {'stage1': {}, 'stage2': {}})

    if current_stage == 1:
        stage1_start_epoch = completed_epochs
        stage2_start_epoch = 0

        if stage1_start_epoch >= EPOCHS_STAGE1:
            current_stage = 2
            stage1_start_epoch = EPOCHS_STAGE1
            stage2_start_epoch = 0
        else:
            print(f"Resuming Stage 1 from epoch {stage1_start_epoch}/{EPOCHS_STAGE1}")
    else:
        stage1_start_epoch = EPOCHS_STAGE1
        stage2_start_epoch = completed_epochs

        if stage2_start_epoch >= EPOCHS_STAGE2:
            print(f"Training already complete!")
            exit(0)
        else:
            print(f"Resuming Stage 2 from epoch {stage2_start_epoch}/{EPOCHS_STAGE2}")

# ============================================================================
# STAGE 1: TRAIN DECODER ONLY
# ============================================================================

if current_stage == 1 and stage1_start_epoch < EPOCHS_STAGE1:
    print("\n" + "="*80)
    print(f"STAGE 1: DECODER TRAINING")
    print("="*80)

    encoder.trainable = False
    for layer in encoder.layers:
        layer.trainable = False

    enc_out = encoder(img_inp)
    fused_out = context_fusion(enc_out, context_inp)
    dec_out = decoder([seq_inp, fused_out])
    model = tf.keras.Model([img_inp, context_inp, seq_inp], dec_out)

    if stage1_start_epoch > 0:
        checkpoint = get_latest_checkpoint(1)
        if checkpoint:
            model.load_weights(checkpoint)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    class EpochCheckpoint(tf.keras.callbacks.Callback):
        def __init__(self, stage, start_epoch):
            super().__init__()
            self.stage = stage
            self.start_epoch = start_epoch

        def on_epoch_end(self, epoch, logs=None):
            actual_epoch = self.start_epoch + epoch + 1

            if self.stage == 1:
                self.model.save_weights(STAGE1_CHECKPOINT)
            else:
                self.model.save_weights(STAGE2_CHECKPOINT)

            save_training_progress(self.stage, actual_epoch, training_history)

    epoch_checkpoint = EpochCheckpoint(stage=1, start_epoch=stage1_start_epoch)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-7)

    remaining_epochs = EPOCHS_STAGE1 - stage1_start_epoch

    history1 = model.fit(
        dataset,
        epochs=remaining_epochs,
        callbacks=[epoch_checkpoint, early_stop, reduce_lr],
        verbose=1
    )

    for key, values in history1.history.items():
        if key not in training_history['stage1']:
            training_history['stage1'][key] = []
        training_history['stage1'][key].extend(values)

    current_stage = 2
    stage2_start_epoch = 0

# ============================================================================
# STAGE 2: FINE-TUNE ENTIRE MODEL
# ============================================================================

if current_stage == 2 and stage2_start_epoch < EPOCHS_STAGE2:
    print("\n" + "="*80)
    print(f"STAGE 2: FINE-TUNING")
    print("="*80)

    encoder.trainable = True
    for layer in encoder.layers:
        layer.trainable = True

    enc_out = encoder(img_inp)
    fused_out = context_fusion(enc_out, context_inp)
    dec_out = decoder([seq_inp, fused_out])
    model = tf.keras.Model([img_inp, context_inp, seq_inp], dec_out)

    if stage2_start_epoch == 0:
        checkpoint = get_latest_checkpoint(1)
        if checkpoint:
            model.load_weights(checkpoint)
    else:
        checkpoint = get_latest_checkpoint(2)
        if checkpoint:
            model.load_weights(checkpoint)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    epoch_checkpoint = EpochCheckpoint(stage=2, start_epoch=stage2_start_epoch)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-7)

    remaining_epochs = EPOCHS_STAGE2 - stage2_start_epoch

    history2 = model.fit(
        dataset,
        epochs=remaining_epochs,
        callbacks=[epoch_checkpoint, early_stop, reduce_lr],
        verbose=1
    )

    for key, values in history2.history.items():
        if key not in training_history['stage2']:
            training_history['stage2'][key] = []
        training_history['stage2'][key].extend(values)

# ============================================================================
# SAVE FINAL MODEL
# ============================================================================

print("SAVING FINAL MODEL")

final_weights = "/content/drive/MyDrive/MSCOCO/trainingdata/final_model.weights.h5"
model.save_weights(final_weights)

tokenizer_path = "/content/drive/MyDrive/MSCOCO/trainingdata/tokenizer.pkl"
with open(tokenizer_path, 'wb') as f:
    pickle.dump(tokenizer, f)

config = {
    'IMG_SIZE': IMG_SIZE,
    'PATCH_SIZE': PATCH_SIZE,
    'NUM_PATCHES': NUM_PATCHES,
    'EMBED_DIM': EMBED_DIM,
    'NUM_HEADS': NUM_HEADS,
    'MLP_DIM': MLP_DIM,
    'ENCODER_LAYERS': ENCODER_LAYERS,
    'DECODER_LAYERS': DECODER_LAYERS,
    'MAX_LEN': MAX_LEN,
    'CONTEXT_DIM': CONTEXT_DIM,
    'MSCOCO_OBJECTS_DIM': MSCOCO_OBJECTS_DIM,
    'MSCOCO_STUFF_DIM': MSCOCO_STUFF_DIM,
    'SCENE_STATS_DIM': SCENE_STATS_DIM,
    'VOCAB_SIZE': VOCAB_SIZE
}

config_path = "/content/drive/MyDrive/MSCOCO/trainingdata/model_config.pkl"
with open(config_path, 'wb') as f:
    pickle.dump(config, f)

history_path = "/content/drive/MyDrive/MSCOCO/trainingdata/training_history.pkl"
with open(history_path, 'wb') as f:
    pickle.dump(training_history, f)

print(f"Model weights: {final_weights}")
print(f"Tokenizer: {tokenizer_path}")
print(f"Configuration: {config_path}")
print(f"Training history: {history_path}")

print("\n" + "="*80)
print("TRAINING COMPLETE!")
print("="*80)
print(f"Architecture: Image + 180D MSCOCO Context")
print(f"  • 80D: Objects (person, car, dog, ...)")
print(f"  • 91D: Stuff (sky, grass, water, beach, ...)")
print(f"  • 9D: Scene statistics")
print(f"Total epochs: {EPOCHS_STAGE1 + EPOCHS_STAGE2}")
print("="*80)



# Save the entire model as a `.keras` zip archive.
model.save('/content/drive/MyDrive/MSCOCO/trainingdata/final_model.keras')
model.save('/content/drive/MyDrive/MSCOCO/trainingdata/final_model.h5')







