# Multimodal Image Captioning: Code Workflow

This repository contains the implementation of a multimodal image captioning framework developed for a Master's thesis. The system combines visual, contextual, and emotional information to generate image captions.

The complete workflow includes:

- scene-context extraction;
- human and pet emotion recognition;
- multimodal feature preparation;
- caption-model training;
- quantitative evaluation;
- caption generation for unseen images.

---

## Overall Workflow

```text
MS COCO images and captions
          │
          ├── scene_understanding.py
          │       └── 180-D context features
          │
          ├── ferplus.py
          │       └── trained human emotion model
          │
          ├── pet_emotion_recognition.py
          │       └── trained pet emotion model
          │
          ├── pet_img_emotion.py
          │       └── image-level emotion labels
          │
          └── merge_pet_img_dataset.py
                  └── captions + emotion + context
                              │
                              ▼
                    main.py + model.py
                              │
                              ▼
                     Four caption models
                              │
                              ├── evaluation.py
                              │       └── metric scores
                              │
                              └── inference.py
                                      └── captions for a new image
```

---

## 1. Human Emotion Model: `ferplus.py`

This file trains the human facial emotion classifier.

### Step 1: Load the FERPlus dataset

The training, validation, and test folders are loaded.

### Step 2: Preprocess the images

Each image is:

- converted to grayscale;
- resized to `48 × 48`;
- normalized by dividing pixel values by `255`.

Training images are also augmented using:

- rotation;
- width and height shifting;
- zooming;
- shearing;
- horizontal flipping.

### Step 3: Build the CNN model

The model starts with a convolution layer and then uses several residual blocks. Each residual block contains convolution layers and a shortcut connection.

The number of filters gradually increases from `64` to `128` and then to `256`.

```text
Input image
    ↓
Convolution + Batch Normalization + ReLU
    ↓
Residual blocks
    ↓
Global Average Pooling
    ↓
Dense layer
    ↓
Dropout
    ↓
Softmax emotion output
```

### Step 4: Train the model

The model is trained using:

- Adam optimizer;
- categorical cross-entropy loss;
- early stopping;
- learning-rate reduction;
- model checkpointing.

### Step 5: Evaluate the model

The final model is evaluated using:

- accuracy;
- classification report;
- confusion matrix.

---

## 2. Pet Emotion Model: `pet_emotion_recognition.py`

This file trains the pet facial-expression classifier.

### Step 1: Load MobileNetV2

A pretrained MobileNetV2 model is loaded without its original classification layer.

ImageNet weights are used, and the MobileNetV2 base layers are frozen during training.

### Step 2: Add classification layers

```text
MobileNetV2
    ↓
Flatten
    ↓
Dense 100
    ↓
Dense 70
    ↓
Dense 40
    ↓
Softmax output
```

### Step 3: Use four pet emotion classes

```text
Angry
Happy
Relaxed
Sad
```

### Step 4: Preprocess pet images

Each image is:

- resized to `224 × 224`;
- normalized;
- augmented using shearing, zooming, and horizontal flipping.

### Step 5: Train and save the model

The model is trained with early stopping and saved in both `.h5` and `.keras` formats.

---

## 3. Scene-Context Extraction: `scene_understanding.py`

This file creates context features for MS COCO images.

### Step 1: Load image names and captions

The code reads the MS COCO 2014 annotation JSON file and links every image ID with its filename and captions.

### Step 2: Extract the 80-D object vector

YOLOv8 detects objects from the 80 MS COCO object classes.

For each detected class, its count is recorded. The counts are normalized so that the values are between `0` and `1`.

```text
YOLOv8 detections
        ↓
80-D object vector
```

### Step 3: Extract the 91-D stuff vector

SegFormer performs semantic segmentation.

Its ADE20K labels are mapped to MS COCO-Stuff categories.

For each mapped category, the code calculates how much of the image is covered by that category.

```text
SegFormer segmentation
        ↓
ADE20K labels
        ↓
COCO-Stuff mapping
        ↓
91-D stuff vector
```

### Step 4: Calculate nine scene statistics

The code calculates nine additional values that describe the image, including:

- number of detected object classes;
- number of detected stuff classes;
- mean values;
- standard deviations;
- maximum values;
- combined scene density.

### Step 5: Create the final context vector

```text
80 object features
+ 91 stuff features
+ 9 scene statistics
= 180-D context vector
```

The final context features are saved in `context.csv`.

---

## 4. Image-Level Emotion Extraction: `pet_img_emotion.py`

This file assigns an emotion to each MS COCO image.

### Step 1: Detect people and animals

YOLO detects:

- people;
- birds;
- cats;
- dogs;
- horses;
- sheep;
- cows;
- elephants;
- bears;
- zebras;
- giraffes.

Detected regions are saved as separate image crops.

### Step 2: Predict human emotion

Each person crop is:

```text
converted to grayscale
→ resized to 48 × 48
→ normalized
→ passed to the FERPlus model
```

The human model predicts one of eight emotion classes.

### Step 3: Predict pet emotion

Each animal crop is:

```text
resized to 224 × 224
→ normalized
→ passed to the pet emotion model
```

The four pet classes are mapped to the common emotion classes.

```text
Pet angry   → angry
Pet happy   → happy
Pet relaxed → surprise
Pet sad     → sad
```

### Step 4: Combine predictions

When an image contains several people or animals, their emotion predictions are combined.

If both humans and animals are detected, the human and pet predictions are merged.

If no person or animal is detected, the image is assigned the `neutral` emotion.

### Step 5: Save results

The generated CSV contains:

```text
filename
caption
emotion
```

---

## 5. Merge Context and Emotion Data: `merge_pet_img_dataset.py`

This file joins the image-level emotion CSV with `context.csv`.

The merge is performed using:

```text
emotion CSV: filename
context CSV: image_name
```

After merging, each row contains:

```text
filename
caption
emotion
f_0 ... f_179
```

The script also checks for:

- duplicate image names;
- missing context features;
- unsuccessful matches.

---

## 6. Captioning Architecture: `model.py`

This file defines the shared image-captioning architecture.

### Step 1: Prepare the input image

The input image size is:

```text
224 × 224 × 3
```

### Step 2: Create ViT patches

The Vision Transformer divides the image into `16 × 16` patches.

```text
224 / 16 = 14 patches per side
14 × 14 = 196 patches
```

Each patch is converted into a `256`-dimensional embedding.

Therefore, the ViT output is:

```text
196 × 256
```

### Step 3: Process visual features

The Transformer encoder uses self-attention to learn relationships between the image patches.

### Step 4: Add context and emotion features

The four experiments use different input combinations:

```text
Image Only
Image + Context
Image + Emotion
Image + Context + Emotion
```

The context and emotion features are projected into the same feature space before fusion.

### Step 5: Generate captions

The Transformer decoder uses:

- masked self-attention for the partial caption;
- cross-attention for the image-related features;
- a vocabulary output layer for next-word prediction.

---

## 7. Caption-Model Training: `main.py`

This file controls the full caption-model training process.

### Step 1: Load the merged dataset

The merged CSV containing captions, emotion labels, and context features is loaded.

### Step 2: Clean captions

Captions are converted to lowercase and surrounded by special tokens.

```text
startseq a dog is running endseq
```

### Step 3: Create the tokenizer

The tokenizer builds the vocabulary and converts words into integer IDs.

```text
startseq → 2
dog      → 47
running  → 193
endseq   → 3
```

Captions are padded to a fixed maximum length.

### Step 4: Prepare context features

The original `180-D` context vectors are reduced using PCA.

The reduced features are scaled and stored as `40-D` context vectors.

The PCA model and scaler are saved for inference.

### Step 5: Prepare emotion features

Emotion labels are converted into eight-dimensional one-hot vectors.

```text
happy → [0, 0, 0, 0, 1, 0, 0, 0]
sad   → [0, 0, 0, 0, 0, 0, 1, 0]
```

### Step 6: Create caption training samples

Caption generation is trained autoregressively.

For example:

```text
Full caption:
startseq a dog runs endseq
```

Training pairs:

```text
Input: startseq
Target: a

Input: startseq a
Target: dog

Input: startseq a dog
Target: runs
```

Padding tokens are ignored during loss calculation.

### Step 7: Train four models

The same ViT and Transformer decoder structure is used for all experiments.

Only the input modalities are changed.

The trained models are saved in separate folders.

Supporting files are also saved:

```text
tokenizer.pkl
config.pkl
context_pca.pkl
context_scaler.pkl
emotion_to_index.pkl
```

---

## 8. Model Evaluation: `evaluation.py`

This file evaluates the trained caption models.

### Step 1: Load each model

The four trained models and their supporting files are loaded.

### Step 2: Generate captions

Beam search is used instead of selecting only the most likely word at every step.

Beam search keeps several possible caption sequences and selects the strongest final sequence.

### Step 3: Compare with reference captions

Generated captions are compared with the original MS COCO captions using:

- BLEU-1;
- BLEU-2;
- BLEU-3;
- BLEU-4;
- METEOR;
- ROUGE-L.

The same evaluation method is used for all four experiments.

---

## 9. Inference on an Unseen Image: `inference.py`

This file runs the complete framework on a new image.

### Step 1: Load saved components

The script loads:

- tokenizer;
- configuration;
- PCA model;
- context scaler;
- emotion mapping;
- caption models;
- YOLO;
- SegFormer;
- human emotion model;
- pet emotion model.

### Step 2: Extract context features

For the new image, the script creates:

```text
80-D object vector
+ 91-D stuff vector
+ 9-D scene statistics
= 180-D raw context
```

The raw vector is transformed into the context format used during training.

### Step 3: Extract emotion features

YOLO detects people and animals.

The human and pet models predict their emotions, and the results are converted into an eight-dimensional emotion vector.

### Step 4: Prepare the image

The image is:

- decoded;
- resized;
- converted to floating point;
- normalized.

### Step 5: Generate a caption

Caption generation begins with:

```text
startseq
```

At each step, the decoder predicts the next word.

Beam search keeps the strongest caption candidates until:

- `endseq` is generated; or
- the maximum caption length is reached.

### Step 6: Generate captions from all experiments

The inference process is repeated for:

- Image Only;
- Image + Context;
- Image + Emotion;
- Image + Context + Emotion.

The generated captions can then be saved in a CSV file.

---

## Experimental Models

| Model | Visual Features | Context Features | Emotion Features |
|---|---:|---:|---:|
| Image Only | Yes | No | No |
| Image + Context | Yes | Yes | No |
| Image + Emotion | Yes | No | Yes |
| Image + Context + Emotion | Yes | Yes | Yes |

---

## Main Feature Dimensions

| Feature | Dimension |
|---|---:|
| Input image | `224 × 224 × 3` |
| ViT patch embeddings | `196 × 256` |
| YOLO object vector | `80-D` |
| SegFormer stuff vector | `91-D` |
| Scene statistics | `9-D` |
| Raw context vector | `180-D` |
| PCA context vector | `40-D` |
| Emotion vector | `8-D` |

---

## Evaluation Metrics

The caption models are evaluated using:

- BLEU-1;
- BLEU-2;
- BLEU-3;
- BLEU-4;
- METEOR;
- ROUGE-L.

---


## Summary

The repository implements a complete multimodal image captioning workflow. It combines:

- ViT visual features;
- YOLOv8 and SegFormer context features;
- human and pet emotion features;
- Transformer-based caption generation.

The four experiments make it possible to measure how context and emotion information affect the quality of generated captions.
