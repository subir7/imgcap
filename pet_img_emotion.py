import os
import shutil
import cv2
from pycocotools.coco import COCO
import pandas as pd
import numpy as np
import string
import math
import tensorflow as tf
from ultralytics import YOLO

def get_img(IMG_PATH):

    img = cv2.imread(IMG_PATH, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(
            f"OpenCV could not read the image: {IMG_PATH}. "
            "Check the filename, extension, and folder path."
        )

    #print("Original image shape:", img.shape)

    img_resized = cv2.resize(
        img,
        (48, 48),
        interpolation=cv2.INTER_AREA
    )

    #print("Resized image shape:", img_resized.shape)

    x = img_resized.astype("float32") / 255.0

    # (48, 48) -> (48, 48, 1)
    x = np.expand_dims(x, axis=-1)

    # (48, 48, 1) -> (1, 48, 48, 1)
    x = np.expand_dims(x, axis=0)

    prediction = img_emotion_model.predict(x, verbose=0)
    #print(prediction)

    predicted_index = int(np.argmax(prediction[0]))

    return predicted_index


def get_pet(PET_IMG_PATH):
    #img = image.load_img(PET_IMG_PATH, target_size=(224, 224))
    img = cv2.imread(PET_IMG_PATH)
    img = cv2.resize(
        img,
        (224, 224),
        interpolation=cv2.INTER_AREA
    )
    #x = image.img_to_array(img)
    img=img/255
    img = np.expand_dims(img, axis=0)
    preds = pet_emotion_model.predict(img)
    preds=np.argmax(preds, axis=1)
    #['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'suprise']
    # Return human like emotion number
    if preds == 0:
        #print("Angry")
        preds = 0
    elif preds == 1:
        #print("Happy")
        preds = 4
    elif preds == 2:
        #print("Relaxed")
        preds = 7
    elif preds == 3:
        #print("Sad")
        preds = 6
    else:
        preds = 5

    return preds

img_emotion_model = tf.keras.models.load_model('/kaggle/input/datasets/subirdas7/emotion1/image_emotion.keras')
pet_emotion_model = tf.keras.models.load_model('/kaggle/input/datasets/subirdas7/emotion1/pet_emotion.keras')

#import logging
#logging.getLogger("ultralytics").setLevel(logging.ERROR)

YOLO_MODEL = YOLO("yolo26n.pt")
#YOLO_MODEL.to('cuda')
ANIMAL_CLASSES = {
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
}


def check_animal_dirs(crop_dir):
    """Return existing animal crop directories."""

    animal_dirs = []

    if not os.path.isdir(crop_dir):
        return animal_dirs

    for animal in ANIMAL_CLASSES:
        animal_dir = os.path.join(crop_dir, animal)

        if os.path.isdir(animal_dir):
            animal_dirs.append(animal_dir)

    return animal_dirs


def get_emotion_img_pet(img_path):
    runs_dir = "/kaggle/working/runs"

    # Remove previous YOLO outputs.
    if os.path.exists(runs_dir):
        shutil.rmtree(runs_dir)

    frame = cv2.imread(img_path, cv2.IMREAD_COLOR)

    if frame is None:
        print(f"Could not read image: {img_path}")
        return 0

    results = YOLO_MODEL.predict(
        source=frame,
        classes=[0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
        conf=0.20,
        imgsz=960,
        agnostic_nms=True,
        save_crop=True,
        project="/kaggle/working/runs/detect",
        name="predict",
        exist_ok=True,
        verbose=True,
    )

    if not results:
        #print ("No object detected.")
        return 5 # If No object detect assigned emotion set to neutral 5

    # Use the actual directory returned by YOLO.
    save_dir = str(results[0].save_dir)
    crop_dir = os.path.join(save_dir, "crops")

    #print("YOLO save directory:", save_dir)
    #print("Crop directory:", crop_dir)

    if not os.path.isdir(crop_dir):
        #print("No crop directory was created.")
        return 5 # If No object detect assigned emotion set to neutral 5

    human_scores = []
    pet_scores = []

    # ---------------------------------------
    # Process person crops
    # ---------------------------------------
    person_dir = os.path.join(crop_dir, "person")

    if os.path.isdir(person_dir):
        for img_name in os.listdir(person_dir):
            person_img_path = os.path.join(person_dir, img_name)

            if not os.path.isfile(person_img_path):
                continue

            emotion_score = get_img(person_img_path)
            human_scores.append(emotion_score)

    # ---------------------------------------
    # Process animal crops
    # ---------------------------------------
    animal_dirs = check_animal_dirs(crop_dir)

    for animal_dir in animal_dirs:
        for img_name in os.listdir(animal_dir):
            animal_img_path = os.path.join(animal_dir, img_name)

            if not os.path.isfile(animal_img_path):
                continue

            pet_emotion_score = get_pet(animal_img_path)
            pet_scores.append(pet_emotion_score)

    # ---------------------------------------
    # Calculate average scores
    # ---------------------------------------
    average_human_emotion = 0
    average_pet_emotion = 0

    if human_scores:
        average_human_emotion = sum(human_scores) / len(human_scores)

    if pet_scores:
        average_pet_emotion = sum(pet_scores) / len(pet_scores)

    if human_scores and pet_scores:
        return (average_pet_emotion + average_human_emotion) / 2
    else:
      # Current behavior:
      # Return human emotion if a person exists.
      if human_scores:
          return average_human_emotion

      # Otherwise return animal emotion.
      if pet_scores:
          return average_pet_emotion

    return 5 # if No object detected assigned emotion set to Neutral 5

annotation_file = "/kaggle/working/annotations/captions_train2014.json"
IMG_DIR = (
    "/kaggle/input/datasets/jeffaudi/coco-2014-dataset-for-yolov3/coco2014/images/train2014"
)

EMOTION_LIST = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

COCO_EMOTION = []

coco = COCO(annotation_file)
image_ids = sorted(coco.getImgIds())
iterated_image_count = 0

for img_id in image_ids[:15000]:
    img = coco.loadImgs(img_id)[0]

    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns = coco.loadAnns(ann_ids)

    for ann in anns:
        caption = ann["caption"]

        caption = caption.translate(
            str.maketrans("", "", string.punctuation)
        )

        record_key = (caption, img["file_name"])

        existing_keys = [
            (row[1], row[0])
            for row in COCO_EMOTION
        ]

        if record_key not in existing_keys:
            emotion = get_emotion_img_pet(
                os.path.join(IMG_DIR, img["file_name"])
            )

            #emotion = get_emotion_img_pet(
            #    os.path.join("/kaggle/input/datasets/jeffaudi/coco-2014-dataset-for-yolov3/coco2014/images/train2014/COCO_train2014_000000222016.jpg")
            #)
            emotion =  EMOTION_LIST[math.floor(emotion + 0.5)]
            #print(emotion)
            COCO_EMOTION.append([
                img["file_name"],
                caption,
                emotion
            ])

            #print(caption)
            #print(COCO_EMOTION)

    iterated_image_count += 1
    print("Total image processed:", iterated_image_count)

    #if iterated_image_count % 1000 == 0:
        #print("Total image processed:", iterated_image_count)
        #print(COCO_EMOTION)
        #break



df = pd.DataFrame(
    COCO_EMOTION,
    columns=["filename", "caption", "emotion"]
)

df.to_csv("/kaggle/working/coco_pet_img_emotions.csv", index=False)
#print(df)

