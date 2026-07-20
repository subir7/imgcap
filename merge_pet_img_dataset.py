import pandas as pd


PET_IMG_EMOTION_CSV = "/kaggle/input/notebooks/trdas911/pet-img-emotion/coco_pet_img_emotions.csv"
CONTEXT_CSV = "/kaggle/input/datasets/trdas911/context/context.csv"

OUTPUT_CSV = "/kaggle/working/pet_img_emotion_with_context.csv"


# Load CSVs
pet_img_df = pd.read_csv(PET_IMG_EMOTION_CSV)
context_df = pd.read_csv(CONTEXT_CSV)

# Check duplicates in context.csv
duplicates = context_df["image_name"].duplicated().sum()

if duplicates > 0:
    raise ValueError(
        f"context.csv contains {duplicates} duplicate image_name values."
    )

# Merge
merged_df = pet_img_df.merge(
    context_df,
    left_on="filename",
    right_on="image_name",
    how="left"
)

# Remove duplicate filename column from context.csv
merged_df.drop(columns=["image_name"], inplace=True)

# -----------------------------
# Check missing matches
# -----------------------------
missing = merged_df.iloc[:, 3:].isna().all(axis=1).sum()

print(f"Rows in pet_img_emotion.csv : {len(pet_img_df)}")
print(f"Rows after merge           : {len(merged_df)}")
print(f"Rows without context       : {missing}")


merged_df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved to:\n{OUTPUT_CSV}")
