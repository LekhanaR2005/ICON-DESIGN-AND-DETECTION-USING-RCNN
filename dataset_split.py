import os
import pandas as pd
from sklearn.model_selection import train_test_split


# 1. Dataset Folder Path (uploaded folder)

dataset_path = "dataset_augmented"   # path to your uploaded dataset folder


# 2. Read Dataset Structure

classes = sorted(os.listdir(dataset_path))

print("Total classes:", len(classes))

data = []

# loop through folders
for class_id, class_name in enumerate(classes):

    class_folder = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_folder):
        continue

    for img in os.listdir(class_folder):

        img_path = os.path.join(class_folder, img)

        data.append({
            "image_path": img_path,
            "class_name": class_name,
            "class_id": class_id
        })

# convert to dataframe
df = pd.DataFrame(data)

print("Total images:", len(df))
print(df.head())


# 3. Train / Val / Test Split

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["class_id"],
    random_state=42
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["class_id"],
    random_state=42
)

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# 4. Save Annotation Files

os.makedirs("dataset_final", exist_ok=True)

train_df.to_csv("dataset_final/train_labels.csv", index=False)
val_df.to_csv("dataset_final/val_labels.csv", index=False)
test_df.to_csv("dataset_final/test_labels.csv", index=False)

print("Annotation files saved!")


# 5. Save Class Mapping

class_mapping = {i: name for i, name in enumerate(classes)}

mapping_df = pd.DataFrame(list(class_mapping.items()),
                          columns=["class_id", "class_name"])

mapping_df.to_csv("dataset_final/class_mapping.csv", index=False)

print("Class mapping saved!")

print("Dataset preparation completed successfully!")