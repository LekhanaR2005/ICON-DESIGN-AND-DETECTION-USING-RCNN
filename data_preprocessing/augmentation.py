import os
import cv2
import albumentations as A

# Input dataset folder
input_folder = r"dataset_clean"

# Output folder
output_folder = r"dataset_augmented"

os.makedirs(output_folder, exist_ok=True)

# Augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=25, p=0.7),
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(p=0.3),
    A.Blur(blur_limit=3, p=0.3),
    A.RandomScale(scale_limit=0.2, p=0.5)
])

num_augmented = 3

# Read category folders
for category in os.listdir(input_folder):

    category_path = os.path.join(input_folder, category)

    if not os.path.isdir(category_path):
        continue

    save_category = os.path.join(output_folder, category)
    os.makedirs(save_category, exist_ok=True)

    for image_name in os.listdir(category_path):

        image_path = os.path.join(category_path, image_name)
        image = cv2.imread(image_path)

        if image is None:
            continue

        for i in range(num_augmented):

            augmented = transform(image=image)
            aug_image = augmented["image"]

            new_name = f"{image_name.split('.')[0]}_aug_{i}.png"
            save_path = os.path.join(save_category, new_name)

            cv2.imwrite(save_path, aug_image)

print("Data Augmentation Completed!")