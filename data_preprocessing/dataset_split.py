import os
import shutil
from sklearn.model_selection import train_test_split

dataset_path = "dataset_augmented"

train_dir = "dataset_split/train"
test_dir = "dataset_split/test"

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

classes = sorted(os.listdir(dataset_path))

for class_name in classes:

    class_folder = os.path.join(dataset_path, class_name)

    if not os.path.isdir(class_folder):
        continue

    images = os.listdir(class_folder)

    # 50-50 split
    train_imgs, test_imgs = train_test_split(images, test_size=0.5, random_state=42)

    # create class folders
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # copy train images
    for img in train_imgs:
        src = os.path.join(class_folder, img)
        dst = os.path.join(train_dir, class_name, img)
        shutil.copy(src, dst)

    # copy test images
    for img in test_imgs:
        src = os.path.join(class_folder, img)
        dst = os.path.join(test_dir, class_name, img)
        shutil.copy(src, dst)

print("Dataset successfully split into train and test folders.")