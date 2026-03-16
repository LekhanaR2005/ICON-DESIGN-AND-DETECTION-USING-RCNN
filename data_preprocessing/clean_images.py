import os
import cv2
from PIL import Image

input_folder = "dataset_raw"
output_folder = "dataset_clean"

for root, dirs, files in os.walk(input_folder):

    # create same folder structure in output
    relative_path = os.path.relpath(root, input_folder)
    output_path = os.path.join(output_folder, relative_path)

    os.makedirs(output_path, exist_ok=True)

    for file in files:

        input_path = os.path.join(root, file)

        try:
            # verify image
            img = Image.open(input_path)
            img.verify()

            # read using opencv
            img = cv2.imread(input_path)

            if img is None:
                continue

            # resize
            img = cv2.resize(img, (224, 224))

            # convert BGR → RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            save_path = os.path.join(output_path, file)

            cv2.imwrite(save_path, img)

            print("Processed:", save_path)

        except:
            print("Skipped:", input_path)