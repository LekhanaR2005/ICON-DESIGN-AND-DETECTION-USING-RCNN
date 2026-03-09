import os
import cv2
from PIL import Image

input_folder = "dataset_raw"
output_folder = "dataset_clean"

for root, dirs, files in os.walk(input_folder):

    for file in files:

        input_path = os.path.join(root, file)

        # get class folder name
        class_name = os.path.basename(root)

        # create class folder in dataset_clean
        class_output = os.path.join(output_folder, class_name)
        os.makedirs(class_output, exist_ok=True)

        try:
            img = Image.open(input_path)
            img.verify()

            img = cv2.imread(input_path)

            if img is None:
                continue

            # resize
            img = cv2.resize(img, (224, 224))

            # convert BGR → RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            output_path = os.path.join(class_output, file)

            cv2.imwrite(output_path, img)

            print("Processed:", class_name, "/", file)

        except:
            print("Skipped:", file)