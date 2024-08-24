import os
import cv2
from PIL import Image
import numpy as np

def train_classifier(data_dir):

    if not os.path.exists(data_dir):
        print(f"Directory {data_dir} does not exist.")
        return

    path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]


    if len(path) == 0:
        print(f"No images found in {data_dir}.")
        return

    faces = []
    ids = []

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img, 'uint8')
        id = int(os.path.split(image)[1].split(".")[1])

        faces.append(imageNp)
        ids.append(id)

    ids = np.array(ids)


    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces, ids)
    clf.write("classifier.xml")

train_classifier("data")
