import os
import numpy as np
from PIL import Image
import cv2
# from cv2 import LBPHFaceRecognizer_create
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "image")

#face cascade
face_cascade = cv2.CascadeClassifier("E:/scikit/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}
x_train = []
y_label = []


for root, dirs, files in os.walk(image_dir):
	for file in files:
		if file.endswith("png") or file.endswith("jpg"):
			path = os.path.join(root, file)
			label = os.path.basename(os.path.dirname(path)).replace(" ","-").lower()
			# print(label)
			if not label in label_ids:
				label_ids[label] = current_id
				current_id += 1
			id_ = label_ids[label]
			# print(label_ids)
			

			# x_train.append(path)
			# y_label.append(label) # dirname name is your label..we need to convert it an numpy array, Gray
			pil_image = Image.open(path).convert('L')
			image_array = np.array(pil_image, 'uint8')
			# print(image_array)
			faces = face_cascade.detectMultiScale(image_array, 1.3, 5)

			for (x,y,w,h) in faces:
				roi = image_array[y:y+h, x:x+w]
				x_train.append(roi)
				y_label.append(id_)

# print(y_label)
# print(x_train)


with open("labels.pickle", "wb") as f:
	pickle.dump(label_ids, f)
recognizer.train(x_train, np.array(y_label))
recognizer.save("trainning.yml")
