import cv2
import numpy as np
import os
import shutil
import time
import pickle

face_cascade = cv2.CascadeClassifier("E:/scikit/haarcascade_frontalface_alt2.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainning.yml")

# load label
labels = {}
with open("labels.pickle", "rb") as f:
	labels = pickle.load(f)
	labels = {v:k for k, v in labels.items()}
cap = cv2.VideoCapture(0)

while True:
	# capture frame by frame
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for x,y,w,h in faces:
		# print(x,y,w,h)
		
		roi_gray = gray[y:y+h, x:x+w]
		roi_color = frame[:y:y+h, x:x+w]
		
		id_, conf = recognizer.predict(roi_gray)
		if conf >= 45 and conf <= 85:
			print(id_)
			print(labels[id_])
			font = cv2.FONT_HERSHEY_SIMPLEX
			name = labels[id_]
			color = (0, 255, 0)
			stroke = 2
			cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)


		# img_item = "new.png"
		# cv2.imwrite(roi_gray)
		
		color = (255, 0, 0) #BGR
		stroke = 2
		end_cord_x = x + w
		end_cord_y = y + h
		cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
	# Display the result frame
	cv2.imshow('frame', frame)
	
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break



cap.release()
cv2.destroyAllWindows()