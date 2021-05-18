import numpy as np
import cv2
from classifiers import *


classifier = DeepFake_Detector_Model()
classifier.load('weights/wt.h5')

video_path = r"C:\Users\Varun Chandra\Desktop\BDA Code\test_videos\putin.mp4"


cap = cv2.VideoCapture(video_path)
face_cascade = cv2.CascadeClassifier('face_detector.xml')


while(True):
	# Capture Frame By Frame
	ret, frame = cap.read()

	if ret == True:

		# Detect Face
		faces = face_cascade.detectMultiScale(frame, 1.1, 4)

		for (x, y, w, h) in faces: 
			# Extracting face 
			fc_img = frame[y - 20: y + h + 20, x - 20: x + w + 20]

			# Classifying the face
			fc_img = fc_img * 1./255
			fc_img = cv2.resize(fc_img, dsize = (256, 256), interpolation = cv2.INTER_CUBIC)
			fc_img = fc_img.reshape((1, 256, 256, 3))

			prediction = classifier.predict(fc_img)

			if(prediction >= 0.5):
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
				cv2.putText(frame, 'Real', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

			else:
				cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
				cv2.putText(frame, 'Fake', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)



		# Display the resulting frame

		cv2.imshow('frame',frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	else:
		break

cap.release()
cv2.destroyAllWindows()