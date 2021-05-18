import numpy as np
import cv2
from classifiers import *


classifier = DeepFake_Detector_Model()
classifier.load('weights/wt.h5')

# image_path = r"C:\Users\Varun Chandra\Desktop\BDA Code\test_images\real\real00772.jpg"

image_path = r"C:\Users\Varun Chandra\Desktop\BDA Code\test_images\df\df01254.jpg"

frame = cv2.imread(image_path)


fc_img = frame * 1./255
fc_img = cv2.resize(fc_img, dsize = (256, 256), interpolation = cv2.INTER_CUBIC)
fc_img = fc_img.reshape((1, 256, 256, 3))

prediction = classifier.predict(fc_img)


if(prediction >= 0.5):
	res = "REAL"
	print("REAL")

else:
	res = "FAKE"
	print("FAKE")


# Display the resulting frame

cv2.putText(frame, res, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


cv2.imshow('frame',frame)

cv2.waitKey(5000) 
		
cv2.destroyAllWindows()