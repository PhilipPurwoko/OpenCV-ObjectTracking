print('Importing Library...')
import cv2
import matplotlib.pyplot as plt
import numpy as np

def blit(text,image):
	cv2.putText(image,text,(25,25),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)

print('Starting Camera...')
videoCapture = cv2.VideoCapture(0)
while True:
	res,frame = videoCapture.read()

	# Convert image to gray
	gray = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)

	# Gamma corection
	gray = np.array(255 * (gray / 255) ** 1.2 , dtype='uint8')

	# Apply threshold
	thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 19)
	thresh = cv2.bitwise_not(thresh)

	# Dilatation + erosion
	kernel = np.ones((15,15), np.uint8)
	image_dilatation = cv2.dilate(thresh, kernel, iterations=1)
	image_erosion = cv2.erode(image_dilatation,kernel, iterations=1)

	# clean all noise after dilatation and erosion
	iamge_erosion = cv2.medianBlur(image_erosion, 7)

	# Labeling
	ret, labels = cv2.connectedComponents(image_erosion)
	object_count = str(ret-1)

	label_hue = np.uint8(179 * labels / np.max(labels))
	blank_ch = 255 * np.ones_like(label_hue)

	labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
	labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
	labeled_img[label_hue == 0] = 0

	blit(f'Objects Count : {object_count}',labeled_img)
	cv2.imshow('Camera',labeled_img)

	if cv2.waitKey(1) == 27:
		break

videoCapture.release()
cv2.destroyAllWindows()