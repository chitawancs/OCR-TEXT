from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import cv2
import time

def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < 0.1:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute thภe sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences)


def det_txt_ocr(img_path):
	try:
		# load the input image and grab the image dimensions
		image = cv2.imread(img_path)

		#angle for rotation
		# fix------
		ag = 357
		# fix------

		#rotation
		num_rows, num_cols = image.shape[:2]
		rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), ag, 1)
		image = cv2.warpAffine(image, rotation_matrix, (num_cols, num_rows))

		orig = image.copy()
		(origH, origW) = image.shape[:2]

		# set the new width and height and then determine the ratio in change
		# for both the width and height
		# fix ---------------------------
		inW = 320
		inH = 160
		# -------------------------------

		(newW, newH) = (inW,inH)
		rW = origW / float(newW)
		rH = origH / float(newH)

		# resize the image and grab the new image dimensions
		image = cv2.resize(image, (newW, newH))
		(H, W) = image.shape[:2]

		# define the two output layer names for the EAST detector model that
		# we are interested -- the first is the output probabilities and the
		# second can be used to derive the bounding box coordinates of text


		# construct a blob from the image and then perform a forward pass of
		# the model to obtain the two output layer sets
		# fix ---------------------------
		blobsize = 0.5
		# -------------------------------

		blob = cv2.dnn.blobFromImage(image,blobsize, (W, H),
			(123.68, 116.78, 103.94), swapRB=True, crop=False)
		netdet.setInput(blob)
		(scores, geometry) = netdet.forward(layerNames)

		# decode the predictions, then  apply non-maxima suppression to
		# suppress weak, overlapping bounding boxes
		(rects, confidences) = decode_predictions(scores, geometry)
		boxes = non_max_suppression(np.array(rects),probs=confidences,)

		results = []

		# 1 round
		for (startX, startY, endX, endY) in boxes:
			# scale the bounding box coordinates based on the respective
			# ratios
			startX = int(startX * rW)
			startY = int(startY * rH)
			endX = int(endX * rW)
			endY = int(endY * rH)

			# in order to obtain a better OCR of the text we can potentially
			# apply a bit of padding surrounding the bounding box -- here we
			# are computing the deltas in both the x and y directions
			# fix ---------------------------
			pX = 0.0
			pY = 0.2
			# -------------------------------

			dX = int((endX - startX) * pX)
			dY = int((endY - startY) * pY)

			# apply padding to each side of the bounding box, respectively
			startX = 0
			startY = max(0, startY - dY)
			endX = origW
			endY = min(origH, endY + (dY * 2))

			# extract the actual padded ROI
			roi = orig[startY:endY, startX:endX]
			pd = orig[endY:origH, startX:endX]

			# in order to apply Tesseract v4 to OCR text we must supply
			pytesseract.pytesseract.tesseract_cmd = r'Tesseract-OCR\tesseract'
			config = ("-l thafast --oem 1 --psm 7")
			text = pytesseract.image_to_string(roi, config=config)

			# add the bounding box coordinates and OCR'd text to the list
			# of results
			results.append(((startX, startY, endX, endY), text))
			# just 1r
			break

		# sort the results bounding box coordinates from top to bottom
		results = sorted(results, key=lambda r:r[0][1])

		# loop over the results
		for ((startX, startY, endX, endY), text) in results:
			# display the text OCR by Tesseract
			x = list(text)
			for i, item in enumerate(x):
				if ord(item) == 46 or ord(item) == 91 or ord(item) == 93:
					x[i] = " "
				if ord(item) == 124:
					x[i] = " "
				if ord(item) > 3630:
					x[i] = " "
			# text out put
			text_out = "".join([c if ord(c) > 44 else "" for c in x]).strip()
			# print("--OCR--")
			#print(text_out)

			output = orig.copy()
			cv2.rectangle(output, (startX, startY), (endX, endY),(127, 255, 0), 1)
			# sv pd
			idx = 1
			write_name = r'pd\pd_' + str(idx) + '.png'
			cv2.imwrite(write_name, pd)
			idx = + 1
			# show the output image
			# cv2.imshow("Text Detection", output)
			# cv2.moveWindow("Text Detection", 600, 300)
			# cv2.imshow("Text ROI", roi)
			# cv2.moveWindow("Text ROI", 600, 400)
			# cv2.imshow("PV", pv)
			# cv2.moveWindow("PV", 600, 500)
			# (origH2, origW2) = pv.shape[:2]
			# print(origH2)
			# print(origW2)
			cv2.waitKey(0)

			return text_out
	except Exception as e:
		text_out = "type error: " + str(e)
		# print("can't detect text")
		return text_out

def run_ocr(input):
	t = time.time()
	#####################
	ocr = det_txt_ocr(input)
	#####################
	print(ocr)
	print(time.time() - t)
	return ocr

layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]
netdet = cv2.dnn.readNet('frozen_east_text_detection.pb')

