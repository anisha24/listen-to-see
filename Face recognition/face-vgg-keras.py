import keras_vggface
import tensorflow
import cv2
import mtcnn
from matplotlib import pyplot
from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN

def extract_face(filename, required_size=(224, 224)):
	# load image from file
	pixels = pyplot.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the facepython 
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array

for i in range(38):
	imgname = '000'
	if i<10:
		imgname = imgname + '0' + str(i)
	else:
		imgname = '000' + str(i) 
	pixels = extract_face('./dataset/anish/' + imgname + '.jpeg')
# plot the extracted face
pyplot.imshow(pixels)
# show the plot
pyplot.show()