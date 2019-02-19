import cv2
import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import mnist
from keras import backend as k
from keras import layers
from keras import Sequential
from keras.models import load_model, model_from_json



#load mnist dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data() #everytime loading data won't be so easy :)

def format_training_data():
	#reshaping
	#this assumes our data format
	#For 3D data, "channels_last" assumes (conv_dim1, conv_dim2, conv_dim3, channels) while 
	#"channels_first" assumes (channels, conv_dim1, conv_dim2, conv_dim3).

	img_rows, img_cols = 28, 28
	if k.image_data_format() == 'channels_first':
	    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
	    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
	    input_shape = (1, img_rows, img_cols)
	else:
	    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
	    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
	    input_shape = (img_rows, img_cols, 1)
	#more reshaping
	X_train = X_train.astype('float32')
	X_test = X_test.astype('float32')
	X_train /= 255
	X_test /= 255
	print('X_train shape:', X_train.shape) #X_train shape: (60000, 28, 28, 1)


	#set number of categories
	num_category = 10
	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_category)
	y_test = keras.utils.to_categorical(y_test, num_category)



def build_model():
	##model building
	model = Sequential()
	#convolutional layer with rectified linear unit activation
	model.add(layers.Conv2D(32, kernel_size=(3, 3),
	                 activation='relu',
	                 input_shape=input_shape))
	#32 convolution filters used each of size 3x3
	#again
	model.add(layers.Conv2D(64, (3, 3), activation='relu'))
	#64 convolution filters used each of size 3x3
	#choose the best features via pooling
	model.add(layers.MaxPooling2D(pool_size=(2, 2)))
	#randomly turn neurons on and off to improve convergence
	model.add(layers.Dropout(0.25))
	#flatten since too many dimensions, we only want a classification output
	model.add(layers.Flatten())
	#fully connected to get all relevant data
	model.add(layers.Dense(128, activation='relu'))
	#one more dropout for convergence' sake :) 
	model.add(layers.Dropout(0.5))
	#output a softmax to squash the matrix into output probabilities
	model.add(layers.Dense(num_category, activation='softmax'))


	#Adaptive learning rate (adaDelta) is a popular form of gradient descent rivaled only by adam and adagrad
	#categorical ce since we have multiple classes (10) 
	model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adadelta(),
	              metrics=['accuracy'])


	batch_size = 128
	num_epoch = 10
	#model training
	model_log = model.fit(X_train, y_train,
	          batch_size=batch_size,
	          epochs=num_epoch,
	          verbose=1,
	          validation_data=(X_test, y_test))


	score = model.evaluate(X_test, y_test, verbose=0)
	print('Test loss:', score[0]) #Test loss: 0.0296396646054
	print('Test accuracy:', score[1]) #Test accuracy: 0.9904


	#Save the model
	# serialize model to JSON
	model_digit_json = model.to_json()
	with open("model_digit.json", "w") as json_file:
	    json_file.write(model_digit_json)
	# serialize weights to HDF5
	model.save_weights("model_digit.h5")
	print("Saved model to disk")





def make_coordinates(image, line_parameters):
	slope, intercept = line_parameters
	y1 = image.shape[0]
	y2 = int(y1 * (4/5))
	x1 = int((y1 - intercept)/slope)
	x2 = int((y2 - intercept)/slope)
	return np.array([x1, y1, x2, y2])

# maybe not relevent for tekken image
def average_slope_intercept(image, lines):
	left_fit = []
	right_fit = []
	for line in lines:
		x1, y1, x2, y2 = line.reshape(4)
		parameters = np.polyfit((x1, x2), (y1, y2), 1)
		slope = parameters[0]
		intercept = parameters[1]
		if slope < 0:
			left_fit.append((slope, intercept))
		else:
			right_fit.append((slope, intercept))
	left_fit_average = np.average(left_fit, axis=0)
	right_fit_average = np.average(right_fit, axis=0)
	left_line = make_coordinates(image, left_fit_average)
	right_line = make_coordinates(image, right_fit_average)
	return np.array([left_line, right_line])

# detect lines with canny
def canny(image):
	gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	# gray = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
	blur = cv2.GaussianBlur(image, (5, 5), 0)
	canny = cv2.Canny(blur, 50, 150)
	canny = cv2.bitwise_not(canny)
	return canny

def display_lines(image, lines):
	line_image = np.zeros_like(image)
	if lines is not None:
		for line in lines:
			x1, y1, x2, y2 = line.reshape(4)
			cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
	return line_image

def region_of_interest(image):
	height = image.shape[0]
	width = image.shape[1]
	polygons = np.array([
		# [(0, 800), (0, 700), (width, 700), (width, 800)]
		# [(325, 775), (325, 735), (420, 735), (420, 775)]
		[(358, 854), (358, 817), (410, 817), (410, 854)]
		# [(355, 860), (355, 810), (450, 810), (450, 860)]
	])
	mask = np.zeros_like(image)
	cv2.fillPoly(mask, polygons, 255)
	masked_image = cv2.bitwise_and(image, mask) #
	return masked_image


# original_image = cv2.imread("example.JPG")
# tekken_image = np.copy(original_image)
# canny_image = canny(tekken_image)
# cropped_image = region_of_interest(canny_image)


# # detect lines
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(tekken_image, lines)
# line_image = display_lines(tekken_image, lines)
# combo_image = cv2.addWeighted(tekken_image, 0.8, line_image, 1, 1)

json_file = open('model_digit.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("model_digit.h5")

model.compile(loss=keras.losses.categorical_crossentropy,
	              optimizer=keras.optimizers.Adadelta(),
	              metrics=['accuracy'])

# img_rows, img_cols = 28, 28
# if k.image_data_format() == 'channels_first':
#     cropped_image = cv2.resize(cropped_image, (28, 28))
#     cropped_image = np.reshape(cropped_image, [1, 1, 28, 28])
#     input_shape = (1, img_rows, img_cols)
# else:
#     cropped_image = cv2.resize(cropped_image, (28, 28))
#     cropped_image = np.reshape(cropped_image, [1, 28, 28, 1])
#     input_shape = (img_rows, img_cols, 1)

# cropped_image = cropped_image.astype('float32')
# cropped_image /= 255

# classes = model.predict_classes(cropped_image)
# print("Wins = " + str(classes)) #X_train shape: (60000, 28, 28, 1)



# cv2.imshow("result", cropped_image)
# cv2.waitKey(0)
# plt.imshow(canny_image)
# plt.show()


# Video Processing
cap = cv2.VideoCapture("example_video.mp4")
while(cap.isOpened()):
	_, frame = cap.read()
	canny_image = canny(frame)
	# plt.imshow(canny_image)
	# plt.show()
	cropped_image = region_of_interest(canny_image)
	cv2.imshow("result", cropped_image)
	# # detect lines
	# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
	# # averaged_lines = average_slope_intercept(frame, lines)
	# line_image = display_lines(frame, lines)
	# combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

	img_rows, img_cols = 28, 28
	if k.image_data_format() == 'channels_first':
	    cropped_image = cv2.resize(cropped_image, (28, 28))
	    cropped_image = np.reshape(cropped_image, [1, 1, 28, 28])
	    input_shape = (1, img_rows, img_cols)
	else:
	    cropped_image = cv2.resize(cropped_image, (28, 28))
	    cropped_image = np.reshape(cropped_image, [1, 28, 28, 1])
	    input_shape = (img_rows, img_cols, 1)

	cropped_image = cropped_image.astype('float32')
	cropped_image /= 255

	classes = model.predict_classes(cropped_image)
	print("Wins = " + str(classes)) #X_train shape: (60000, 28, 28, 1)

	

	
	if cv2.waitKey(1) == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()