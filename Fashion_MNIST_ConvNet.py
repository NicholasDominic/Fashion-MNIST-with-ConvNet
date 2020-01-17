import numpy as np, cv2, itertools
from imutils import build_montages as bm
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report as cr, confusion_matrix as cfm
from keras.models import Model
from keras.layers import Input, Activation, Dense, Conv2D, MaxPooling2D, ZeroPadding2D, Flatten
from keras.optimizers import Adam, Nadam
from keras.utils import np_utils as nu
from keras.utils.np_utils import to_categorical as tc
from keras.callbacks import TensorBoard
from keras import backend as K
from keras.datasets import fashion_mnist

def split_train_data(train_data, train_label):
	# Split into 48,000 train data and 12,000 validation data
	partial_train_data = train_data[:48000]
	partial_train_label = train_label[:48000]
	validation_data = train_data[48000:]
	validation_label = train_label[48000:]
	return partial_train_data, partial_train_label, validation_data, validation_label

def data_exploration(train, val, test):
	print("\nFASHION-MNIST DATASET EXPLORATION")
	print("Size of Train Data: ", train.shape)
	print("Size of Validation Data: ", val.shape)
	print("Size of Test Data: ", test.shape)
	print()

def convert_into_4D(partial_train_data, validation_data, test_data):
	# If we are using "channels first" ordering, then reshape the design
	# matrix such that the matrix is: num_samples x depth x rows x columns
	if K.image_data_format() == "channels_first":
		partial_train_data = train_data.reshape((partial_train_data.shape[0], 1, 28, 28))
		validation_data = validation_data.reshape((validation_data.shape[0], 1, 28, 28))
		test_data = test_data.reshape((test_data.shape[0], 1, 28, 28))
		
	# otherwise, we are using "channels last" ordering, so the design
	# matrix shape should be: num_samples x rows x columns x depth
	else:
		partial_train_data = partial_train_data.reshape((partial_train_data.shape[0], 28, 28, 1))
		validation_data = validation_data.reshape((validation_data.shape[0], 28, 28, 1))
		test_data = test_data.reshape((test_data.shape[0], 28, 28, 1))
	
	return partial_train_data, validation_data, test_data

def optimize(cnn_model):
	# Adam Optimizer and Cross Entropy Loss
	cnn_model.compile(optimizer=Nadam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
	print(cnn_model.summary())

	# Use TensorBoard
	# Step 1. In terminal, script "tensorboard --logdir ./Graph" (without "")
	# Step 2. In Chrome, type url localhost:6006 (or any IP emerged when Step 1 applied)

	# Train for 100 Epochs and use TensorBoard Callback
	result = cnn_model.fit(
		partial_train_data, partial_train_label,
		batch_size=number_of_batch,
		epochs=number_of_epoch, verbose=1, # verbose = 0 (silent) / verbose = 1 (progress bar)
		validation_data=(validation_data, validation_label),
		callbacks=[TensorBoard(log_dir='./Graph')])

	# Save Weights
	cnn_model.save_weights('{}.ckpt'.format(cnn_model.name))
	return cnn_model, result

def cnn_model_one():
	cnn_model_label = "ConvNet Model 1"
	inputs = Input(shape=(28, 28, 1))
	conv_layer = ZeroPadding2D(padding=(2,2))(inputs)
	conv_layer = Conv2D(16, (5, 5), strides=(1,1), activation='relu')(conv_layer)
	conv_layer = MaxPooling2D((2, 2))(conv_layer)
	conv_layer = Conv2D(32, (3, 3), strides=(1,1), activation='relu')(conv_layer)
	conv_layer = Conv2D(32, (3, 3), strides=(1,1), activation='relu')(conv_layer)
	conv_layer = MaxPooling2D((2, 2))(conv_layer)
	conv_layer = Conv2D(64, (3, 3), strides=(1,1), activation='relu')(conv_layer)
	flatten = Flatten()(conv_layer) # Flatten feature map to Vector with 576 element
	fc_layer = Dense(256, activation='relu')(flatten) # Fully-connected layer
	fc_layer = Dense(64, activation='relu')(fc_layer) 
	outputs = Dense(10, activation='softmax')(fc_layer)

	model_1 = Model(name=cnn_model_label, inputs=inputs, outputs=outputs)
	return model_1

def cnn_model_two():
	cnn_model_label = "ConvNet Model 2"
	inputs = Input(shape=(28, 28, 1))
	conv_layer = Conv2D(16, (14, 14), strides=(1,1), activation='relu')(inputs)
	conv_layer = Conv2D(16, (7, 7), strides=(1,1), activation='relu')(inputs)
	conv_layer = Conv2D(16, (5, 5), strides=(1,1), activation='relu')(inputs)
	conv_layer = MaxPooling2D((2, 2))(conv_layer)
	conv_layer = Conv2D(32, (3, 3), strides=(1,1), activation='tanh')(conv_layer)
	conv_layer = MaxPooling2D((2, 2))(conv_layer)
	conv_layer = Conv2D(32, (3, 3), strides=(1,1), activation='relu')(conv_layer)
	conv_layer = MaxPooling2D((2, 2))(conv_layer)
	flatten = Flatten()(conv_layer)
	fc_layer = Dense(256, activation='relu')(flatten)
	fc_layer = Dense(64, activation='tanh')(fc_layer) 
	outputs = Dense(10, activation='softmax')(flatten)
	
	model_2 = Model(name=cnn_model_label, inputs=inputs, outputs=outputs)
	return model_2

def training_and_validation_loss(loss):
	plt.figure("Training & Val Loss")
	plt.plot(loss.history['loss'])
	plt.plot(loss.history['val_loss'])
	plt.title("Model Loss")
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend(['Train', 'Test'])
	plt.show()

def training_and_validation_accuracy(acc):
	plt.figure("Training & Val Accuracy")
	plt.plot(acc.history['accuracy'])
	plt.plot(acc.history['val_accuracy'])
	plt.title("Model Accuracy")
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.legend(['Train', 'Test'])
	plt.show()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	plt.figure("Confusion Matrix")
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	legend = np.arange(len(classes))
	plt.xticks(legend, classes, rotation=90)
	plt.yticks(legend, classes)
	
	if normalize: cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
	
	plt.tight_layout()
	plt.ylabel('ACTUAL')
	plt.xlabel('PREDICTIVE')
	plt.show()

def random_test_images(a):
	images = []
	for i in np.random.choice(np.arange(0, len(test_label)), size=(16)):
		probs = a.predict(test_data[np.newaxis, i])
		prediction = probs.argmax(axis=1)
		label = label_names[prediction[0]]

		if K.image_data_format() == "channels_first": image = (test_data[i][0] * 255).astype("uint8")
		else: image = (test_data[i] * 255).astype("uint8")

		color = (0, 255, 0) # Label color default = green (correct)
		if prediction[0] != np.argmax(test_label[i]): color = (0, 0, 255)  # Label color = red (incorrect)

		image = cv2.merge([image] * 3) # merge the channels into one image
		image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR) # resize from 28x28 to 96x96
		cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
		images.append(image)

	cv2.imshow("Fashion MNIST - Random Image Test", bm(images, (96, 96), (4, 4))[0])
	cv2.waitKey(0)

############################### INITIALIZATION ###############################
learning_rate = 1e-3
number_of_epoch = 100
number_of_batch = 256
label_names = ["Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

############################## DATA PREPARATION ##############################
# NOTES: Fashion MNIST is already organized into 60,000 training/ 10,000 testing splits
(train_data, train_label), (test_data, test_label) = fashion_mnist.load_data()

# Scale/normalize data to the range of [0, 1]
train_data = train_data.astype("float32") / 255.0
test_data = test_data.astype("float32") / 255.0

# One-hot encode the training and testing labels
train_label = tc(train_label, 10)
test_label = tc(test_label, 10)

partial_train_data, partial_train_label, validation_data, validation_label = split_train_data(train_data, train_label)
data_exploration(partial_train_data, validation_data, test_data)
partial_train_data, validation_data, test_data = convert_into_4D(partial_train_data, validation_data, test_data)

############################### SELECT CNN MODEL ##############################
select_model = 0
while select_model < 1 or select_model > 2:
	select_model = int(input("Select CNN Model [1-2]: "))

if select_model == 1:
	model_1, result_1 = optimize(cnn_model_one()) # Train CNN Model 1

	prediction_1 = model_1.predict(test_data)
	print("Evaluate Test")
	model_1.evaluate(test_data, test_label)
	print(cr(test_label.argmax(axis=1), prediction_1.argmax(axis=1), target_names=label_names)) # Classification Report

	# Training and Validation Curves
	training_and_validation_accuracy(result_1)
	training_and_validation_loss(result_1)

	# Confusion Matrix Visualization
	prediction_class_1 = np.argmax(prediction_1, axis=1) # Convert predictions classes to one hot vectors 
	test_label_cfm = np.argmax(test_label, axis=1) # Convert validation observations to one hot vectors
	confusion_mtx = cfm(test_label_cfm, prediction_class_1) # Compute the confusion matrix
	plot_confusion_matrix(confusion_mtx, classes=label_names) # Plot the confusion matrix

	random_test_images(model_1)
else:
	model_2, result_2 = optimize(cnn_model_two()) # Train CNN Model 2

	prediction_2 = model_2.predict(test_data)
	print("Evaluate Test")
	model_2.evaluate(test_data, test_label)
	print(cr(test_label.argmax(axis=1), prediction_2.argmax(axis=1), target_names=label_names)) # Classification Report

	# Training and Validation Curves
	training_and_validation_accuracy(result_2)
	training_and_validation_loss(result_2)

	# Confusion Matrix Visualization
	prediction_class_2 = np.argmax(prediction_2, axis=1) # Convert predictions classes to one hot vectors 
	test_label_cfm = np.argmax(test_label, axis=1) # Convert validation observations to one hot vectors
	confusion_mtx = cfm(test_label_cfm, prediction_class_2) # Compute the confusion matrix
	plot_confusion_matrix(confusion_mtx, classes=label_names) # Plot the confusion matrix

	random_test_images(model_2)

# REFERENCES
# https://www.pyimagesearch.com/2019/02/11/fashion-mnist-with-keras-and-deep-learning/
# https://www.kaggle.com/fuzzywizard/fashion-mnist-cnn-keras-accuracy-93
# https://medium.com/@samuelsena/pengenalan-deep-learning-part-7-convolutional-neural-network-cnn-b003b477dc94