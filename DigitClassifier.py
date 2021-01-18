# Class for digit classifier using MNIST CNN

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

import os
import numpy as np
import cv2


class DigitClassifier:
    def __init__(self, weights_file='model_weights.h5', epochs=10, batch_size=128):
        self.weights_file = weights_file
        self.epochs = epochs
        self.batch_size = batch_size
        if not os.path.isfile(self.weights_file):
            self.model_built = False
            self.train_model()
        else:
            self.model_built = True
            self.model = tf.keras.models.load_model(self.weights_file)

    def get_model_structure(self, width, height, depth, classes):
        model = Sequential()

        model.add(Conv2D(32, (5, 5), padding="same",
                         input_shape=(height, width, depth)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(64))
        model.add(Activation("relu"))
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))

        return model

    def train_model(self):
        print("Downloading MNIST...")
        ((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()

        trainData = trainData.reshape((trainData.shape[0], 28, 28, 1)) # So they are in grayscale
        testData = testData.reshape((testData.shape[0], 28, 28, 1))
        trainData = trainData.astype("float32") / 255.0 # Now values in [0, 1]
        testData = testData.astype("float32") / 255.0

        le = LabelBinarizer()
        trainLabels = le.fit_transform(trainLabels)
        testLabels = le.transform(testLabels)
        opt = Adam(lr=1e-3)
        model_trained = self.get_model_structure(28, 28, 1, 10) # 10 classes=0...9 values
        model_trained.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
        print("Training CNN...")
        H = model_trained.fit(
            trainData, trainLabels,
            validation_data=(testData, testLabels),
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1)

        print("Evaluating CNN...")
        predictions = model_trained.predict(testData)
        print(classification_report(
            testLabels.argmax(axis=1),
            predictions.argmax(axis=1),
            target_names=[str(x) for x in le.classes_]))

        print("Saving model in ", self.weights_file, " file")
        model_trained.save(self.weights_file, save_format="h5")
        self.model_built = True

    def predictDigitImage(self, digit_image):
        if self.model_built:
            roi = cv2.resize(digit_image, (28, 28))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            return self.model.predict(roi).argmax(axis=1)[0]
        else:
            return False
