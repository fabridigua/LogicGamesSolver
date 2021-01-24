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

        self.puzzles = []
        self.puzzles_seen = 0

        self.exclude_classes = [0]

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

        trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))  # So they are in grayscale
        testData = testData.reshape((testData.shape[0], 28, 28, 1))
        trainData = trainData.astype("float32") / 255.0  # Now values in [0, 1]
        testData = testData.astype("float32") / 255.0

        le = LabelBinarizer()
        trainLabels = le.fit_transform(trainLabels)
        testLabels = le.transform(testLabels)
        opt = Adam(lr=1e-3)
        model_trained = self.get_model_structure(28, 28, 1, 10)  # 10 classes=0...9 values
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

    def predictDigitImage(self, digit_image, showRoi=False):
        if self.model_built:
            roi = cv2.resize(digit_image, (28, 28))
            if showRoi:
                cv2.imshow('Prediction image', roi)
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)
            preds = self.model.predict(roi)
            for c in self.exclude_classes:
                preds[0][c] = 0
            return preds.argmax(axis=1)[0]
        else:
            return False

    def analyze_boards(self, digit_images, info):
        board_structure = {}
        cells = []
        [[cells.append(str(i) + str(j)) for j in range(info['GRID_LEN'])] for i in range(info['GRID_LEN'])]

        for idx, digit in enumerate(digit_images):
            if digit is not None:
                prediction = self.predictDigitImage(digit)
                board_structure[cells[idx]] = str(prediction)

        return board_structure

    def save_puzzle(self, puzzle):
        if self.puzzles_seen < 7:
            self.puzzles.append(puzzle)
        else:
            self.puzzles[self.puzzles_seen % 7] = puzzle
        self.puzzles_seen += 1

    def get_sudoku_digits(self, info):
        digits_found = {}
        predictions = []
        for board in self.puzzles:
            digits = self.analyze_boards(board, info)
            print(digits)
            predictions.append(digits)

        print("len predictions ", len(predictions))
        for pos in predictions[len(predictions) - 1]:
            predicted_in_pos = [pred[pos] for pred in predictions if pos in pred]
            predicted_value_count = max(np.bincount(predicted_in_pos))
            predicted_value = np.bincount(predicted_in_pos).argmax()
            # 4 => 9 heuristic: it happens that 4 is predicted as 9 cause the font
            if 4 in predicted_in_pos and 9 in predicted_in_pos:
                predicted_value = 4
            if len(set(predicted_in_pos)) > 1:
                second_predicting = [x for x in predicted_in_pos if x is not predicted_value]
                second_predicted = np.bincount(second_predicting).argmax()
                if len(second_predicting) == predicted_value_count:
                    print("Strange prediction for ", pos, " => ", predicted_value, " or ", second_predicted, "? <= ",
                          predicted_in_pos)
                    continue
            #print(pos, " => ", predicted_in_pos, " ==> ", predicted_value)
            digits_found[pos] = str(predicted_value)
        return digits_found

    def get_skyscrapers_digits(self, info):
        digits_found = {}
        predictions = []
        for board in self.puzzles:
            digits = self.analyze_skyscrapers_boards(board, info)
            print(digits)
            predictions.append(digits)

        print("len predictions ", len(predictions))

        for pos in predictions[len(predictions) - 1]:
            predicted_in_pos = [pred[pos] for pred in predictions if pos in pred]
            predicted_value_count = max(np.bincount(predicted_in_pos))
            predicted_value = np.bincount(predicted_in_pos).argmax()
            # 4 => 9 heuristic: it happens that 4 is predicted as 9 cause the font
            if 4 in predicted_in_pos and 9 in predicted_in_pos:
                predicted_value = 4
            # if len(set(predicted_in_pos)) > 1:
            #     second_predicting = [x for x in predicted_in_pos if x is not predicted_value]
            #     second_predicted = np.bincount(second_predicting).argmax()
            #     if len(second_predicting) == predicted_value_count:
            #         print("Strange prediction for ", pos, " => ", predicted_value, " or ", second_predicted, "? <= ",
            #               predicted_in_pos)
            #         continue
            # print(pos, " => ", predicted_in_pos, " ==> ", predicted_value)
            digits_found[pos] = str(predicted_value)
        return digits_found

    def analyze_skyscrapers_boards(self, digit_images, info):
        if info['game'] == 'skyscrapers':
            self.exclude_classes = [x for x in range(10) if x not in [ n for n in range(1, info['GRID_LEN'] + 1)]]

        board_structure = {}

        cells = []
        grid_len = info['GRID_LEN']  # Ex. 4
        exclude = ['00', '0' + str(grid_len + 1), str(grid_len + 1) + '0', str(grid_len + 1) + str(grid_len + 1)]
        for i in range(1, grid_len + 1):
            [exclude.append(str(i) + str(j)) for j in range(1, grid_len + 1)]
        for j in range(grid_len+2):
            for i in range(grid_len+2):
                if not str(j) + str(i) in exclude:
                    cells.append(str(j) + str(i))

        for idx, digit in enumerate(digit_images):
            if digit is not None:
                imshow = False if cells[idx] != '51' else True
                prediction = self.predictDigitImage(digit, imshow)
                if cells[idx] == '51':
                    cv2.imshow('digit ', digit)
                board_structure[cells[idx]] = str(prediction)

        return board_structure

# 01
# 02
# 03
# 04
# 10
# 15
# 20
# 25
# 30
# 35
# 40
# 45
# 51
# 52
# 53
# 54
