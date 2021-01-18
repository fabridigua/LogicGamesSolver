# Class for puzzle detection

import cv2
import numpy as np
from skimage.segmentation import clear_border
import operator

class PuzzleDetector:
    def __init__(self, game_info, game='sudoku'):
        self.GAME = game
        self.game_info = game_info  # Es. GRID_LEN

    def detectGameBoard(self, image):
        if self.GAME == 'sudoku':
            return self.detectSudokuBoard(image)
        elif self.GAME == 'stars':
            return self.detectStarsBoard(image)
        elif self.GAME == 'cities':
            return self.detectCitiesBoard(image)
        else:
            return None

    def detectSudokuBoard(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 3)
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh = cv2.bitwise_not(thresh, thresh)

        kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
        proc = cv2.dilate(thresh, kernel)

        # cv2.imshow("Puzzle Thresh proc", proc)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = sorted(cnts, key=cv2.contourArea, reverse=True)
        polygon = contours[0]  # The first contour refers to the biggest

        approx = cv2.approxPolyDP(polygon, 0.02 * cv2.arcLength(polygon, True), True)

        output = img.copy()
        cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)

        bottom_right_index, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in
                                         polygon]), key=operator.itemgetter(1))
        top_left_index, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in
                                     polygon]), key=operator.itemgetter(1))
        bottom_left_index, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in
                                        polygon]), key=operator.itemgetter(1))
        top_right_index, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in
                                      polygon]), key=operator.itemgetter(1))

        top_left = tuple(polygon[top_left_index][0])
        top_right = tuple(polygon[top_right_index][0])
        bottom_right = tuple(polygon[bottom_right_index][0])
        bottom_left = tuple(polygon[bottom_left_index][0])

        cv2.circle(output, top_left, 4, (0, 0, 255), -1)
        cv2.circle(output, top_right, 4, (0, 0, 255), -1)
        cv2.circle(output, bottom_right, 4, (0, 0, 255), -1)
        cv2.circle(output, bottom_left, 4, (0, 0, 255), -1)

        src_polygon = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
        square_side = max(
            [
                self.distance(top_left, top_right),
                self.distance(top_left, bottom_left),
                self.distance(bottom_left, bottom_right),
                self.distance(bottom_right, top_right)
            ]
        )

        dst_polygon = np.array([[0, 0], [square_side - 1, 0], [square_side - 1, square_side - 1], [0, square_side - 1]], dtype='float32')
        m = cv2.getPerspectiveTransform(src_polygon, dst_polygon)
        img = cv2.warpPerspective(img, m, (int(square_side), int(square_side)))

        squares = []
        grid_len = self.game_info['GRID_LEN']  # Es. 9
        side = img.shape[:1]
        side = side[0] / grid_len
        for j in range(grid_len):
            for i in range(grid_len):
                p1 = (int(i * side), int(j * side))  # Top left corner of a box
                p2 = (int((i + 1) * side), int((j + 1) * side))  # Bottom right corner
                squares.append((p1, p2))

        digits = []
        for square in squares:
            square_roi = img[square[0][0]:square[1][0], square[0][1]:square[1][1]]
            # cv2.imshow("Puzzle square_roi " + str(idx), square_roi)
            digits.append(self.extract_digit(square_roi))

        cv2.imshow("Sudoku Puzzle Found", output)
        cv2.imshow("Sudoku Puzzle Image", img)

        self.grid_image = img
        self.grid_digit_images = digits

    def detectStarsBoard(self, image):
        pass

    def detectCitiesBoard(self, image):
        pass

    def distance(self, p1, p2):
        a = p2[0] - p1[0]
        b = p2[1] - p1[1]
        return np.sqrt((a ** 2) + (b ** 2))

    def extract_digit(self, cell_roi):
        cell_roi = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(cell_roi, 0, 255,
                               cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = clear_border(thresh)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]

        if len(cnts) == 0:  # Empty cell
            return None

        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        (h, w) = thresh.shape
        percentFilled = cv2.countNonZero(mask) / float(w * h)

        if percentFilled < 0.05:  # 0.03
            return None

        return cv2.bitwise_and(thresh, thresh, mask=mask)