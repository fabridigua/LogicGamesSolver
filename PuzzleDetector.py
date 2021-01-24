# Class for puzzle detection

import cv2
import numpy as np
from skimage.segmentation import clear_border
import operator


class PuzzleDetector:
    def __init__(self, game_info):
        self.game_info = game_info  # Es. GRID_LEN
        self.grid_digit_images = None

    def detectGameBoard(self, image):
        if self.game_info['game'] == 'sudoku':
            return self.detectSudokuBoard(image)
        elif self.game_info['game'] == 'stars':
            return self.detectStarsBoard(image)
        elif self.game_info['game'] == 'skyscrapers':
            return self.detectSkyscrapersBoard(image)
        else:
            return None

    def detectSudokuBoard(self, img):
        polygon, output = self.findPolygon(img)

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

        dst_polygon = np.array([[0, 0], [square_side - 1, 0], [square_side - 1, square_side - 1], [0, square_side - 1]],
                               dtype='float32')
        m = cv2.getPerspectiveTransform(src_polygon, dst_polygon)
        img = cv2.warpPerspective(img, m, (int(square_side), int(square_side)))

        squares = []
        grid_len = self.game_info['GRID_LEN']  # Ex. 9
        side = img.shape[:1]
        side = side[0] / grid_len
        for j in range(grid_len):
            for i in range(grid_len):
                p1 = (int(i * side), int(j * side))  # Top left corner of a box
                p2 = (int((i + 1) * side), int((j + 1) * side))  # Bottom right corner
                squares.append((p1, p2))

        digits = []
        for idx, square in enumerate(squares):
            square_roi = img[square[0][1]:square[1][1], square[0][0]:square[1][0]]
            extracted_digit = self.extract_digit(square_roi) if idx != 77 else self.extract_digit(square_roi, True)
            if extracted_digit is not None:
                if idx == 77:
                    cv2.imshow("Puzzle square_roi ", extracted_digit)
            digits.append(extracted_digit)

        output = cv2.putText(output, "Press Space when the puzzle is well seen", (30, output.shape[0] - 20),
                             cv2.FONT_HERSHEY_DUPLEX, 0.75, color=(0, 255, 255))

        cv2.imshow("Sudoku Puzzle Found", img)
        cv2.imshow("Sudoku Puzzle Image", output)

        self.grid_image = img
        self.grid_digit_images = digits

    def detectStarsBoard(self, img):
        polygon, output = self.findPolygon(img)

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

        dst_polygon = np.array([[0, 0], [square_side - 1, 0], [square_side - 1, square_side - 1], [0, square_side - 1]],
                               dtype='float32')
        m = cv2.getPerspectiveTransform(src_polygon, dst_polygon)
        img = cv2.warpPerspective(img, m, (int(square_side), int(square_side)))
        warped = img.copy()

        output = cv2.putText(output, "Press Space when the puzzle is well seen", (30, output.shape[0] - 20),
                             cv2.FONT_HERSHEY_DUPLEX, 0.75, color=(0, 255, 255))

        img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
        kernel = np.ones((6, 6), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
        _, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
        num_labels, labels = cv2.connectedComponents(img)

        # Map component labels to hue val, 0-179 is the hue range in OpenCV
        if num_labels == 0:
            return
        label_hue = np.uint8(179 * labels / np.max(labels))  # todo divide for 0
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # Converting cvt to BGR
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

        # set bg label to black
        labeled_img[label_hue == 0] = 0
        cv2.imshow("Stars Puzzle - Area detection", labeled_img)

        grid_len = self.game_info['GRID_LEN']  # Ex. 8
        side = img.shape[:1]
        side = side[0] / grid_len
        areas = [[] for a in range(grid_len)]
        if num_labels == grid_len + 1:
            for j in range(grid_len):
                for i in range(grid_len):
                    p1 = (int(i * side), int(j * side))  # Top left corner of a box
                    p2 = (int((i + 1) * side), int((j + 1) * side))  # Bottom right corner
                    area_label = labels[p1[1] + int((p2[1] - p1[1]) / 2), p1[0] + int((p2[0] - p1[0]) / 2)] - 1
                    areas[area_label].append(str(j) + str(i))
        cv2.imshow("Stars Puzzle Image", output)

        self.grid_image = warped
        self.grid_digit_images = areas

    def detectSkyscrapersBoard(self, img):
        polygon, output = self.findPolygon(img)

        rect = cv2.minAreaRect(polygon)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(output, [box], 0, (0, 0, 255), 2)

        top_left = tuple(box[1])
        top_right = tuple(box[2])
        bottom_right = tuple(box[3])
        bottom_left = tuple(box[0])

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

        dst_polygon = np.array([[0, 0], [square_side - 1, 0], [square_side - 1, square_side - 1], [0, square_side - 1]],
                               dtype='float32')
        m = cv2.getPerspectiveTransform(src_polygon, dst_polygon)
        img = cv2.warpPerspective(img, m, (int(square_side), int(square_side)))

        squares = []
        grid_len = self.game_info['GRID_LEN']  # Ex. 9
        side = img.shape[:1]
        side = side[0] / (grid_len + 2)
        exclude = ['00', '0' + str(grid_len + 1), str(grid_len + 1) + '0', str(grid_len + 1) + str(grid_len + 1)]
        for i in range(1, grid_len + 1):
            [exclude.append(str(i) + str(j)) for j in range(1, grid_len + 1)]
        for j in range(grid_len+2):
            for i in range(grid_len+2):
                if not str(j) + str(i) in exclude:
                    p1 = (int(i * side), int(j * side))  # Top left corner of a box
                    p2 = (int((i + 1) * side), int((j + 1) * side))  # Bottom right corner
                    squares.append((p1, p2))
                    # print(str(j) + str(i))

        digits = []
        for idx, square in enumerate(squares):
            square_roi = img[square[0][1]:square[1][1], square[0][0]:square[1][0]]
            extracted_digit = self.extract_digit(square_roi) if idx != 3 else self.extract_digit(square_roi, True)
            if extracted_digit is not None:
                if idx == len(squares) - 1:
                    cv2.imshow("Puzzle extracted_digit ", extracted_digit)
                digits.append(extracted_digit)

        output = cv2.putText(output, "Press Space when the puzzle is well seen", (30, output.shape[0] - 20),
                             cv2.FONT_HERSHEY_DUPLEX, 0.75, color=(0, 255, 255))

        cv2.imshow("Sudoku Puzzle Found", img)
        cv2.imshow("Sudoku Puzzle Image", output)

        self.grid_image = img
        self.grid_digit_images = digits

    def findPolygon(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 3)
        thresh = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        thresh = cv2.bitwise_not(thresh, thresh)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours = sorted(cnts, key=cv2.contourArea, reverse=True)
        polygon = contours[0]

        approx = cv2.approxPolyDP(polygon, 0.02 * cv2.arcLength(polygon, True), True)

        output = img.copy()
        cv2.drawContours(output, [approx], -1, (0, 255, 0), 2)
        return polygon, output

    def distance(self, p1, p2):
        a = p2[0] - p1[0]
        b = p2[1] - p1[1]
        return np.sqrt((a ** 2) + (b ** 2))

    def extract_digit(self, cell_roi, show=False):
        cell_roi = cv2.cvtColor(cell_roi, cv2.COLOR_BGR2GRAY)
        if show:
            cv2.imshow("DEBUG", cell_roi)
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

        if percentFilled < 0.02:  # 0.03
            return None

        # thresh = cv2.erode(thresh, np.ones((1, 1), np.uint8))
        thresh = cv2.medianBlur(thresh, 3)

        return cv2.bitwise_and(thresh, thresh, mask=mask)

    def get_stars_areas(self, puzzles):
        areas_count = []
        for puzzle in puzzles:
            count = 0
            for p in puzzles:
                count += 1 if len([area for area in p if area in puzzle]) == len(p) else 0
            areas_count.append(count)
        areas = puzzles[areas_count.index(max(areas_count))]
        return areas
