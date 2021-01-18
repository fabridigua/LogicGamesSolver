from PuzzleDetector import PuzzleDetector
from Solver import Solver
from DigitClassifier import DigitClassifier

import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

info = {
    'GRID_LEN': 9
}

detector = PuzzleDetector(info)

while True:
    _, frame = cap.read()

    # cv2.imshow('PuzzleFinder', frame)

    detector.detectGameBoard(frame)

    k = cv2.waitKey(20)
    if k == 27 & 0xFF == ord('q'):
        break
    elif k == 32:
        pass

cap.release()
cv2.destroyAllWindows()
