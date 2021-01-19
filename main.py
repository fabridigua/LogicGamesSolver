from PuzzleDetector import PuzzleDetector
from Solver import Solver
from DigitClassifier import DigitClassifier

import cv2


info = {
    'game': 'sudoku',
    'GRID_LEN': 9,
    'SQUARE_LEN': 3
}

puzzle_detected = False
puzzle_analyzed = False
puzzle_solved = False


detector = PuzzleDetector(info)
classifier = DigitClassifier()
solver = Solver(info)

# 1. Board detection phase
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while not puzzle_detected:
    _, frame = cap.read()

    detector.detectGameBoard(frame)

    k = cv2.waitKey(20)
    if k == 27 & 0xFF == ord('q'):
        break
    elif k == 32:
        puzzle_detected = True
        cv2.destroyAllWindows()

print('Digits found ', str(len([d for d in detector.grid_digit_images if d is not None])))

# #DEBUG: draw and analyze all digits found:
# for idx, digit in enumerate(detector.grid_digit_images):
#     if digit is not None:
#         cv2.imshow("Digit "+str(idx), digit)

# 2. Board analyze phase
digits_found = classifier.analyze_boards(detector.grid_digit_images, info)
print(digits_found)

# 3. Game solution phase
data = {
    'variables_found': digits_found
}

wait_image = detector.grid_image.copy()
wait_image = cv2.putText(wait_image, "Solving..", (int(wait_image.shape[1]/3), int(wait_image.shape[0]/2)),
                             cv2.FONT_HERSHEY_DUPLEX, 0.75, color=(0, 255, 255))
cv2.imshow("Game solved", wait_image)

solved = solver.solveGame(data)

if solved != "SOLVED":
    print("Error: ", solved)

cv2.imshow("Game solved", solver.drawResult(detector.grid_image))

cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()
