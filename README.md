# LogicGamesSolver
 Python tool to solve logic games with AI, Computer Vision and a bit of Deep Learning. 

----------------

<img src="imgs/screen_boards_solved.png" alt="Scre" style="zoom:100%;" />

------

## Table of contents

* [Basic Overview](#basic-overview)
* [Project structure](#project-structure)
* [System Requirements](#system-requirements)
* [Setup](#setup)
* [How it works](#how-it-works)
* [Games included](#games-included)
* [References](#references)

## Basic Overview

This project mixes Computer Vision and Artificial Intelligence to solve logic puzzle games like *Sudoku*, *Stars Battle* and *Skyscrapers*.

The execution consists of 2 phases:

| 1. Board Detection                                           | 2. Game solving                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="imgs/screen_stars_board_found.png" style="zoom:100%;" /> | <img src="imgs/screen_stars_board_solved.png" style="zoom:100%;" /> |
| The software detects the board showed by the user (in real-time or analyzing a local image). Then analyzes the structure to understand the needed informations to solve the game. | The informations collected are then used to solve the puzzle considering it as a Constraints Satisfaction Problem using a Backtracking algorithm to find the solution, given the game rules. |

**Language**: *Python*

**Frameworks**: *OpenCV*, *Tensorflow*

**Algorithms**: Constraints Satisfaction Problem Backtracking, Digits Classifier (CNN), Image Contours Finding and Warping 

## Project structure

```
LogicGamesSolver
│   README.md
│   LICENSE
│
└───imgs
│   │   screen_XXX	Screens for project explaination
│   │   ...
│	
│   main.py		Main file to execute the software
│   DigitClassifier.py		Class for digit classification with a pretrained CNN			
│   PuzzleDetector.py	Class for puzzle detection and analyze from an image
│   Solver.py		Class for solving the games given puzzle's informations and rules 		
```

## System Requirements

- **Python 3.8**

- **Numpy 1.19.2**

- **OpenCV 4.0.1**

- **Tensorflow 2.3.0**

  

## Setup

To run the project, clone it with [Git] and run the `main.py` file with desired parameters:

[Git]: https://git-scm.com/downloads	"Git download page"

```
$ git clone https://github.com/fabridigua/LogicGamesSolver
$ cd LogicGamesSolver
$ python main.py sudoku 9 3
```

There are 4 arguments you can set: 

[**game** name of the game you want to solve] = sudoku | skyscrapers | stars [default: sudoku]

[**grid_len** number of squares in the side of the puzzle's grid] = *Integer* es. 5 [default: 9]

[**square_len** for sudoku, number of squares in an area] = *Integer* es. 2 [default: 3]

[**is_realtime** for coose between local or real-time execution] = *Boolean* es. true [default true]

Note that if you set **is_realtime** to *false* you have to put an image file named *input_[**game**]*.*png* es. `input_stars.png` 

## How it works

This project touches many fields of study:

1. **Computer Vision** for board detection 
2. **Deep Learning** for puzzle analyzing
3. **Artificial Intelligence** for puzzle solving

##### 1. Board Detection

The software analyzes the image in input looking for the biggest *contour* in the scene.

<img src="imgs\screen_sudoku_realtime_detection.png" style="zoom:80%;" />

Once found the puzzle, its image is warped applying a perspective transformation to make the puzzle image plane.

In *real-time* mode,  the user must press `space` key to go ahead when the software is recognizing well the puzzle.

##### 2. Puzzle Analyzing 

The software analyzes the puzzle board to retrieve the needed informations to solve the game.

If the puzzle includes numbers (like *sudoku* and *skyscrapers*) a **Convolutional Neural Network** for digit classifying (trained with the fabulous *MNIST* dataset) is executed.

<img src="imgs\screen_stars_realtime_analyzing.png" style="zoom:100%;" />

For the *stars* game, the board image is processed to find the inner areas. Using OpenCV methods, the boldest contours are highlighted and then the *connected components* are analyzed (and drawn in different colors) to find the areas of the grid.

##### 3. Game solving

Once collected all the informations about the structure of the puzzle, the game is represented as a CSP (Constraint Satisfaction Problem) and solved applying a backtracking algorithm. By the game point of view, the CSP consists in 3 elements:

	1. *Variables*: the grid cells to fill
 	2. *Domains*: the sets of values that each cell can assume
 	3. *Constraints*: the game's rules

If the input is correct, the algorithm finds the solution with 100% of accuracy, but it can takes a long time basing on grid length (and so the number of variables) and size of domains. For a normal *Sudoku* scheme it take 3-5 seconds.