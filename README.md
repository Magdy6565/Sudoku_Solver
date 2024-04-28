Sudoku Solver with GUI
This project is a Sudoku solver implemented in Python with a graphical user interface (GUI) using Pygame. It provides functionality to generate Sudoku puzzles, solve them using a backtracking search algorithm, and display the solutions on the GUI.

Table of Contents
Overview
Requirements
How to Use
Modes
Backend
GUI
Overview
The project consists of two main components: the backend solver and the graphical user interface (GUI). The backend solver is responsible for generating, solving, and validating Sudoku puzzles using various algorithms. The GUI provides a visual representation of the puzzle, allows users to interact with it, and displays the solution.

Requirements
Python 3.x
Pygame library
How to Use
To use the Sudoku solver with GUI:

Clone this repository to your local machine.
Install Python 3.x if you haven't already.
Install the Pygame library using pip:
bash
Copy code
pip install pygame
Run the main.py file:
bash
Copy code
python main.py
Choose the mode and difficulty level from the GUI to generate or solve Sudoku puzzles.
Modes
The GUI offers three modes:

AI Generate and Solve: Automatically generates a Sudoku puzzle and solves it using the backtracking search algorithm.
User Generate and AI Solve: Allows the user to input their own Sudoku puzzle, then solves it using the backtracking search algorithm.
User Generate and Solve: Allows the user to input their own Sudoku puzzle and solve it manually.
Backend
The backend solver contains functions for generating, solving, and validating Sudoku puzzles. It includes the following key functions:

print_sudoku: Prints a given Sudoku puzzle to the console.
node_consistency: Ensures that each cell in the puzzle contains a valid value according to Sudoku rules.
ac3: Implements the AC-3 algorithm for arc consistency.
bts: Implements the backtracking search algorithm for solving Sudoku puzzles.
GUI
The graphical user interface (GUI) is built using Pygame and provides a visual representation of the Sudoku puzzle. It includes the following features:

Interactive grid for inputting Sudoku values.
Buttons for selecting the mode and difficulty level.
Error messages for invalid inputs or solutions.
