#  Importing needed libraries and modules 
import pygame
import numpy as np
import sys
import random
from collections import deque
import copy
import time
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Start Backend@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
variables = []
for i in range (9) : 
    for j in range (9): 
        variables.append( (i,j) ) 
domains ={v : set(range(1,10)) for v in variables}
domain_stack = []
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def print_sudoku(puzzle):
    for i in range(9):
        if i % 3 == 0 and i != 0:
            print("-" * 21)
        for j in range(9):
            if j % 3 == 0 and j != 0:
                print("|", end=" ")
            print(puzzle[i, j] if puzzle[i, j] != -1 else ".", end=" ")
        print()   

def node_consistency(csp):
    global domains
    domains ={v : set(range(1,10)) for v in variables}
    for v in domains:
        row, col = v
        if csp[row, col] != -1:
            domains[v] = set([csp[row, col]])
def neighbors(X):

    row, col = X
    box_row = row // 3
    box_col = col // 3

    row_neighbors = [(row, i) for i in range(9) if i != col]
    col_neighbors = [(i, col) for i in range(9) if i != row]
    box_neighbors = [(box_row * 3 + i, box_col * 3 + j) for i in range(3) for j in range(3) if (box_row * 3 + i, box_col * 3 + j) != X]

    return list(set(row_neighbors + col_neighbors + box_neighbors))        
    # get all variables and domain of each one
def ac3(csp, arcs=None):
    global variables, domains
    # node_consistency(csp)
    n = len(variables)
    if arcs is not None:
        queue = deque(arcs)
    else:
        queue = deque()
        # Add constraints for rows and columns
        for i in range(9):
            for j in range(9):
                for k in range(j + 1, 9):
                    # Row constraints
                    queue.append(((i, j), (i, k)))
                    # Column constraints
                    queue.append(((j, i), (k, i)))

        # Add constraints for blocks
        for block_row in range(3):
            for block_col in range(3):
                for i in range(3):
                    for j in range(3):
                        for u in range(i, 3):
                            start_v = j + 1 if u == i else 0
                            for v in range(start_v, 3):
                                queue.append((
                                    (block_row * 3 + i, block_col * 3 + j),
                                    (block_row * 3 + u, block_col * 3 + v)
                                ))

    while queue:
        X, Y = queue.popleft()
        if revise(X, Y):
            if len(domains[X]) == 0:
                # print(f"Cells {X} and {Y} are not arc consistent")                
                return False
            for Z in neighbors(X):
                if Z != Y:
                    queue.append((Z, X))
    return True
def revise(X, Y):
        global domains
        revised = False
        if len( domains[Y] ) == 1 and next(iter( domains[Y] )) in domains[X] : 
                # print(f"{next(iter( domains[Y] ))} removed from cell {X} due to cell {Y} ")            
                domains[X].remove(next(iter( domains[Y] )))
                # print(f"now domain of cell {X} => {domains[X]}")
                revised = True
        return revised
    
def Inference(assignment, X , puzzle):
            exit_value   = ac3(puzzle , arcs=[(Y, X) for Y in neighbors(X)])
            inferences = dict()
            if not exit_value:
                return exit_value, inferences
            for v in domains:
                if v not in assignment and len(domains[v]) == 1:
                    inferences[v] = next(iter(domains[v]))
            return exit_value, inferences 
 

def value_consistent_with_assignment(row , col , puzzle , value) :  
    rowvalues  = puzzle[row] 
    if value in rowvalues : 
        return  False
    colvalues = []
    for i in range (9) : 
          colvalues.append(puzzle[i, col] )
    if value in colvalues :
        return False 
    startsquarerow = (row //3 ) * 3
    startsquarecol = (col // 3 ) *3
    for i in range (startsquarerow , startsquarerow + 3  ) : 
        for j in range (startsquarecol ,  startsquarecol +3 ) : 
            if value  == puzzle[i,j] : 
                return False  
    
    return True 
def is_valid_move(board, row, col, num):
    # Check if the number is not present in the same row
    if num in board[row]:
        return False
    
    # Check if the number is not present in the same column
    if num in [board[i][col] for i in range(9)]:
        return False
    
    # Check if the number is not present in the 3x3 subgrid
    subgrid_row, subgrid_col = 3 * (row // 3), 3 * (col // 3)
    for i in range(subgrid_row, subgrid_row + 3):
        for j in range(subgrid_col, subgrid_col + 3):
            if board[i][j] == num:
                return False
    
    return True
def count_conflicts(  num , row, col    , board) : 
    # row w col w num , dy l cell l hl3bh fyha w num da l rkml l hl3bo
    # num da bykon gy mn domain values
    count = 0
    for i in range(9):
        if i != col and not is_valid_move(board, row, i, num):
            count += 1
        if i != row and not is_valid_move(board, i, col, num):
            count += 1

    for i in range(row - row % 3, row - row % 3 + 3):
        for j in range(col - col % 3, col - col % 3 + 3):
            if (i != row or j != col) and not is_valid_move(board, i, j, num):
                count += 1
    return count
    
    
def domain_values(var, assignment, puzzle):
    # 3ayz a3ml lcv 
    global domains
    # domain = list(range(1,10))
    domain  = domains[var]
    row , col = var
    conflicts_dict = {value: count_conflicts(value, row , col, puzzle) for value in domain}
    sorted_values = sorted(domain, key=lambda value: conflicts_dict[value])
    return sorted_values


def get_domain_values(board, row, col):
    domain_values = [num for num in range(1, 10) if is_valid_move(board, row, col, num)]
    return domain_values
def  select_unassigned_var(assignment , puzzle) :  
    # i want  to select variable with smallest domain 
    #  MRV
    choosen_row = -1 
    choosen_col =-1
    counter = 0 
    least_value = float('inf')
    for i in range (9) : 
        for j in range (9) : 
            if puzzle[i,j] == -1 :
                counter  = len  ( get_domain_values(puzzle , i , j))
                if counter < least_value : 
                    choosen_row = i  
                    choosen_col = j 
                    least_value = counter  
                counter  = 0 

    return choosen_row , choosen_col    

def is_complete_assignemnt(puzzle) :
    count_minus_one = np.sum(puzzle == -1)
    return True if count_minus_one == 0  else False 


def bts (assignment   ,  puzzle) : 
    global domains
    # if assignment complete  --> return assignemnt 
    if is_complete_assignemnt(puzzle) :
        return True  , assignment 
    row , col  = select_unassigned_var(assignment , puzzle)
    # print(f"MRV is ({row,col})")

    domain_values_sorted = domain_values( (row, col) , assignment , puzzle ) 
    # print(f"Attempting to fill cell {(row,col)} with domain values {domains[(row,col)]}")

    for value in domain_values_sorted: 
        # print(f"Trying value {value} for cell ({row,col}) ")

        if value_consistent_with_assignment(row , col , puzzle , value) : 
            domain_stack.append(copy.deepcopy(domains))
            puzzle[row,col] = value 
            assignment[(row, col)] = value
            exit_value , inferences =  Inference(assignment , (row,col)  , puzzle)
            if exit_value :  
                    assignment = {**assignment, **inferences}

            _ , result = bts(assignment, puzzle)
            if result:
                return True , result
            # print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!BACKTRACK!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            domains = domain_stack.pop()
            del assignment[(row, col)]
            puzzle[row,col] = -1
            assignment = {k : v for k, v in assignment.items() if k not in inferences} 
        else : 
                        pass
                        # print(f"value {value} for cell ({row,col}) is not consistent with assignemnt !!")

    
    return False , None  
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@END BACKEND@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@START GUI@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

# Initialize Pygame
pygame.init()
# Constants
WIDTH, HEIGHT = 540, 540
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GRID_LINE = 2
CELL_SIZE = WIDTH // 9
WIN_WIDTH, WIN_HEIGHT = 820 , 542
BUTTON_WIDTH = 50
BUTTON_HEIGHT = 50
BUTTON_GAP = 10
BUTTON_TEXT_COLOR = (255, 255, 255)
BUTTON_FONT = pygame.font.Font(None, 24)
BUTTON_COLOR = (0, 128, 255)
ERROR_FONT = pygame.font.SysFont(None, 26)
screen = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))

# Game Difficulties
EASY = [35,40]
MEDIUM = [30,35]
HARD = [25,30]
DIFFICULTY_OPTIONS = ["Easy", "Medium", "Hard"]

# Function to draw the Sudoku grid with bold blocks around each 3x3 grid
def draw_grid():
    screen.fill(WHITE)
    for x in range(0, WIDTH, CELL_SIZE):
        for y in range(0, HEIGHT, CELL_SIZE):
            # Draw vertical line
            if x % (3 * CELL_SIZE) == 0:
                pygame.draw.line(screen, BLACK, (x, y), (x, y + CELL_SIZE), 4)  # Bold vertical line
            else:
                pygame.draw.line(screen, BLACK, (x, y), (x, y + CELL_SIZE), GRID_LINE)  # Normal vertical line
            # Draw horizontal line
            if y % (3 * CELL_SIZE) == 0:
                pygame.draw.line(screen, BLACK, (x, y), (x + CELL_SIZE, y), 4)  # Bold horizontal line
            else:
                pygame.draw.line(screen, BLACK, (x, y), (x + CELL_SIZE, y), GRID_LINE)  # Normal horizontal line
    
    # Draw line extending full height after rightmost cells
    pygame.draw.line(screen, BLACK, (WIDTH, 0), (WIDTH , HEIGHT), 2)  # Bold vertical line
     # Draw line extending full width after bottom cells
    pygame.draw.line(screen, BLACK, (0, HEIGHT), (WIDTH, HEIGHT), 2)  # Bold horizontal line
    





# Function to draw the Sudoku values with gray background for cells containing numbers
def draw_values(puzzle):
    font = pygame.font.SysFont(None, 40)
    for i in range(9):
        for j in range(9):
            if puzzle[i, j] != -1:
                pygame.draw.rect(screen, (200, 200, 200), (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
                text = font.render(str(puzzle[i, j]), True, BLACK)
                text_rect = text.get_rect(center=(j * CELL_SIZE + CELL_SIZE // 2, i * CELL_SIZE + CELL_SIZE // 2))
                screen.blit(text, text_rect)
    # Redraw grid lines
    for x in range(0, WIDTH, CELL_SIZE):
        if x % (3 * CELL_SIZE) == 0:
            pygame.draw.line(screen, BLACK, (x, 0), (x, HEIGHT), 4)  # Bold vertical line
        else:
            pygame.draw.line(screen, BLACK, (x, 0), (x, HEIGHT), GRID_LINE)  # Normal vertical line
    for y in range(0, HEIGHT, CELL_SIZE):
        if y % (3 * CELL_SIZE) == 0:
            pygame.draw.line(screen, BLACK, (0, y), (WIDTH, y), 4)  # Bold horizontal line
        else:
            pygame.draw.line(screen, BLACK, (0, y), (WIDTH, y), GRID_LINE)  # Normal horizontal line

def create_buttons():
    font = pygame.font.SysFont(None, 40)
    input_buttons = []
    rows = 3
    cols = 3
    button_width = 60
    button_height = 60
    gap = 10
    start_x = WIN_WIDTH - (cols * (button_width + gap)) - 30
    start_y = 50
    for i in range(1, 10):
        col = (i - 1) % cols
        row = (i - 1) // cols
        x = start_x + col * (button_width + gap)
        y = start_y + row * (button_height + gap)
        button = pygame.Rect(x, y, button_width, button_height)
        text = font.render(str(i), True, BLACK)
        text_rect = text.get_rect(center=button.center)
        input_buttons.append((button, text, text_rect))
    return input_buttons

def create_control_buttons(button_name,button_name2):
    control_buttons =[]
    # Draw buttons
    pygame.draw.rect(screen, BLACK, (580, 370, 200, 50), border_radius=5)
    pygame.draw.rect(screen, BLACK, (580, 450, 200, 50), border_radius=5)
    pygame.draw.rect(screen, BLACK, (750, 10, 50, 20), border_radius=5)  # Back button

    # Add text to buttons
    button_font = pygame.font.SysFont(None, 24)
    solve_text = button_font.render(button_name2, True, WHITE)
    solve_button_rect= pygame.Rect((580, 450, 200, 50))
    solve_text_centerd =solve_text.get_rect(center=solve_button_rect.center)
    screen.blit(solve_text, solve_text_centerd)
    control_buttons.append(solve_button_rect)
    
    
    reset_text = button_font.render(button_name, True, WHITE)
    reset_button_rect= pygame.Rect((580, 370, 200, 50))
    reset_text_centerd =reset_text.get_rect(center=reset_button_rect.center)
    screen.blit(reset_text, reset_text_centerd)
    control_buttons.append(reset_button_rect)
    
    back_text = button_font.render("Back", True, WHITE)
    back_button_rect= pygame.Rect((750, 10, 50, 20))
    back_text_centered = back_text.get_rect(center=back_button_rect.center)
    screen.blit(back_text, back_text_centered)
    control_buttons.append(back_button_rect)
    

    return control_buttons

def get_initial_indices(puzzle):
    indices = []
    for i in range(9):
        for j in range(9):
            if puzzle[i, j] != -1:
                indices.append((i, j))
    return indices

# Function to display error message
def display_error_message(message):
    # Create a window for the error message
    error_window = pygame.Surface((400, 100))
    error_window.fill(RED)  # Background color
    error_rect = error_window.get_rect(center=(WIN_WIDTH // 2, WIN_HEIGHT // 2))

    # Render error message text
    text_surface = ERROR_FONT.render(message, True, WHITE)
    text_rect = text_surface.get_rect(center=(error_rect.width // 2, error_rect.height // 2))
    error_window.blit(text_surface, text_rect)

    # Main loop to display error message
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False  # Close error message window if ESC key is pressed

        # Display error message window
        screen.blit(error_window, error_rect.topleft)
        pygame.display.flip()
        
def generate_random_puzzle(difficulty_range):
    global domains , domain_stack
    domains ={v : set(range(1,10)) for v in variables}

    domain_stack.clear()
    empty_cells = 81 - np.random.randint(difficulty_range[0],difficulty_range[1])
    def solve(board):
        for i in range(9):
            for j in range(9):
                if board[i, j] == -1: 
                    nums = list(range(1, 10))
                    random.shuffle(nums) 
                    for num in nums:
                        if is_valid_move(board, i, j, num):
                            board[i, j] = num
                            if solve(board):
                                return True
                            board[i, j] = -1  # Backtrack
                    return False
        return True
    
    def make_some_cells_empty(board, num_empty):
        if num_empty >= 81:
            return np.full_like(board, -1) 
        
        empty_board = np.copy(board)
        indices = list(range(81))
        random.shuffle(indices) 
        
        for idx in indices[:num_empty]:
            row, col = divmod(idx, 9)
            empty_board[row, col] = -1 
        
        return empty_board

    # Initialize an empty Sudoku board
    puzzle = np.full((9, 9), -1, dtype=int)

    # Solve the Sudoku puzzle
    solve(puzzle)
    # print(puzzle)
    puzzle = make_some_cells_empty(puzzle, empty_cells) 

    return puzzle


def Ai_gen_and_solve(difficulty):
    global domains , domain_stack
    puzzle=generate_random_puzzle(difficulty)
    initial_puzzle=puzzle
    print(initial_puzzle)
    buttons = create_buttons()
    control_buttons = create_control_buttons("Regenerate Puzzle","Solve Puzzle")
    selected_number = None
    indices=get_initial_indices(initial_puzzle)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                # print(x ,y)
                for i, (button, _, _) in enumerate(buttons):
                    if button.collidepoint(x, y):
                        selected_number = i + 1  # Assign the selected number from the button
                        print(selected_number)

                row, col = y // CELL_SIZE, x // CELL_SIZE
                if 0 <= row < 9 and 0 <= col < 9:  # Check bounds
                    if selected_number is not None:  # Check if a number is selected
                        if (row,col) not in indices:  # Check if clicked cell is empty
                            if is_valid_move(puzzle,row,col,selected_number):
                                puzzle[row, col] = selected_number  # Assign the selected number to the cell
                            elif selected_number == -1:
                                puzzle[row, col] = selected_number
                            else:
                                display_error_message("Invalid Input: Check Constraints of Game")
                mouse_pos = pygame.mouse.get_pos()
                if control_buttons[0].collidepoint(mouse_pos):
                    if len(get_initial_indices(puzzle))==81:
                        error_count = 0
                        for i in range(9):
                            for j in range(9):
                                if puzzle[i,j] != -1:
                                    value = puzzle[i,j]
                                    puzzle[i,j]=-1
                                    if is_valid_move(puzzle,i,j,value):
                                        puzzle[i,j]=value
                                    else:
                                        puzzle[i,j]=value
                                        error_count+=1
                        if error_count != 0:
                            display_error_message("Incorrect Solution")
                        else:
                            display_error_message("Correct Solution")
                    else:
                        # define_domain(puzzle)
                        one_value_array = []
                        for i in range (9) : 
                            for j in range (9) : 
                                if puzzle[i , j]  != -1 : 
                                    one_value_array.append((i,j))
                        
                        
                        node_consistency(puzzle)
                        x = ac3(puzzle)
                        new_arcs  = []
                        # exit_value   = ac3(puzzle , arcs=[(Y, X) for Y in neighbors(X)])
                        for cell in one_value_array : 
                            for n in neighbors(cell) : 
                                new_arcs.append((n,cell))
                        tp = ac3(puzzle , new_arcs)
                        assignment={}
                        # print(domains)
                        # print(domain_stack)
                        start_time  = time.time()
                        solvable , solution = bts(assignment, puzzle)
                        end_time  = time.time()
                        print(f"Time Taken is  {end_time - start_time}")
                        if solvable:
                            print("Solution:")
                            print_sudoku(puzzle)
                            # domains ={v : set(range(1,10)) for v in variables}

                            # domain_stack.clear()

                        else:
                            # print("No solution exists.")
                            print("Hello There ")
                            print(puzzle)
                            display_error_message("No Solution Exists")
                elif control_buttons[1].collidepoint(mouse_pos):
                    # ai_solve()
                    puzzle=generate_random_puzzle(difficulty)
                    indices=get_initial_indices(puzzle)
                    # print('2')
                elif control_buttons[2].collidepoint(mouse_pos):
                    main()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DELETE:
                    selected_number = -1


        draw_grid()
        draw_values(puzzle)
        for button, text, text_rect in buttons:
            pygame.draw.rect(screen, WHITE, button)
            pygame.draw.rect(screen, BLACK, button, 2)
            screen.blit(text, text_rect)
        create_control_buttons("Regenerate Puzzle","Solve Puzzle")
        
        
        # Update the display
        pygame.display.flip()

def user_gen_ai_solve():
    puzzle = np.full((9, 9), -1)
    indices=[]
    buttons = create_buttons()
    control_buttons = create_control_buttons("Empty Puzzle","Solve Puzzle")
    selected_number = None
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                print(x ,y)
                for i, (button, _, _) in enumerate(buttons):
                    if button.collidepoint(x, y):
                        selected_number = i + 1  # Assign the selected number from the button
                        print(selected_number)

                row, col = y // CELL_SIZE, x // CELL_SIZE
                if 0 <= row < 9 and 0 <= col < 9:  # Check bounds
                    if selected_number is not None:  # Check if a number is selected
                        if (row,col) not in indices:  # Check if clicked cell is empty
                            if is_valid_move(puzzle,row,col,selected_number):
                                puzzle[row, col] = selected_number  # Assign the selected number to the cell
                            elif selected_number == -1:
                                puzzle[row, col] = selected_number
                            else:
                                display_error_message("Invalid Input: Check Constraints of Game")
                                
                mouse_pos = pygame.mouse.get_pos()
                if control_buttons[0].collidepoint(mouse_pos):
                    if len(get_initial_indices(puzzle))==0:
                        display_error_message("Can't Solve Empty Board")
                    else:
                        # define_domain(puzzle)
                        one_value_array = []
                        for i in range (9) : 
                            for j in range (9) : 
                                if puzzle[i , j]  != -1 : 
                                    one_value_array.append((i,j))
                        
                        
                        node_consistency(puzzle)
                        x = ac3(puzzle)
                        new_arcs  = []
                        # exit_value   = ac3(puzzle , arcs=[(Y, X) for Y in neighbors(X)])
                        for cell in one_value_array : 
                            for n in neighbors(cell) : 
                                new_arcs.append((n,cell))
                        tp = ac3(puzzle , new_arcs)
                        assignment={}
                        # print(domains)
                        # print(domain_stack)
                        start_time  = time.time()
                        solvable , solution = bts(assignment, puzzle)
                        end_time  = time.time()
                        print(f"Time Taken is  {end_time - start_time}")
  
                        if solvable:
                            print("Solution:")
                            print_sudoku(puzzle)
                        else:
                            # print("No solution exists.")
                            display_error_message("No Solution Exists")

                elif control_buttons[1].collidepoint(mouse_pos):
                    puzzle = np.full((9, 9), -1)
                    
                elif control_buttons[2].collidepoint(mouse_pos):
                    main()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DELETE:
                    selected_number = -1

        draw_grid()
        draw_values(puzzle)
        for button, text, text_rect in buttons:
            pygame.draw.rect(screen, WHITE, button)
            pygame.draw.rect(screen, BLACK, button, 2)
            screen.blit(text, text_rect)
        create_control_buttons("Empty Puzzle","Solve Puzzle")
        
        
        # Update the display
        pygame.display.flip()

def user_gen_user_solve():
    puzzle = np.full((9, 9), -1)
    indices=[]
    buttons = create_buttons()
    control_buttons = create_control_buttons("Empty Puzzle","Solve Puzzle")
    selected_number = None
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                
                x, y = pygame.mouse.get_pos()
                # print(x ,y)
                for i, (button, _, _) in enumerate(buttons):
                    if button.collidepoint(x, y):
                        selected_number = i + 1  # Assign the selected number from the button

                row, col = y // CELL_SIZE, x // CELL_SIZE
                if 0 <= row < 9 and 0 <= col < 9:  # Check bounds
                    if selected_number is not None:  # Check if a number is selected
                        if  selected_number in  domains[(row,col)] :
                            puzzle[row, col] = selected_number  # Assign the selected number to the cell
                        else : 
                                print("Not in domain choose another valueb")
                        
                                
                mouse_pos = pygame.mouse.get_pos()
                if control_buttons[0].collidepoint(mouse_pos):
                    if len(get_initial_indices(puzzle)) == 81:
                        error_count = 0
                        for i in range(9):
                            for j in range(9):
                                if puzzle[i,j] != -1:
                                    value = puzzle[i,j]
                                    puzzle[i,j]=-1
                                    if is_valid_move(puzzle,i,j,value):
                                        puzzle[i,j]=value
                                    else:
                                        puzzle[i,j]=value
                                        error_count+=1
                        if error_count != 0:
                            display_error_message("Incorrect Solution")
                        else:
                            display_error_message("Correct Solution")
                    else:
                        display_error_message("Please Fill Empty Cells")
                    

                elif control_buttons[1].collidepoint(mouse_pos):
                    puzzle = np.full((9, 9), -1)
                    
                elif control_buttons[2].collidepoint(mouse_pos):
                    main()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_DELETE:
                    selected_number = -1

        draw_grid()
        draw_values(puzzle)
        for button, text, text_rect in buttons:
            pygame.draw.rect(screen, WHITE, button)
            pygame.draw.rect(screen, BLACK, button, 2)
            screen.blit(text, text_rect)
        create_control_buttons("Empty Puzzle","Check Answer")
        
        
        # Update the display
        pygame.display.flip()


# Function to display difficulty selection window
def display_difficulty_selection(screen):
    difficulty_selected = None
    font = pygame.font.SysFont(None, 40)

    # Create a window for difficulty selection
    difficulty_window = pygame.Surface((400, 200))
    difficulty_window.fill(WHITE)  # Background color
    difficulty_rect = difficulty_window.get_rect(center=(WIN_WIDTH // 2, WIN_HEIGHT // 2))

    screen.fill(WHITE)
    background_image = pygame.image.load("background.jpg")  
    background_image = pygame.transform.scale(background_image, (WIN_WIDTH, WIN_HEIGHT))
    screen.blit(background_image, (0,0))
    buttons=[pygame.Rect((WIN_WIDTH - 350) // 2, 100, 350, 50),
            pygame.Rect((WIN_WIDTH - 350) // 2, 200, 350, 50),
            pygame.Rect((WIN_WIDTH - 350) // 2, 300, 350, 50)]
    button_texts = ["Easy","Medium","Hard"]

    # Main loop for difficulty selection
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                for i, button_rect in enumerate(buttons):
                    if button_rect.collidepoint(mouse_pos):
                        difficulty_selected = button_texts[i]
                        running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    main()
        for i, button in enumerate(buttons):
            pygame.draw.rect(screen, BUTTON_COLOR, button, border_radius=10)
            text = BUTTON_FONT.render(button_texts[i], True, WHITE)
            text_rect = text.get_rect(center=button.center)
            screen.blit(text, text_rect)
        
        pygame.display.flip()
        
        
        
    return difficulty_selected




# Main function to handle events
def main():
    pygame.init()
    # Set up the display
    screen.fill((255, 255, 255))
    pygame.display.set_caption("Sudoku Solver")
    background_image = pygame.image.load("background.jpg")  
    background_image = pygame.transform.scale(background_image, (WIN_WIDTH, WIN_HEIGHT))
    screen.blit(background_image, (0,0))
    
    
    buttons=[pygame.Rect((WIN_WIDTH - 350) // 2, 100, 350, 50),
             pygame.Rect((WIN_WIDTH - 350) // 2, 200, 350, 50),
             pygame.Rect((WIN_WIDTH - 350) // 2, 300, 350, 50)]
    button_texts = [
    "Mode 1: AI Generate And Solve",
    "Mode 2: User Generate And AI Solve",
    "Mode 3: User Generate And User Solve"
]

    running=True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if buttons[0].collidepoint(mouse_pos):
                    selected_difficulty = display_difficulty_selection(screen)
                    if selected_difficulty == "Easy":
                        Ai_gen_and_solve(EASY)
                    if selected_difficulty == "Medium":
                        Ai_gen_and_solve(MEDIUM)
                    if selected_difficulty == "Hard":
                        Ai_gen_and_solve(HARD)
                elif buttons[1].collidepoint(mouse_pos):
                    user_gen_ai_solve()
                elif buttons[2].collidepoint(mouse_pos):
                    user_gen_user_solve()



        # Draw buttons
        for i, button in enumerate(buttons):
            pygame.draw.rect(screen, BUTTON_COLOR, button, border_radius=10)
            text = BUTTON_FONT.render(button_texts[i], True, WHITE)
            text_rect = text.get_rect(center=button.center)
            screen.blit(text, text_rect)
        
        pygame.display.flip()



# Run the main loop
if __name__ == "__main__":
    main()
