import tkinter as tk
import random

def create_minesweeper_grid(rows, cols, mines):
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    mine_positions = random.sample(range(rows*cols), mines)
    for pos in mine_positions:
        row, col = divmod(pos, cols)
        grid[row][col] = 'X'
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if 0 <= row+i < rows and 0 <= col+j < cols and grid[row+i][col+j] != 'X':
                    grid[row+i][col+j] += 1
    return grid

def reveal_square(row, col):
    if buttons[row][col]['text'] != ' ' or game_over[0]:
        return
    buttons[row][col]['text'] = str(grid[row][col])
    buttons[row][col]['relief'] = tk.SUNKEN
    if grid[row][col] == 'X':
        buttons[row][col]['background'] = 'red'
        game_over[0] = True
    elif grid[row][col] == 0:
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if 0 <= row+i < rows and 0 <= col+j < cols:
                    reveal_square(row+i, col+j)

rows, cols, mines = 10, 10, 10
grid = create_minesweeper_grid(rows, cols, mines)
game_over = [False]

root = tk.Tk()
buttons = [[tk.Button(root, text=' ', width=2, command=lambda row=row, col=col: reveal_square(row, col)) 
            for col in range(cols)] for row in range(rows)]
for i in range(rows):
    for j in range(cols):
        buttons[i][j].grid(row=i, column=j)
root.mainloop()