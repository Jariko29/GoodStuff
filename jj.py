#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <limits>
#include <stack>

using namespace std;

const int GRID_SIZE = 3;

class Move {
public:
    int inputrow;
    int inputcol;
    Move();
    void Display();
    void ReverseGame(int);
    void playGame();
    stack<Move> undoStack;
    stack<Move> redoStack;
private:
    int grid[GRID_SIZE][GRID_SIZE];
    bool isGameComplete();
    void undoMove();
    void redoMove();
    void performMove(Move&);
};

Move::Move() {
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++)
            grid[row][col] = 9;
    }
}

int Difficulty() {
    int difficulty = 0;
    cout << "Give difficulty (1-9) :";
    cin >> difficulty;
    while (difficulty < 1 || difficulty > 9) {
        cout << "\nWrong input, give correct difficulty level (1-9) :";
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cin >> difficulty;
    }
    return difficulty;
}

int Row() {
    int row = 0;
    cout << "Enter row(1-3) :";
    cin >> row;
    while (row < 1 || row > 3) {
        cout << "\nWrong input, give the correct row (1-3) :";
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cin >> row;
    }
    return row - 1;
}

int Col() {
    int col = 0;
    cout << "Enter col(1-3) :";
    cin >> col;
    while (col < 1 || col > 3) {
        cout << "\nWrong input, give the correct col (1-3) :";
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cin >> col;
    }
    return col - 1;
}

int Choice() {
    int choice = 0;
    cin >> choice;
    while (choice < 1 || choice > 3) {
        cout << "\nWrong input, give a correct choice (1-3) :";
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cin >> choice;
    }
    return choice;
}

void Move::Display() {
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            cout << setw(3) << grid[row][col];
        }
        cout << endl;
    }
    cout << endl;
}

void Move::ReverseGame(int diff) {
    srand(time(NULL));

    while (diff != 0) {
        int row = rand() % GRID_SIZE;
        int col = rand() % GRID_SIZE;
      
        // Subtract one from all cells in the same row and column
        for (int i = 0; i < GRID_SIZE; i++) {
            grid[row][i]--;
            grid[i][col]--;
        }
        grid[row][col]++;
        diff--;
    }
}

bool Move::isGameComplete() {
    for (int row = 0; row < 3; row++) {
        for (int col = 0; col < 3; col++) {
            if (grid[row][col] != 9)
                return false;
        }
    }
    return true;
}

void Move::performMove(Move& move) {
    for (int i = 0; i < GRID_SIZE; i++) {
        grid[move.inputrow][i]++;
        grid[i][move.inputcol]++;
    }
    grid[move.inputrow][move.inputcol]--;
}

void Move::playGame() {
    int diff = Difficulty();
    ReverseGame(diff);
    while (!isGameComplete()) {
        Display();
        int row = Row();
        int col = Col();
        Move playerMove;
        playerMove.inputrow = row;
        playerMove.inputcol = col;
        performMove(playerMove);
        undoStack.push(playerMove);
        while (!redoStack.empty())
            redoStack.pop();
    }
}

void Move::undoMove() {
    if (!undoStack.empty()) {
        Move move = undoStack.top();
        undoStack.pop();

        // Decrease all digits in the same row and column by one
        for (int i = 0; i < 3; i++) {
            // Decrease digit and handle wraparound
            grid[move.inputrow][i] = (grid[move.inputrow][i] - 1 + 10) % 10;
            grid[i][move.inputcol] = (grid[i][move.inputcol] - 1 + 10) % 10;
        }
        // Push the move onto the redo stack
        redoStack.push(move);
    }
}

void Move::redoMove() {
    if (!redoStack.empty()) {
        Move move = redoStack.top();
        redoStack.pop();
        for (int i = 0; i < 3; i++) {
            // Decrease digit and handle wraparound
            grid[move.inputrow][i] = (grid[move.inputrow][i] + 1 + 10) % 10;
            grid[i][move.inputcol] = (grid[i][move.inputcol] + 1 + 10) % 10;
        }
        // Push the move onto the redo stack
        undoStack.push(move);
    }
}

int main() {
    // Test the fixed code
    Move move;
    move.playGame();
    return 0;
}