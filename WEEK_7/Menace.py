import random
import json
from collections import defaultdict

# Represents the tic-tac-toe board and game logic
class Board:
    def __init__(self):
        self.board = [" "] * 9  # A list of 9 spaces to represent the 3x3 grid

    def __str__(self):
        # Creating a board display
        display = "\n"
        for i in range(0, 9, 3):
            display += f" {self.board[i]} | {self.board[i+1]} | {self.board[i+2]} \n"
            if i < 6:
                display += "---+---+---\n"
        return display

    def get_valid_moves(self):
        # Returns a list of valid positions where a move can be made (i.e., empty spaces)
        return [i for i in range(9) if self.board[i] == " "]

    def play_move(self, position, marker):
        # Place the marker ('X' or 'O') on the board at the given position
        self.board[position] = marker

    def check_win(self):
        # List of all possible win conditions (3 in a row)
        win_conditions = [
            (0, 1, 2),
            (3, 4, 5),
            (6, 7, 8),  # Horizontal
            (0, 3, 6),
            (1, 4, 7),
            (2, 5, 8),  # Vertical
            (0, 4, 8),
            (2, 4, 6),
        ]  # Diagonal

        # Check if any win condition is met (all positions have the same non-empty marker)
        for a, b, c in win_conditions:
            if self.board[a] == self.board[b] == self.board[c] and self.board[a] != " ":
                return True
        return False

    def check_draw(self):
        # A draw occurs when there are no empty spaces left and no winner
        return " " not in self.board

    def get_board_state(self):
        # Returns the current board state as a string (used for tracking in MENACE)
        return "".join(self.board)


# Represents MENACE, which plays and learns the game of tic-tac-toe
class Menace:
    def __init__(self, marker):
        self.marker = marker  # 'X' or 'O', depending on which side MENACE is playing
        self.matchboxes = defaultdict(
            lambda: [1] * 9
        )  # Track board states and move weights
        self.current_game_moves = []  # Stores the moves MENACE made in the current game

    def load_matchboxes(self, filename):
        # Load pre-trained matchbox states from a JSON file, if available
        try:
            with open(filename, "r") as file:
                self.matchboxes = json.load(file)
        except FileNotFoundError:
            print("No saved matchboxes found. Starting fresh.")

    def save_matchboxes(self, filename):
        # Save matchbox states to a JSON file for future use
        with open(filename, "w") as file:
            json.dump(self.matchboxes, file)

    def select_move(self, board):
        # Get the current board state as a string
        state = board.get_board_state()
        # If the state isn't in matchboxes, initialize it (equal weight for all moves)
        if state not in self.matchboxes:
            self.matchboxes[state] = [
                1 if board.board[i] == " " else 0 for i in range(9)
            ]

        # Select a move probabilistically, weighted by the matchbox (non-empty valid moves)
        move_weights = self.matchboxes[state]
        valid_moves = board.get_valid_moves()
        move = random.choices(valid_moves, [move_weights[i] for i in valid_moves])[0]

        # Store this move for potential reinforcement later
        self.current_game_moves.append((state, move))
        return move

    def reward_win(self):
        # If MENACE wins, reward the moves made in this game by adding beads (weights)
        for state, move in self.current_game_moves:
            self.matchboxes[state][move] += 3  # Add 3 beads for winning moves

    def reward_draw(self):
        # If the game is a draw, add fewer beads (neutral outcome)
        for state, move in self.current_game_moves:
            self.matchboxes[state][move] += 1  # Add 1 bead for draw moves

    def penalize_loss(self):
        # If MENACE loses, remove beads (penalize losing moves)
        for state, move in self.current_game_moves:
            if self.matchboxes[state][move] > 1:
                self.matchboxes[state][move] -= 1  # Remove 1 bead for losing moves

    def reset_game(self):
        # Clear the moves from the last game
        self.current_game_moves = []


# Represents a human player
class HumanPlayer:
    def __init__(self, marker):
        self.marker = marker

    def select_move(self, board):
        # Ask the human player for their move
        valid_moves = board.get_valid_moves()
        move = -1
        while move not in valid_moves:
            try:
                move = int(input(f"Choose your move ({valid_moves}): "))
            except ValueError:
                continue
        return move

# Main function to play a game of tic-tac-toe
def play_game(player1, player2):
    board = Board()
    current_player = player1

    while True:
        # Display the current board state
        print(board)

        # Get the current player's move and play it
        move = current_player.select_move(board)
        board.play_move(move, current_player.marker)

        # Check if the current player has won
        if board.check_win():
            print(board)
            print(f"{current_player.marker} wins!")
            if isinstance(current_player, Menace):
                current_player.reward_win()
            return current_player.marker

        # Check if the game is a draw
        if board.check_draw():
            print(board)
            print("It's a draw!")
            if isinstance(player1, Menace):
                player1.reward_draw()
            if isinstance(player2, Menace):
                player2.reward_draw()
            return "Draw"

        # Switch turns
        current_player = player2 if current_player == player1 else player1


# Example gameplay
if __name__ == "__main__":
    # Initialize MENACE and human players
    menace = Menace(marker="X")
    human = HumanPlayer(marker="O")

    # Load or train MENACE before the game
    menace.load_matchboxes("menace_data.json")

    # Play 100 games against the human
    for _ in range(100):
        menace.reset_game()
        play_game(menace, human)

    # Save MENACE's learned strategies
    menace.save_matchboxes("menace_data.json")