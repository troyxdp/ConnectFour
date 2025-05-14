import numpy as np

class ConnectFour():

    def __init__(self):
        self.board = np.zeros((6,7)) # bottom row is 0; top row is 5
        self.winner = 0 # -1 for player 2, 0 for incomplete, 1 for player 1, 2 for draw
        self.is_player_one_turn = True

    def make_move(self, col : int):
        if col not in range(0, 7):
            raise Exception
        for r in range(6):
            if self.board[r][col] == 0:
                self.board[r][col] = 1 if self.is_player_one_turn else -1
                self.is_player_one_turn = not self.is_player_one_turn
                return
        raise Exception

    def get_moves(self):
        # set all moves to illegal
        moves = [False for i in range(len(self.board[0]))]
        for i in range(len(moves)):
            # set move to legal when top space of a column is open
            if self.board[5][i] == 0:
                moves[i] = True
        return moves

    def calculate_winner(self):
        # check columns
        for c in range(len(self.board[0])):
            for r in range(3):
                if self.board[r][c] == 0:
                    continue
                if self.board[r][c] == self.board[r+1][c] and self.board[r+1][c] == self.board[r+2][c] and self.board[r+2][c] == self.board[r+3][c]:
                    if self.board[r][c] == 1:
                        self.winner = 1
                        return 1
                    self.winner = -1
                    return -1

        # check rows
        for r in range(len(self.board)):
            for c in range(4):
                if self.board[r][c] == 0:
                    continue
                if self.board[r][c] == self.board[r][c+1] and self.board[r][c+1] == self.board[r][c+2] and self.board[r][c+2] == self.board[r][c+3]:
                    if self.board[r][c] == 1:
                        self.winner = 1
                        return 1
                    self.winner = -1
                    return -1

        # check BL to TR diagonals
        for r in range(3):
            for c in range(4):
                if self.board[r][c] == 0:
                    continue
                if self.board[r][c] == self.board[r+1][c+1] and self.board[r+1][c+1] == self.board[r+2][c+2] and self.board[r+2][c+2] == self.board[r+3][c+3]:
                    if self.board[r][c] == 1:
                        self.winner = 1
                        return 1
                    self.winner = -1
                    return -1

        # check TL to BR diagonals
        for r in range(3):
            for c in range(4):
                if self.board[r+3][c] == 0:
                    continue
                if self.board[r+3][c] == self.board[r+2][c+1] and self.board[r+2][c+1] == self.board[r+1][c+2] and self.board[r+1][c+2] == self.board[r][c+3]:
                    if self.board[r+3][c] == 1:
                        self.winner = 1
                        return 1
                    self.winner = -1
                    return -1

        return 0

    def is_over(self):
        return self.calculate_winner() != 0 or not np.any(self.get_moves())

    def reset_board(self):
        self.board = np.zeros((6,7))
        self.winner = 0
        self.is_player_one_turn = True

    def __str__(self):
        to_ret = ''
        for r in range(5, -1, -1):
            for c in range(len(self.board[0])):
                if self.board[r][c] == 1:
                    to_ret += 'X'
                elif self.board[r][c] == -1:
                    to_ret += 'O'
                else:
                    to_ret += '.'
            to_ret += '\n'
        to_ret = to_ret[:-1]
        return to_ret

    def __repr__(self):
        return self.__str__()

    def get_is_player_one_turn(self):
        return self.is_player_one_turn

    def get_winner(self):
        return self.winner

if __name__ == '__main__':
    game = ConnectFour()
    print(game)
    while not game.is_over():  
        if game.get_is_player_one_turn():
            print("Player one's turn")
        else:
            print("Player two's turn")

        move = int(input("Please enter a move: "))
        try:
            game.make_move(move)
            print(game)
        except Exception:
            print("Error: illegal move enterred. Please enter a legal move")
    winner = game.get_winner()
    if winner == 1:
        print("Player one wins!")
    elif winner == -1:
        print("Player two wins!")
    else:
        print("Draw")