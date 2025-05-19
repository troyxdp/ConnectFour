import numpy as np
from game import ConnectFour
from neural_network import NeuralNetwork
import copy

class Agent():
    def __init__(
            self, 
            is_player_one: bool,
            game=None, 
            results_scores={'win': 30, 'loss': -30, 'draw': -3, 'ongoing': -0.001, 'three_in_a_row': 0.75}
        ):
        # Check results scores is valid
        if results_scores['win'] is None or results_scores['loss'] is None or results_scores['draw'] is None or results_scores['ongoing'] is None or results_scores['three_in_a_row'] is None:
            raise Exception

        # Initialize values
        self._game = game
        self._is_player_one = is_player_one
        self._results_scores = results_scores
        self._prev_state_threes_player = 0
        self._prev_state_threes_opponent = 0

    def evaluate_reward(self):
        # REWARDS FOR GAME RESULTS
        # Player 1 wins
        reward = 0
        if self._game.get_winner() == 1:
            if self._is_player_one:
                reward += self._results_scores['win']
            else:
                reward += self._results_scores['loss']
        # Player 2 wins
        elif self._game.get_winner() == -1:
            if self._is_player_one:
                reward += self._results_scores['loss']
            else:
                reward += self._results_scores['win']
        # Draw
        elif self._game.get_winner() == 2:
            reward += self._results_scores['draw']
        # Game is not complete
        reward += self._results_scores['ongoing']

        # REWARDS FOR GETTING 3 IN A ROW
        curr_state_threes_player = 0
        curr_state_threes_opponent = 0
        board_state = self._game.get_board() 
        # Check rows
        for i in range(len(board_state)):
            for j in range(len(board_state[0]) - 2):
                if board_state[i][j] == board_state[i][j+1] and board_state[i][j+1] == board_state[i][j+2] and not board_state[i][j] == 0:
                    if board_state[i][j] == 1 and self._is_player_one or board_state[i][j] == -1 and not self._is_player_one:
                        curr_state_threes_player += 1
                    else:
                        curr_state_threes_opponent += 1
        # Check columns
        for i in range(len(board_state) - 2):
            for j in range(len(board_state[0])):
                if board_state[i][j] == board_state[i+1][j] and board_state[i+1][j] == board_state[i+2][j] and not board_state[i][j] == 0:
                    if board_state[i][j] == 1 and self._is_player_one or board_state[i][j] == -1 and not self._is_player_one:
                        curr_state_threes_player += 1
                    else:
                        curr_state_threes_opponent += 1
        # Check BL-TR diagonals
        for i in range(len(board_state) - 2):
            for j in range(len(board_state[0]) - 2):
                if board_state[i][j] == board_state[i+1][j+1] and board_state[i+1][j+1] == board_state[i+2][j+2] and not board_state[i][j] == 0:
                    if board_state[i][j] == 1 and self._is_player_one or board_state[i][j] == -1 and not self._is_player_one:
                        curr_state_threes_player += 1
                    else:
                        curr_state_threes_opponent += 1
        # Check TL-BR diagonals
        for i in range(len(board_state) - 2):
            for j in range(len(board_state[0]) - 2):
                if board_state[i+2][j] == board_state[i+1][j+1] and board_state[i+1][j+1] == board_state[i][j+2] and not board_state[i+2][j] == 0:
                    if board_state[i+2][j] == 1 and self._is_player_one or board_state[i+2][j] == -1 and not self._is_player_one:
                        curr_state_threes_player += 1
                    else:
                        curr_state_threes_opponent += 1

        # Calculate the new number of threes achieved for opponent and player and update reward accordingly
        reward += (curr_state_threes_player - self._prev_state_threes_player) * self._results_scores['three_in_a_row']
        reward -= (curr_state_threes_opponent - self._prev_state_threes_opponent) * self._results_scores['three_in_a_row']

        # Update counts of number of threes
        self._prev_state_threes_player = curr_state_threes_player
        self._prev_state_threes_opponent = curr_state_threes_opponent

        # Return reward value
        return reward
        

    def sense(self):
        return self._game.get_moves()

    def act(self, eps):
        pass

    def reinforce(self):
        pass

    def set_game(self, game):
        self._game = game

    def set_player_number(self, player_num: int):
        if player_num not in range(1, 3):
            raise Exception
        if player_num == 1:
            self._is_player_one = True
        else:
            self._is_player_one = False

    def is_player_one(self):
        return self._is_player_one

    def reset_prev_state_threes_counts(self):
        self._prev_state_threes_player = 0
        self._prev_state_threes_opponent = 0



class RandomAgent(Agent):
    def __init__(
            self,  
            is_player_one: bool, 
            results_scores={'win': 30, 'loss': -30, 'draw': -3, 'ongoing': -0.001, 'three_in_a_row': 1},
            game=None
        ):
        super().__init__(game, is_player_one, results_scores)

    def act(self, eps):
        # Get number of legal moves
        moves = self.sense()
        num_legal_moves = 0
        for move in moves:
            if move:
                num_legal_moves += 1

        # Select a random legal move
        selected_move_num = np.random.randint(low=0, high=num_legal_moves)
        for i, move in enumerate(moves):
            if move:
                selected_move_num -= 1
                if selected_move_num < 0:
                    return i

        # No legal moves
        raise Exception("Error: no legal moves available")



class ConnectFourAgent(Agent):
    def __init__(
            self, 
            is_player_one: bool, 
            nn: NeuralNetwork,
            results_scores={'win': 30, 'loss': -30, 'draw': -3, 'ongoing': -0.001, 'three_in_a_row': 0.75}, 
            game=None
        ):
        super().__init__(game, is_player_one, results_scores)
        self.nn_pred = nn
        self.nn_target = nn

    def act(self, eps):
        # Select move
        rand_val = np.random.rand()
        if rand_val < eps: # Select a random legal move
            # Get number of legal moves
            moves = self.sense()
            num_legal_moves = 0
            for move in moves:
                if move:
                    num_legal_moves += 1

            # Select a random legal move
            selected_move_num = np.random.randint(low=0, high=num_legal_moves)
            for i, move in enumerate(moves):
                if move:
                    selected_move_num -= 1
                    if selected_move_num < 0:
                        return i

            # No legal moves
            raise Exception("Error: no legal moves available")

        else: # Select the move with the highest Q-Value
            # Get Q-Values
            board_state = self._game.get_board().flatten()
            nn_input = np.concatenate((board_state, np.array([1 if self._game.get_is_player_one_turn() else -1])))
            q_values = self.nn_pred.feed_forward(nn_input)

            # Sort (Action, Q-Value) in descending order
            q_val_actions = np.zeros((7, 2))
            for i in range(len(q_val_actions)):
                q_val_actions[i][0] = i
                q_val_actions[i][1] = q_values[i]

            for o in range(len(q_val_actions) - 1):
                for i in range(o+1, len(q_val_actions)):
                    if q_val_actions[i][1] > q_val_actions[o][1]:
                        temp = q_val_actions[i].copy()
                        q_val_actions[i] = q_val_actions[o].copy()
                        q_val_actions[o] = temp
            
            # Choose legal move
            for action in q_val_actions:
                if self._game.is_legal_move(int(action[0])):
                    return int(action[0])
            
            # Raise exception because no legal move was found
            raise Exception("Error: no legal moves available")

    def get_nn_pred_q_values(self, nn_input):
        return self.nn_pred.feed_forward(nn_input)

    def get_nn_target_q_values(self, nn_input):
        return self.nn_target.feed_forward(nn_input)

    def copy_to_target(self):
        self.nn_target = copy.copy(self.nn_pred)

    def reinforce(self, lr, delta):
        self.nn_pred.update_network(lr, delta)