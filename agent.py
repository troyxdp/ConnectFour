import numpy as np
from game import ConnectFour
from neural_network import NeuralNetwork

class Agent():
    def __init__(self, game: ConnectFour, is_player_one: bool, results_scores={'win': 1, 'loss': -1, 'draw': 0, 'ongoing': 0}):
        # Check results scores is valid
        if results_scores['win'] is None or results_scores['loss'] is None or results_scores['draw'] is None or results_scores['ongoing'] is None:
            raise Exception

        # Initialize values
        self._game = game
        self._is_player_one = is_player_one
        self._results_scores = results_scores

    def evaluate_reward(self):
        # Player 1 wins
        if self._game.get_winner() == 1:
            if self._is_player_one:
                return self._results_scores['win']
            else:
                return self._results_scores['loss']
        # Player 2 wins
        elif self._game.get_winner() == -1:
            if self._is_player_one:
                return self._results_scores['loss']
            else:
                return self._results_scores['win']
        # Draw
        elif self._game.get_winner() == 2:
            return self._results_scores['draw']
        # Game is not complete
        return self._results_scores['ongoing']

    def sense(self):
        return self._game.get_moves()

    def act(self):
        pass

    def reinforce(self):
        pass



class RandomAgent(Agent):
    def __init__(
            self, 
            game: ConnectFour, 
            is_player_one: bool, 
            results_scores={'win': 1, 'loss': -1, 'draw': 0, 'ongoing': 0}
        ):
        super().__init__(game, is_player_one, results_scores)

    def act(self):
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
                if selected_move_num == 0:
                    return i



class ConnectFourAgent(Agent):
    def __init__(
            self, 
            game: ConnectFour, 
            s_player_one: bool, 
            results_scores={'win': 1, 'loss': -1, 'draw': 0, 'ongoing': 0}, 
            nn: NeuralNetwork,
            eps: float,
            gamma: float,
            alpha: float
        ):
        super().__init__(game, is_player_one, results_scores)
        self.nn = nn
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha

    def act(self):
        # Get Q-Values
        board_state = self.game.get_board().flatten()
        nn_input = np.concatenate(board_state, [1 if game.get_is_player_one_turn() else -1])
        q_values = nn.feed_forward(nn_input)

        # Select move
        rand_val = np.random.rand()
        if rand_val < eps: # Select a random legal move
            # Get legal moves
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
                    if selected_move_num == 0:
                        return i
        else: # Select the move with the highest Q-Value
            # Sort (Action, Q-Value) tuple in descending order
            q_val_actions = zip(np.arange(len_q_values), q_values)
            for o in range(len(q_val_actions) - 1):
                for i in range(o+1, len(q_val_actions)):
                    if q_val_actions[i][1] > q_val_actions[o][1]
                        temp = q_val_actions[i].copy()
                        q_val_actions[i] = q_val_actions[o].copy()
                        q_val_actions[o] = temp
            
            # Choose legal move
            for action in q_val_actions:
                if self.game.is_legal_move(action[0]):
                    return action[0]
            
            # Raise exception because no legal move was found
            raise Exception

