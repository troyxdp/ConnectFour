import numpy as np
from game import ConnectFour

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