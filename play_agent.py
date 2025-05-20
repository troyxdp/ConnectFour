from agent import ConnectFourAgent, UserAgent
from neural_network import NeuralNetwork
from game import ConnectFour
import numpy as np
import os

if __name__ == '__main__':
    # Get number of neurons of best model
    num_neurons = np.load(os.path.join(os.getcwd(), 'best_model', 'num_neurons.npy'), allow_pickle=True)

    # Get path of best model
    dir = input("Please input the directory path of the model you would like to play against: ")
    if not os.path.isdir(dir):
        raise Exception("Please enter a valid directory path")

    # Get activation functions
    activation_functions = []
    with open(os.path.join(dir, 'activation_functions.txt'), 'r') as f:
            act_fns = f.readlines()
            for act_fn in act_fns:
                activation_functions.append(act_fn.strip())

    # Instantiate neural network
    nn = NeuralNetwork(num_neurons, activation_functions)

    # Load weights from best model
    nn.load_network(dir)

    # Get whether user wants to be player 1 or not
    wants_player_one = input("Enter TRUE if you would like to be Player 1: ")
    if wants_player_one.lower() == 'true':
        is_user_player_one = True
    else:
        is_user_player_one = False

    # Initialize game
    game = ConnectFour()

    # Initialize agents
    user_agent = UserAgent(is_player_one=is_user_player_one, game=game)
    bot_agent = ConnectFourAgent(is_player_one=(not is_user_player_one), game=game, nn=nn)

    # Play game
    is_player_one_turn = True
    while not game.is_over():
        print(game)
        move = -1
        if (is_player_one_turn and is_user_player_one) or (not is_player_one_turn and not is_user_player_one):
            move = user_agent.act(0)
        else:
            move = bot_agent.act(0)
        game.make_move(move)
        is_player_one_turn = not is_player_one_turn
        print()
    print(game)

    # Display result
    if (game.get_winner() == 1 and is_user_player_one) or (game.get_winner() == -1 and not is_user_player_one):
        print("YOU WIN!")
    elif game.get_winner() != 0:
        print("YOU LOSE!")
    else:
        print("DRAW!")
    