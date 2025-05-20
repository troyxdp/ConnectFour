from agent import Agent, ConnectFourAgent, RandomAgent
from neural_network import NeuralNetwork
from game import ConnectFour
import os
import numpy as np

def match(agent_1: Agent, agent_2: Agent):
    score = [0, 0]
    for i in range(7): # play games with different starting moves
        # Initialize game
        game = ConnectFour()

        # Initialize agents for game
        agent_1.set_game(game)
        agent_2.set_game(game)
        agent_1.set_player_number(1)
        agent_2.set_player_number(2)
        agent_1.reset_prev_state_threes_counts()
        agent_2.reset_prev_state_threes_counts()

        # Play game starting with player one dropping piece in column i
        game.make_move(i)
        is_player_one_turn = False
        while not game.is_over():
            if is_player_one_turn:
                move = agent_1.act(0)
                try:
                    game.make_move(move)
                except Exception:
                    raise Exception("Error: Player 1 could not make move")
            else:
                move = agent_2.act(0)
                try:
                    game.make_move(move)
                except Exception:
                    raise Exception("Error: Player 2 could not make move")
            is_player_one_turn = not is_player_one_turn
                
        # Update score according to who won
        if game.get_winner() == 1:
            score[0] += 1
        elif game.get_winner() == -1:
            score[1] += 1

    # Return score of match
    return score

def load_agents(dir_name: str):
    agents = []
    dirs = sorted(os.listdir(dir_name))
    for dir in dirs:
        # Check that dir is a directory
        if not os.path.isdir(os.path.join(dir_name, dir)):
            continue

        # Get number of neurons
        num_neurons = np.load(os.path.join(dir_name, dir, 'num_neurons.npy'), allow_pickle=True)
        activation_functions = ['relu', 'relu', 'linear']

        # Set activation functions - commenting out because I encountered an error where all the activation functions in activation_functions.txt were saved as 'linear'
        # with open(os.path.join(dir_name, dir, 'activation_functions.txt'), 'r') as f:
        #     act_fns = f.readlines()
        #     for act_fn in act_fns:
        #         activation_functions.append(act_fn.strip())

        # Initialize network
        nn = NeuralNetwork(num_neurons, activation_functions)
        nn.load_network(os.path.join(dir_name, dir))
        agent = ConnectFourAgent(is_player_one=True, nn=nn)
        agents.append((dir, agent))
    agents.append(('random', RandomAgent(is_player_one=True)))
    return agents

def scores_to_string(scores):
    to_ret = ''
    for player in scores:
        for score in player:
            to_ret += score
            to_ret += ' '
        to_ret = to_ret[:-1]
        to_ret += '\n'
    to_ret = to_ret[;-1]
    return to_ret

def run_tournament(dir_name: str, result_scores={'win': 3, 'draw': 1, 'lose': 0}):
    # Load agents
    agents = load_agents(dir_name)
    print("Names of agents in tournament:")
    for agent in agents:
        print(agent[0])
    print(f"Number of agents in tournament: {len(agents)}")

    # Initialize scores and totals
    scores = np.zeros((len(agents), len(agents)))
    totals = []

    # Run matches
    print("\nRunning tournament...")
    for i, agent_i in enumerate(agents): # agent 1
        for j in range(i + 1, len(agents)): # agent 2
            # Get agents
            agent_o = agents[j]
            agent_1 = agent_i[1]
            agent_2 = agent_o[1]
            print(f"Agent {agent_i[0]} vs Agent {agent_o[0]}")

            # Make them play one another with agent_1 as Player 1 and agent_2 as Player 2
            match_1_scores = match(agent_1, agent_2)
            print(f"Scores with Agent '{agent_i[0]}' as Player 1 and Agent '{agent_o[0]}' as Player 2:")
            print(match_1_scores)

            # Make them play on another with agent_2 as Player 1 and agent_1 as Player 2
            match_2_scores = match(agent_2, agent_1)
            print(f"Scores with Agent '{agent_o[0]}' as Player 1 and Agent '{agent_i[0]}' as Player 2:")
            print(match_2_scores)

            # Final scores
            match_scores = [match_1_scores[0] + match_2_scores[1], match_1_scores[1] + match_2_scores[0]]
            print(f"Overall Score of Agent '{agent_i[0]}' vs Agent '{agent_o[0]}':")
            print(match_scores)
            if match_scores[0] > match_scores[1]:
                scores[i][j] += result_scores['win']
                scores[j][i] += result_scores['lose']
                print(f"Agent '{agent_i[0]}' wins!")
            elif match_scores[1] > match_scores[0]:
                scores[i][j] += result_scores['lose']
                scores[j][i] += result_scores['win']
                print(f"Agent '{agent_o[0]}' wins!")
            else:
                scores[i][j] += result_scores['draw']
                scores[j][i] += result_scores['draw']
                print("Draw!")
            print('')
            
    with open(os.path.join(dir_name, 'results.txt'), 'w') as f:
        print("Scores:")
        print(scores_to_string(scores))
        f.write(scores_to_string(scores))

        # Get totals
        for r in range(len(scores)):
            total = 0
            agent_name = agents[r][0]
            for c in range(len(scores)):
                total += scores[r][c]
            totals.append([agent_name, total])

        to_write = '\n\n'
        for total in totals:
            to_write += total[0]
            to_write += ': '
            to_write += str(total[1])
            to_write += '\n'
        to_write = to_write[:-1]
        f.write(to_write)

        # Sort totals
        for o in range(len(totals) - 1):
            for i in range(o + 1, len(totals)):
                if totals[i][1] > totals[o][1]:
                    temp = totals[i].copy()
                    totals[i]= totals[o].copy()
                    totals[o] = temp.copy()
    
        to_write = '\n\n'
        for total in totals:
            to_write += total[0]
            to_write += ': '
            to_write += str(total[1])
            to_write += '\n'
        to_write = to_write[:-1]
        f.write(to_write)
        print(to_write)

    # Display winner
    print(f"Tournament winner is Agent {totals[0][0]}")



if __name__ == '__main__':
    dir_name = '/home/troyxdp/Documents/University Work/Artificial Intelligence/Project/combined_results'
    run_tournament(dir_name)