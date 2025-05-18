import numpy as np
from game import ConnectFour
from agent import ConnectFourAgent, RandomAgent, Agent

def train_agent(lr: float, gamma: float, eps: float, batch_size: int, max_mem_len: int, target_update_freq: int, episodes: int, trainee_agent: Agent, training_agent: Agent):
    memory = []
    for i in range(episodes):
        game = ConnectFour()
        is_trainee_player_one = True if np.random.rand() > 0.5 else False # choose random player number for agents
        if is_trainee_player_one:
            trainee_agent.set_player_number(1)
            training_agent.set_player_number(2)
        else:
            trainee_agent.set_player_number(2)
            training_agent.set_player_number(1)
        is_player_one_turn = True
        while not game.is_over():
            # Check whose turn it is and get them to make move
            if (not training_agent.is_player_one() and is_player_one_turn) or (training_agent.is_player_one() and not is_player_one_turn): # Is opponent turn
                move = training_agent.act(eps)
                try:
                    game.make_move(move)
                except Exception:
                    print("Error: could not make move")
                    return
                is_player_one_turn = not is_player_one_turn
            else: # Is trainee agent turn
                # Get values for memory buffer
                curr_state = game.get_board().copy()
                move = trainee_agent.act(eps)
                next_state = game.make_move(move)
                reward = trainee_agent.evaluate_reward()
                memory.append((curr_state, move, next_state, reward))
