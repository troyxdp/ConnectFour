import numpy as np
from game import ConnectFour
from agent import ConnectFourAgent, RandomAgent, Agent
from neural_network import NeuralNetwork
import random
import yaml
import os
import matplotlib.pyplot as plt

# Created with assistance from https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae
def update_deep_q_network(
        lr: float, 
        gamma: float, 
        batch_size: int, 
        trainee_agent: ConnectFourAgent, 
        memory: list
    ):
    update_batch = random.sample(memory, batch_size)
    for item in update_batch:
        # Create input for neural networks
        curr_state_input = np.concatenate((item['s'].flatten(), np.array([1 if item['t'] else -1])))
        next_state_input = np.concatenate((item['s`'].flatten(), np.array([-1 if item['t'] else 1])))

        # Get curr state q values from training network and next state q values from target network
        curr_state_q_values = trainee_agent.get_nn_pred_q_values(curr_state_input)
        next_state_q_values = trainee_agent.get_nn_target_q_values(next_state_input)

        # Get max q value from next state q values obtained from target network
        max_q_val = next_state_q_values[0]
        max_q_val_action = 0
        for i in range(1, len(next_state_q_values)):
            if next_state_q_values[i] > max_q_val:
                max_q_val = next_state_q_values[i]
                max_q_val_action = i

        # Get Q Value of action taken for current state in item
        q_sa = curr_state_q_values[item['a']]

        # Get the derivative of the error - use for updating the network
        mse_dx = item['r'] + gamma * max_q_val - q_sa
        error_prime = np.zeros(len(curr_state_q_values))
        error_prime[item['a']] = mse_dx

        # Update the prediction neural network of trainee agent
        trainee_agent.reinforce(lr, error_prime)



# Created with assistance from https://medium.com/@samina.amin/deep-q-learning-dqn-71c109586bae 
def train_agent(
        lr: float, 
        gamma: float, 
        eps: float, 
        eps_final: float, 
        batch_size: int, 
        max_mem_len: int, 
        target_update_freq: int, 
        episodes: int, 
        model_save_freq: int, 
        save_dir: str, 
        trainee_agent: ConnectFourAgent, 
        training_agent: Agent
    ):
    memory = []
    iterations = 0
    episode_cumulative_rewards = []
    trainee_wins = 0
    for i in range(episodes):
        print(f"\nEPISODE {i + 1}")
        # Create new game
        game = ConnectFour()

        # Assign player numbers to agents
        is_trainee_player_one = True if np.random.rand() > 0.5 else False # choose random player number for agents
        if is_trainee_player_one:
            trainee_agent.set_player_number(1)
            training_agent.set_player_number(2)
            print("Trainee is Player 1")
        else:
            print("Trainee is Player 2")
            trainee_agent.set_player_number(2)
            training_agent.set_player_number(1)

        # Set game in agents to game of episode
        trainee_agent.set_game(game)
        training_agent.set_game(game)

        # Reset three in a row counts for trainee agent
        trainee_agent.reset_prev_state_threes_counts()

        # Get the epsilon value for choosing actions for the current episode
        curr_eps = eps - ((eps - eps_final)/episodes) * i
        print(f"Current Epsilon Value = {curr_eps}")

        # Run episode
        episode_cumulative_reward = 0
        is_player_one_turn = True
        while not game.is_over():
            # Check whose turn it is and get them to make move
            if (training_agent.is_player_one() and is_player_one_turn) or (not training_agent.is_player_one() and not is_player_one_turn): # Is opponent turn
                # print("Training agent turn")
                # Get move
                move = training_agent.act(curr_eps) # setting the epsilon value here so that some exploration can also happen using the training agent
                try:
                    game.make_move(move)
                except Exception:
                    raise Exception
                    
            else: # Is trainee agent turn
                # print("Trainee agent turn")
                # Get values for memory buffer
                curr_state = game.get_board().copy()
                try:
                    move = trainee_agent.act(curr_eps)
                except Exception:
                    raise Exception
                next_state = game.make_move(move)
                reward = trainee_agent.evaluate_reward()
                memory.append({
                    's': curr_state, 
                    't': is_player_one_turn, # needed for getting neural network inputs
                    'a': move, 
                    'r': reward,
                    's`': next_state
                })

                # Update cumulative reward for episode
                episode_cumulative_reward += reward

                # Check whether to update using memory
                if len(memory) >= max_mem_len:
                    update_deep_q_network(lr, gamma, batch_size, trainee_agent, memory)

                # Check whether to copy prediction weights to target weights
                if iterations == target_update_freq:
                    trainee_agent.copy_to_target()
                    iterations = 0
                else:
                    iterations += 1
            
            # Update the turn tracker
            is_player_one_turn = not is_player_one_turn

        # Print result of game
        winner = game.get_winner()
        if winner == 1 and training_agent.is_player_one():
            print("Training Agent wins as Player 1!")
        elif winner == 1 and not training_agent.is_player_one():
            print("User's Agent wins as Player 1!")
            trainee_wins += 1
        elif winner == -1 and training_agent.is_player_one():
            print("User's Agent wins as Player 2!")
            trainee_wins += 1
        elif winner == -1 and not training_agent.is_player_one():
            print("Training Agent wins as Player 2!")
        else:
            print("Draw!")

        # Update cumulative rewards
        episode_cumulative_rewards.append(episode_cumulative_reward)
        print(f"Cumulative rewards of episode {i}: {episode_cumulative_reward}")

        # Save target neural network periodically
        if (i % model_save_freq == 0):
            model_num = int(i / model_save_freq)
            model_path = os.path.join(save_dir, f'model_{model_num}')
            os.mkdir(model_path)
            trainee_agent.nn_target.save_network(model_path)
            print(f"NUMBER OF WINS: {trainee_wins}/{model_save_freq}")
            trainee_wins = 0

    # Save final weights
    model_path = os.path.join(save_dir, 'final_model')
    os.mkdir(model_path)
    trainee_agent.nn_target.save_network(model_path)

    # Plot rewards over episodes
    x = range(episodes)
    plt.plot(x, episode_cumulative_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Rewards")
    plt.title("Cumulative Rewards over Episodes")
    plt.show()

    # Print target neural network
    print(trainee_agent.nn_target)




if __name__ == '__main__':
    # Load hyperparameters file and retrieve values
    with open('hyperparameters.yaml', 'r') as f:
        hyp = yaml.safe_load(f)
    
    # Create trainee agent
    num_neurons = tuple(hyp['num_neurons'])
    activation_functions = tuple(hyp['activation_functions'])
    trainee_agent = ConnectFourAgent(
        is_player_one=True,
        nn=NeuralNetwork(num_neurons, activation_functions)
    )

    # Create training agent
    training_agent = RandomAgent(is_player_one=False)

    # Get save directory
    save_dir = input("Please input the name of a directory to save the trained model to: ")
    if not (os.path.exists(save_dir) or os.path.exists(os.path.join(os.getcwd(), save_dir))):
        print("Please input a valid directory!")
        save_dir = input("Please input the name of a directory to save the trained model to: ")
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(os.getcwd(), save_dir)
    if not os.path.exists(save_dir):
        raise Exception("Error: save path does not exist!")

    # Start training cycle
    train_agent(
        lr=hyp['lr'], 
        gamma=hyp['gamma'], 
        eps=hyp['eps'], 
        eps_final=hyp['eps_final'], 
        batch_size=hyp['batch_size'], 
        max_mem_len=hyp['max_mem_len'], 
        target_update_freq=hyp['target_update_freq'], 
        episodes=hyp['episodes'], 
        model_save_freq=hyp['model_save_freq'], 
        save_dir=save_dir, 
        trainee_agent=trainee_agent, 
        training_agent=training_agent
    )
