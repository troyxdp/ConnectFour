from neural_network import NeuralNetwork
import os

nn = NeuralNetwork((43, 128, 128, 7), ('relu', 'relu', 'linear'))
for dir_name in sorted(os.listdir('/home/troyxdp/Documents/University Work/Artificial Intelligence/Project/combined_results')):
    print(f'{dir_name}')
    path = os.path.join('/home/troyxdp/Documents/University Work/Artificial Intelligence/Project/combined_results', dir_name)
    if not os.path.isdir(path):
        continue
    try:
        nn.load_network(path)
        print(nn)
        print()
    except:
        nn = NeuralNetwork((43, 256, 128, 7), ('relu', 'relu', 'linear'))
        try:
            nn.load_network(path)
            print(nn)
            print()
        except:
            nn = NeuralNetwork((43, 256, 256, 7), ('relu', 'relu', 'linear'))
            try:
                nn.load_network(path)
                print(nn)
                print()
            except:
                nn = NeuralNetwork((43, 128, 128, 7), ('relu', 'relu', 'linear'))
                try:
                    nn.load_network(path)
                    print(nn)
                    print()
                except:
                    print("Error")