from neural_network import NeuralNetwork

nn = NeuralNetwork((43, 128, 128, 7), ('relu', 'relu', 'linear'))
for dir_name in sorted(os.listdir('/home/troyxdp/Documents/University Work/Artificial Intelligence/Project/architecture')):
    print(f'{dir_name}')
    path = os.path.join('/home/troyxdp/Documents/University Work/Artificial Intelligence/Project/architecture', dir_name)
    if not os.path.isdir(path):
        continue
    nn.load_network(path)
    print(nn)
    print()