from neural_network.py import NeuralNetwork

def main():
  network = NeuralNetwork()
  network.train()
  network.write_to_testing_file()

main()
