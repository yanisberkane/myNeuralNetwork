#!/usr/bin/python3

import argparse
import sys
import numpy as np

ERROR_CODE = 84
SUCCES_CODE = 0
NB_NEW_NETWORKS_PARAMS = 3

expected_results = {
    "1-0": [1, 0, 0, 0],
    "0-1": [0, 1, 0, 0],
    "1/2-1/2": [0, 0, 1, 0],
    "overflow": [0, 0, 0, 1]
}

class ActivationLayer:
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        self.input = None
        self.output = None

    def forward(self, data):
        self.input = data
        self.output = self.activation(data)
        return self.output

    def backward(self, out_error, rate):
        return self.activation_prime(self.input) * out_error

class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.loas_of_value = None
        self.loass_prime = None

    def add_layers(self, layer):
        self.layers.append(layer)

    def set_loss_to_use(self, loss, loss_prime):
        self.loas_of_value = loss
        self.loass_from_prime = loss_prime

    def get_prediction(self, data):
        nb_data = len(data)
        res = []
        for i in range(nb_data):
            output = data[i]
            for layer in self.layers:
                output = layer.forward(output)
            res.append(output)
        return res
    
    def train(self, data, expected, rate, epochs):
        nb_data = len(data)
        for i in range(epochs):
            error = 0
            for j in range(nb_data):
                output = data[j]
                for layer in self.layers:
                    output = layer.forward(output)
                error += self.loas_of_value(expected[j], output)
                out_error = self.loass_from_prime(expected[j], output)
                for layer in reversed(self.layers):
                    out_error = layer.backward(out_error, rate)
            error /= nb_data
            print("epoch: ", i + 1, "/", epochs, " error: ", error)

    def mse(self,y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def mse_prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

    def tanh(self, x):
        return np.tanh(x)

    def tanh_prime(self, x):
        return 1 - np.tanh(x) ** 2

class CustomParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('Error: %s\n' % message)
        sys.exit(ERROR_CODE)

def parse_args():
    parser = CustomParser(description="Program that gets the expected value and variance")
    first_group = parser.add_mutually_exclusive_group(required=True)
    second_group = parser.add_mutually_exclusive_group(required=True)

    first_group.add_argument("--new", type=int, nargs="*", help="Creates a new neural network with random weights.")
    first_group.add_argument("--load", type=str, help="Loads a neural network from a folder.")
    second_group.add_argument("--train", required=False, action='store_true', help="Launches the neural network in training mode. Each board in FILE must contain inputs to send to the neural network, as well as the expected output.")
    second_group.add_argument("--predict", required=False, action='store_true', help="Launches the neural network in prediction mode. Each board in FILE must contain inputs to send to the neural network.")
    parser.add_argument("--save", type=str, required=False, help="Save neural network internal state into SAVEFILE.")
    parser.add_argument("FILE", type=str, help="FILE containing chessboards.")
    return parser.parse_args()

def open_file(file) -> str:
    try:
        with open(file, "r") as f:
            return f.read()
    except:
        print("Error: file not found")
        sys.exit(ERROR_CODE)

def parse_chessboard(chessboard_config) -> tuple:
    boards = []
    res = None
    is_chackmate = None
    fen = None

    try:
        chessboards = chessboard_config.split("\n\n")
        for chessboard in chessboards:
            chessboard_infos = chessboard.split("\n")
            for chessboard_info in chessboard_infos:
                if chessboard_info.startswith("RES"):
                    res = chessboard_info.split(": ")[1]
                elif chessboard_info.startswith("CHECKMATE"):
                    is_chackmate = True if chessboard_info.split(": ")[1] == "True" else False
                elif chessboard_info.startswith("FEN"):
                    fen = chessboard_info.split(": ")[1].split()[0]
            boards.append((res, is_chackmate, fen))
    except Exception as e:
        print(e)
        sys.exit(ERROR_CODE)

    return boards

def convert_fen_to_input(fen) -> list:
    line = []
    pieces = {
        "P": 1, "N": 3, "B": 3, "R": 4, "Q": 8, "K": 11,
        "p": -1, "n": -3, "b": -3, "r": -4, "q": -8, "k": -11
    }

    try:
        chessboard_lines = fen.split("/")
        for i in range(len(chessboard_lines)):
            if (chessboard_lines[i].isdigit()):
                line.extend([0] * int(chessboard_lines[i]))
            else:
                for j in range(len(chessboard_lines[i])):
                    if chessboard_lines[i][j].isdigit():
                        line.extend([0] * int(chessboard_lines[i][j]))
                    else:
                        line.append(pieces[chessboard_lines[i][j]])
    
    except Exception as e:
        print(e)
        sys.exit(ERROR_CODE)

    return np.array([line])

def format_chessboards(chessboards_str_infos) -> list:
    chessboards = []
    chessboard = {"res": None, "is_chackmate": None, "fen": []}

    try:
        for i in range(len(chessboards_str_infos)):
            chessboard["res"] = expected_results[chessboards_str_infos[i][0]]
            chessboard["is_chackmate"] = chessboards_str_infos[i][1]
            chessboard["fen"] = convert_fen_to_input(chessboards_str_infos[i][2])
            chessboards.append(chessboard)
            chessboard = {"res": None, "is_chackmate": None, "fen": []}
    except Exception as e:
        print(e)
        sys.exit(ERROR_CODE)

    return chessboards

def init_network(settings : list) -> NeuralNetwork:
    network = NeuralNetwork()

    if (len(settings) < 2):
        print("Error: invalid number of parameters for --new")
        sys.exit(ERROR_CODE)

    # Generate the network
    print("Creating a new network with", settings[0], "inputs and", settings[-1], "outputs")
    for i in range(len(settings)):
        if (settings[i] < 1):
            print("Error: invalid number of neurons for layer", i)
            sys.exit(ERROR_CODE)
        if (i == len(settings) - 1):
            network.add_layers(FullyConnectedLayer(settings[i], settings[i]))
        else:
            network.add_layers(FullyConnectedLayer(settings[i], settings[i + 1]))
        print("Layer", i, ":", settings[i], "neurons") # Debug
        network.add_layers(ActivationLayer(network.tanh, network.tanh_prime))

    return network

def save_network(filename : str, settings : list, network : NeuralNetwork):
    try:
        file = open(filename, "w")

    except OSError:
        print("Error: could not open file")
        sys.exit(ERROR_CODE)

    try:
        file.write("Settings:\n")
        file.write(str(settings) + "\n")

        for i in range(0, len(network.layers), 2):
            if (i == 0):
                file.write("Input layer:\n")
            elif (i == len(network.layers) - 2):
                file.write("Output layer:\n")
            else:
                file.write("Hidden layer " + str(i // 2) + ":\n")
            file.write("Weights:\n")
            file.write(str(network.layers[i].weights.tolist()) + "\n")
            file.write("Bias:\n")
            file.write(str(network.layers[i].bias.tolist()) + "\n")
    except Exception as e:
        print(e)
        sys.exit(ERROR_CODE)

    file.close()
    return

def load_network(filename) -> NeuralNetwork:
    network = NeuralNetwork()
    settings = []

    try:
        file = open(filename, "r")
    except OSError:
        print("Error: could not open file")
        sys.exit(ERROR_CODE)

    try:
        # Read settings
        file.readline() # Skip "Settings:"
        settings = file.readline()
        settings = settings[1:-2].split(", ")
        print("Loading a network with", settings[0], "inputs and", settings[-1], "outputs")
        for i in range(len(settings)):
            settings[i] = int(settings[i])
        network = init_network(settings)

        # Read weights and bias
        file.readline() # Skip "Input layer:"
        file.readline() # Skip "Weights:"
        network.layers[0].weights = np.array(eval(file.readline()))
        file.readline() # Skip "Bias:"
        network.layers[0].bias = np.array(eval(file.readline()))

        for i in range(2, len(network.layers) - 2, 2):
            file.readline() # Skip "Hidden layer i:"
            file.readline() # Skip "Weights:"
            network.layers[i].weights = np.array(eval(file.readline()))
            file.readline() # Skip "Bias:"
            network.layers[i].bias = np.array(eval(file.readline()))

        file.readline() # Skip "Output layer:"
        file.readline() # Skip "Weights:"
        network.layers[len(network.layers) - 2].weights = np.array(eval(file.readline()))
        file.readline() # Skip "Bias:"
        network.layers[len(network.layers) - 2].bias = np.array(eval(file.readline()))
    except Exception as e:
        print(e)
        sys.exit(ERROR_CODE)

    return network, settings

def set_args_in_context(args):
    network : NeuralNetwork = None
    network_settings = []
    if args.new:
        network_settings = args.new
        network = init_network(args.new)
    if args.load:
        network, network_settings = load_network(args.load)
    chessboards_str_infos = parse_chessboard(open_file(args.FILE))
    formatted_chessboards = format_chessboards(chessboards_str_infos)

    x_train = np.array([chessboard['fen'] for chessboard in formatted_chessboards])
    y_train = np.array([chessboard["res"] for chessboard in formatted_chessboards])
    if args.train:
        print("Training...")
        network.set_loss_to_use(network.mse, network.mse_prime)
        network.train(x_train, y_train, epochs=1000, rate=0.1)

    #if args.predict:
    print("Predicting...")
    out = network.get_prediction(x_train)
    print("expected :\n", y_train)
    print("predictions :\n", np.round(out, 2))

    if args.save:
        save_network(args.save, network_settings, network)
    return

def main():
    args = parse_args()
    set_args_in_context(args)
    return SUCCES_CODE

if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(e)
        sys.exit(ERROR_CODE)
