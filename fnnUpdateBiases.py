import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import sys
np.set_printoptions(threshold=np.nan)

def sigmoid(x):

    return 1 / (1 + np.exp(-x))


def weights_init(first_dim, second_dim, method):
    """Determins the type of weight initialization used by the layers, taking its dimensions into account"""

    if method == 1:
        weights = np.random.uniform(low=(-1 / np.sqrt(first_dim)), high=(1 / np.sqrt(first_dim)), size=(first_dim, second_dim))
    elif method == 2:
        weights = (np.random.randn(first_dim, second_dim) * 0.2) - 0.1
    elif method == 3:
        weights = (np.random.randn(first_dim, second_dim) * (1/np.sqrt(first_dim)))
    else:
        weights = (np.random.randn(first_dim, second_dim))
    return weights


def layer_init(input_dim, output_dim, method, first_layer_dim=100, ratio=0.5, arch=None):
    """Initializes the hidden layer architecture of the neural network. If you have a particular architecture in mind,
    it can be passed as a list of integers corresponding to the number of neurons in each layer using the arch keyword.
    If you simply want to pass a first hidden layer dimension and the rate at which it decrease, use the first_layer_dim
    and ratio parameters. Biases are also initialized as a vector of dimensions (next_layer, 1)"""

    if arch:
        arch.insert(0, input_dim)
        arch.append(output_dim)
        layer_dims = arch
    else:
        layers = first_layer_dim
        layer_dims = [input_dim]
        while layers > output_dim:
            layer_dims.append(layers)
            layers = int(layers * ratio)
        layer_dims.append(output_dim)

    weights = []
    biases = []
    for i in range(len(layer_dims) - 1):
        weights.append(weights_init(layer_dims[i], layer_dims[i+1], method))
        biases.append(np.ones(layer_dims[i+1]))


    return weights, biases


def test(weights, biases, classes, test_image, test_label=None):
    '''If you supply a test label, it will preform a train/test split accuracy evaluation. If no prediction label is
    given, it will produce a validation file that can be uploaded to kaggle'''

    predicted_label = []
    actual_label = []

    for input_layer in test_image:
        next_layer = input_layer
        for weight, bias in zip(weights, biases):
            next_layer = sigmoid(next_layer.dot(weight) + bias)
            # print(next_layer)
        # print(next_layer)
        class_index = list(next_layer).index(max(next_layer))
        # print(class_index)
        predicted_label.append(classes[class_index])

    if test_label:
        for item in test_label:
            class_index = list(item).index(1)
            actual_label.append(classes[class_index])
        tally = 0

        for i in range(len(test_label)):
            if actual_label[i] == predicted_label[i]:
                tally += 1
        accuracy = (tally * 100)/len(predicted_label)
        print('Accuracy: {}%'.format(accuracy))
        print(predicted_label[:100])
        print(actual_label[:100])
        return accuracy

    else:
        writer = open('FNNtest.csv', 'w')
        writer.write('Id,Label\n')
        for i, row in enumerate(predicted_label):
            writer.write(str(i + 1) + "," + str(row) + "\n")
        writer.close()


def train(train_x, train_y, test_x, test_y, classes, epochs, alpha, weight_init_method, first_layer_dim=100, ratio=0.5, arch=None):
    """Main function of the neural network, with tuneable hyperparameters of: epochs, learning rate, weight
    initialization method and architecture."""

    input_dim = len(train_x[0])
    output_dim = len(train_y[0])

    # initializes layer weights
    weights, biases = layer_init(input_dim, output_dim, weight_init_method, first_layer_dim, ratio, arch)
    for iter in range(epochs):
        error = 0
        shuff_train_x = []
        shuff_train_y = []
        shuffled_index = list(range(len(train_x)))
        np.random.shuffle(shuffled_index)
        for index in shuffled_index:
            shuff_train_x.append(train_x[index])
            shuff_train_y.append(train_y[index])

        shuff_train_x = np.array(shuff_train_x)

        for batch_i in range(int(len(train_x) / batch_size)):
            batch_x = shuff_train_x[(batch_i * batch_size):(batch_i + 1) * batch_size]
            batch_y = shuff_train_y[(batch_i * batch_size):(batch_i + 1) * batch_size]
            # input layer
            next_layer = batch_x
            layer_list = [next_layer]
            # takes the previous layer, adds a bias and applies the sigmoid function, stores the new layer, repeat for each non-input layer
            for weight, bias in zip(weights, biases):
                next_layer = sigmoid(next_layer.dot(weight) + bias)
                layer_list.append(next_layer)

            for count, layer in enumerate(layer_list[::-1]):
                # Last layer initiates backprop
                if count == 0:
                    # print(layer)
                    last_layer = (layer - batch_y) * layer * (1 - layer)
                    last_bias = np.sum(layer, axis=0)
                    # print("bias", last_bias.shape)
                    # print("weight", last_layer.shape)
                    layer_change = [last_layer]
                    bias_change = [last_bias]

                # backprop pushes through successively earlier layers
                else:
                    last_layer = last_layer.dot(weights[-count].T) * layer * (1 - layer)
                    last_bias = np.sum(layer, axis=0)
                    # print("bias",last_bias.shape)
                    # print("weight", last_layer.shape)
                    layer_change.append(last_layer)
                    bias_change.append(last_bias)

            # weights and biases are updated.
            for j in range(len(weights)):
                change_index = -2 - j
                weights[j] += layer_list[j].T.dot(layer_change[change_index]) * -alpha
                biases[j] += bias_change[change_index] * -alpha
                # print(biases[j])


            error += (np.sum(np.abs(layer_change[0])))

        sys.stdout.write("\rIter: " + str(iter) + " Loss: " + str(error))
        if iter % 100 == 0 and iter > 1:
            print("")
            accuracy = test(weights, biases, classes, test_x, test_y)
            writer = open("mnistResults.csv", 'a')
            writer.write(str([iter, alpha, weight_init_method, str(arch[1:-1]).replace(",",":").replace('\'', ""),
                              error, accuracy]).replace("[", "").replace("]", "") + "\n")
            writer.close()

    return weights


# Load and split #############################################################################

# data = np.loadtxt("train_data.csv", delimiter=',')
# labels = data[:, 0]
# images = data[:, 1:]
# images = images/255
# pickle.dump(images[:5000], open("downsized_images.pkl", 'wb'))
# pickle.dump(labels[:5000], open("downsized_labels.pkl", 'wb'))
images = pickle.load(open("mnist_images.pkl", 'rb'))
labels = pickle.load(open("mnist_labels.pkl", 'rb'))

one_hot = []
classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
           11, 12, 13, 14, 15, 16, 17, 18, 20,
           21, 24, 25, 27, 28, 30, 32, 35, 36,
           40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for example in labels:
    one_hot_example = np.zeros(len(classes))
    one_hot_example[classes.index(example)] = 1

    one_hot.append(one_hot_example.astype(int))
labels = one_hot


X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.20, random_state=2)

# Hyper-parameters ###########################################################################

batch_size = 100
epochs = 301
lr = [0.1, 0.01, 0.001, 0.0001]

for weight_init in range(1, 3):
    for alpha in lr:
        train(X_train, y_train, X_test, y_test, classes, epochs, alpha, weight_init, arch=[500])
        train(X_train, y_train, X_test, y_test, classes, epochs, alpha, weight_init, arch=[200, 100])
        train(X_train, y_train, X_test, y_test, classes, epochs, alpha, weight_init, arch=[500, 300, 200, 100])