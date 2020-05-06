import numpy
import scipy.special        # for sigmoid activation function also called expit
import matplotlib.pyplot    # for ploting the numbers into image, refer to exMatplotLib vice versa
import scipy.ndimage


class NeuralNetwork:
    def __init__(self, input_node, hidden_node, output_node, learning_rate):
        self.i_node = input_node
        self.h_node = hidden_node
        self.o_node = output_node
        self.lr = learning_rate
        self.wih = numpy.random.normal(0.0, pow(self.h_node, -0.5), (self.h_node, self.i_node))
        self.who = numpy.random.normal(0.0, pow(self.o_node, -0.5), (self.o_node, self.h_node))
        self.activation_func = lambda x: scipy.special.expit(x)
        self.inverse_activation_func = lambda x: scipy.special.logit(x)

    def train(self, input_list, target_list):
        target = numpy.array(target_list, ndmin=2).T
        inputs = numpy.array(input_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        output_errors = target - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot(output_errors * final_outputs * (1.0 - final_outputs), hidden_outputs.T)
        self.wih += self.lr * numpy.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), inputs.T)

        pass

    def query(self, input_list):

        # below ndmin is to tell that at least there must be 2 dimension in array
        # for ex: if input_list = [1, 2, 3] then 'numpy.array(input_list, ndmin=2)' will turn it into [[1, 2, 3]]
        # '.T' is for calculating the transpose of array returned by numpy.array which will make inputs be as follows
        # [[1]
        #  [2]
        #  [3]]

        inputs = numpy.array(input_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_func(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_func(final_inputs)

        return final_outputs

    def reverse_query(self, label):

        final_outputs = numpy.zeros(10).T + 0.01
        final_outputs[label] *= 99

        final_inputs = self.inverse_activation_func(final_outputs)
        hidden_outputs = numpy.dot(self.who.T, final_inputs)

        # rescale the hidden_outputs to 0.01 to 0.99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        hidden_inputs = self.inverse_activation_func(hidden_outputs)
        img_inputs = numpy.dot(self.wih.T, hidden_inputs)

        img_inputs -= numpy.min(img_inputs)
        img_inputs /= numpy.max(img_inputs)
        img_inputs *= 0.98
        img_inputs += 0.01

        return img_inputs


input_nodes = 784
hidden_nodes = 100         # for more accuracy turn hidden_nodes to 200 it is set to 100 to decrease time taken to run
output_nodes = 10
learning_rate = 0.1

n = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_data = open('mnist_train.csv', 'r')
training_data_list = training_data.readlines()
training_data.close()

epochs = 1                  # for better performance put epochs = 5 it is set to 1 to decrease time

for _ in range(epochs):
    for record in training_data_list:
        # numpy.asfarray() is the function that converts string('0') into floating(0.) point numbers
        # reshape is to reshape 2D array and it takes tuple of (row, column)
        all_values = record.split(',')
        scaled_input = ((numpy.asfarray(all_values[1:])/255.0) * 0.99) + 0.01

        # to increase the accuracy of the network we rotate each training image twice by 10 degrees clock wise
        # and anticlockwise

        # img_plus10 rotate image by 10 degree anticlockwise
        #img_plus10 = scipy.ndimage.interpolation.rotate(scaled_input.reshape((28, 28)), 10, cval=0.01, reshape=False)

        # img_minus10 rotate image by 10 degree clockwise
        #img_minus10 = scipy.ndimage.interpolation.rotate(scaled_input.reshape((28, 28)), -10, cval=0.01, reshape=False)
        target = numpy.zeros(10) + 0.01
        target[int(all_values[0])] = 0.99
        n.train(scaled_input, target)
        #n.train(img_minus10.reshape(784), target)
        #n.train(img_plus10.reshape(784), target)

test_data_file = open("mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

for test in test_data_list:
    all_values = test.split(',')
    correct_value = all_values[0]
    scaled_input = ((numpy.asfarray(all_values[1:]) / 255.0) * 0.99) + 0.01
    outputs = n.query(scaled_input)

    # numpy.argmax() is the function that returns the index of the largest number in the array
    output_label = numpy.argmax(outputs)

    if int(correct_value) == output_label:
        scorecard.append(1)
    else:
        scorecard.append(0)

scorecard_array = numpy.asarray(scorecard)
print("Accuracy =", scorecard_array.sum()/scorecard_array.size * 100, "%")


# -----------------------------------------x---------------------x-------------


# for k in range(2, 8):
#     test = matplotlib.pyplot.imread(f'test{k}.png')
#     x = ((1 - numpy.array(test))*0.99) + 0.01
#     a = []
#     for i in range(len(x)):
#         for j in range(len(x[0])):
#             a.append(x[i][j][0])
#
#     print(numpy.argmax(n.query(a)))
#     matplotlib.pyplot.imshow(numpy.array(a).reshape((28, 28)), cmap="Greys")
#     matplotlib.pyplot.show()


# ---------------------------x--------------------x-------------------------------


# for q in range(10):
#     matplotlib.pyplot.imshow(n.reverse_query(q).reshape((28, 28)), cmap='Greys')
#     matplotlib.pyplot.show()
