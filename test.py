import network
import samples
import utils
import numpy as np

epochs = 3
train_times = 50000
test_times = 10000
tests = 1
total_accuracy = 0
test_samples = samples.load('testing')

"""
Trains network for 1 epoch and then runs a test, repeating a set number of
times(tests). Returns accuracy of network after most recent test.
"""
for test in range(tests):
    net = network.Network(utils.NET_STRUCTURE, utils.NET_ALPHA, utils.NET_BIAS)

    for epoch in range(epochs):
        training_samples = samples.load('training')
        print("test " + str(test + 1) + ", epoch " + str(epoch + 1))
        for j in range(train_times):
            label = [0] * 10
            index, sample = training_samples.next()
            label[index] = 1
            net.train(sample, label)

    correct = 0
    for i in range(test_times):
        index, sample = test_samples.next()
        correct += int(net.classify(sample)) == index
    total_accuracy + correct / float(test_times)

print("accuracy: " + str(total_accuracy * 100) + "%")
