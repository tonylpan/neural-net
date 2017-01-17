import utils
import network
import samples
import Tkinter

class Visualizer:

    def __init__(self, root, network, samples):
        self.epochs = 0
        self.root = root
        self.network = network
        self.samples = samples
        self.testSamples = samples.load('testing')
        self.root.configure(bg = utils.BG_COLOR, padx = 10, pady = 10)


    """
    Handles the creation and placement of graphical elements
    on the interface.
    """
    def gui_init(self):
        epoch_button = Tkinter.Button(self.root, text = "Train 1 Epoch",
                 command = lambda: self.train_epoch(), relief = 'flat', bd = 2,\
                 font = utils.FONT, bg = utils.BUTTON_COLOR, highlightthickness = 0,\
                 highlightbackground = utils.EDGE_COLOR, fg = utils.TEXT_COLOR)
        epoch_button.grid(row = 1, column = 2, padx = 10, pady = 10, sticky = 'S')

        next_button = Tkinter.Button(self.root, text = "Next Sample",
                 command = lambda: self.display_sample(), relief = 'flat', bd = 2,\
                 font = utils.FONT, bg = utils.BUTTON_COLOR, highlightthickness = 0,\
                 highlightbackground = utils.EDGE_COLOR, fg = utils.TEXT_COLOR)
        next_button.grid(row = 3, column = 2, padx = 10, pady = 10, sticky = 'N')

        self.display = Tkinter.Canvas(self.root, width = 596, height = 596,\
                       bg = utils.BG_COLOR, highlightthickness = 0)

        self.sampleDisplay = Tkinter.Canvas(self.root, width = 168, height = 168,\
                       bg = utils.BG_COLOR, highlightthickness = 0)

    """
    Load in test data, and run the network on data, returning overall
    accuracy.
    """
    def get_test_data(self):
        self.testSamples = self.samples.load('testing')
        samples = self.samples.load('testing')
        data = []
        accuracy = 0

        for test in range(utils.TEST_TIMES):
            label, sample = samples.next()
            datum = self.network.classify(sample)
            accuracy += datum == label
            data.append((datum, label))

        return (data, accuracy * 100 / float(utils.TEST_TIMES))

    """
    Display a representation of the network's accuracy, where red blocks
    signify incorrectly classified data, while green blocks signify correctly
    classified data.
    """
    def display_data(self, data):
        self.display.delete('all')
        offset = 4
        size = 2

        for i in range(utils.TEST_TIMES):
            x = (size + offset) * (i % 100)
            y = (size + offset) * (i / 100)
            color = utils.GREEN_COLOR if data[i][0] == data[i][1] else utils.RED_COLOR
            self.display.create_rectangle(x, y, x + size, y + size,\
                                         fill = color, width = 0)
        self.display.grid(row = 0, column = 0, columnspan = 3, padx = 10, pady = 10)

    #Draw the MNIST image to the interface, and update.
    def display_sample(self):
        self.sampleDisplay.delete('all')
        size = 6
        label, sample = next(self.testSamples, (None, None))

        if label is None and sample is None:
            self.testSamples = self.samples.load('testing')
            label, sample = next(self.testSamples, (None, None))

        for i in range(784):
            color = tuple([max(33, sample[i] * 64)] * 3)
            self.sampleDisplay.create_rectangle((i % 28) * size, \
            (i / 28) * size, (i % 28 + 1) * size, (i / 28 + 1) * size,\
            fill = "#%02x%02x%02x" % color, outline = utils.BG_COLOR)

        output = self.network.classify(sample)
        self.sampleDisplay.grid(row = 1, column = 0, rowspan = 3, padx = 10, pady = 10, sticky = 'W')
        entry = Tkinter.Entry(self.root, highlightbackground = utils.BG_COLOR,\
                bd = 0, width = 1, font = utils.CLASS_FONT, justify = 'center')

        entry.insert(0, "%s" % str(output))
        entry.configure(state = 'disabled', disabledbackground = utils.BG_COLOR,\
        disabledforeground = utils.GREEN_COLOR if output == label else utils.RED_COLOR)
        entry.grid(row = 1, column = 1, rowspan = 3, padx = 10, pady = 10)

        self.root.update_idletasks()
        self.root.update()

    def display_title(self, accuracy):
        msg = "Epoch %s, %s percent accurate" % (self.epochs, accuracy)
        self.root.title(msg)

    #Update all elements of the interface
    def gui_update(self):
        data, accuracy = self.get_test_data()
        self.display_data(data)
        self.display_title(accuracy)
        self.display_sample()
        self.root.update_idletasks()
        self.root.update()

    """
    Trains for 1 epoch, and then updates the interface to reflect the
    network's change in accuracy.
    """
    def train_epoch(self):
        self.epochs += 1
        training_samples = self.samples.load('training')
        msg = "Training..."
        self.root.title(msg)
        self.root.update_idletasks()
        self.root.update()

        for train in range(utils.TRAIN_TIMES):
            label = [0] * 10
            index, sample = training_samples.next()
            label[index] = 1
            self.network.train(sample, label)
        self.gui_update()

def main(*args):
    bias = True
    net = network.Network(utils.NET_STRUCTURE, 0.01, bias)
    root = Tkinter.Tk()
    visualizer = Visualizer(root, net, samples)
    visualizer.gui_init()
    visualizer.gui_update()
    root.mainloop()

main()
