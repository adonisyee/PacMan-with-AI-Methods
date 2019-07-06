import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w,x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        temp = nn.DotProduct(self.w,x)

        if (nn.as_scalar(temp) > -0.000001):
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        count1 = 0
        count2 = 0
        mistakecount = 0
        while True:

            for x, y in dataset.iterate_once(1):

                if self.get_prediction(x) != nn.as_scalar(y):
                    self.w.update(x, nn.as_scalar(y))
                    count2 -= 1

                count1 += 1
                count2 += 1
            if count1 == count2:
                break
            count1 = 0
            count2 = 0

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learningRate = -0.01
        self.hiddenLayerSize = 50
        self.batch_sizes = 10


        self.W1 = nn.Parameter(1, self.hiddenLayerSize)
        self.b1 = nn.Parameter(1, self.hiddenLayerSize)
        self.W2 = nn.Parameter(self.hiddenLayerSize, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        product1 = nn.Linear(x, self.W1)
        product2 = nn.AddBias(product1, self.b1)
        product3 = nn.ReLU(product2)
        product4 = nn.Linear(product3, self.W2)
        product5 = nn.AddBias(product4, self.b2)
        return product5

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        loss = nn.SquareLoss(self.run(x), y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        for x, y in dataset.iterate_forever(self.batch_sizes):
            loss = self.get_loss(x,y)
            if nn.as_scalar(loss) < .001:
                break
            grad_w1, grad_b1, grad_w2, grad_b2 = nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2])
            self.W1.update(grad_w1, self.learningRate)
            self.b1.update(grad_b1, self.learningRate)
            self.W2.update(grad_w2, self.learningRate)
            self.b2.update(grad_b2, self.learningRate)



class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learningRate = -0.05
        self.hiddenLayerSize = 100
        self.batch_sizes = 5


        self.W1 = nn.Parameter(784, self.hiddenLayerSize)
        self.b1 = nn.Parameter(1, self.hiddenLayerSize)
        self.W2 = nn.Parameter(self.hiddenLayerSize, 10)
        self.b2 = nn.Parameter(1, 10)
        #self.W2 = nn.Parameter(self.hiddenLayerSize, self.hiddenLayerSize)
        #self.b2 = nn.Parameter(1, self.hiddenLayerSize)
        #self.W3 = nn.Parameter(self.hiddenLayerSize, 10)
        #self.b3 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        product1 = nn.Linear(x, self.W1)
        product2 = nn.AddBias(product1, self.b1)
        product3 = nn.ReLU(product2)
        product4 = nn.Linear(product3, self.W2)
        product5 = nn.AddBias(product4, self.b2)
        #product6 = nn.ReLU(product5)
        #product7 = nn.Linear(product5, self.W3)
        #product8 = nn.AddBias(product7, self.b3)
        return product5

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        loss = nn.SoftmaxLoss(self.run(x), y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        iteration = 1
        for x, y in dataset.iterate_forever(self.batch_sizes):
            loss = self.get_loss(x,y)
            if iteration  % 40 == 0:
                if dataset.get_validation_accuracy() >= .975:
                    return
            grad_w1, grad_b1, grad_w2, grad_b2 = nn.gradients(loss, [self.W1, self.b1, self.W2, self.b2])
            #, grad_w3, grad_b3
            #, self.W3, self.b3
            self.W1.update(grad_w1, self.learningRate)
            self.b1.update(grad_b1, self.learningRate)
            self.W2.update(grad_w2, self.learningRate)
            self.b2.update(grad_b2, self.learningRate)
            #self.W3.update(grad_w3, self.learningRate)
            #self.b3.update(grad_b3, self.learningRate)
            iteration += 1

class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learningRate = -0.05
        self.hiddenLayerSize = 50
        self.batch_sizes = 10

        self.W1 = nn.Parameter(self.num_chars, self.hiddenLayerSize)
        self.b1 = nn.Parameter(1, self.hiddenLayerSize)
        self.Wh = nn.Parameter(self.hiddenLayerSize, self.hiddenLayerSize)
        self.W2 = nn.Parameter(self.hiddenLayerSize, self.hiddenLayerSize)
        self.bd = nn.Parameter(1, 5)
        #self.W3 = nn.Parameter(self.hiddenLayerSize, self.hiddenLayerSize)
        #self.b3 = nn.Parameter(1, self.hiddenLayerSize)
        self.b2 = nn.Parameter(1, self.hiddenLayerSize)
        self.Wd = nn.Parameter(self.hiddenLayerSize, 5)



    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        for i in range(len(xs)):
            if i == 0:
                temp = nn.Linear(xs[0], self.W1)
                hi = nn.AddBias(temp, self.b1)
            else:
                hi = nn.Add(nn.Linear(xs[i], self.W1), nn.Linear(hi, self.Wh))
        
        hi = nn.Linear(hi, self.W2)
        hi = nn.AddBias(hi, self.b2)
        hi = nn.ReLU(hi)
        #hi = nn.Linear(hi, self.W3)
        #hi = nn.AddBias(hi, self.b3)
        hi = nn.Linear(hi, self.Wd)
        hi = nn.AddBias(hi, self.bd)
        return hi

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        loss = nn.SoftmaxLoss(self.run(xs), y)
        return loss


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        iteration = 1
        for x, y in dataset.iterate_forever(self.batch_sizes):
            loss = self.get_loss(x,y)
            if iteration  % 40 == 0:
                if dataset.get_validation_accuracy() >= .86:
                    return
            grad_w1, grad_b1, grad_wh, grad_wd, grad_bd, grad_w2, grad_b2= nn.gradients(loss, [self.W1, self.b1, self.Wh, self.Wd, self.bd, self.W2, self.b2])
            #grad_w3, grad_b3
            #self.W3, self.b3
            self.W1.update(grad_w1, self.learningRate)
            self.b1.update(grad_b1, self.learningRate)
            self.Wh.update(grad_wh, self.learningRate)
            self.Wd.update(grad_wd, self.learningRate)
            self.b2.update(grad_b2, self.learningRate)
            self.W2.update(grad_w2, self.learningRate)
            self.bd.update(grad_bd, self.learningRate)
            #self.W3.update(grad_w3, self.learningRate)
            #self.b3.update(grad_b3, self.learningRate)

            iteration += 1
