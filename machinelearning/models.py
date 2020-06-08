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
        # return 1 if dotproduct is greater than 0 else return -1
        return 1 if nn.as_scalar(nn.DotProduct(self.w,x)) >= 0.0 else -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        #running until 100%
        # follow instructions
        while True:
            flag = True
            for x,y in dataset.iterate_once(1):
                if(nn.as_scalar(y)) != self.get_prediction(x):
                    flag = False
                    self.w.update(x, nn.as_scalar(y))
            # if there is one is not accurate it will keep going
            if flag:
                break


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.m1 = nn.Parameter(1, 100)
        self.m2 = nn.Parameter(100, 1)
        self.b1 = nn.Parameter(1, 100)
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
        xm1 = nn.Linear(x, self.m1)
        reluy =nn.ReLU(nn.AddBias(xm1,self.b1))
        xm2 = nn.Linear(reluy,self.m2)
        predicted_y = nn.AddBias(xm2, self.b2)
        return predicted_y

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
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"

        loss1= 0
        while True:
            flag = True
            for x, y in dataset.iterate_once(1):
                # print(x,y)
                loss = self.get_loss(x, y)

                if nn.as_scalar(loss) > 0.02:
                    # loss1 =nn.as_scalar(loss)
                    flag = False
                grad_wrt_m1,grad_wrt_m2,grad_wrt_b1,grad_wrt_b2 = nn.gradients(loss, [self.m1,self.m2,self.b1,self.b2])
                # print(grad_wrt_m1,grad_wrt_b1)

                self.m1.update(grad_wrt_m1, -0.005)
                self.m2.update(grad_wrt_m2, -0.005)
                self.b1.update(grad_wrt_b1, -0.005)
                self.b2.update(grad_wrt_b2, -0.005)
            if flag:
                # print(loss1)
                break


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
        self.m1 = nn.Parameter(784,100)
        self.m2 = nn.Parameter(100,10)
        self.b1 = nn.Parameter(1,100)
        self.b2 = nn.Parameter(1,10)

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

        xm1 = nn.Linear(x, self.m1)
        # print("1",xm1)
        reluy = nn.ReLU(nn.AddBias(xm1, self.b1))
        # print("2",reluy)
        xm2 = nn.Linear(reluy, self.m2)
        # print("3",xm2)
        predicted_y = nn.AddBias(xm2, self.b2)
        # print(predicted_y)
        return predicted_y

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

        #nn.SoftmaxLoss computes a batched softmax loss, used for classification problems
        return nn.SoftmaxLoss(self.run(x),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(1):
                loss = self.get_loss(x,y)
                grad_wrt_m1, grad_wrt_m2, grad_wrt_b1, grad_wrt_b2 = nn.gradients(loss, [self.m1,self.m2,self.b1,self.b2])
                self.m1.update(grad_wrt_m1, -0.005)
                self.m2.update(grad_wrt_m2, -0.005)
                self.b1.update(grad_wrt_b1, -0.005)
                self.b2.update(grad_wrt_b2, -0.005)
            if dataset.get_validation_accuracy() >= 0.97:
                break

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
        self.w = nn.Parameter(self.num_chars, 300)
        self.whid = nn.Parameter(300,300)
        self.wfin = nn.Parameter(300, 5)

    def run(self, xs):
        """
        Runs the model for a batch of examples.w1q

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
        # for x in xs:
        #The first layer of finitial will begin by multiplying the vector x0 by some weight matrix W to produce z0=x0⋅W
        z = nn.Linear(xs[0], self.w)
        #you should replace this computation with zi=xiW+hiWhidden using an nn.Add operation

        for x in (xs[1:]):
            z = nn.ReLU(nn.Add(nn.Linear(x,self.w),nn.Linear(z,self.whid)))

        return nn.Linear(z,self.wfin)


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
        return nn.SoftmaxLoss(self.run(xs),y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x, y in dataset.iterate_once(2):
                loss = self.get_loss(x,y)
                grad_wrt_w, grad_wrt_whid,grad_wrt_wfin = nn.gradients(loss, [self.w,self.whid,self.wfin])
                self.w.update(grad_wrt_w, -0.005)
                self.whid.update(grad_wrt_whid, -0.005)
                self.wfin.update(grad_wrt_wfin, -0.005)
                # self.b.update(grad_wrt_b, -0.005)
            if dataset.get_validation_accuracy() >= 0.86:
                break

