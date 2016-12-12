# Initial package for artificial neural networks

# imports

import numpy as np
import random

# Define activation functions

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return x*(x>0)


# define softmax function

def softmax(xv):
    tot=np.sum(np.exp(xv))
    return np.exp(xv)/tot



# define perceptron neuron

class Perceptron:
    
    # Constructor. learn_speed = learning speed parameter. num_weights = number of
    def __init__(self, learn_speed, num_weights):
        self.speed = learn_speed
        self.weights = []
        for x in range(0, num_weights):
            self.weights.append(random.random()*2-1)

    # Feedforward activation function
    def feed_forward(self, inputs,activate):
        sum = 0
        # multiply inputs by weights and sum them
        for x in range(0, len(self.weights)):
            sum += self.weights[x] * inputs[x]
            # return the 'activated' sum
            return self.activate(sum)


    def train(self, inputs, desired_output):
        guess = self.feed_forward(inputs)
        error = desired_output - guess

        for x in range(0, len(self.weights)):
            self.weights[x] += error*inputs[x]*self.speed

            
class Trainer:

    def __init__(self):
        self.perceptron = Perceptron(0.01, 3)

    def f(self, x):
        return 0.5*x + 10 # line: f(x) = 0.5x + 10

    def train(self):
        for x in range(0, 1000000):
            x_coord = random.random()*500-250
            y_coord = random.random()*500-250
            line_y = self.f(x_coord)

            if y_coord > line_y: # above the line
                answer = 1
                self.perceptron.train([x_coord, y_coord,1], answer)
                else: # below the line
                answer = -1
                self.perceptron.train([x_coord, y_coord,1], answer)
            return self.perceptron # return our trained perceptron
            
            

# Support functions

# create matrix of indicator variables (=0 or 1)
# this supposes we are working with categorical variables, with values 0..k-1
# where k = number of categories
# this is needed when working with softmaxes


def y2indicator(yv,k):
    n=len(yv)
    ind=np.zeros(n,k)
    for i in xrange(n):
        ind[i,yv[i]]=1
    return ind


# main program

def main():
    a=np.array( [1,2,3,4,1,2,3] )
    print softmax(a)

if __name__ == "__main__": main()
        
            
     
    
    


