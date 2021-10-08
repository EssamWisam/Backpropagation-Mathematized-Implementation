import numpy as np

class ANN:

    def __init__(self, structure):
        #structure = [5, 3, 2] means our NN has three layers with 5, 3, 2 neurons respectively including the input layer.
        #the input layer has no weights or biases.
        self.num_layers = len(structure) 
        #for each layer except the first randomize with y neurons randomize a (y, 1) vector for the bias vector:
        self.Bₙ = [np.random.randn(l, 1) for l in structure[1:]] 
        #for each two consecutive layers with x and y neurons respectively randomize a (y,x) matrix for the weight matrix:
        self.Wₙ = [np.random.randn(l, next_l) for l, next_l in zip(structure[:-1], structure[1:])]
        
    def backprop(self, x, y):
        #Given an observation this computes (მJⳆმBₙₛ, მJⳆმWₙₛ) that will be used to minimize the cost.
        მJⳆმBₙₛ = [np.zeros(b.shape) for b in self.Bₙ]
        მJⳆმWₙₛ = [np.zeros(W.shape) for W in self.Wₙ]

        # forward pass (computing z for all layers)
        Zₙ = []                     # list to store all the z vectors, layer by layer
        Aₙ = []                     # list to store all the , layer by layer

        for b, W in zip(self.Bₙ, self.Wₙ):
            z = W.T @ a + b if Zₙ else W.T @ x + b
            a = σ(z)
            Zₙ.append(z)
            Aₙ.append(a)

        #Zₙ and Aₙ are now ready.

        # backward pass (computing δ and consequently მJⳆმBₙₛ and მJⳆმWₙₛ layer by layer )
        H = self.num_layers-2
        for L in range(H, -1, -1):
            δ =  σࠤ(Zₙ[L]) * (self.Wₙ[L+1] @ δ) if L != H else (Aₙ[L]- y) * σࠤ(Zₙ[L])
            მJⳆმBₙₛ[L] = δ
            მJⳆმWₙₛ[L] = Aₙ[L-1] @ δ.T  if L != 0 else x @ δ.T
        
        return (მJⳆმBₙₛ, მJⳆმWₙₛ)
    
    def gradient_descent(self, mini_batch, λ):
        #Given a mini batch this updates the weights and biases by applying SGD
        მJⳆმBₙ= [np.zeros(b.shape) for b in self.Bₙ]
        მJⳆმWₙ = [np.zeros(W.shape) for W in self.Wₙ]

        for x, y in mini_batch:
            მJⳆმBₙₛ, მJⳆმWₙₛ = self.backprop(x, y)
            მJⳆმBₙ = [მJⳆმb + მJⳆმbₛ for მJⳆმb, მJⳆმbₛ in zip(მJⳆმBₙ, მJⳆმBₙₛ)]  
            მJⳆმWₙ = [მJⳆმW + მJⳆმWₛ for მJⳆმW, მJⳆმWₛ in zip(მJⳆმWₙ, მJⳆმWₙₛ)]

        d = len(mini_batch)
        self.Wₙ = [W - λ/d * მJⳆმW for W, მJⳆმW in zip(self.Wₙ, მJⳆმWₙ)]
        self.Bₙ = [b - λ/d * მJⳆმb for b, მJⳆმb in zip(self.Bₙ, მJⳆმBₙ)]


    def feedforward(self, x ):
        #Loop starting from the 2nd layer to the last. For the 2nd layer the Aₙ are simply the input.
        a = x
        for b, W in zip( self.Bₙ, self.Wₙ):
            a = σ(W.T @ a + b)
        return a
        #This will return the Aₙ of the last layer (the output)

            
   



def ᐁJ(aᴺ, y):                  #We can use this instead of using (Aᴺ[l]-y) in the backpropagation loop.
    return (aᴺ-y)


def σ(z):
    return 1.0/(1.0+np.exp(-z))


def σࠤ(z):
    return σ(z)*(1-σ(z))

