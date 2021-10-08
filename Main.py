import numpy as np
from Dataloader import load_data
from NeuralNet import ANN

def train(model, epochs, training_data, validation_data, λ):
   for j in range(epochs):
      for mini_batch in training_data:
            model.gradient_descent(mini_batch, λ)          #update the parameters after learning from the mini_batch.
      validate(model, validation_data, epoch=j)            #whenever an epoch is done test the neural network on the validation set.

def validate(model, validation_data, epoch):
   validation_results = [(np.argmax(model.feedforward(x)), np.argmax(y)) for (x, y) in validation_data]   #the index is the number itself
   accuracy = sum(int(yᴺ == y) for (yᴺ, y) in validation_results) / len(validation_data)          
   print(f"Epoch {epoch} is complete with validation accuracy: {accuracy:.0%} ")

def test(model, testing_data):
   test_results = [(np.argmax(model.feedforward(x)), np.argmax(y)) for (x, y) in testing_data]   
   accuracy = sum(int(yᴺ == y) for (yᴺ, y) in test_results) / len(testing_data)               
   print(f"The accuracy of your model is {accuracy:.0%}")


training_data, validation_data, testing_data = load_data(mini_batch_size=16)
model = ANN([784, 30, 10])
train(model, epochs=5, training_data=training_data, validation_data=validation_data, λ=3.0)

test(model, testing_data)       #uncomment only when there's no more hyperparameter-tuning to be done.

