this page will write some summaries about deep learning optimizer. 

### Which optimizer to use?
If your input data is sparse, then you likely achieve the best results using one of the adaptive learning-rate methods. An additional benefit is that you won't need to tune the learning rate but likely achieve the best results with the default value.

In summary, RMSprop is an extension of Adagrad that deals with its radically diminishing learning rates. It is identical to Adadelta, except that Adadelta uses the RMS of parameter updates in the numinator update rule. Adam, finally, adds bias-correction and momentum to RMSprop. Insofar, RMSprop, Adadelta, and Adam are very similar algorithms that do well in similar circumstances. Kingma et al. [15] show that its bias-correction helps Adam slightly outperform RMSprop towards the end of optimization as gradients become sparser. Insofar, Adam might be the best overall choice.

Interestingly, many recent papers use vanilla SGD without momentum and a simple learning rate annealing schedule. As has been shown, SGD usually achieves to find a minimum, but it might take significantly longer than with some of the optimizers, is much more reliant on a robust initialization and annealing schedule, and may get stuck in saddle points rather than local minima. Consequently, if you care about fast convergence and train a deep or complex neural network, you should choose one of the adaptive learning rate methods.


# reference
1. [An overview of gradient descent optimization algorithms](http://ruder.io/optimizing-gradient-descent/)  and its [review paper on arxiv](https://arxiv.org/pdf/1609.04747.pdf) :+1: 
2. [CS231n Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/neural-networks-3/)
> Summary

> * Gradient check your implementation with a small batch of data and be aware of the pitfalls.
> * As a sanity check, make sure your initial loss is reasonable, and that you can achieve 100% training accuracy on a very small portion of the data
> * During training, monitor the loss, the training/validation accuracy, and if you’re feeling fancier, the magnitude of updates in relation to parameter values (it should be ~1e-3), and when dealing with ConvNets, the first-layer weights.
> * The two recommended updates to use are either SGD+Nesterov Momentum or Adam.
> * Decay your learning rate over the period of the training. For example, halve the learning rate after a fixed number of epochs, or whenever the validation accuracy tops off.
> * Search for good hyperparameters with random search (not grid search). Stage your search from coarse (wide hyperparameter ranges, training only for 1-5 epochs), to fine (narrower rangers, training for many more epochs)
> * Form model ensembles for extra performance

3. [Overview	of	mini-batch	gradient	descent (Geoffrey	Hinton)](http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

4. [TensorFlow Optimizers ipynb](https://github.com/vsmolyakov/experiments_with_python/blob/master/chp03/tensorflow_optimizers.ipynb)

5. [stackoverflow discussion-How to set adaptive learning rate for GradientDescentOptimizer?](https://stackoverflow.com/questions/33919948/how-to-set-adaptive-learning-rate-for-gradientdescentoptimizer)

6. [tensoflow official training optimizer](https://www.tensorflow.org/api_guides/python/train#Decaying_the_learning_rate)



