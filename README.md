# Linear-Regression-Raw-Tensorflow-With-Two-Variables

Builds a simple raw linear regression model with basic tensorflow for 2 variables.

# Libraries Required

1. numpy.      # install by : pip install numpy
2. tensorflow. # install by : pip install tensorflow
3. matplotlib  # install by : pip install matplotlib or sudo apt-get install matplotlib

# Overview

This is a simple tutorial program developed, focusing on beginners to Machine Learning and Tensorflow, the code snippet has been well commented.
- A bit of a recall, during school times, in Linear algebra we have come accross the mathematical equation for a straight line ie, y = m\*x + c, next to that we learned that we can represent linear problems with multiple variables.. ie, y = m1\*x1 + m2\*x2 + c, even though the formula gets long, the concept is pretty simple.


# Training data

- when x1 = 1 and x2 = 1 some calculation done on 'x', and 'y' was predicted as 2. !!
- when x1 = 2 and x2 = 2 some calculation done on 'x', and 'y' was predicted as 4. !!
- when x1 = 3 and x2 = 3 some calculation done on 'x', and 'y' was predicted as 6. !!
- when x1 = 4 and x2 = 4 some calculation done on 'x', and 'y' was predicted as 8. !!
- .. ..
- .. ..

so from the above its clear that adding the 'x's gives 'y's ie,
when x1=1 and x2=1,
-  y = 1 + 1 + 0 = 2,
-  y = m1 * x1 + m2 * x2 + c,
-  x1 = 1, x2 = 1
-  m1 = 1, m2 = 1
-  c = 0,
-  therefore, y = (1 * 1) + (1 * 1) + 0 = 2

So, what we have with us is the training data, and testing data for validating the training, the aim of this program is to find the values for 'm1', 'm2' and 'c' which is 'W1'(weight1), 'W2'(weight2) and 'b'(bias) respectively.

# Solution

And thus finding the values of 'W1', 'W2' and 'b', and substituting them to the above equation y = (W1 * X1) + (W2 * X2) + b, gives the vaue of 'y' for any substituion of 'X1' and 'X2' in it.

# Challenge - tweek

Tweek some parameters like (learning_rate, training_epoch, optimizers, amount of train data) to obtain close results, by reducing the cost function and precising the values for Weights and Bias, that makes the program more optimized.. and keep in mind, this configurations keeps changing depending on the problem, train dataset etc.

Please keep me posted for any doubts and updates.
