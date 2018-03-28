# basicNN
Neural Networks that can approximate any functions given a sample of data points. These Neural Network uses Conjugate Gradient as an update method and is written in C. Must include the INPUT.txt in the same folder as the .cpp file.

This code was used to generate the data in this research paper:
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.119.150601#fulltext

Last change since August 2016

# INPUT.txt
Must include the following:

    trainfile - training file name
    OPTION - which method of training (mentioned more below)
    ITERATIONS - number of iterations to run
    SAVE - save progress every mentioned iterations
    CUTOFF - number of maximum conjugate gradients to compute before ending iteration

Possible OPTIONs:

    -2 = training network with old weights + noise
    -1 = training network with old weights
     0 = output cost function of data with old weights
     1 = training network with new weights

To add old weights, follow the output weight's format. An example is shown in the sample INPUT.txt.

# Gradient CG Network Streamlined.cpp
A Neural Network that approximates a function space based a sample of the gradients of the function.

# CG Network Streamlined.cpp
A Neural Network that approximates a function space based a sample of the function.
