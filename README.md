# Public Security
TCSS 600 - Image Classification for Public Security

Alex Pawlak

# Overview
This code contains the script used to train, validate, and test the Convolutional Neural Network. The output of the script shows the training and validation accuracy and loss for each epoch, graphs corresponding to the accuracies and losses, and a confusion matrix for the test results.

# Prerequisites
Prior to running the script Python3, Keras and Tensorflow must be installed on machine.

Requires the image folders to be stored in the following relative directories:
- ./train/safe
- ./train/threat
- ./validation/safe
- ./validation/threat
- ./test/safe
- ./test/threat

# Usage
python3 publicSecurity.py
