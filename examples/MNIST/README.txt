MNIST example
=============

This folder contains a full-featured example for an MNIST model. It does
not reach a specific performance, but is rather meant to be instructive
with basic infrastructure that can be used for new data.

All files are executable and encapsulate one aspect of the model. They can
all be run with `--help` to get more information.

To run the training, simply run

    ./train.py testrun --model_name=basic

This will run the training and store the results in the folder results/testrun.
The model is exchangeable, and must be a Python module in 'models' that has a
`MODEL` property.

Happy training!
