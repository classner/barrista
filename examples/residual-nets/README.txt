Residual Network example
========================

This folder contains a full-featured example for deep residual networks.
It reaches comparable performance to the network described in
"Deep Residual Learning for Image Recognition", He et al., 2015.

All files are executable and encapsulate one aspect of the model. They can
all be run with `--help` to get more information.

To run the training, simply run

    ./train.py testrun --model_name=msra3

This will run the training and store the results in the folder results/testrun.
The model is exchangeable, and must be a Python module in 'models' that has a
`MODEL` property.

The `msra3` creates a residual network with 3 residual blocks per network
part with the same image size. The proposed network has 3 such parts. In
total, this corresponds to the 20 layer network from the original paper.
The constructing method simply takes this number of blocks as parameter.
`msra9` thus constructs the 50 layer network, and you can easily play
around with a lot deeper architectures.

Happy training!
