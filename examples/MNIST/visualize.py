#!/usr/bin/env python
"""Create visualizations."""
# pylint: disable=no-member, invalid-name, wrong-import-position
from __future__ import print_function

import os
import json
import logging
import click

import numpy as np
_LOGGER = logging.getLogger(__name__)
try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    print("Matplotlib could not be imported!")
    MPL_AVAILABLE = False


def _sorted_ar(inf, key):
    iters = []
    vals = []
    for values in inf:
        if values.has_key(key):
            iters.append(int(values['NumIters']))
            vals.append(float(values[key]))
    sortperm = np.argsort(iters)
    arr = np.array([iters, vals]).T
    return arr[sortperm, :]


def _get_information(logfile):
    _LOGGER.info("Getting log information from %s...", logfile)
    with open(logfile, 'r') as infile:
        perfdict = json.load(infile)

    train_ce = _sorted_ar(perfdict['train'], 'train_loss')
    train_ac = _sorted_ar(perfdict['train'], 'train_accuracy')
    test_ce = _sorted_ar(perfdict['test'], 'test_loss')
    test_ac = _sorted_ar(perfdict['test'], 'test_accuracy')
    return train_ce, train_ac, test_ce, test_ac


@click.group()
def cli():
    """Create visualizations for model results."""
    pass

@cli.command()
@click.argument('model_name', type=click.STRING)
@click.option('--display', is_flag=True, default=False,
              help='Do not write the output, but display the plot.')
def performance(model_name, display=False):
    """Create performance plots."""
    _LOGGER.info('Creating performance plot for model `%s`.',
                 model_name)
    if display:
        outfile = None
    else:
        outfile = os.path.join('results', model_name, 'performance.png')
    draw_perfplots(os.path.join('results', model_name, 'barrista_model.json'),
                   outfile)
    _LOGGER.info("Done.")


def draw_perfplots(logfile, outfile=None):
    """Draw the performance plots."""
    train_ce, train_ac, test_ce, test_ac =\
        _get_information(logfile)

    if not MPL_AVAILABLE:
        raise Exception("This method requires Matplotlib!")
    _, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
    # Loss.
    ax1.set_title("Loss")
    ax1.plot(train_ce[:, 0], train_ce[:, 1],
             label='Training', c='b', alpha=0.7)
    ax1.plot(test_ce[:, 0], test_ce[:, 1],
             label='Test', c='g', alpha=0.7)
    ax1.scatter(test_ce[:, 0], test_ce[:, 1],
                c='g', s=50)

    ax1.set_ylabel('Cross-Entropy-Loss')
    ax1.grid()
    # Accuracy.
    ax2.set_title("Accuracy")
    ax2.plot(train_ac[:, 0], train_ac[:, 1],
             label='Training', c='b', alpha=0.7)
    ax2.plot(test_ac[:, 0], test_ac[:, 1],
             label='Test', c='g', alpha=0.7)
    ax2.scatter(test_ac[:, 0], test_ac[:, 1],
                c='g', s=50)
    ax2.set_ylabel('Accuracy')
    ax2.grid()

    ax1.legend()
    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cli()
