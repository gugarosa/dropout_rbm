import argparse

import numpy as np
import torch

import utils.loader as l
import utils.objects as m
import utils.opt as o
import utils.target as t


def get_arguments():
    """Gets arguments from the command line.

    Returns:
        A parser with the input arguments.

    """

    # Creates the ArgumentParser
    parser = argparse.ArgumentParser(usage='Optimizes a model over the validation set.')

    parser.add_argument('dataset', help='Dataset identifier', choices=['mnist', 'semeion', 'usps'])

    parser.add_argument('mh', help='Meta-heuristic identifier', choices=['ba', 'cs', 'fa', 'pso'])

    parser.add_argument('-n_visible', help='Number of visible units', type=int, default=784)

    parser.add_argument('-n_hidden', help='Number of hidden units', type=int, default=400)

    parser.add_argument('-steps', help='Number of CD steps', type=int, default=1)

    parser.add_argument('-lr', help='Learning rate', type=float, default=0.1)

    parser.add_argument('-momentum', help='Momentum', type=float, default=0.0002)

    parser.add_argument('-decay', help='Weight decay', type=float, default=0.5)

    parser.add_argument('-temp', help='Temperature', type=float, default=1)

    parser.add_argument('-batch_size', help='Batch size', type=int, default=20)

    parser.add_argument('-device', help='CPU or GPU usage', choices=['cpu', 'cuda'])

    parser.add_argument('-epochs', help='Number of training epochs', type=int, default=25)

    parser.add_argument('-n_agents', help='Number of meta-heuristic agents', type=int, default=15)

    parser.add_argument('-n_iter', help='Number of meta-heuristic iterations', type=int, default=25)

    parser.add_argument('-seed', help='Seed identifier', type=int, default=0)

    return parser.parse_args()


if __name__ == '__main__':
    # Gathers the input arguments
    args = get_arguments()

    # Gathering common variables
    dataset = args.dataset
    seed = args.seed

    # Gathering RBM variables
    n_visible = args.n_visible
    n_hidden = args.n_hidden
    steps = args.steps
    lr = args.lr
    momentum = args.momentum
    decay = args.decay
    T = args.temp
    device = args.device
    batch_size = args.batch_size
    epochs = args.epochs
    model = m.get_model('drbm').obj

    # Gathering optimization variables
    meta = args.mh
    n_agents = args.n_agents
    n_iterations = args.n_iter
    meta_heuristic = m.get_mh(meta).obj
    hyperparams = m.get_mh(meta).hyperparams

    # Checks for the name of device
    if device == 'cpu':
        # Updates accordingly
        use_gpu = False
    else:
        # Updates accordingly
        use_gpu = True

    # Loads the data
    train, val, _ = l.load_dataset(name=dataset)

    # Defining torch and numpy seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initializes the optimization target
    opt_fn = t.reconstruction(model, train, val, n_visible, n_hidden, steps, lr, momentum, decay, T, use_gpu, batch_size, epochs)

    # Running the optimization task
    history = o.optimize(meta_heuristic, opt_fn, n_agents, n_iterations, hyperparams)

    # Saves the history object to an output file
    history.save(f'models/{meta}_{n_hidden}hid_{lr}lr_drbm_{dataset}_{seed}.pkl')
