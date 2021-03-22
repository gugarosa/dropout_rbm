import torch
import torchvision as tv

# A constant used to hold a dictionary of possible datasets
DATASETS = {
    'mnist': tv.datasets.MNIST,
    'semeion': tv.datasets.SEMEION,
    'usps': tv.datasets.USPS
}


def load_semeion(val_split=0.2):
    """Loads Semeion dataset.

    Args:
        val_split (float): Percentage of split for the validation set.

    Returns:
        Training, validation and testing sets of loaded dataset.

    """

    # Defining the torch seed
    torch.manual_seed(0)

    # Loads the training data
    data = DATASETS['semeion'](root='./data', download=True,
                               transform=tv.transforms.ToTensor())

    # Splitting the data into training/validation/test
    train, val, test = torch.utils.data.random_split(data, [int(len(
        data) * (1 - 2 * val_split) + 1), int(len(data) * val_split), int(len(data) * val_split)])

    return train, val, test


def load_dataset(name='mnist', val_split=0.22, seed=0):
    """Loads a dataset.

    Args:
        name (str): Name of dataset to be loaded.
        val_split (float): Percentage of split for the validation set.
        seed (int): Randomness seed.

    Returns:
        Training, validation and testing sets of loaded dataset.

    """

    # Defining the torch seed
    torch.manual_seed(seed)

    # Checks if it is supposed to load custom datasets
    if name == 'semeion':
        return load_semeion(val_split)

    # Loads the training data
    train = DATASETS[name](root='./data', train=True, download=True,
                           transform=tv.transforms.Compose(
                               [tv.transforms.ToTensor()])
                           )

    # Splitting the training data into training/validation
    if name == 'mnist':
        train, val = torch.utils.data.random_split(
            train, [int(len(train) * (1 - val_split)), int(len(train) * val_split)])
    elif name == 'usps':
        train, val = torch.utils.data.random_split(
            train, [int(len(train) * (1 - val_split) + 1), int(len(train) * val_split)])

    # Loads the testing data
    test = DATASETS[name](root='./data', train=False, download=True,
                          transform=tv.transforms.Compose(
                              [tv.transforms.ToTensor()])
                          )

    return train, val, test
