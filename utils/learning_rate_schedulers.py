# utils/learning_rate_schedulers.py

import torch.optim as optim

def cosine_lr(optimizer, epoch, lr_max, lr_min, total_epochs):
    """
    Implements a cosine annealing learning rate scheduler.

    Args:
    - optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be adjusted.
    - epoch (int): The current epoch.
    - lr_max (float): The maximum learning rate.
    - lr_min (float): The minimum learning rate.
    - total_epochs (int): The total number of epochs.

    Returns:
    - None: Adjusts the learning rate of the optimizer.
    """
    lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + torch.cos(torch.tensor(epoch / total_epochs * 3.141592653589793)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
