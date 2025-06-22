# # utils/helpers.py (or utils/io.py)

# import os
# from datetime import datetime


# def create_output_dir(config):
#     """
#     Creates the output directory if it doesn't exist.
#     If the path is empty, it will generate a default path based on the current date and time.
#     """
#     # Check if output_dir is provided in config, otherwise generate a default path
#     output_path = config.output_dir if hasattr(config, 'output_dir') else None
    
#     if not output_path:
#         # Generate default output path with date-time
#         output_path = f'./outputs/{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
#     if not os.path.exists(output_path):
#         os.makedirs(output_path)
#         print(f"Directory created at: {output_path}")
#     else:
#         print(f"Directory already exists at: {output_path}")
    
#     return output_path





# def create_checkpoint_dir(checkpoint_path):
#     """
#     Creates the checkpoint directory if it doesn't exist.

#     Args:
#     - checkpoint_path (str): The path of the checkpoint directory to create.

#     Returns:
#     - str: The path of the created or existing checkpoint directory.
#     """
#     if not os.path.exists(checkpoint_path):
#         os.makedirs(checkpoint_path)
#         print(f"Checkpoint directory created at: {checkpoint_path}")
#     else:
#         print(f"Checkpoint directory already exists at: {checkpoint_path}")
#     return checkpoint_path



# # utils/helpers.py

# import torch

# def flatten_temporal_batch_dims(x):
#     """
#     Flattens the batch dimensions for temporal sequences.

#     Args:
#     - x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, features).

#     Returns:
#     - torch.Tensor: Flattened tensor of shape (batch_size * sequence_length, features).
#     """
#     batch_size, seq_len, features = x.size()
#     return x.view(batch_size * seq_len, features)
import os
from os import path
import datetime
import shutil
import torch
import numpy as np


def flatten_temporal_batch_dims(outputs, targets):
    def is_nested_list(x):
        return isinstance(x, list) and len(x) > 0 and isinstance(x[0], list)

    # Flatten outputs
    for k in outputs:
        if is_nested_list(outputs[k]):
            outputs[k] = [i for step_t in outputs[k] for i in step_t]

    # Flatten targets
    if isinstance(targets, list) and len(targets) > 0 and isinstance(targets[0], list):
        targets = [i for step_t in targets for i in step_t]

    return outputs, targets



def create_output_dir(config):
    if config.output_dir:
        output_dir_path = config.output_dir
    else:
        root = '/home/nazir/NeurIPS2023_SOC/outputs'
        output_dir_path = path.join(root, 'runs', config.dataset_name, config.version)

    os.makedirs(output_dir_path, exist_ok=True)
    shutil.copyfile(src=config.config_path, dst=path.join(output_dir_path, 'config.yaml'))
    return output_dir_path



def create_checkpoint_dir(output_dir_path):
    checkpoint_dir_path = path.join(output_dir_path, 'checkpoints')
    os.makedirs(checkpoint_dir_path, exist_ok=True)
    return checkpoint_dir_path

def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr

def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length

def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster