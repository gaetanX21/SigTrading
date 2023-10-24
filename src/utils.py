import numpy as np
import torch
import signatory
import itertools


def get_level_k_words(k: int, channels:int) -> np.ndarray:
    """
    This method returns all possible words of length k with letters in {0,...,channels-1}.
    """
    return np.array(list(itertools.product(range(channels), repeat=k)))


def compute_lead_lag_transform(batch_path: torch.Tensor) -> torch.Tensor:
    """"
    """
    batch_path_doubled = batch_path.repeat_interleave(2, dim=1) # each path is doubled (with neighbors equal)
    batch_lead = batch_path_doubled[:, 1:, :] # remove the first point of each path
    batch_lag = batch_path_doubled[:, :-1, :] # remove the last point of each path

    # concatenate lead and lag paths
    batch_path_LL = torch.cat((batch_lead, batch_lag), dim=2)
    return batch_path_LL

def compute_signature(batch_path: torch.Tensor, depth: int) -> torch.Tensor:
    """
    """
    signature = signatory.signature(batch_path, depth, scalar_term=True)
    return signature