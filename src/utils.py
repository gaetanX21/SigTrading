import numpy as np
import torch
import signatory
import itertools
import time
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List

sns.set_theme()


def timeit(func):
    """
    Decorator to time a function, displaying the time it took to run it iff it took more than 1 second.
    """

    def timed(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        stop = time.time()
        duration = stop - start
        if duration > 1:
            print(
                "\033[91m"
                + f"function {func.__name__} took {duration:.2f}s"
                + "\033[0m"
            )
        return output

    return timed


def word_to_i(word: List[int], d: int) -> int:
    """
    Given a word written in the alphabet {0, 1, ..., d-1}, return its index in the lexicographic order (basically).
    Recall that word is a list of integers. (with integer i representing (i-1)-th letter of the alphabet)

    The purpose of this utility function is to go from word indices (as defined in the paper) to integer indices
    (as defined by our implementation), which is useful for indexing signature objects since we represent them as flat tensors.
    """
    k = len(word)  # we're accessing the k-th signature term
    s = sum(d**i for i in range(k))  # 1 + d + d^2 + ... + d^(k-1)
    c = 0
    for i, letter in enumerate(word):
        c += letter * d ** (k - 1 - i)
    return s + c


def get_length_k_words(k: int, channels: int) -> np.ndarray:
    """
    This method returns all possible words of length k with letters in {0,...,channels-1}.
    Recall that a word is a list of integers. (with integer i representing (i-1)-th letter of the alphabet)
    """
    alphabet = list(range(channels))
    return list(
        list(tup) for tup in itertools.product(alphabet, repeat=k)
    )  # we want lists, not tuples, for indexing, otherwise we cannot concatenate them easily afterwards


def get_length_leq_k_words(k: int, channels: int) -> np.ndarray:
    """
    This method returns all possible words of length less or equal to k with letters in {0,...,channels-1}.
    """
    alphabet = list(range(channels))
    words = []
    for i in range(k + 1):
        words_length_i = get_length_k_words(i, channels)
        words.append(words_length_i)
    # put all words in a single list
    words_single_list = [word for sublist in words for word in sublist]
    return words_single_list


def get_number_of_words_k(k: int, channels: int) -> int:
    """
    This method returns the number of words of length k with letters in {0,...,channels-1}.
    """
    return channels**k


def get_number_of_words_leq_k(k: int, channels: int) -> int:
    """
    This method returns the number of words of length less or equal to k with letters in {0,...,channels-1}.
    """
    return sum(get_number_of_words_k(i, channels) for i in range(k + 1))


@timeit
def compute_lead_lag_transform(batch_path: torch.Tensor) -> torch.Tensor:
    """
    Computes the lead-lag transform of a batch of paths.
    """
    batch_path_doubled = batch_path.repeat_interleave(
        2, dim=1
    )  # each path is doubled (with neighbors equal)
    batch_lead = batch_path_doubled[:, 1:, :]  # remove the first point of each path
    batch_lag = batch_path_doubled[:, :-1, :]  # remove the last point of each path

    # concatenate lead and lag paths
    batch_path_LL = torch.cat((batch_lead, batch_lag), dim=2)
    return batch_path_LL


@timeit
def compute_signature(
    batch_path: torch.Tensor, depth: int, no_batch: bool = False
) -> torch.Tensor:
    """
    Computes the signature of a batch of paths, using signatory.
    """
    if depth == 0:
        if no_batch:
            return torch.ones(1)
        else:
            return torch.ones(batch_path.shape[0], 1)
    if no_batch:
        sig = signatory.signature(
            batch_path.unsqueeze(0), depth, scalar_term=True
        ).squeeze(0)
    else:
        sig = signatory.signature(batch_path, depth, scalar_term=True)

    return sig


def shuffle_product(word1, word2):
    """
    Given two words, return the shuffle product of the two.
    """
    if len(word1) == 0:
        return word2
    if len(word2) == 0:
        return word1

    if len(word1) == 1:
        return [word2[:k] + word1 + word2[k:] for k in range(len(word2) + 1)]
    elif len(word2) == 1:
        return [word1[:k] + word2 + word1[k:] for k in range(len(word1) + 1)]

    else:
        # we use ua ⧢ vb = (u ⧢ vb)a + (ua ⧢ v)b
        # word1 = ua, word2 = vb
        u, a = word1[:-1], word1[-1]
        v, b = word2[:-1], word2[-1]

        shuffle_left = shuffle_product(u, word2)  # (u ⧢ vb)
        left_term = [word + [a] for word in shuffle_left]  # (u ⧢ vb)a

        shuffle_right = shuffle_product(word1, v)  # (ua ⧢ v)
        right_term = [word + [b] for word in shuffle_right]

        return left_term + right_term  # union of the two


def plot_cum_pnl(cum_pnl: torch.tensor) -> None:
    """
    Plots the cumulative PnL of a trading strategy.
    """
    plt.figure(figsize=(10, 5))
    n_assets = cum_pnl.shape[1]
    T = cum_pnl.shape[0]  # number of time steps
    for i in range(n_assets):
        plt.plot(
            np.arange(T),
            cum_pnl[:, i],
            label=f"cumulative PnL from trading on asset {i}",
        )
    cum_pnl_all = torch.sum(cum_pnl, axis=1)
    plt.plot(np.arange(T), cum_pnl_all, label="cumulative PnL from trading all assets")
    plt.legend()
    plt.xlabel("time")
    plt.ylabel("cumulative PnL")
    plt.show()


def print_signature(flat: torch.tensor, channels: int, depth: int) -> None:
    """
    Receives a {depth}-truncated signature of a signal with {channels} dimension and prints it in a pretty way.
    """
    N = flat.shape[0]
    s = 0
    for k in range(depth + 1):
        s += channels**k
    assert N == s, "The signature is not of the expected length."

    for k in range(depth + 1):
        print(f"Level {k}:")
        index_0 = get_number_of_words_leq_k(k - 1, channels)
        level_k = flat[index_0 : index_0 + get_number_of_words_k(k, channels)]
        print(level_k)
