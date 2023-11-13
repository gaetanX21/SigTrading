import numpy as np
import torch
import signatory
import itertools
import time


def timeit(func):
    def timed(*args, **kwargs):
        start = time.time()
        output = func(*args, **kwargs)
        stop = time.time()
        duration = stop - start
        print("\033[94m" + f"function {func.__name__} took {duration:.2f}s" + "\033[0m")
        return output

    return timed


def word_to_i(word, d):
    """
    Given a word written in the alphabet {0, 1, ..., d-1}, return its index in the lexicographic order (basically).
    """
    k = len(word)  # we're accessing the k-th signature term
    s = sum(d**i for i in range(k))  # 1 + d + d^2 + ... + d^(k-1)
    c = 0
    for i, w in enumerate(word):
        c += int(w) * d ** (k - 1 - i)
    return s + c


def get_length_k_words(k: int, channels: int) -> np.ndarray:
    """
    This method returns all possible words of length k with letters in {0,...,channels-1}.
    """
    alphabet = [str(i) for i in range(channels)]
    return np.array(["".join(i) for i in itertools.product(alphabet, repeat=k)])


def get_length_leq_k_words(k: int, channels: int) -> np.ndarray:
    """
    This method returns all possible words of length less or equal to k with letters in {0,...,channels-1}.
    """
    alphabet = [str(i) for i in range(channels)]
    words = []
    for i in range(k + 1):
        words_length_i = get_length_k_words(i, channels)
        words.append(words_length_i)
    return np.concatenate(words)


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
    """ " """
    batch_path_doubled = batch_path.repeat_interleave(
        2, dim=1
    )  # each path is doubled (with neighbors equal)
    batch_lead = batch_path_doubled[:, 1:, :]  # remove the first point of each path
    batch_lag = batch_path_doubled[:, :-1, :]  # remove the last point of each path

    # concatenate lead and lag paths
    batch_path_LL = torch.cat((batch_lead, batch_lag), dim=2)
    return batch_path_LL


@timeit
def compute_signature(batch_path: torch.Tensor, depth: int) -> torch.Tensor:
    """ """
    signature = signatory.signature(batch_path, depth, scalar_term=True)
    return signature


def shuffle_product(word1, word2):
    """
    Given two words, return the shuffle product of the two.
    TESTED, WORKS.
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
        left_term = [word + a for word in shuffle_left]  # (u ⧢ vb)a

        shuffle_right = shuffle_product(word1, v)  # (ua ⧢ v)
        right_term = [word + b for word in shuffle_right]

        return left_term + right_term  # union of the two
