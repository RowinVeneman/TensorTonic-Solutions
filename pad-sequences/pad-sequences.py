import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    if max_len:
        # Truncate
        seqs = [seq[0:max_len] for seq in seqs]
    if not max_len:
        max_len = max(len(seq) for seq in seqs)
    padded_ars = [np.pad(seq, (0, max_len - len(seq)), constant_values=pad_value) for seq in seqs]
    return np.array(padded_ars)