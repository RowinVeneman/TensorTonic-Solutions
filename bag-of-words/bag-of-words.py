import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    unique, counts = np.unique(tokens, return_counts=True)
    res = np.zeros(len(vocab), dtype=int)
    for word, count in zip(unique, counts):
        if word in vocab:
            res[vocab.index(word)] = count
    return res
            
            