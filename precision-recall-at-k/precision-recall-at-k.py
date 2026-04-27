def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    recommended = set(recommended[0:k])
    relevant = set(relevant)
    
    hits = len(recommended & relevant)
    # hits / k
    precision = hits / k

    # hits / number of relevant items
    recall = hits / len(relevant)
    return [precision, recall]
