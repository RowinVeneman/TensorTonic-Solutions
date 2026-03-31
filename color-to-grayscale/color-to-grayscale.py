def color_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    H = len(image)
    W = len(image[0])
    res = [[0 for _ in range(W)] for _ in range(H)]
    for i in range(H):
        for j in range(W):
            res[i][j] = image[i][j][0] * 0.299 + 0.587 * image[i][j][1] + 0.114 * image[i][j][2]
    return res