def centroid_and_average_distance(X):
    # Compute centroid and unpack it into cx and cy
    cx, cy = np.mean(X, axis=0)

    # Compute distances to centroid
    distances = np.sqrt(np.sum((X - [cx, cy]) ** 2, axis=1))

    # Compute average distance
    average_distance = np.mean(distances)

    return cx, cy, average_distance
