import numpy as np
from scipy.signal import convolve2d


def getFeature(x, featureType, featureTransform='none'):
    if featureType == 'pixel':
        features = zeroFeatures(x)
    elif featureType == 'hog':
        features = hogFeatures(x)
    elif featureType == 'lbp':
        features = lbpFeatures(x)

    if featureTransform == 'sqrt':
        features = np.sqrt(features)
    elif featureTransform == 'l2':
        features = features / np.linalg.norm(features, axis=0)

    return features


def zeroFeatures(x):
    return np.stack([x[:, :, i].flatten() for i in range(x.shape[2])]).transpose((1, 0))


def hogFeatures(x):
    features = []
    sobelFilter = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    for i in range(x.shape[2]):
        image = x[:, :, i]

        gx = convolve2d(image, sobelFilter, mode='same')
        gy = convolve2d(image, sobelFilter.T, mode='same')

        m = np.sqrt(gx ** 2 + gy ** 2)
        o = np.arctan2(gy, gx) * (180 / np.pi)
        o[o < 0] = o[o < 0] + 180

        cell_size = 4

        cells_x = m.shape[1] // cell_size
        cells_y = m.shape[0] // cell_size
        image_feature = []
        for i in range(cells_y):
            for j in range(cells_x):
                cell_magnitude = m[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
                cell_angle = o[i * cell_size:(i + 1) * cell_size, j * cell_size:(j + 1) * cell_size]
                hist, _ = np.histogram(cell_angle, bins=8, range=(0, 180), weights=cell_magnitude)
                image_feature.append(hist)

        image_feature = np.array(image_feature).flatten()

        features.append(image_feature)

    features = np.array(features).transpose((1, 0))

    return features


def lbpFeatures(x):
    features = []
    for i in range(x.shape[2]):
        image = x[:, :, i]
        height, width = image.shape[0], image.shape[1]
        image_features = np.empty_like(image)
        for i in range(height):
            for j in range(width):
                image_features[i, j] = apply_bit_representation(image, i, j)
        hist, _ = np.histogram(image_features.flatten(), 256, (0, 255))

        features.append(hist)

    features = np.array(features).transpose((1, 0))

    return features


x, y = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))


def apply_bit_representation(image, i, j):
    center = image[i, j]
    mul = 1
    value = 0
    for x_i in reversed(range(3)):
        for y_j in range(3):
            if x[x_i, y_j] != 0 or y[x_i, y_j] != 0:
                value += mul * calculate_value(image, center, i + x[x_i, y_j], j + y[x_i, y_j])
                mul *= 2

    return value


def calculate_value(img, center, x_i, y_j):
    value = 0
    try:
        if img[x_i][y_j] > center:
            value = 1
    except:
        pass
    return value
