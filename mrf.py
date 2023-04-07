import numpy as np


class MRFDenoiser:
    def __init__(self, alpha=1.0, beta=1.0, iter=1000):
        self.alpha = alpha
        self.beta = beta
        self.iter = iter

    def boundary_checker(image, x, y):
        if 0 <= x <= image.shape[0] - 1 and 0 <= y <= image.shape[1] - 1:
            return image[x][y]
        else:
            return 0

    def prob_calc(self, imageX, imageY, i, j, y_val):
        prob = self.alpha * imageX[i][j] * y_val
        prob += self.beta * y_val * self.boundary_checker(imageY, i - 1, j)
        prob += self.beta * y_val * self.boundary_checker(imageY, i, j - 1)
        prob += self.beta * y_val * self.boundary_checker(imageY, i + 1, j)
        prob += self.beta * y_val * self.boundary_checker(imageY, i, j + 1)
        return prob

    def denoise(self, image):
        width, height = image.shape
        denoised_image = np.copy(image)
        for i in range(self.iter):
            for w in range(width):
                for h in range(height):
                    pos_prob = self.prob_calc(image, denoised_image, w, h, 1)
                    neg_prob = self.prob_calc(image, denoised_image, w, h, -1)
                    if neg_prob > pos_prob:
                        denoised_image[w][h] = -1
                    else:
                        denoised_image[w][h] = 1

        return denoised_image
