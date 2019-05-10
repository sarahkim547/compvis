"""
This file contains the code for the post-processing of the images that were input into the neural network
"""
import numpy as np
from matplotlib import pyplot as plt


def fillInterior(net_output):
    """
    Completes the boundary filling procedure
    :param net_output: This is the output from the neural net with the probability maps, a numpy array
    :return filled_img: This is the filled out image, a numpy array
    """
    # Threshold the inside class probability map at 0.5
    thresh = 0.5
    height, width, _ = net_output.shape
    filled_img = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if net_output[i, j, 1] >= thresh:
                filled_img[i, j] = 1

    # Fill the 25 pixel border to ensure the next algorithm doesn't crash
    seeded_points = filled_img
    filled_img[0:25, :] = 1
    filled_img[-25:, :] = 1
    filled_img[:, 0:25] = 1
    filled_img[:, -25:] = 1

    # Iteratively fill in the gaps for each pixel
    increments_y = [1, 1, -1, -1]
    increments_x = [1, -1, 1, -1]
    for i in range(height):
        for j in range(width):
            # Check if the pixel is a seeded nucleus, and only then do the growing
            if seeded_points[i, j] == 1:
                queue = []
                prev_bdry_prob = 0
                curr_bdry_prob = 0
                num_bdry_pts = 4
                # Compute previous inside class probability and queue in all the new points
                for t in range(4):
                    queue.append([i + increments_y[t], j + increments_x[t]])
                    curr_bdry_prob += net_output[i + increments_y[t], j + increments_x[t], 2]/4
                while (len(queue) > 0) or (curr_bdry_prob > prev_bdry_prob):
                    new_point = queue.pop()
                    filled_img[new_point[0], new_point[1]] = 1
                    prev_bdry_prob = curr_bdry_prob
                    curr_bdry_prob = (curr_bdry_prob*num_bdry_pts - net_output[i, j, 2])/(num_bdry_pts - 1)
                    num_bdry_pts -= 1
                    # Check the points around new_point to see if they should be queued in
                    for t in range(4):
                        inc_point = [new_point[0] + increments_y[t], new_point[1] + increments_x[t]]
                        # First see if it's already filled in
                        if filled_img[inc_point[0], inc_point[1]] != 1:
                            # Then queue them in and update the boundary probability
                            queue.append(inc_point)
                            curr_bdry_prob = (curr_bdry_prob*num_bdry_pts + net_output[inc_point[0], inc_point[1]])/(num_bdry_pts + 1)
                            num_bdry_pts += 1

    return filled_img


def showImg(img):
    plt.imshow(img, interpolation='nearest')
    plt.show()


def main():
    img = np.load('Net Output/PrognosisTMABlock1_A_1_4_H&E.npy')
    filled_img = fillInterior(img)
    showImg(filled_img)


if __name__ == '__main__':
    main()
