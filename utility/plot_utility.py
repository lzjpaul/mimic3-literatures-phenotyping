import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

def draw_pl(x=[], y=[], x_y=[], type='o'):
    if len(x_y) != 0:
        x = np.array(x_y)[:, 0]
        y = np.array(x_y)[:, 1]
    pl.plot(x, y, type)
    pl.show()


if __name__ == '__main__':
    x = [1, 2, 3, 4, 5]  # Make an array of x values
    y = [1, 4, 9, 16, 25]  # Make an array of y values for each x value

    x_y = [[1,2],[3,4],[5,6]]
    print list(np.array(x_y)[:, 0].flatten())
    draw_pl(x_y=x_y)