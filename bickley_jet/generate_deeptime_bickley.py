import deeptime
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

def get_bickley_data(n_particles):
    # get dataset
    dataset = deeptime.data.bickley_jet(n_particles, n_jobs=8)
    
    # 401 timesteps worth of n_particles, for x and y positions
    Y = dataset.data
    
    # re-shaping into x and y; n_particles; time_steps
    Y_x = Y[:, :, 0].T
    Y_y = Y[:, :, 1].T
    
    # Y_new = n x m x t 
    # n = x-y positions of particles, m is number of particles, number of timesteps generated (401)
    Y_new = np.zeros((2, Y.shape[1], Y.shape[0]))

    Y_new[0] = Y_x
    Y_new[1] = Y_y
    
    return Y_new


def main(n_particles, save_data, time_steps):
    # get data, reshaped into training format
    Y_new = get_bickley_data(n_particles)
    
    # plot colored by the particles x positions
    for i in range(time_steps):
        plt.figure()
        plt.scatter(*Y_new[:, :, i], c=Y_new[:, :, 0][0])
        plt.title('i=' + str(i))
    plt.show()
    
    if save_data:
        np.save('bickley_data_' + str(n_particles) + '.npy', Y_new)
        a = np.load('bickley_deeptime.npy')
    
    
if __name__ == "__main__":
    n_particles = 2000
    save_data = False 
    time_steps = 50
    
    main(n_particles, save_data, time_steps)
    
    
    
    