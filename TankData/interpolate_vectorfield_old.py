import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Create a function taking in date and resolution
# and returning a grid of the interpolated velocity field
def create_vec_field(date, resolution, swap_xy=False):
    # Loading the original trajectory data
    df = pd.read_csv("data/data_rigid_bodies_"+str(date)+".csv")
    
    drifters = df["id_"].unique()
    num_drifters = drifters.shape[0]
    drifter_pos = dict()
    for i in range(num_drifters):
        drifter_pos[drifters[i]] = []
    time_stamp = df["timestamp"].unique()

    for i in range(num_drifters):
        drifter_df = df.loc[df['id_'] == drifters[i]]
        if swap_xy==False:
            # y axis is longer than x axis in the following order
            drifter_trajectory = np.array((drifter_df.iloc[:,0],drifter_df.iloc[:,4],drifter_df.iloc[:,2],drifter_df.iloc[:,3] )).T
        else:
            # makeing the x axis longer than y axis
            drifter_trajectory = np.array((drifter_df.iloc[:,0],drifter_df.iloc[:,2],drifter_df.iloc[:,4],drifter_df.iloc[:,3] )).T
        drifter_pos[drifters[i]].append(drifter_trajectory)

    # drifter_pos[drifters[0]] is a list of length 1
    # print(drifter_pos[drifters[0]][0].shape,'drifter_pos') #shape (39671, 4)

    # Create a grid to interpolate the data onto
    if swap_xy==False:
        xmin = -0.5
        xmax = 3.5
        ymin = -4
        ymax = 4
    else:
        xmin = -4
        xmax = 4
        ymin = -0.5
        ymax = 3.5
    x = np.arange(xmin, xmax, resolution)
    y = np.arange(ymin, ymax, resolution)
    X, Y = np.meshgrid(x, y)

    # Calculate the velocity field on the grid using linear interpolation
    grid_U = np.zeros_like(X)
    grid_V = np.zeros_like(X)
    for i in range(num_drifters):
        # Take the discrete gradient of the trajectory
        dt_series = drifter_pos[drifters[i]][0][:-1,0]-drifter_pos[drifters[i]][0][1:,0]
        U = drifter_pos[drifters[i]][0][1:,1]-drifter_pos[drifters[i]][0][:-1,1]
        U = U/dt_series
        V = drifter_pos[drifters[i]][0][1:,2]-drifter_pos[drifters[i]][0][:-1,2]
        V = V/dt_series

        interp_U = griddata((drifter_pos[drifters[i]][0][:-1,1], drifter_pos[drifters[i]][0][:-1,2]), U, (X, Y), method='linear', fill_value=0)
        interp_V = griddata((drifter_pos[drifters[i]][0][:-1,1], drifter_pos[drifters[i]][0][:-1,2]), V, (X, Y), method='linear', fill_value=0)
        # interp_V = griddata((x_traj[i], y_traj[i]), V, (X, Y), method='linear', fill_value=0)
        grid_U += interp_U
        grid_V += interp_V
    # Take the average of the velocity field over all drifters
    grid_U = grid_U/num_drifters
    grid_V = grid_V/num_drifters

    # Save the velocity field using np.save
    if swap_xy==False:
        np.save('data/velocity_field_'+str(date)+'_res_'+str(resolution)+'.npy', np.array((X, Y, grid_U, grid_V)))
    else:
        np.save('data/velocity_field_'+str(date)+'_res_'+str(resolution)+'_xy_swap.npy', np.array((X, Y, grid_U, grid_V)))

    # Plot the vector field
    fig, ax = plt.subplots()
    flow_size = np.sqrt(grid_U**2 + grid_V**2)
    qq = ax.quiver(X, Y, grid_U, grid_V,flow_size)
    plt.colorbar(qq, cmap = plt.cm.jet,label='flow speed w[m/s]')
    ax.set_xlabel('$x[m]$')
    ax.set_ylabel('$y[m]$')
    ax.set_aspect('equal')
    plt.savefig('data/velocity_field_'+str(date)+'.png')
    plt.show()

if __name__ == "__main__":
    create_vec_field(date='12_08_2022_12_52_06',resolution = 0.05)