import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Create a function taking in date and resolution
# and returning a grid of the interpolated velocity field
def create_vec_field(date, resolution, swap_xy=False):
    print('###running create_vec_field###')
    print('date', date)
    print('resolution', resolution)
    print('swap_xy', swap_xy)

    threshold = 0.3

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
            # drifter_trajectory = np.array((drifter_df.iloc[:,0],drifter_df.iloc[:,4],drifter_df.iloc[:,2],drifter_df.iloc[:,3] )).T
            drifter_trajectory = np.array((drifter_df.iloc[:,0],drifter_df.iloc[:,2],-drifter_df.iloc[:,4],drifter_df.iloc[:,3] )).T

        else:
            # makeing the x axis longer than y axis
            drifter_trajectory = np.array((drifter_df.iloc[:,0],drifter_df.iloc[:,2],drifter_df.iloc[:,4],drifter_df.iloc[:,3] )).T
        drifter_pos[drifters[i]].append(drifter_trajectory)

    # drifter_pos[drifters[0]] is a list of length 1
    # print(drifter_pos[drifters[0]][0].shape,'drifter_pos') #shape (39671, 4)

    # Create a grid to interpolate the data onto
    if swap_xy==False:
        # xmin = -0.5
        # xmax = 3.5
        # ymin = -4
        # ymax = 4
        xmin = -3
        xmax = 3
        ymin = -3.5
        ymax = 0.5
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
        x_traj = drifter_pos[drifters[i]][0][:-1,1]
        y_traj = drifter_pos[drifters[i]][0][:-1,2]

        print(U.shape, V.shape, dt_series.shape, x_traj.shape, y_traj.shape, 'before removing U, V, dt_series, x_traj, y_traj')
        flow_size = np.sqrt(U**2+V**2)
        # First, figure out the index where flow size is larger than threshold
        # Then, remove the data points where flow size is larger than threshold
        # Remove the dt series as well
        index = np.where(flow_size>threshold)
        U = np.delete(U, index)
        V = np.delete(V, index)
        dt_series = np.delete(dt_series, index)
        x_traj = np.delete(x_traj, index)
        y_traj = np.delete(y_traj, index)
        print(index,'index')
        print(len(index[0]),'number of data points removed')
        print(U.shape, V.shape, dt_series.shape, x_traj.shape, y_traj.shape, 'after removing U, V, dt_series, x_traj, y_traj')


        # Concatenate the velocity data over different drifters
        if i==0:
            concatenate_U = U
            concatenate_V = V
            # print(concatenate_U.shape, concatenate_V.shape, 'concatenate_U, concatenate_V')
            concatenate_x_traj = x_traj
            concatenate_y_traj = y_traj
        else:
            concatenate_U = np.concatenate((concatenate_U, U))
            concatenate_V = np.concatenate((concatenate_V, V))
            # print(concatenate_U.shape, concatenate_V.shape, 'concatenate_U, concatenate_V')
            concatenate_x_traj = np.concatenate((concatenate_x_traj, x_traj))
            concatenate_y_traj = np.concatenate((concatenate_y_traj, y_traj))
            # print(concatenate_x_traj.shape, concatenate_y_traj.shape, 'concatenate_x_traj, concatenate_y_traj')
            # exit()
            
    # Interpolate the velocity field on the grid
    grid_U = griddata((concatenate_x_traj, concatenate_y_traj), concatenate_U, (X, Y), method='linear', fill_value=0)
    grid_V = griddata((concatenate_x_traj, concatenate_y_traj), concatenate_V, (X, Y), method='linear', fill_value=0)

    print('xmin',X.min(),'xmax',X.max(),'ymin',Y.min(),'ymax',Y.max())
    grid_info = np.array((xmin, xmax, ymin, ymax, X, Y, grid_U, grid_V))
    np.save('data/grid_info_'+str(date)+'_res_'+str(resolution)+'concatenate.npy', grid_info)
    xmin, xmax, ymin, ymax, X, Y, grid_U, grid_V = np.load('data/grid_info_'+str(date)+'_res_'+str(resolution)+'concatenate.npy', allow_pickle=True)
    print('xmin',xmin,'xmax',xmax,'ymin',ymin,'ymax',ymax)
    print(X.shape, Y.shape, grid_U.shape, grid_V.shape, 'X, Y, grid_U, grid_V')
    # Save the velocity field using np.save
    if swap_xy==False:
        np.save('data/velocity_field_'+str(date)+'_res_'+str(resolution)+'concatenate.npy', np.array((X, Y, grid_U, grid_V)))
    else:
        np.save('data/velocity_field_'+str(date)+'_res_'+str(resolution)+'concatenate_xy_swap.npy', np.array((X, Y, grid_U, grid_V)))

    # Plot the vector field
    fig, ax = plt.subplots()
    flow_size = np.sqrt(grid_U**2 + grid_V**2)
    qq = ax.quiver(X, Y, grid_U, grid_V,flow_size)
    plt.colorbar(qq, cmap = plt.cm.jet,label='flow speed w[m/s]')
    ax.set_xlabel('$x[m]$')
    ax.set_ylabel('$y[m]$')
    ax.set_aspect('equal')
    plt.savefig('data/velocity_field_'+str(date)+'concatenate.png')
    plt.show()

if __name__ == "__main__":
    create_vec_field(date='02_26_2023_13_05_55',resolution = 0.05,swap_xy=False)