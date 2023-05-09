#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
import numpy as np

df = pd.read_csv("data_rigid_bodies_07_14_2022_15_16_48.csv")

timestamp_set = set()
previous_markerset = set()
current_marketset = set()

prv_tstamp = 0.0
curr_tstamp = 0.0
marker_positions = dict()
curr_markers_arr = np.array([0])
curr_markers_pos_arr = np.array([[0.0,0.0,0.0]])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(90, 90)

count = 1
for i in range(0,len(df)):
    curr_tstamp = df.iloc[i,0]
    if curr_tstamp != prv_tstamp:
        count += 1
        if count%10==0:
            ax.set_xlim3d([-0.5, 3.5])
            ax.set_xlabel('X')

            ax.set_ylim3d([-4.0, 4.0])
            ax.set_ylabel('Y')

            ax.set_zlim3d([-2.0, 2.0])
            ax.set_zlabel('Z')
            try:
                ax.scatter(curr_markers_pos_arr[:,0], curr_markers_pos_arr[:,1], curr_markers_pos_arr[:,2], s=4)
            except:
                print("err")
            plt.pause(0.001)
        if count%5 == 0 and count%10!=0:
            plt.cla()
        prv_tstamp = curr_tstamp
        curr_markers_pos_arr = np.array([[df.iloc[i,4], df.iloc[i,2], df.iloc[i,3]]])
        if count % 10 == 0:
            ax.view_init(90, 90)
            filename = '3d_vis_' + str(count) + '.png'
            plt.savefig(filename, dpi=300)
            print(count)

    else:
        curr_markers_pos_arr = np.vstack((curr_markers_pos_arr, np.array([df.iloc[i,4], df.iloc[i,2], df.iloc[i,3]])))
        prv_tstamp = curr_tstamp
        if count % 10 == 0:
            ax.view_init(90, 90)
            filename = '3d_vis_' + str(count) + '.png'
            plt.savefig(filename, dpi=75)
            print(count)



        

