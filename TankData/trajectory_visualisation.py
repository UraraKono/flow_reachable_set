#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

date='02_26_2023_13_05_55'

df = pd.read_csv("data/data_rigid_bodies_"+str(date)+".csv")

drifters = df["id_"].unique()
num_drifters = drifters.shape[0]
drifter_pos = dict()
for i in range(num_drifters):
    drifter_pos[drifters[i]] = []
time_stamp = df["timestamp"].unique()

fig = plt.figure(dpi = 150)#, figsize=plt.figaspect(1)
ax = Axes3D(fig)
ax.view_init(90, 90)
ax.set_xlim3d([-0.5, 3.5])
ax.set_xlabel('X(m)')
ax.set_ylim3d([-4.0, 4.0])
ax.set_ylabel('Y(m)')
ax.set_zlim3d([-2.0, 2.0])
ax.set_zlabel('Z(m)')
plt.title("Drifter trajectories in a Single Gyre flow simulated in Tank")

for i in range(num_drifters):
    drifter_df = df.loc[df['id_'] == drifters[i]]
    drifter_trajectory = np.array((drifter_df.iloc[:,0],drifter_df.iloc[:,4],drifter_df.iloc[:,2],drifter_df.iloc[:,3] )).T
    drifter_pos[drifters[i]].append(drifter_trajectory)

#Plotting and save images
# ax.plot3D(drifter_pos[drifters[0]][0][:,1], drifter_pos[drifters[0]][0][:,2], drifter_pos[drifters[0]][0][:,3], 'green')
# ax.plot3D(drifter_pos[drifters[1]][0][:,1], drifter_pos[drifters[1]][0][:,2], drifter_pos[drifters[1]][0][:,3], 'blue')
# ax.plot3D(drifter_pos[drifters[2]][0][:,1], drifter_pos[drifters[2]][0][:,2], drifter_pos[drifters[2]][0][:,3], 'yellow')
# ax.plot3D(drifter_pos[drifters[3]][0][:,1], drifter_pos[drifters[3]][0][:,2], drifter_pos[drifters[3]][0][:,3], 'black')
# ax.plot3D(drifter_pos[drifters[4]][0][:,1], drifter_pos[drifters[4]][0][:,2], drifter_pos[drifters[4]][0][:,3], 'orange')

print("done")
line1 = plt.plot(drifter_pos[drifters[0]][0][:1, 1], drifter_pos[drifters[0]][0][:1, 2], drifter_pos[drifters[0]][0][:1, 3],'green',label='Drifter 1')[0]
line2 = plt.plot(drifter_pos[drifters[1]][0][:1, 1], drifter_pos[drifters[1]][0][:1, 2], drifter_pos[drifters[1]][0][:1, 3],'blue',label='Drifter 2')[0]
line3 = plt.plot(drifter_pos[drifters[2]][0][:1, 1], drifter_pos[drifters[2]][0][:1, 2], drifter_pos[drifters[2]][0][:1, 3],'orange',label='Drifter 3')[0]
line4 = plt.plot(drifter_pos[drifters[3]][0][:1, 1], drifter_pos[drifters[3]][0][:1, 2], drifter_pos[drifters[3]][0][:1, 3],'black',label='Drifter 4')[0]
line5 = plt.plot(drifter_pos[drifters[4]][0][:1, 1], drifter_pos[drifters[4]][0][:1, 2], drifter_pos[drifters[4]][0][:1, 3],'darkviolet',label='Drifter 5')[0]
Dot1 = plt.plot(drifter_pos[drifters[0]][0][:1, 1], drifter_pos[drifters[0]][0][:1, 2], drifter_pos[drifters[0]][0][:1, 3], lw=2, color='green', marker='o')[0]
Dot2 = plt.plot(drifter_pos[drifters[0]][0][:1, 1], drifter_pos[drifters[0]][0][:1, 2], drifter_pos[drifters[0]][0][:1, 3], lw=2, color='blue', marker='o')[0]
Dot3 = plt.plot(drifter_pos[drifters[0]][0][:1, 1], drifter_pos[drifters[0]][0][:1, 2], drifter_pos[drifters[0]][0][:1, 3], lw=2, color='orange', marker='o')[0]
Dot4 = plt.plot(drifter_pos[drifters[0]][0][:1, 1], drifter_pos[drifters[0]][0][:1, 2], drifter_pos[drifters[0]][0][:1, 3], lw=2, color='black', marker='o')[0]
Dot5 = plt.plot(drifter_pos[drifters[0]][0][:1, 1], drifter_pos[drifters[0]][0][:1, 2], drifter_pos[drifters[0]][0][:1, 3], lw=2, color='darkviolet', marker='o')[0]
ax.legend(loc='lower right')

def animate(iter,line1, line2, line3, line4, line5,Dot1,Dot2, Dot3, Dot4, Dot5):
    iter = iter*100
    line1.set_data(drifter_pos[drifters[0]][0][:iter, 1], drifter_pos[drifters[0]][0][:iter, 2])
    line1.set_3d_properties(drifter_pos[drifters[0]][0][:iter, 3])
    line2.set_data(drifter_pos[drifters[1]][0][:iter, 1], drifter_pos[drifters[1]][0][:iter, 2])
    line2.set_3d_properties(drifter_pos[drifters[1]][0][:iter, 3])
    line3.set_data(drifter_pos[drifters[2]][0][:iter, 1], drifter_pos[drifters[2]][0][:iter, 2])
    line3.set_3d_properties(drifter_pos[drifters[2]][0][:iter, 3])
    line4.set_data(drifter_pos[drifters[3]][0][:iter, 1], drifter_pos[drifters[3]][0][:iter, 2])
    line4.set_3d_properties(drifter_pos[drifters[3]][0][:iter, 3])
    line5.set_data(drifter_pos[drifters[4]][0][:iter, 1], drifter_pos[drifters[4]][0][:iter, 2])
    line5.set_3d_properties(drifter_pos[drifters[4]][0][:iter, 3])
    Dot1.set_data(drifter_pos[drifters[0]][0][iter, 1], drifter_pos[drifters[0]][0][iter, 2])
    Dot1.set_3d_properties(drifter_pos[drifters[0]][0][iter, 3])
    Dot2.set_data(drifter_pos[drifters[1]][0][iter, 1], drifter_pos[drifters[1]][0][iter, 2])
    Dot2.set_3d_properties(drifter_pos[drifters[1]][0][iter, 3])
    Dot3.set_data(drifter_pos[drifters[2]][0][iter, 1], drifter_pos[drifters[2]][0][iter, 2])
    Dot3.set_3d_properties(drifter_pos[drifters[2]][0][iter, 3])
    Dot4.set_data(drifter_pos[drifters[3]][0][iter, 1], drifter_pos[drifters[3]][0][iter, 2])
    Dot4.set_3d_properties(drifter_pos[drifters[3]][0][iter, 3])
    Dot5.set_data(drifter_pos[drifters[4]][0][iter, 1], drifter_pos[drifters[4]][0][iter, 2])
    Dot5.set_3d_properties(drifter_pos[drifters[4]][0][iter, 3])
    return line1, line2, line3, line4, line5,Dot1,Dot2, Dot3, Dot4, Dot5

frs = int(time_stamp.shape[0]/100)
anim =   animation.FuncAnimation(fig, animate,fargs=(line1, line2, line3, line4, line5,Dot1,Dot2, Dot3, Dot4, Dot5,),frames=frs, interval=5, blit=False, repeat=False)
anim.save('./drifter_trajectories_'+str(date)+'.gif', writer='pillow', fps=100,progress_callback=lambda i, n: print(i),)
plt.show()



        

