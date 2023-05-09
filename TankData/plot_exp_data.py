import matplotlib.pyplot as plt
import numpy as np
from interpolate_vectorfield import create_vec_field
import os
from scipy.interpolate import griddata

### plot the flow vector field ###
def get_flow(x,y,t):
    # function that interpolates grid_U and grid_V at location (x,y)
    # x,y: 1D array
    # grid_U, grid_V: 2D array
    # Vx, Vy: 1D array
    Vx = griddata((X.flatten(),Y.flatten()),grid_U.flatten(),(x,y),method='linear')
    Vy = griddata((X.flatten(),Y.flatten()),grid_V.flatten(),(x,y),method='linear')
    return Vx, Vy

date='02_26_2023_13_05_55'
resolution = 0.05
# If the interpolated vector field has already been created, load it
# Otherwise, create it using interpolate_vectorfield.py
if os.path.exists('data/velocity_field_'+str(date)+'_res_'+str(resolution)+'concatenate.npy'):
    print('loading vector field')
    X, Y, grid_U, grid_V = np.load('data/velocity_field_'+str(date)+'_res_'+str(resolution)+'concatenate.npy')
else:
    print('creating vector field')
    create_vec_field(date, resolution)
    X, Y, grid_U, grid_V = np.load('data/velocity_field_'+str(date)+'_res_'+str(resolution)+'concatenate.npy')

xmin = -3
xmax = 3
ymin = -3.5
ymax = 0.5
resolution_p = 0.2
xx_p = np.arange(xmin, xmax, resolution_p)
yy_p = np.arange(ymin, ymax, resolution_p)
Xp, Yp = np.meshgrid(xx_p, yy_p)
[Vx_p, Vy_p] = get_flow(Xp,Yp,0)

fig, ax = plt.subplots()
flow_size = np.sqrt(Vx_p**2+Vy_p**2)
qq=ax.quiver(xx_p,yy_p,Vx_p,Vy_p,flow_size,cmap=plt.cm.jet)
plt.colorbar(qq, cmap = plt.cm.jet,label='flow speed w[m/s]', shrink=1)
ax.set_xlabel('$x[m]$')
ax.set_ylabel('$y[m]$')
ax.set_aspect('equal')

filename = "boat_data/filename_2023_02_27-06_12_24_PM.txt"
f = open(filename,"r")

traj_1 = [[],[]]
traj_2 = [[],[]]
dt_list = []
start_index = 30
end_index = -60

for line in f.readlines():
    l = line.strip().split(", ")
    if l[0] == '2':
        dt_list.append(float(l[1]))
        traj_1[0].append(float(l[2]))
        traj_1[1].append(float(l[3]))
    if l[0] == '3':
        traj_2[0].append(float(l[2]))
        traj_2[1].append(float(l[3]))

print(len(traj_1[0]),len(traj_2[0]))
print(len(traj_1[0][start_index:end_index]),len(traj_2[0][start_index:end_index]))

docking_x = (traj_1[0][end_index]+traj_2[0][end_index])/2
docking_y = (traj_1[1][end_index]+traj_2[1][end_index])/2

dt = dt_list[start_index:end_index]
t = np.cumsum(dt)
print(t)

# ax.scatter(traj_1[0][start_index:end_index],traj_1[1][start_index:end_index],color='red',marker='.')
# ax.scatter(traj_2[0][start_index:end_index],traj_2[1][start_index:end_index],color='blue',marker='.')
# ax.scatter(traj_1[0][start_index],traj_1[1][start_index],color='black',marker='o',label='start of agent 1')
# ax.scatter(traj_2[0][start_index],traj_2[1][start_index],color='black',marker='^',label='start of agent 2')
# ax.scatter(docking_x,docking_y,color='black',marker='x',label='docking point')

# ax.set_xlim(-2,3)
# ax.set_ylim(-3.5,0.5)
# ax.set_title('Actual trajectories of two agents')
# ax.legend(loc = 'lower right')
# plt.savefig('results_tank/actual_trajectories_'+str(date)+'.png',dpi=300,bbox_inches='tight')
# plt.show()






