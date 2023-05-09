import matplotlib.pyplot as plt
import numpy as np
from heapq import heappush, heappop
from skimage import measure
from generate_bickley import bickley_u_v
import math
import matplotlib.animation as animation
import os

### level set propagation in bickley flow
### two agents docking

####### parameters setting #######
## simulation time setting
dt = 0.01
T = 3
## control input
F = np.array([2,1])
## start and goal
n_agents = 2
x_ini = np.array([[2,0.5],[4,-2]])
## gradient_threshold
g_threshold = 50
## threshold to regard the value of phi as 0
zero_threshold = 0
## Whether you plot back-tracked path
TF_back_track=True
## Whether you take the four closest points for back tracking
TF_four_closest=True
## integration method. 1: Fractional Step Method, 2: Euler, 3:RK You should basically choose 2
integral_method = 2
## size of time step for backtracking
size_backtrack = 15
## plot time step
plot_time_step = 5
## time step to reinitialize
freq_reinit = int(0.5/dt)
REACH = False
SHOW_PLOT = True
SHOW_ANIM = False
SAVE_GIF = False
## grid
resolution = 0.2
xmin = 0
xmax = 10
ymin = -3
ymax = 3
xx = np.arange(xmin, xmax, resolution)
yy = np.arange(ymin, ymax, resolution)
X, Y = np.meshgrid(xx, yy)

## meshgrid for plotting
resolution_p = 0.5
xx_p = np.arange(xmin, xmax, resolution_p)
yy_p = np.arange(ymin, ymax, resolution_p)
Xp, Yp = np.meshgrid(xx_p, yy_p)

# phi at every time step
# data is defined as phi(yMinMax, xMinMax)
phi_data = np.zeros((int((ymax-ymin)/resolution), int((xmax-xmin)/resolution), int(T/dt)+1, n_agents))
phi_0 = np.sqrt((X-x_ini[0,0])**2 + (Y-x_ini[0,1])**2)
phi_0_b = np.sqrt((X-x_ini[1,0])**2 + (Y-x_ini[1,1])**2)

phi_data[:,:,0,0] = phi_0
phi_data[:,:,0,1] = phi_0_b

def gradient_upwind(phi,resolution):

    Dx_p = (phi[1:-1,2:] - phi[1:-1,1:-1])/resolution #D_plus_x
    Dx_m = (phi[1:-1,1:-1] - phi[1:-1,:-2])/resolution
    Dx_1 = (Dx_m+Dx_p)/2 
    Dx = np.zeros((int((ymax-ymin)/resolution),int((xmax-xmin)/resolution)))
    Dx[1:-1,1:-1] = Dx_1

    # handling boudary
    # Dx[1:-1,0] = (phi[1:-1,1] - phi[1:-1,0])/resolution
    Dx[:,0] = (phi[:,1] - phi[:,0])/resolution
    # Dx[1:-1,1] = (phi[1:-1,2] - phi[1:-1,1])/resolution
    # Dx[1:-1,-1] = (phi[1:-1,-1] - phi[1:-1,-2])/resolution
    Dx[:,-1] = (phi[:,-1] - phi[:,-2])/resolution

    Dy_p = (phi[2:,1:-1] - phi[1:-1,1:-1])/resolution #D_plus_x
    Dy_m = (phi[1:-1,1:-1] - phi[:-2,1:-1])/resolution

    Dy_1 = (Dy_m+Dy_p)/2
    Dy = np.zeros((int((ymax-ymin)/resolution),int((xmax-xmin)/resolution)))
    Dy[1:-1,1:-1] = Dy_1

    # handling boudary
    # Dy[0,1:-1] = (phi[1,1:-1] - phi[0,1:-1])/resolution
    # Dy[1,1:-1] = (phi[2,1:-1] - phi[1,1:-1])/resolution
    Dy[0,:] = (phi[1,:] - phi[0,:])/resolution
    # Dy[-1,1:-1] = (phi[-1,1:-1] - phi[-2,1:-1])/resolution
    Dy[-1,:] = (phi[-1,:] - phi[-2,:])/resolution

    return Dx, Dy

def reinitialize(phi):
    # getting the index of points on zero-level set
    # phi_reinitialize = phi

    # ## np.whereを使う方法
    # zero_level = np.where(np.absolute(phi)<=0.01)
    # if len(zero_level)==0:
    #     print('Cannot find 0 contour for reinitialization')
    # else:
    #     zero_level_y = zero_level[0]
    #     zero_level_x = zero_level[1]
    #     n_zero_level = len(zero_level_x)
    #     # print(n_zero_level, ':number of zero-level points by np.where')
    #     print(zero_level_x.shape)
    #     points = np.array([zero_level_x,zero_level_y]).T
    #     # print(points.shape,'points.shape')
    #     hull = ConvexHull(points)
    #     zero_level_x_hull = points[hull.vertices,0]
    #     zero_level_y_hull = points[hull.vertices,1]
    #     n_zero_level = zero_level_y_hull.shape[0]
    #     # print(zero_level_x_hull.shape,'whats')

    #     print(n_zero_level, ':number of zero-level points by convex hull')

    #     plt.figure()
    #     plt.scatter(zero_level_x,zero_level_y,c='k')
    #     plt.scatter(zero_level_x_hull,zero_level_y_hull,c='b')
    #     # plt.show()

    #     # store the min of square of distance
    #     big_distance = 100000000
    #     phi_sq_min = np.ones((int((ymax-ymin)/resolution),int((xmax-xmin)/resolution))) * big_distance
    #     for l in range(n_zero_level):
    #         # position of points on zero-level set
    #         ### if you find the 0-level set with np.where
    #         # point_x = xmin + zero_level_x[l]*resolution
    #         # point_y = ymin + zero_level_y[l]*resolution
    #         ### if you find the 0-level set with np.where and convex hull
    #         point_x = xmin + zero_level_x_hull[l]*resolution
    #         point_y = ymin + zero_level_y_hull[l]*resolution
    #         ### if you find the 0-level set with measure.find_contours
    #         # point_x = xmin + zero_level_index[l,1]*resolution
    #         # point_y = ymin + zero_level_index[l,0]*resolution
    #         phi_sq_tmp = (X-point_x)**2 + (Y-point_y)**2
    #         # True if tmp<min, False otherwise
    #         tmp_min_compare = (phi_sq_tmp < phi_sq_min).astype(np.int32)
    #         phi_sq_min = (1-tmp_min_compare)*phi_sq_min + tmp_min_compare*phi_sq_tmp
    #     # phi = np.sqrt(phi_sq_min)
    #     phi = np.sqrt(phi_sq_min)*np.sign(phi)

    ## measure.find_contoursを使う方法
    zero_level_index = measure.find_contours(phi,level=0)
    if len(zero_level_index)==0:
        print('Cannot find 0 contour for reinitialization')
    else:
        zero_level_index = zero_level_index[0][:-1]
        n_zero_level = zero_level_index.shape[0]
        zero_level_x,zero_level_y = zero_level_index[:,1], zero_level_index[:,0]

        # print(n_zero_level, ':number of zero-level points', zero_level,':position of zero-level points')
        
        # store the min of square of distance
        big_distance = 100000000
        phi_sq_min = np.ones((int((ymax-ymin)/resolution),int((xmax-xmin)/resolution))) * big_distance
        for l in range(n_zero_level):
            # position of points on zero-level set
            ### if you find the 0-level set with np.where
            # point_x = xmin + zero_level_x[l]*resolution
            # point_y = ymin + zero_level_y[l]*resolution
            ### if you find the 0-level set with np.where and convex hull
            # point_x = xmin + zero_level_x_hull[l]*resolution
            # point_y = ymin + zero_level_y_hull[l]*resolution
            ### if you find the 0-level set with measure.find_contours
            point_x = xmin + zero_level_index[l,1]*resolution
            point_y = ymin + zero_level_index[l,0]*resolution
            phi_sq_tmp = (X-point_x)**2 + (Y-point_y)**2
            # True if tmp<min, False otherwise
            tmp_min_compare = (phi_sq_tmp < phi_sq_min).astype(np.int32)
            phi_sq_min = (1-tmp_min_compare)*phi_sq_min + tmp_min_compare*phi_sq_tmp
        # phi = np.sqrt(phi_sq_min)
        phi = np.sqrt(phi_sq_min)*np.sign(phi)

    gx, gy = gradient_upwind(phi,resolution)
    dphi_norm = np.sqrt(gx**2+gy**2)

    return phi, dphi_norm

def col_cycler(cols):
    count = 0
    while True:
        yield cols[count]
        count = (count + 1)%len(cols)

# col_iter = col_cycler(['c','m', 'y','k'])
col_iter = col_cycler(['red','orange','green','blue','purple'])
# col_iter = col_cycler(['orange', 'green', 'purple'])

def slope_normal(xf, x1, x2):
    a = x1[0]-x2[0]
    b = x1[1]-x2[1]
    slope = np.array([a,b])
    normal = np.array([b,-a])
    normal = normal/np.sqrt(a**2+b**2) #making a unit vector
    # shortest distance from point xf to the line segment x1x2
    d = np.abs(np.dot(normal, xf-x1))

    return slope, normal, d

def animate(iter, Q, dt, Xp, Yp, X, Y, phi_data, col_iter):
    ax.clear()
    t = iter*dt
    Up, Vp = bickley_u_v(t, Xp, Yp)
    flow_size = np.sqrt(Up**2+Vp**2)
    # Q.set_UVC(Up,Vp,flow_size)
    Q = ax.quiver(Xp, Yp, Up, Vp, flow_size,cmap=plt.cm.jet)
    ax.scatter(x_ini[0,0],x_ini[0,1],c='r',label='start of agent 1')
    ax.scatter(x_ini[1,0],x_ini[1,1],c='b',label='start of agent 2')
    ax.scatter(x_goal[0],x_goal[1],c='g',label='docking point')
    ax.set_title('t={:.1f}'.format(t))
    ax.contour(X, Y, phi_data[:,:,iter+1,0],0,colors='r')
    ax.contour(X, Y, phi_data[:,:,iter+1,1],0,colors='b')
    ax.legend()

    return Q

if SHOW_ANIM:
    fig, ax = plt.subplots()
    Up, Vp = bickley_u_v(0,Xp,Yp)
    flow_size = np.sqrt(Up**2+Vp**2)
    Q = ax.quiver(Xp, Yp, Up, Vp, flow_size,cmap=plt.cm.jet)
    plt.colorbar(Q, cmap = plt.cm.jet)
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-3.3, 3.3)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal')
    # ax.set_title('t=0.0')

for i in range(int(T/dt)):
    # print('i=',i)
    # Vx, Vy = bickley_u_v(i*dt+dt/2, X, Y)
    Vx, Vy = bickley_u_v(i*dt, X, Y)
    for agent in range(n_agents):
        phi = phi_data[:,:,i,agent]

        gx, gy = gradient_upwind(phi,resolution)
        dphi_norm = np.sqrt(gx**2 + gy**2)
        #dphi_data: norm of gradient of phi at time step i
        # dphi_data[:,:,i] = dphi_norm

        # if (dphi_norm > g_threshold).any():
        if i%freq_reinit==0 and i!=0:
                print('-----Reinitialized at iteration',i)
                phi, dphi_norm = reinitialize(phi)

        if integral_method == 1:
            ### fractional step method ###
            phi_1 = phi - dt/2*F[agent]*dphi_norm

            gx_1, gy_1 = gradient_upwind(phi_1,resolution)
            dot_product = Vx*gx_1+Vy*gy_1
            phi_2 = phi_1 - dt*dot_product

            gx_2, gy_2 = gradient_upwind(phi_2,resolution)
            dphi_2_norm = np.sqrt(gx_2**2+gy_2**2)
            phi_next = phi_2 - dt/2*F[agent]*dphi_2_norm
        elif integral_method == 2:
            ### Euler Method ###
            phi_1 = -dt*F[agent]*dphi_norm
            dot_product = Vx*gx + Vy*gy
            phi_2 = -dt*dot_product
            phi_next = phi + phi_1 + phi_2
        elif integral_method == 3:
            ### Runge-Kutta Method ###
            k1 = -F[agent]*dphi_norm-(Vx*gx + Vy*gy)
            phi_0 = phi
            phi = phi + dt*k1/2

            gx, gy = gradient_upwind(phi,resolution)
            dphi_norm = np.sqrt(gx**2 + gy**2)
            k2 = -F*dphi_norm-(Vx*gx + Vy*gy)
            phi = phi + dt*k2/2

            gx, gy = gradient_upwind(phi,resolution)
            dphi_norm = np.sqrt(gx**2 + gy**2)
            k3 = -F[agent]*dphi_norm-(Vx*gx + Vy*gy)
            phi = phi + dt*k3

            gx, gy = gradient_upwind(phi,resolution)
            dphi_norm = np.sqrt(gx**2 + gy**2)
            k4 = -F[agent]*dphi_norm-(Vx*gx + Vy*gy)

            phi_next = phi_0 + 1/6*(k1+2*k2+2*k3+k4)*dt
        else: print('error: invalid integer for the integral method!!')

        phi_data[:,:,i+1,agent] = phi_next
        
        # if i%plot_time_step == 0:
        # # if (T/dt)%100 == 0:
        #     if i!=0:
        #         print('-----zero level set plotting------ i=',i)
        #         ax.contour(X,Y,phi_next,0,colors=next(col_iter))

    # If there's a point whose level is less than zero_threshold in both of the agents,
    # then the goal is reached
    # print(phi_data[:,:,i+1,0] <= zero_threshold and phi_data[:,:,i+1,1] <= zero_threshold)
    (y_dock_index, x_dock_index) = np.where((phi_data[:,:,i+1,0] <= zero_threshold) * (phi_data[:,:,i+1,1] <= zero_threshold))
    if len(x_dock_index) > 0:
        print('two robots docking at i=',i)
        print(len(x_dock_index),'number of negative points')
        t_final = i
        REACH = True
        break

x_goal_index = np.array([int(np.average(x_dock_index)), int(np.average(y_dock_index))]) 
x_goal = np.array([x_goal_index[0]*resolution+xmin, x_goal_index[1]*resolution+ymin])
x_ini_index = np.array([[int((x_ini[0,0]-xmin)/resolution), int((x_ini[0,1]-ymin)/resolution)],[int((x_ini[1,0]-xmin)/resolution), int((x_ini[1,1]-ymin)/resolution)]])
path = np.array([x_goal_index])
path_b = np.array([x_goal_index])
x_tmp = x_goal_index

# backtracking at time step = size_backtrack
# taking piecewise linear contour from multiple closest points
if TF_back_track and TF_four_closest and REACH:
    print('########back propagation#########')
    for agent in range(n_agents):
        print('---------agent=',agent,'---------')
        x_tmp = x_goal_index
        k = t_final
        l = 0
        # while k!=0 and len(measure.find_contours(phi_data[:,:,k+1],level=0)) != 0:
        while k!=0:
            print('k=',k)
            if l%size_backtrack==0:
                print('l=',l)
                # zero_level_index_np_where = np.where(np.absolute(phi_data[:,:,k+1,agent])<=zero_threshold)
                zero_level_index_np_where = np.where(np.absolute(phi_data[:,:,k+1,agent])<=0.1)
                
                if len(zero_level_index_np_where)==0: 
                    print(zero_level_index_np_where)
                    break
                current_phi = phi_data[:,:,k+1,agent]
                zero_level_index_y,zero_level_index_x = zero_level_index_np_where
                n_zero_level = zero_level_index_y.shape[0]
                # print('# of 0-level points',n_zero_level,zero_level_index_x*resolution+xmin,zero_level_index_y*resolution+ymin)
                if n_zero_level<4:
                    break
                d_list = []
                for j in range(zero_level_index_y.shape[0]):
                    d_sq = (zero_level_index_x[j]-x_tmp[0])**2 + (zero_level_index_y[j]-x_tmp[1])**2
                    heappush(d_list,[d_sq, (zero_level_index_x[j],zero_level_index_y[j])])
                # print(d_list)
                d_sq_min_1, x_close_1 = heappop(d_list)

                while d_sq_min_1<0.1 and len(d_list)>=4: 
                    # print('taking the next closest point at k=',k)
                    d_sq_min_1, x_close_1 = heappop(d_list)
                d_sq_min_2, x_close_2 = heappop(d_list)
                while np.array_equal(x_close_1,x_close_2) and len(d_list)>=3:
                    d_sq_min_2, x_close_2 = heappop(d_list)
                d_sq_min_3, x_close_3 = heappop(d_list)
                while np.array_equal(x_close_3,x_close_2) and len(d_list)>=2:
                    d_sq_min_3, x_close_3 = heappop(d_list)
                d_sq_min_4, x_close_4 = heappop(d_list)
                while np.array_equal(x_close_3,x_close_4) and len(d_list)>=1:
                    d_sq_min_4, x_close_4 = heappop(d_list)
                print('four closest points',x_close_1,x_close_2, x_close_3, x_close_4)

                closest_x = np.array([x_close_1[0], x_close_2[0], x_close_3[0], x_close_4[0]])
                closest_y = np.array([x_close_1[1], x_close_2[1], x_close_3[1], x_close_4[1]])
                print('closest_x',closest_x)
                print('closest_y',closest_y)
                if np.all(closest_x == closest_x[0]):
                    print('vertical contour')
                    d = x_tmp[0]-closest_x[0]
                    normal = np.array([-1,0])
                    x_tmp_sub = x_tmp + np.array([d,0])
                    if np.linalg.norm(x_tmp_sub-x_ini_index[agent,:]) > np.linalg.norm(x_tmp-x_ini_index[agent,:]):
                        x_tmp_sub = x_tmp + np.array([-d,0])
                        normal = -normal
                elif np.all(closest_y == closest_y[0]):
                    print('horizontal contour')
                    d = x_tmp[1]-closest_y[0]
                    normal = np.array([0,-1])
                    x_tmp_sub = x_tmp + np.array([0,d])
                    if np.linalg.norm(x_tmp_sub-x_ini_index[agent,:]) > np.linalg.norm(x_tmp-x_ini_index[agent,:]):
                        x_tmp_sub = x_tmp + np.array([0,-d])
                        normal = -normal
                else:
                    lstsq = np.linalg.lstsq(np.vstack([closest_x, np.ones(len(closest_x))]).T, closest_y, rcond=None)[0]
                    print('lstsq',lstsq)
                    slope = lstsq[0]
                    intercept = lstsq[1]
                    d = abs(slope*x_tmp[0]-x_tmp[1]+intercept)/math.sqrt(slope**2+1)
                    normal = np.array([slope, -1])/math.sqrt(slope**2+1)
                    print('normal',normal)
                    x_tmp_sub = x_tmp + d*normal
                    if np.linalg.norm(x_tmp_sub-x_ini_index[agent,:]) > np.linalg.norm(x_tmp-x_ini_index[agent,:]):
                        normal = -normal
                        x_tmp_sub = x_tmp + d*normal
                
                # slope, normal, d = slope_normal(x_tmp,x_close_1,x_close_2)
                # x_tmp_sub = x_tmp + d*normal
                # if np.linalg.norm(x_tmp_sub-x_ini_index) > np.linalg.norm(x_tmp-x_ini_index):
                #     normal = -normal
                #     x_tmp_sub = x_tmp + d*normal
                    
                x_tmp_sub_x = x_tmp_sub[0]
                x_tmp_sub_y = x_tmp_sub[1]
                x = xmin+x_tmp_sub_x*resolution
                y = ymin+x_tmp_sub_y*resolution
                u, v = bickley_u_v(k*dt, x, y)
                x_new = x_tmp_sub-dt*([u,v]+F[agent]*normal)*size_backtrack
                print('x_new',x_new)
                x_tmp = x_new
                if agent==0:
                    path=np.append(path, np.array([x_new]), axis=0)
                else:
                    path_b=np.append(path_b, np.array([x_new]), axis=0)
                # print('x_new',x_new)
        
            k -= 1
            l += 1
    path = np.append(path, np.array([x_ini_index[0,:]]), axis=0)
    path_b = np.append(path_b, np.array([x_ini_index[1,:]]), axis=0)
    print('number of points for path planning',path.shape[0])
    print(path)
    print(path_b)
    
dir_name = 'dock_xini'+str(x_ini[0])+str(x_ini[1])
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

if REACH==True and SHOW_PLOT==True:
    for j in range(int(t_final//plot_time_step)+1):
        t=plot_time_step*j*dt
        if t!=0:
            plot_time_index = plot_time_step*j
            if j==int(t_final//plot_time_step):
                t=t_final*dt
                plot_time_index = t_final
            print('-----zero level set plotting------ t=',t)
            # plt.figure('t='+str(t))
            fig, ax = plt.subplots()
            Up, Vp = bickley_u_v(t, Xp, Yp)
            flow_size = np.sqrt(Up**2+Vp**2)
            Q = ax.quiver(Xp, Yp, Up, Vp, flow_size,cmap=plt.cm.jet)
            plt.colorbar(Q, cmap = plt.cm.jet,label='flow speed w[m/s]')
            ax.scatter(x_ini[0,0],x_ini[0,1],c='r',label='start of agent 1')
            ax.scatter(x_ini[1,0],x_ini[1,1],c='b',label='start of agent 2')
            if j==int(t_final//plot_time_step):
                ax.scatter(x_goal[0],x_goal[1],c='g',label='docking point')
                if TF_back_track:
                    ax.plot(xmin+path[:,0]*resolution,ymin+path[:,1]*resolution,marker='.',markersize=3,c='r')
                    ax.plot(xmin+path_b[:,0]*resolution,ymin+path_b[:,1]*resolution,marker='.',markersize=3,c='b')

            # ax.scatter(x_goal[0],x_goal[1],c='g',label='docking point')
            ax.set_title('t={:.2f}'.format(t))
            ax.contour(X,Y,phi_data[:,:,plot_time_index,0],0,colors='r')
            ax.contour(X,Y,phi_data[:,:,plot_time_index,1],0,colors='b')
            ax.legend()
            plt.savefig(os.path.join(dir_name,'dock_xini'+str(x_ini[0])+str(x_ini[1])+'t={:.2f}.png'.format(t)),dpi=300,bbox_inches='tight')
            # plt.savefig('./bickley_jet/dock_xini'+str(x_ini[0])+str(x_ini[1])+'t={:.2f}.png'.format(t))
            plt.show()


if TF_back_track==True and TF_four_closest==False and REACH==True:
    print('########back propagation#########')
    k = t_final
    l = 0
    # while k!=0 and len(measure.find_contours(phi_data[:,:,k+1],level=0)) != 0:
    while k!=0:
        if l%size_backtrack==0:
            zero_level_index_np_where = np.where(np.absolute(phi_data[:,:,k+1])<=zero_threshold)
            if len(zero_level_index_np_where)==0: 
                print(zero_level_index_np_where)
                break
            current_phi = phi_data[:,:,k+1]
            zero_level_index_y,zero_level_index_x = zero_level_index_np_where
            n_zero_level = zero_level_index_y.shape[0]
            # print('# of 0-level points',n_zero_level,zero_level_index_x*resolution+xmin,zero_level_index_y*resolution+ymin)
            if n_zero_level <2:
                break
            d_list = []
            for j in range(zero_level_index_y.shape[0]):
                d_sq = (zero_level_index_x[j]-x_tmp[0])**2 + (zero_level_index_y[j]-x_tmp[1])**2
                heappush(d_list,[d_sq, (zero_level_index_x[j],zero_level_index_y[j])])
            # print(d_list)
            d_sq_min_1, x_close_1 = heappop(d_list)
            while d_sq_min_1<0.1 and len(d_list)>1: 
                # print('taking the next closest point at k=',k)
                d_sq_min_1, x_close_1 = heappop(d_list)
            d_sq_min_2, x_close_2 = heappop(d_list)
            while np.array_equal(x_close_1,x_close_2) and len(d_list)!=0:
                d_sq_min_2, x_close_2 = heappop(d_list)
            print('two closest points',x_close_1,x_close_2)
            slope, normal, d = slope_normal(x_tmp,x_close_1,x_close_2)
            x_tmp_sub = x_tmp + d*normal
            if np.linalg.norm(x_tmp_sub-x_ini_index) > np.linalg.norm(x_tmp-x_ini_index):
                normal = -normal
                x_tmp_sub = x_tmp + d*normal
            x_tmp_sub_x = x_tmp_sub[0]
            x_tmp_sub_y = x_tmp_sub[1]
            x = xmin+x_tmp_sub_x*resolution
            y = ymin+x_tmp_sub_y*resolution
            x_new = x_tmp_sub-dt*([bickley_u_v(k*dt, x, y)]+F*normal)*size_backtrack
            path=np.append(path, np.array([x_new]), axis=0)
            x_tmp = x_new
            print('x_new',x_new)
        k -= 1
        l += 1
    path = np.append(path, np.array([x_ini_index]), axis=0)
    print('number of points for path planning',path.shape[0])
    ax.plot(xmin+path[:,0]*resolution,ymin+path[:,1]*resolution,marker='.',markersize=3,c='k')

if REACH:
    frs = t_final
else:
    frs = int(T/dt)

if SHOW_ANIM:
    anim = animation.FuncAnimation(fig, animate, fargs=(Q, dt, Xp, Yp, X, Y, phi_data, col_iter),frames=frs, interval=50, blit=False)
if SAVE_GIF:
    anim.save('bickley.gif', writer='pillow', fps=10, progress_callback=lambda i, n: print(i),)

# パラメーターをcsvに保存
# with open('./bickley_jet/dock_xini'+str(x_ini[0])+str(x_ini[1])+'.csv', mode='w') as f:
with open(os.path.join(dir_name,'dock_xini'+str(x_ini[0])+str(x_ini[1])+'.csv'), mode='w') as f:
    f.write('F,'+str(F)+'\n')
    f.write('size_backtrack,'+str(size_backtrack)+'\n')
    f.write('dt,'+str(dt)+'\n')
    f.write('T,'+str(T)+'\n')
    f.write('plot_time_step,'+str(plot_time_step)+'\n')
    f.write('zero_threshold,'+str(zero_threshold)+'\n')
    f.write('size_backtrack,'+str(size_backtrack)+'\n')
    f.write('x1_ini,'+str(x_ini[0])+'\n')
    f.write('x2_ini,'+str(x_ini[1])+'\n')
    f.write('x_goal,'+str(x_goal)+'\n')
    f.write('dock time,'+str(t_final*dt)+'\n')


