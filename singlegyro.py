import matplotlib.pyplot as plt
import numpy as np
import utils
from heapq import heappush, heappop
from skimage import measure
from scipy.spatial import ConvexHull
import math

####### parameters setting #######
## simulation time setting
dt = 0.01
T = 20
## control input
F = 1.0
## start and goal
x_ini = np.array([15,5])
x_goal = np.array([10,17])
## gradient_threshold
g_threshold = 50
## threshold to regard the value of phi as 0
zero_threshold = 0.1
## Whether you plot back-tracked path
TF_back_track=True
## Whether you take the four closest points for back tracking
TF_four_closest=True
## integration method. 1: Fractional Step Method, 2: Euler, 3:RK
integral_method = 2
## size of time step for backtracking
size_backtrack = 25
## plot time step
plot_time_step = 20
## time step to reinitialize
freq_reinit = int(0.5/dt)


## grid
class grid():
    def __init__(self):
        self.resolution = 0.2
        self.xmin = 0
        self.xmax = 25
        self.ymin =0
        self.ymax = 25

self = grid()
resolution = self.resolution
xmin = self.xmin
xmax = self.xmax
ymin = self.ymin
ymax = self.ymax

xx = np.arange(xmin, xmax, resolution)
yy = np.arange(ymin, ymax, resolution)
X, Y = np.meshgrid(xx, yy)

resolution_p = 1
xx_p = np.arange(xmin, xmax, resolution_p)
yy_p = np.arange(ymin, ymax, resolution_p)
Xp, Yp = np.meshgrid(xx_p, yy_p)

# phi at every time step
# data is defined as phi(yMinMax, xMinMax)
phi_data = np.zeros((int((ymax-ymin)/resolution), int((xmax-xmin)/resolution), int(T/dt)+1))
# dphi_data = np.zeros((int((ymax-ymin)/resolution), int((xmax-xmin)/resolution), int(T/dt)+1))
# dphi_norm_list = np.zeros((int((ymax-ymin)/resolution), int((xmax-xmin)/resolution), int(T/dt)+1))
# F_dot_g_list = np.zeros((int((ymax-ymin)/resolution), int((xmax-xmin)/resolution), int(T/dt)+1))
# dphi_2_norm_list = np.zeros((int((ymax-ymin)/resolution), int((xmax-xmin)/resolution), int(T/dt)+1))
# phi_0 = np.sqrt(X**2 + Y**2) #initial position is (0,0)

phi_0 = np.sqrt((X-x_ini[0])**2 + (Y-x_ini[1])**2)
# phi_0[0,:] = 100
# phi_0[-1,:] = 100
# phi_0[:,0] = 100
# phi_0[:,-1] = 100
# structure phi_0(Y,X)
phi_data[:,:,0] = phi_0
# phi_0[0,0], phi_0[0,-1], phi_0[-1,0], phi_0[-1,-1] = 100, 100, 100, 100
# print(X)
# print(Y)
# print(X.shape)
# print(phi_0[-20:,-20:])
# print(np.where(phi_0==100))
# exit()

####### double gyros #######
A        =     1
omega    = (2*np.pi)/10
eps = 0
s = 25
I = 1
sigma = np.sqrt(2*I)

def f(x,t):
    a_t = eps*np.sin(omega*t)
    b_t = 1 - 2*eps*np.sin(omega*t)
    return a_t*(x**2) + b_t*x

def dfdx(x,t):
    a_t = eps*np.sin(omega*t)
    b_t = 1 - 2*eps*np.sin(omega*t)
    return 2*a_t*x + b_t

def get_flow(X,Y,t):
    Vx = -np.pi*A*np.sin(np.pi*f(X,t)/s)*np.cos(np.pi*Y/s)
    Vy = np.pi*A*np.cos(np.pi*f(X,t)/s)*np.sin(np.pi*Y/s)*dfdx(X,t)
    return [Vx,Vy]

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

##### plotting double gyros #####

[Vx, Vy] = get_flow(X,Y,0)
# color = 0.01 * np.sqrt(np.hypot(Vx, Vy))
fig, ax = plt.subplots()
# stream = ax.streamplot(xx,yy,Vx,Vy, color=color, density=2, cmap=plt.cm.inferno,arrowstyle='->',arrowsize=1)
[Vx_p, Vy_p] = get_flow(Xp,Yp,0)
flow_size = np.sqrt(Vx_p**2+Vy_p**2)
qq=ax.quiver(xx_p,yy_p,Vx_p,Vy_p,flow_size,cmap=plt.cm.jet)
plt.colorbar(qq, cmap = plt.cm.jet)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
# ax.set_xlim(5, 15)
# ax.set_ylim(5, 15)
ax.set_aspect('equal')
# plt.show()

for i in range(int(T/dt)):
    # print('i=',i)
    # [Vx, Vy] = get_flow(X,Y,i*dt+dt/2)
    [Vx, Vy] = get_flow(X,Y,0)
    phi = phi_data[:,:,i]
    # if i%50==0:
    #     rang = 4
    #     phi[:int(1/resolution/rang*i/50),:]=phi_data[:int(1/resolution/rang*i/50),:,0]
    #     phi[:,:int(1/resolution/rang*i/50)]=phi_data[:,:int(1/resolution/rang*i/50),0]
    #     phi[-1-int(1/resolution/rang*i/50):,:] = phi_data[-1-int(1/resolution/rang*i/50):,:,0]
    #     phi[:,-1-int(1/resolution/rang*i/50):] = phi_data[:,-1-int(1/resolution/rang*i/50):,0]

    gx, gy = gradient_upwind(phi,resolution)
    dphi_norm = np.sqrt(gx**2 + gy**2)
    #dphi_data: norm of gradient of phi at time step i
    # dphi_data[:,:,i] = dphi_norm

    # if (dphi_norm > g_threshold).any():
    if i%freq_reinit==0 and i!=0:
    # if i>=100 and i%5==0:
            print('-----Reinitialized at iteration',i)
            # print('-----zero level set plotting------ i=',i)
            # ax.contour(X,Y,phi_next,0,colors=next(col_iter),label='i=%.2f'%(i))
            phi, dphi_norm = reinitialize(phi)
            # break

    if integral_method == 1:
        ### fractional step method ###
        phi_1 = phi - dt/2*F*dphi_norm

        gx_1, gy_1 = gradient_upwind(phi_1,resolution)
        dot_product = Vx*gx_1+Vy*gy_1
        phi_2 = phi_1 - dt*dot_product

        gx_2, gy_2 = gradient_upwind(phi_2,resolution)
        dphi_2_norm = np.sqrt(gx_2**2+gy_2**2)
        phi_next = phi_2 - dt/2*F*dphi_2_norm
    elif integral_method == 2:
        ### Euler Method ###
        phi_1 = -dt*F*dphi_norm
        dot_product = Vx*gx + Vy*gy
        phi_2 = -dt*dot_product
        phi_next = phi + phi_1 + phi_2
    elif integral_method == 3:
        ### Runge-Kutta Method ###
        k1 = -F*dphi_norm-(Vx*gx + Vy*gy)
        phi_0 = phi
        phi = phi + dt*k1/2

        gx, gy = gradient_upwind(phi,resolution)
        dphi_norm = np.sqrt(gx**2 + gy**2)
        k2 = -F*dphi_norm-(Vx*gx + Vy*gy)
        phi = phi + dt*k2/2

        gx, gy = gradient_upwind(phi,resolution)
        dphi_norm = np.sqrt(gx**2 + gy**2)
        k3 = -F*dphi_norm-(Vx*gx + Vy*gy)
        phi = phi + dt*k3

        gx, gy = gradient_upwind(phi,resolution)
        dphi_norm = np.sqrt(gx**2 + gy**2)
        k4 = -F*dphi_norm-(Vx*gx + Vy*gy)

        phi_next = phi_0 + 1/6*(k1+2*k2+2*k3+k4)*dt
    else: print('error: invalid integer for the integral method!!')

    phi_data[:,:,i+1] = phi_next
    
    if i%plot_time_step == 0:
    # if (T/dt)%100 == 0:
        if i!=0:
            print('-----zero level set plotting------ i=',i)
            # ax.contour(X,Y,phi_next_abs,0,colors=next(col_iter))
            ax.contour(X,Y,phi_next,0,colors=next(col_iter))
            # plt.figure(2)
            # plt.imshow(phi_next, cmap='hot', interpolation='nearest')
            # plt.colorbar()
            # plt.show()

    # finding the two closest grids on 0-level contour to x_goal
    if phi_next[int((x_goal[1]-ymin)/resolution),int((x_goal[0]-xmin)/resolution)] <= zero_threshold:
        print('0-level set reached the goal at i=',i)
        t_final = i
        break

x_goal_index = np.array([int((x_goal[0]-xmin)/resolution), int((x_goal[1]-ymin)/resolution)])
x_ini_index = np.array([int((x_ini[0]-xmin)/resolution), int((x_ini[1]-ymin)/resolution)])
path = np.array([x_goal_index])
x_tmp = x_goal_index


# backtracking at time step = size_backtrack
# taking piecewise linear contour from multiple closest points
if TF_back_track and TF_four_closest:
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
            if n_zero_level <4:
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
            # print('closest_x',closest_x)
            # print('closest_y',closest_y)
            counts_x = np.unique(closest_x,return_counts=True)
            counts_y = np.unique(closest_y,return_counts=True)
            if np.all(closest_x == closest_x[0]):
                print('vertical contour')
                d = ((x_tmp[0]-closest_x[0]))
                x_tmp_sub = x_tmp + np.array([d,0])
                if np.linalg.norm(x_tmp_sub-x_ini_index) > np.linalg.norm(x_tmp-x_ini_index):
                    x_tmp_sub = x_tmp + np.array([-d,0])
            elif np.all(counts_x[1]==np.array([2,2])):
                print('vertical contour')
                x_mean = np.mean(counts_x[0])
                d = x_tmp[0]-x_mean
                x_tmp_sub = x_tmp + np.array([d,0])
                if np.linalg.norm(x_tmp_sub-x_ini_index) > np.linalg.norm(x_tmp-x_ini_index):
                    x_tmp_sub = x_tmp + np.array([-d,0])
            elif np.all(closest_y == closest_y[0]):
                print('horizontal contour')
                d = ((x_tmp[1]-closest_y[0]))
                x_tmp_sub = x_tmp + np.array([0,d])
                if np.linalg.norm(x_tmp_sub-x_ini_index) > np.linalg.norm(x_tmp-x_ini_index):
                    x_tmp_sub = x_tmp + np.array([0,-d])
            elif np.all(counts_y[1]==np.array([2,2])):
                print('horizontal contour')
                y_mean = np.mean(counts_y[0])
                d = x_tmp[1]-y_mean
                x_tmp_sub = x_tmp + np.array([0,d])
                if np.linalg.norm(x_tmp_sub-x_ini_index) > np.linalg.norm(x_tmp-x_ini_index):
                    x_tmp_sub = x_tmp + np.array([0,-d])
            else:
                lstsq = np.linalg.lstsq(np.vstack([closest_x, np.ones(len(closest_x))]).T, closest_y, rcond=None)[0]
                print('lstsq',lstsq)
                slope = lstsq[0]
                intercept = lstsq[1]
                d = abs(slope*x_tmp[0]-x_tmp[1]+intercept)/math.sqrt(slope**2+1)
                normal = np.array([slope, -1])/math.sqrt(slope**2+1)
                x_tmp_sub = x_tmp + d*normal
                if np.linalg.norm(x_tmp_sub-x_ini_index) > np.linalg.norm(x_tmp-x_ini_index):
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
            # if k%100==0:
            #     ax.scatter(x,y,c='g')
            x_new = x_tmp_sub-dt*(get_flow(x,y,k*dt)+F*normal)*size_backtrack
            path=np.append(path, np.array([x_new]), axis=0)
            x_tmp = x_new
            # print('x_new',x_new)
        
        k -= 1
        l += 1
    path = np.append(path, np.array([x_ini_index]), axis=0)
    print('number of points for path planning',path.shape[0])
    print(path)
    ax.plot(xmin+path[:,0]*resolution,ymin+path[:,1]*resolution,marker='.',markersize=3,c='k')


# backtracking at time step = size_backtrack
if TF_back_track==True and TF_four_closest==False:
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
            # if k%100==0:
            #     ax.scatter(x,y,c='g')
            x_new = x_tmp_sub-dt*(get_flow(x,y,k*dt)+F*normal)*size_backtrack
            path=np.append(path, np.array([x_new]), axis=0)
            x_tmp = x_new
            print('x_new',x_new)
        
        k -= 1
        l += 1
    path = np.append(path, np.array([x_ini_index]), axis=0)
    print('number of points for path planning',path.shape[0])
    print(path)
    ax.plot(xmin+path[:,0]*resolution,ymin+path[:,1]*resolution,marker='.',markersize=3,c='k')


# backtracking at every time step
# if TF_back_track:
#     print('########back propagation#########')
#     k = t_final
#     # while k!=0 and len(measure.find_contours(phi_data[:,:,k+1],level=0)) != 0:
#     while k!=0:
#         zero_level_index_np_where = np.where(np.absolute(phi_data[:,:,k+1])<=zero_threshold)
#         if len(zero_level_index_np_where)==0: 
#             print(zero_level_index_np_where)
#             break
#         print('k =',k)
#         current_phi = phi_data[:,:,k+1]
#         # zero_level_index = measure.find_contours(phi_data[:,:,k+1],level=0)[0][:-1] #This function add some points by linear interpolation to get smooth contours
#         # print('zero level index',zero_level_index)
#         # n_zero_level = len(zero_level_index)
#         # print('number of 0-level points',n_zero_level)
#         # zero_level_index_np_where = np.where(phi_data[:,:,k+1]==0)
#         # print('np where number of 0-level points', zero_level_index_np_where[0].shape)
#         zero_level_index_y,zero_level_index_x = zero_level_index_np_where
#         n_zero_level = zero_level_index_y.shape[0]
#         print('# of 0-level points',n_zero_level,zero_level_index_x*resolution+xmin,zero_level_index_y*resolution+ymin)
#         if n_zero_level <2:
#             break
#         d_list = []
#         # for j in range(n_zero_level):
#         #     d_sq = (zero_level_index[j][1]-x_tmp[0])**2 + (zero_level_index[j][0]-x_tmp[1])**2
#         #     heappush(d_list,[d_sq, (int(zero_level_index[j][1]),int(zero_level_index[j][0]))])
#         for j in range(zero_level_index_y.shape[0]):
#             d_sq = (zero_level_index_x[j]-x_tmp[0])**2 + (zero_level_index_y[j]-x_tmp[1])**2
#             heappush(d_list,[d_sq, (zero_level_index_x[j],zero_level_index_y[j])])
#         print(d_list)
#         d_sq_min_1, x_close_1 = heappop(d_list)
        
#         # print('position of x_close_1', xmin+x_close_1[0]*resolution, ymin+x_close_1[1]*resolution)
#         # If d_sq_min_1 is too small, x_close_1 should be almost the same with x_tmp.
#         # So you take the third closest point
#         while d_sq_min_1<0.1 and len(d_list)>1: 
#             # print('taking the next closest point at k=',k)
#             d_sq_min_1, x_close_1 = heappop(d_list)
#         d_sq_min_2, x_close_2 = heappop(d_list)
#         while np.array_equal(x_close_1,x_close_2) and len(d_list)!=0:
#             d_sq_min_2, x_close_2 = heappop(d_list)
#         print('two closest points',x_close_1,x_close_2)
#         slope, normal, d = slope_normal(x_tmp,x_close_1,x_close_2)
#         # if normal[0]==0&normal[1]==0:
#         #     print('normal == [0,0]')
#         #     print('normal',normal,'slope',slope)
#         #     break

#         # projection of x_tmp onto the 0-level set
#         x_tmp_sub = x_tmp + d*normal
#         if np.linalg.norm(x_tmp_sub-x_ini_index) > np.linalg.norm(x_tmp-x_ini_index):
#             normal = -normal
#             x_tmp_sub = x_tmp + d*normal
#         # if x_tmp_sub == x_close_1 or x_tmp_sub == x_close_2:
#         #     print('まず無いと思うんだけどxf is exactly on x1 or x2. Getting the third closest point')
#         #     _, x_close_3 = heappop(d_list)
#             # ちゃんと向き考えてnormal vector２つ取ってきてそれらの平均を取る
#         x_tmp_sub_x = x_tmp_sub[0]
#         x_tmp_sub_y = x_tmp_sub[1]
#         x = xmin+x_tmp_sub_x*resolution
#         y = ymin+x_tmp_sub_y*resolution
#         if k%10==0:
#             ax.scatter(x,y,c='g')
#         x_new = x_tmp_sub-dt*(get_flow(x,y,k*dt)+F*normal)
#         path=np.append(path, np.array([x_new]), axis=0)
#         x_tmp = x_new
    #     print('x_new',x_new)
    #     k = k-1
    # path = np.append(path, np.array([x_ini_index]), axis=0)
    # print('number of points for path planning',path.shape[0])
    # print(path)
    # ax.plot(xmin+path[:,0]*resolution,ymin+path[:,1]*resolution,marker='.',markersize=3,c='k')

ax.scatter(x_ini[0],x_ini[1],c='b',label='start')
ax.scatter(x_goal[0],x_goal[1],c='r',label='goal')
# ax.scatter(zero_level_index_x*resolution+xmin,zero_level_index_y*resolution+ymin,c='k')
ax.legend()
# plt.title('single gyro w/ control F='+str(F))
# plt.savefig('./results_singlegyro/singlegyro_F_'+str(F)+'.png')
plt.title('single gyro w/ control F='+str(F)+',size of backtrack ='+str(size_backtrack)+' plot every '+str(plot_time_step*dt)+'[s]')
plt.savefig('./results_singlegyro/singlegyro_F_'+str(F)+',size_backtrack_ ='+str(size_backtrack)+'.png')
plt.show()

# plt.figure(2)
# plt.imshow(phi_next, cmap='hot', interpolation='nearest')
# plt.colorbar()

# plt.figure(3)
# plt.plot(xx,phi_data[int((10-ymin)/resolution),:,800],label='original')
# plt.plot(xx,phi_next_abs[int((10-ymin)/resolution),:],label='abs')
# plt.plot(xx,np.ones_like(phi_next_abs[int((10-ymin)/resolution),:])*zero_threshold,label = 'zero threshold')
# plt.legend()

# plt.show()


# plt.figure(4)
# plt.plot(phi_data[int((x_ini[1]-ymin)/resolution),int((x_ini[0]+1-xmin)/resolution),:],label='x_ini+(1,0)')
# plt.plot(phi_data[int((x_ini[1]-ymin)/resolution),int((x_ini[0]-xmin)/resolution),:],label='x_ini')
# plt.plot(zero_threshold,label='zero threshold')
# plt.xlabel('time step')
# plt.ylabel('phi')
# plt.legend()
# plt.show()

#plt.plot(dphi_norm_list[int((x_ini[1])/resolution),int((x_ini[0]+1)/resolution),:],label='x_ini+(1,0) 1st eq')
#plt.plot(dphi_norm_list[int((x_ini[1])/resolution),int((x_ini[0])/resolution),:],label='x_ini 1st eq')
#plt.plot(dphi_2_norm_list[int((x_ini[1])/resolution),int((x_ini[0]+1)/resolution),:],label='x_ini+(1,0) 1st eq')
#plt.plot(dphi_2_norm_list[int((x_ini[1])/resolution),int((x_ini[0])/resolution),:],label='x_ini 1st eq')
#plt.title('|grad phi| at x_ini+(1,0) and xini')
#plt.xlabel('time step')
#plt.ylabel('|grad phi|')
#plt.legend()
#
#plt.plot(dphi_data[int((x_ini[1])/resolution),int((x_ini[0]+1)/resolution),:],label='x_ini+(1,0)')
#plt.plot(dphi_data[int((x_ini[1])/resolution),int((x_ini[0])/resolution),:],label='x_ini')
#plt.legend()
