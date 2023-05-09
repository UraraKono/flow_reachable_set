import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def bickley_u_v(t,X,Y):

    U0 = 5.4138                        # Mm/day
    L0 = 1.77                          # Mm
    r0 = 6.371                         # Mm
    c1 = 0.1446*U0
    c2 = 0.2053*U0
    c3 = 0.4561*U0
    # lx = 6.371e6*pi
    # ly = 1.77e6
    ep1 = 0.075
    ep2 = 0.4
    ep3 = 0.3

    k1 = 2/r0
    k2 = 4/r0
    k3 = 6/r0

    f1 = ep1 * np.exp(-1j*k1*c1*t)
    f2 = ep2 * np.exp(-1j*k2*c2*t)
    f3 = ep3 * np.exp(-1j*k3*c3*t)
    F1 = f1 * np.exp(1j*k1*X)
    F2 = f2 * np.exp(1j*k2*X)
    F3 = f3 * np.exp(1j*k3*X)
    G = np.real(F1+F2+F3)
    G_x = np.real(1j*k1*F1+1j*k2*F2+1j*k3*F3)
    u = U0 / (np.cosh(Y/L0))**2 + 2*U0*np.sinh(Y/L0)/((np.cosh(Y/L0))**3)*G
    v = U0*L0*(1/np.cosh(Y/L0))**2*G_x

    return u, v

def animate(iter, Q, dt, X, Y):
    # iter is in range(frs)
    t = iter*dt
    U, V = bickley_u_v(t,X,Y)
    flow_size = np.sqrt(U**2+V**2)
    Q.set_UVC(U,V,flow_size)

    ax.set_title('t='+str(int(10*t/10)))

    return Q

if __name__ == "__main__":
    plotinterval = 0.2
    xx = np.arange(0,10,plotinterval)
    yy = np.arange(-3,3,plotinterval)
    X, Y = np.meshgrid(xx,yy)
    dt = 0.1
    T = 10

    fig, ax = plt.subplots()
    U, V = bickley_u_v(0,X,Y)
    flow_size = np.sqrt(U**2+V**2)
    Q = ax.quiver(X, Y, U, V, flow_size,cmap=plt.cm.jet)
    plt.colorbar(Q, cmap = plt.cm.jet)
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(-3.3, 3.3)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_aspect('equal')
    ax.set_title('t=0.0')

    frs = int(T/dt)
    anim = animation.FuncAnimation(fig, animate, fargs=(Q, dt, X, Y),frames=frs, interval=200, blit=False)
    anim.save('bickley.gif', writer='pillow', fps=10, progress_callback=lambda i, n: print(i),)

    plt.show()

    