"""
@author : braveniuniu
@when : 2023-11-08
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import csv
from rbf.sputils import expand_rows
from rbf.pde.fd import weight_matrix
from rbf.pde.geometry import contains
from rbf.pde.nodes import poisson_disc_nodes  

# Define the problem domain with line segments.
vert = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0],
                 [2.0, 2.0], [1.0, 2.0], [0.0, 2.0]])
smp = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 0]])


spacing = 0.07  # approximate spacing between nodes

n = 25  # stencil size. Increase this will generally improve accuracy

phi = 'phs3'  # radial basis function used to compute the weights. Odd
# order polyharmonic splines (e.g., phs3) have always performed
# well for me and they do not require the user to tune a shape
# parameter. Use higher order polyharmonic splines for higher
# order PDEs.

order = 2  # Order of the added polynomials. This should be at least as
# large as the order of the PDE being solved (2 in this case). Larger
# values may improve accuracy
#
# generate nodes
nodes, groups, _ = poisson_disc_nodes(spacing, (vert, smp))
N = nodes.shape[0]

# create the components for the "left hand side" matrix.
A_interior = weight_matrix(
    x=nodes[groups['interior']],
    p=nodes,
    n=n,
    diffs=[[2, 0], [0, 2]],
    phi=phi,
    order=order)
A_boundary = weight_matrix(
    x=nodes[groups['boundary:all']],
    p=nodes,
    n=1,
    diffs=[0, 0])
# Expand and add the components together
A = expand_rows(A_interior, groups['interior'], N)
A += expand_rows(A_boundary, groups['boundary:all'], N)


def F(a1, b1, c1, x, y):
    return c1 * np.exp(-((x - a1) ** 2 + (y - b1) ** 2) / 0.01)
def PDE(source, a,b,c,x, y):
    out = 0

    for i in range(source):
        out += F(a[i], b[i], c[i], x, y)
    return out

types=5
source=4
source_data=100

Data_list = []
Source=[]
for i in range(types):
    #热源位置
    a = np.random.uniform(0.1, 1.9, source)
    b = np.random.uniform(0.1, 1.9, source)
    print("Types{}".format(i))
    print("a", a)
    print("b", b)

    data_list=[]

    for j in range(source_data):
        c = np.random.uniform(1, 10, source)
        print(j,"c", c)
        Source.append([a,b,c])
        yr = PDE(source,a,b,c,nodes[:,0], nodes[:,1])

        # create "right hand side" vector
        d = np.zeros((N,))
        d[groups['interior']] = yr[groups['interior']]
        d[groups['boundary:all']] = np.random.uniform(1, 10)

        # find the solution at the nodes
        u_soln = spsolve(A, d)

        # Create a grid for interpolating the solution
        xg, yg = np.meshgrid(np.linspace(0.0, 2.02, 50), np.linspace(0.0, 2.02, 50))
        points = np.array([xg.flatten(), yg.flatten()]).T
        print(points.shape)
        print(xg.shape)
        print(xg.flatten().shape)



        # We can use any method of scattered interpolation (e.g.,
        # scipy.interpolate.LinearNDInterpolator). Here we repurpose the RBF-FD method
        # to do the interpolation with a high order of accuracy
        I = weight_matrix(
            x=points,
            p=nodes,
            n=n,
            diffs=[0, 0],
            phi=phi,
            order=order)
        u_itp = I.dot(u_soln)

        # mask points outside of the domain
        F1 = PDE(source,a,b,c,points[:,0], points[:,1])

        # u_itp[~contains(points, vert, smp)] = np.nan
        # F1[~contains(points, vert, smp)] = np.nan 
        data_list.append(u_itp.tolist())


        ug = u_itp.reshape((50, 50))  # fold back into a grid
        F11 = F1.reshape((50, 50))




        fig, ax = plt.subplots()
        #p = ax.contourf(xg, yg, ug, np.linspace(-1e-6, 0.3, 9), cmap='viridis')
        f1 = ax.contourf(xg, yg, F11, cmap='viridis')

        #ax.plot(nodes[:, 0], nodes[:, 1], 'ko', markersize=4)

        # for i in range(8):
        #     ax.plot(a[i], b[i], 'ko', markersize=8, alpha=c[i]/max(c))
        # for s in smp:
        #     ax.plot(vert[s, 0], vert[s, 1], 'k-', lw=2)

        ax.set_aspect('equal')
        ax.set_xlim(-0.05, 2.05)
        ax.set_ylim(-0.05, 2.05)
        # fig.colorbar(f1, ax=ax)
        fig.tight_layout()
        plt.savefig('./result/sourceplot/types{}F{}.png'.format(i,j))
        plt.show()
        # make a contour plot of the solution
        fig, ax = plt.subplots()


        p = ax.contourf(xg, yg, ug, cmap='viridis')
        #ax.plot(nodes[:, 0], nodes[:, 1], 'ko', markersize=4)

        ax.set_aspect('equal')
        ax.set_xlim(-0.05, 2.05)
        ax.set_ylim(-0.05, 2.05)
        fig.colorbar(p, ax=ax)
        fig.tight_layout()
        plt.savefig('./result/dataplot/types{}u{}.png'.format(i,j))
        plt.show()
    Data_list.append(data_list)

T = np.array(Data_list)
Source =np.array(Source)
  
X = np.array(points)
file_name='Heat'+'_Types'+str(types)+'_source'+str(source)+'_number'+str(source_data)+'.npz'
np.savez(file_name, T=T, points=points, Source=Source)

# data = np.load('/mnt/jfs/xiangzixue/pythonProject/hyperdeeponet/data_generation/Heat_Types5_source4_number100.npz')#2000功率1000点的热场
# data_T, data_x, data_z = data['T'], data['X'], data['Source']


