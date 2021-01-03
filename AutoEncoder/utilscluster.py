import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
irange = range
    
def plot_latent_variable3d(X, Y, epoch, name):
    plt.figure(figsize=(16, 11.5))
    ax = plt.axes(projection='3d')
    ax.view_init(25, 25)
    
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))  
    
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'cyan', 'lime', 'yellow']
    for l, c in enumerate(color):
        inds = np.where(Y==l)
        ax.scatter3D(X[inds, 0], X[inds, 1], X[inds, 2], color = c, label = l, linewidth = 0)
    plt.xlim([X[:, 0].min() - 0.1, X[:, 0].max() + 0.1])
    plt.ylim([X[:, 1].min()- 0.1, X[:, 1].max() + 0.1])
    ax.set_zlim3d([X[:, 2].min()- 0.1, X[:, 2].max() + 0.1])
#    ax.legend()
    plt.xticks([])
    plt.yticks([])
    ax.set_zticks([])
    plt.savefig('3d_visualization/{}_epoch_{}.pdf'.format(name, epoch)) 
    plt.show()    
