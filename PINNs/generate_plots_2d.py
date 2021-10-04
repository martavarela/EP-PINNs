import matplotlib.pyplot as plt
import pylab
import numpy as np
import matplotlib.animation as animation
from PIL import Image
import io

def plot_2D(data_list,dynamics, model, animation, fig_name):

    plot_2D_cell(data_list, dynamics, model, fig_name[1:])
    plot_2D_grid(data_list,dynamics, model, fig_name[1:])
    if animation:
        generate_2D_animation(dynamics, model, fig_name[1:])
    return 0
    
def plot_2D_cell(data_list, dynamics, model, fig_name):
    
    ## Unpack data
    observe_data, observe_train, v_train, v = data_list[0], data_list[1], data_list[2], data_list[3]
    
    ## Pick a random cell to show
    cell_x = dynamics.max_x*0.75
    cell_y = dynamics.max_y*0.75
        
    ## Get data for cell
    idx = [i for i,ix in enumerate(observe_data) if (observe_data[i][0:2]==[cell_x,cell_y]).all()]
    observe_geomtime = observe_data[idx]
    v_GT = v[idx]
    v_predict = model.predict(observe_geomtime)[:,0:1]
    t_axis = observe_geomtime[:,2]
    
    ## Get data for points used in training process
    idx_train = [i for i,ix in enumerate(observe_train) if (observe_train[i][0:2]==[cell_x,cell_y]).all()]
    v_trained_points = v_train[idx_train]
    t_markers = (observe_train[idx_train])[:,2]
    
    ## create figure
    plt.figure()
    plt.plot(t_axis, v_GT, c='b', label='GT')
    plt.plot(t_axis, v_predict, c='r', label='Predicted')
    # If there are any trained data points for the current cell 
    if len(t_markers):
        plt.scatter(t_markers, v_trained_points, marker='x', c='black',s=6, label='Observed')
    plt.legend(loc='upper right')
    plt.xlabel('t')
    plt.ylabel('V')
    
    ## save figure
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    png2 = Image.open(png1)
    png2.save(fig_name + "_cell_plot_2D.tiff")
    png1.close()
    return 0

def plot_2D_grid(data_list,dynamics, model, fig_name):
    
    grid_size = 200
    rand_t = dynamics.max_t/2
    
    ## Get data
    x = np.linspace(dynamics.min_x,dynamics.max_x, grid_size)
    y = np.linspace(dynamics.min_y,dynamics.max_y, grid_size)
    t = np.ones_like(x)*rand_t
    
    X, T, Y = np.meshgrid(x,t,y)
    X_data = X.reshape(-1,1)
    Y_data = Y.reshape(-1,1)
    T_data = T.reshape(-1,1)
    data = np.hstack((X_data, Y_data, T_data))
    
    v_pred = model.predict(data)[:,0:1]
    X, Y = np.meshgrid(x,y)
    Z = np.zeros((grid_size,grid_size))
    for i in range(grid_size):
        Z[i,:] = (v_pred[(i*grid_size):((i+1)*grid_size)]).reshape(-1)
    
    ## create figure
    plt.figure()
    contour = plt.contourf(X,Y,Z, cmap=plt.cm.bone)
    plt.xlabel('x')
    plt.ylabel('y')
    cbar = plt.colorbar(contour)
    cbar.ax.set_ylabel('V')
    
    ## save figure
    png1 = io.BytesIO()
    plt.savefig(png1, format="png", dpi=500, pad_inches = .1, bbox_inches = 'tight')
    png2 = Image.open(png1)
    png2.save(fig_name + "_grid_plot_2D.tiff")
    png1.close()
    return 0   
    
def generate_2D_animation(dynamics, model, fig_name):
    
    grid_size = 200
    nT = int(dynamics.max_t)
    n_frames = 2*nT
    x = np.linspace(dynamics.min_x,dynamics.max_x, grid_size)
    y = np.linspace(dynamics.min_y,dynamics.max_y, grid_size)
    X, Y = np.meshgrid(x,y)
    Z_0 = np.zeros((grid_size,grid_size))
    
    def get_animation_data(i):
        ## predict V values for each frame (each time step)
        t = np.ones_like(x)*(nT/n_frames)*(i+1)
        X, T, Y = np.meshgrid(x,t,y)
        X_data = X.reshape(-1,1)
        Y_data = Y.reshape(-1,1)
        T_data = T.reshape(-1,1)
        data = np.hstack((X_data, Y_data, T_data))
        v_pred = model.predict(data)[:,0:1]
        Z = np.zeros((grid_size,grid_size))
        for i in range(grid_size):
            Z[i,:] = (v_pred[(i*grid_size):((i+1)*grid_size)]).reshape(-1)
        return Z
    
    ## Create base screen
    fig = pylab.figure()
    ax = pylab.axes(xlim=(dynamics.min_x, dynamics.max_x), ylim=(dynamics.min_y, dynamics.max_y), xlabel='x', ylabel='y')
    levels = np.arange(0,1.15,0.1) 
    contour = pylab.contourf(X, Y, Z_0, levels = levels, cmap=plt.cm.bone)
    cbar = pylab.colorbar()
    cbar.ax.set_ylabel('V')
    
    def animate(i):
        ## create a frame
        Z = get_animation_data(i)
        contour = pylab.contourf(X, Y, Z, cmap=plt.cm.bone)
        plt.title('t = %.1f' %((nT/n_frames)*(i+1)))
        return contour
    
    anim = animation.FuncAnimation(fig, animate, frames=n_frames, repeat=False)
    anim.save(fig_name+'_2D_Animation.mp4', writer=animation.FFMpegWriter(fps=10))
    return 0
    