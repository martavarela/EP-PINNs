import scipy.io
import deepxde as dde
from deepxde.backend import tf
import numpy as np

class system_dynamics():
    
    def __init__(self):
        
        ## PDE Parameters
        self.a = 0.01
        self.b = 0.15
        self.D = 0.1
        self.k = 8
        self.mu_1 = 0.2
        self.mu_2 = 0.3
        self.epsilon = 0.002

        ## Geometry Parameters
        self.min_x = 0.1
        self.max_x = 10            
        self.min_y = 0.1 
        self.max_y = 10
        self.min_t = 1
        self.max_t = 70
        self.spacing = 0.1

    def generate_data(self, file_name, dim):
        
        data = scipy.io.loadmat(file_name)
        if dim == 1:
            t, x, Vsav, Wsav = data["t"], data["x"], data["Vsav"], data["Wsav"]
            X, T = np.meshgrid(x, t)
        elif dim == 2:
            t, x, y, Vsav, Wsav = data["t"], data["x"], data["y"], data["Vsav"], data["Wsav"]
            X, T, Y = np.meshgrid(x,t,y)
            Y = Y.reshape(-1, 1)
        else:
            raise ValueError('Dimesion value argument has to be either 1 or 2')
        self.max_t = np.max(t)
        self.max_x = np.max(x)
        X = X.reshape(-1, 1)
        T = T.reshape(-1, 1)
        V = Vsav.reshape(-1, 1)
        W = Wsav.reshape(-1, 1)    
        if dim == 1:     
            return np.hstack((X, T)), V, W
        return np.hstack((X, Y, T)), V, W

    def geometry_time(self, dim):
        if dim == 1:
            geom = dde.geometry.Interval(self.min_x, self.max_x)
            timedomain = dde.geometry.TimeDomain(self.min_t, self.max_t)
            geomtime = dde.geometry.GeometryXTime(geom, timedomain)    
        elif dim == 2:
            geom = dde.geometry.Rectangle([self.min_x,self.min_y], [self.max_x,self.max_y])
            timedomain = dde.geometry.TimeDomain(self.min_t, self.max_t)
            geomtime = dde.geometry.GeometryXTime(geom, timedomain)
        else:
            raise ValueError('Dimesion value argument has to be either 1 or 2')
        return geomtime

    def params_to_inverse(self,args_param):
        
        params = []
        if not args_param:
            return self.a, self.b, self.D, params
        ## If inverse:
        ## The tf.variables are initialized with a positive scalar, relatively close to their ground truth values
        if 'a' in args_param:
            self.a = tf.math.exp(tf.Variable(-3.92))
            params.append(self.a)
        if 'b' in args_param:
            self.b = tf.math.exp(tf.Variable(-1.2))
            params.append(self.b)
        if 'd' in args_param:
            self.D = tf.math.exp(tf.Variable(-1.6))
            params.append(self.D)
        return params

    def pde_1D(self, x, y):
    
        V, W = y[:, 0:1], y[:, 1:2]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=1)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=1)
        ## Coupled PDE+ODE Equations
        eq_a = dv_dt -  self.D*dv_dxx + self.k*V*(V-self.a)*(V-1) +W*V 
        eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))
        return [eq_a, eq_b]

    def pde_1D_2cycle(self,x, y):
    
        V, W = y[:, 0:1], y[:, 1:2]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=1)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=1)
    
        x_space,t_space = x[:, 0:1],x[:, 1:2]
        t_stim_1 = tf.equal(t_space, 0)
        t_stim_2 = tf.equal(t_space, int(self.max_t/2))
        x_stim = tf.less_equal(x_space, 5*self.spacing)
    
        first_cond_stim = tf.logical_and(t_stim_1, x_stim)
        second_cond_stim = tf.logical_and(t_stim_2, x_stim)
    
        I_stim = tf.ones_like(x_space)*0.1
        I_not_stim = tf.ones_like(x_space)*0
        Istim = tf.where(tf.logical_or(first_cond_stim,second_cond_stim),I_stim,I_not_stim)
        ## Coupled PDE+ODE Equations
        eq_a = dv_dt -  self.D*dv_dxx + self.k*V*(V-self.a)*(V-1) +W*V -Istim
        eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))
        return [eq_a, eq_b]
    
    def pde_2D(self, x, y):
    
        V, W = y[:, 0:1], y[:, 1:2]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
        ## Coupled PDE+ODE Equations
        eq_a = dv_dt -  self.D*(dv_dxx + dv_dyy) + self.k*V*(V-self.a)*(V-1) +W*V 
        eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))
        return [eq_a, eq_b]

    def pde_2D_heter(self, x, y):
    
        V, W, var = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
        dv_dx = dde.grad.jacobian(y, x, i=0, j=0)
        dv_dy = dde.grad.jacobian(y, x, i=0, j=1)
        
        ## Heterogeneity
        D_heter = tf.math.sigmoid(var)*0.08+0.02;
        dD_dx = dde.grad.jacobian(D_heter, x, i=0, j=0)
        dD_dy = dde.grad.jacobian(D_heter, x, i=0, j=1)
        
        ## Coupled PDE+ODE Equations
        eq_a = dv_dt -  D_heter*(dv_dxx + dv_dyy) -dD_dx*dv_dx -dD_dy*dv_dy + self.k*V*(V-self.a)*(V-1) +W*V 
        eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))
        return [eq_a, eq_b]
 
    def pde_2D_heter_forward(self, x, y):
                
        V, W, D = y[:, 0:1], y[:, 1:2], y[:, 2:3]
        dv_dt = dde.grad.jacobian(y, x, i=0, j=2)
        dv_dxx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        dv_dyy = dde.grad.hessian(y, x, component=0, i=1, j=1)
        dw_dt = dde.grad.jacobian(y, x, i=1, j=2)
        dv_dx = dde.grad.jacobian(y, x, i=0, j=0)
        dv_dy = dde.grad.jacobian(y, x, i=0, j=1)
        
        ## Heterogeneity
        dD_dx = dde.grad.jacobian(D, x, i=0, j=0)
        dD_dy = dde.grad.jacobian(D, x, i=0, j=1)
        
        ## Coupled PDE+ODE Equations
        eq_a = dv_dt -  D*(dv_dxx + dv_dyy) -dD_dx*dv_dx -dD_dy*dv_dy + self.k*V*(V-self.a)*(V-1) +W*V 
        eq_b = dw_dt -  (self.epsilon + (self.mu_1*W)/(self.mu_2+V))*(-W -self.k*V*(V-self.b-1))
        return [eq_a, eq_b]   
 
    def IC_func(self,observe_train, v_train):
        
        T_ic = observe_train[:,-1].reshape(-1,1)
        idx_init = np.where(np.isclose(T_ic,1))[0]
        v_init = v_train[idx_init]
        observe_init = observe_train[idx_init]
        return dde.PointSetBC(observe_init,v_init,component=0)
    
    def BC_func(self,dim, geomtime):
        if dim == 1:
            bc = dde.NeumannBC(geomtime, lambda x:  np.zeros((len(x), 1)), lambda _, on_boundary: on_boundary, component=0)
        elif dim == 2:
            bc = dde.NeumannBC(geomtime, lambda x:  np.zeros((len(x), 1)), self.boundary_func_2d, component=0)
        return bc
    
    def boundary_func_2d(self,x, on_boundary):
            return on_boundary and ~(x[0:2]==[self.min_x,self.min_y]).all() and  ~(x[0:2]==[self.min_x,self.max_y]).all() and ~(x[0:2]==[self.max_x,self.min_y]).all()  and  ~(x[0:2]==[self.max_x,self.max_y]).all() 
   
    def modify_inv_heter(self, x, y):                
        domain_space = x[:,0:2]
        D = tf.layers.dense(tf.layers.dense(tf.layers.dense(tf.layers.dense(tf.layers.dense(tf.layers.dense(domain_space, 60,
                            tf.nn.tanh), 60, tf.nn.tanh), 60, tf.nn.tanh), 60, tf.nn.tanh), 60, tf.nn.tanh), 1, activation=None)        
        return tf.concat((y[:,0:2],D), axis=1)    
    
    def modify_heter(self, x, y):
        
        x_space, y_space = x[:, 0:1], x[:, 1:2]
        
        x_upper = tf.less_equal(x_space, 54*0.1)
        x_lower = tf.greater(x_space,32*0.1)
        cond_1 = tf.logical_and(x_upper, x_lower)
        
        y_upper = tf.less_equal(y_space, 54*0.1)
        y_lower = tf.greater(y_space,32*0.1)
        cond_2 = tf.logical_and(y_upper, y_lower)
        
        D0 = tf.ones_like(x_space)*0.02 
        D1 = tf.ones_like(x_space)*0.1
        D = tf.where(tf.logical_and(cond_1, cond_2),D0,D1)
        return tf.concat((y[:,0:2],D), axis=1)

