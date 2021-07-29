from DGM_model import DGM

import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import random

def set_all_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)



class BlackScholes_americanPut_DGM:
    '''
    This class calculates the Black Scholes American Put Option price using DGM model
    '''

    def __init__(self):
        
        # Put Option params
        self.S0 = 50  # Initial price
        self.K = 50  # Strike
        self.r = 0.05  # Interest rate
        self.sigma = 0.5  # Volatility

        # Bounds for solving PDE
        self.S_low = 0.0 + 1e-10  # lower bound for stock price
        self.S_high = 2 * self.K  # upper bound for spot price        
        self.t_low = 0 + 1e-10  # lower bound time
        self.T = 1  # Final time
        self.n_interior = 1000 # Number of interior points to sample
        self.n_terminal = 100 # Number of terminal points to sample

        # Finite difference parameters
        self.n_FD_steps = 10000

        # DGM params
        self.n_layers = 3
        self.n_nodes = 50
        self.lr = 0.001

        # Model training params
        self.n_resampling_steps = 50  # number of resampling steps
        self.n_optimization_steps = 30  # number of optimization steps before resampling

        # Plot options
        self.n_plot = 100  # Points on plot grid for each dimension
        self.save_fig = False
        self.fig_name = 'plots/BlackScholes_AmericanPut.png'


    def _BS_put(self, S, t):
        """
        Method calculates analytical solution for European put option
        """

        d1 = (np.log(S / self.K) + (self.r + self.sigma ** 2 / 2) * (self.T - t)) / (self.sigma * np.sqrt(self.T - t))
        d2 = d1 - (self.sigma * np.sqrt(self.T - t))

        putPrice = self.K * np.exp(-self.r * (self.T - t)) * stats.norm.cdf(-d2) - S * stats.norm.cdf(-d1)

        return putPrice


    def _AmericanPutFD(self, n_FD_steps):
        """
        American put option price under Black-Scholes model using finite differences(implicit BTCS)

        1) Convert to a log space grid using the transformation S = log(S) on the BS PDE
        2) dV/dt = (V(h,t) - V(h,t-1)) / dt
        3) d2V/dS2 = (V(h-1,t) - 2V(h,t) + V(h+1,t)) / dx*dx
        4) dV/dS = (V(h+1,t) - V(h-1,t)) / 2*dx
        """

        # time grid
        dt = self.T / n_FD_steps
        t = np.arange(0, self.T + dt, dt)

        # some useful constants
        dx = self.sigma * np.sqrt(3 * dt) # Converting to log(S) from S
        alpha = 0.5 * self.sigma ** 2 * dt / (dx) ** 2
        beta = (self.r - 0.5 * self.sigma ** 2) * dt / (2 * dx)

        # log space grid
        x_max = np.ceil(10 * self.sigma * np.sqrt(self.T) / dx) * dx
        x_min = - np.ceil(10 * self.sigma * np.sqrt(self.T) / dx) * dx
        x = np.arange(x_min, x_max + dx, dx) # final space points on grid
        n_x = len(x) - 1
        i = np.arange(1, n_x)

        # American option price grid
        put_price = np.nan * np.ones([n_x + 1, n_FD_steps + 1])  # put price at each time step
        put_price[:, -1] = np.maximum(self.K - self.S0 * np.exp(x), 0)  # set payoff at maturity
        put_cur_price = np.nan * np.ones(n_x + 1)

        S = self.S0 * np.exp(x) # Transformation from log(S) to S for final output

        # calculating optimal exercise boundary
        exer_boundary = np.ones(n_FD_steps + 1) * np.nan
        exer_boundary[-1] = self.K

        # step backwards through time
        for j in np.arange(n_FD_steps, 0, -1):

            # computing the PDE values at time step n for all values of x
            put_cur_price[i] = put_price[i, j] - self.r * dt * put_price[i, j] + beta * (put_price[i + 1, j] - put_price[i - 1, j]) + alpha * (
                        put_price[i + 1, j] - 2 * put_price[i, j] + put_price[i - 1, j])
            put_cur_price[0] = 2 * put_cur_price[1] - put_cur_price[2] # initial values
            put_cur_price[n_x] = 2 * put_cur_price[n_x - 1] - put_cur_price[n_x - 2] # Final Boundary values

            # compare with intrinsic values
            exer_val = self.K - S
            put_price[:, j - 1] = np.maximum(put_cur_price, exer_val)
            # Replace value of option with payoff at all places where option value is less than payoff,
            # which cannot happen as it provides arbitrage.

            # determine optimal exercise boundaries
            check_idx = (put_cur_price > exer_val) * 1
            idx = check_idx.argmax() - 1
            if max(check_idx) > 0:
                exer_boundary[j - 1] = S[idx]

        return t, S, put_price, exer_boundary


    def _sampler(self):
        """
        Samples points uniformly from the interior and on the terminal conditions
        """

        # Interior sampling
        t_int_points = np.random.uniform(low=self.t_low, high=self.T, size=[self.n_interior, 1])
        S_int_points = np.random.uniform(low=self.S_low, high=self.S_high * 1.5, size=[self.n_interior, 1])

        # Terminal sampling
        t_ter_points = self.T * np.ones((self.n_terminal, 1))
        S_ter_points = np.random.uniform(low=self.S_low, high=self.S_high * 1.5, size=[self.n_terminal, 1])

        return t_int_points, S_int_points, t_ter_points, S_ter_points


    def _DGMmodel(self):
        t_int_points = Input((1))
        s_int_points = Input((1))
        t_ter_points = Input((1))
        s_ter_points = Input((1))

        X = self.model_internal(t_int_points, s_int_points)
        X_1 = self.model_internal(t_int_points, s_int_points)
        Y = self.model_internal(t_ter_points, s_ter_points)

        return Model(inputs=[t_int_points, s_int_points, t_ter_points, s_ter_points], outputs=[X, X_1, Y])


    @tf.function
    def _loss(self, t_int_points, s_int_points, t_ter_points, s_ter_points):
        """
        Method computes loss function.
        """

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            with tf.GradientTape() as Vss:
                Vss.watch(s_int_points)
                with tf.GradientTape(persistent=True) as Vst:
                    Vst.watch(t_int_points)
                    Vst.watch(s_int_points)
                    V, value, calc_payoff = self.model([t_int_points, s_int_points, t_ter_points, s_ter_points], training=True)
                V_t = Vst.gradient(V, t_int_points)[0]
                V_s = Vst.gradient(V, s_int_points)[0]
            V_ss = Vss.gradient(V_s, s_int_points)[0]

            diff_V = V_t + 0.5 * self.sigma ** 2 * s_int_points ** 2 * V_ss + (
                    self.r * s_int_points * V_s - self.r * V)

            # Loss 1: calculate L2-norm of PDE differential operator with inequality constraint
            payoff = tf.nn.relu(self.K - s_int_points)
            L1 = tf.reduce_mean(input_tensor=tf.square(diff_V * (value - payoff)))
            L2 = tf.reduce_mean(input_tensor=tf.square(tf.nn.relu(diff_V)))  # Minimizes -min(-f,0) = max(f,0)

            # Loss 2: boundary condition
            L3 = tf.reduce_mean(
                input_tensor=tf.square(tf.nn.relu(-(value - payoff))))  # Minimizes -min(-f,0) = max(f,0)

            # Loss 3: terminal condition loss
            L4 = tf.reduce_mean(input_tensor=tf.square(calc_payoff - tf.nn.relu(self.K - s_ter_points)))

            loss_value = L1 + L2 + L3 + L4

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, self.model.trainable_variables)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss_value, L1, L2, L3, L4


    def train(self):

        self.model_internal = DGM.DGMNet(self.n_nodes, self.n_layers, 1)
        self.model = self._DGMmodel()
        print(self.model.summary())

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr)

        # Custom training loop
        for i in range(self.n_resampling_steps):

            # sampling
            t_int_points, s_int_points, t_ter_points, s_ter_points = self._sampler()
            t_int_points = tf.convert_to_tensor(t_int_points, dtype=tf.float32)
            s_int_points = tf.convert_to_tensor(s_int_points, dtype=tf.float32)
            t_ter_points = tf.convert_to_tensor(t_ter_points, dtype=tf.float32)
            s_ter_points = tf.convert_to_tensor(s_ter_points, dtype=tf.float32)

            # optimization steps
            for _ in range(self.n_optimization_steps):
                loss_value, L1, L2, L3, L4 = self._loss(t_int_points, s_int_points, t_ter_points, s_ter_points)
            print(float(loss_value), float(L1), float(L2), float(L3), float(L4), i)
            self.plotting()


    def plotting(self, save_fig = False):

        self.save_fig = save_fig

        # figure options
        plt.figure(figsize=(10, 20))

        # time values at which to examine density
        time_steps = [self.t_low, self.T / 3, 2 * self.T / 3, 0.999 * self.T]

        # solution using finite differences
        (t, S, price, exer_bd) = self._AmericanPutFD(self.n_FD_steps)
        S_idx = S < self.S_high
        t_idx = [round(t * self.n_FD_steps) for t in time_steps]

        S_plot = S[S_idx].reshape(-1,1) # here stock plotting points for model inference are taken from FD(no uniform distribution)

        total_predicted_price = np.zeros(len(time_steps) * len(S_plot))
        total_actual_price = np.zeros(len(time_steps) * len(S_plot))

        for i, curr_t in enumerate(time_steps):

            # specify subplot
            plt.subplot(2, 2, i + 1)

            # simulate process at current t
            europeanOptionValue = self._BS_put(S_plot, curr_t)
            actual_price = price[S_idx, t_idx[i]]

            t_plot = curr_t * np.ones_like(S_plot)

            t_int_points = tf.convert_to_tensor(t_plot, dtype=tf.float32)
            s_int_points = tf.convert_to_tensor(S_plot, dtype=tf.float32)

            predicted_price = self.model_internal(t_int_points, s_int_points)

            plt.plot(S_plot, actual_price, color='r', label='Actual Price', linewidth=1, linestyle=':')
            plt.plot(S_plot, predicted_price, color='c', label='Predicted Price')

            total_actual_price[i * len(S_plot): (i+1) * len(S_plot)] = actual_price
            total_predicted_price[i * len(S_plot): (i + 1) * len(S_plot)] = predicted_price[:, 0]

            # subplot options
            plt.ylim(ymin=0.0, ymax=self.K)
            plt.xlim(xmin=0.0, xmax=self.S_high)
            plt.xlabel(r"Spot Price", fontsize=15, labelpad=10)
            plt.ylabel(r"Option Price", fontsize=15, labelpad=20)
            plt.title(f"t = {curr_t : .2f}", fontsize=18, y=1.03)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)

            if i == 0:
                plt.legend(loc='upper right', prop={'size': 16})

        # adjust space between subplots
        plt.subplots_adjust(wspace=0.3, hspace=0.4)

        #print(mean_absolute_error(total_predicted_price, total_actual_price))
        print(r2_score(total_predicted_price, total_actual_price))

        if self.save_fig:
            plt.savefig(self.fig_name)

#set_all_seeds(121)
BS = BlackScholes_americanPut_DGM()
BS.train()
BS.plotting(save_fig=True)
K.clear_session()