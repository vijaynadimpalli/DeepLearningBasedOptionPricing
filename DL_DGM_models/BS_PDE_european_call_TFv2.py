from DGM_model import DGM

import numpy as np
from scipy import stats
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




class BlackScholes_europeanCall_DGM_v2:
    """
    This class calculates the Black Scholes European Call Option price using DGM model and Tensorflow 2
    """

    def __init__(self):

        # Call Option params
        self.S0 = 50  # Initial price
        self.K = 50  # Strike
        self.r = 0.05  # Interest rate
        self.sigma = 0.25  # Volatility

        # Bounds for solving PDE
        self.S_low = 0.0 + 1e-10  # lower bound for stock price
        self.S_high = 2 * self.K  # upper bound for spot price        
        self.t_low = 0 + 1e-10  # lower bound time
        self.T = 1  # Final time
        self.n_interior = 1000 # Number of interior points to sample
        self.n_terminal = 100 # Number of terminal points to sample

        # DGM params
        self.n_layers = 3
        self.n_nodes = 50
        self.lr = 0.001

        # Model training params
        self.n_resampling_steps = 50  # number of resampling steps
        self.n_optimization_steps = 30  # number of optimization steps before resampling

        # Plotting Steps
        self.n_plot = 41  # Points on plot grid for each dimension
        self.save_fig = False
        self.fig_name = 'plots/BlackScholes_EuropeanCall.png'


    def _BS_call(self, S, t):
        """
        Method calculates analytical solution for European call option
        """

        d1 = (np.log(S / self.K) + (self.r + self.sigma ** 2 / 2) * (self.T - t)) / (self.sigma * np.sqrt(self.T - t))
        d2 = d1 - (self.sigma * np.sqrt(self.T - t))

        callPrice = S * stats.norm.cdf(d1) - self.K * np.exp(-self.r * (self.T - t)) * stats.norm.cdf(d2)

        return callPrice


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
        S_int_points = Input((1))
        t_ter_points = Input((1))
        S_ter_points = Input((1))

        X = self.model_internal(t_int_points, S_int_points)
        Y = self.model_internal(t_ter_points, S_ter_points)

        return Model(inputs=[t_int_points, S_int_points, t_ter_points, S_ter_points], outputs=[X, Y])


    @tf.function
    def _loss(self, t_int_points, S_int_points, t_ter_points, S_ter_points):
        """ 
        Method computes loss function.
        """

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:            

            with tf.GradientTape() as Vss:
                Vss.watch(S_int_points)
                with tf.GradientTape(persistent=True) as Vst:
                    Vst.watch(t_int_points)
                    Vst.watch(S_int_points)
                    V, calc_payoff = self.model([t_int_points, S_int_points, t_ter_points, S_ter_points], training=True)
                V_t = Vst.gradient(V, t_int_points)[0]
                V_s = Vst.gradient(V, S_int_points)[0]
            V_ss = Vss.gradient(V_s, S_int_points)[0]

            diff_V = V_t + 0.5 * self.sigma ** 2 * S_int_points ** 2 * V_ss + (
                    self.r * S_int_points * V_s - self.r * V)
            
            # Loss 1: calculate L2-norm of PDE differential operator
            L1 = tf.reduce_mean(input_tensor=tf.square(diff_V))

            # Loss 3: terminal condition loss
            # Calculates L2 norm between fitted payoff and target
            L3 = tf.reduce_mean(input_tensor=tf.square(calc_payoff - tf.nn.relu(S_ter_points - self.K)))

            loss_value = L1 + L3

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, self.model.trainable_variables)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss_value, L1, L3


    def train(self):


        self.model_internal = DGM.DGMNet(self.n_nodes, self.n_layers, 1)
        self.model = self._DGMmodel()
        print(self.model.summary())

        self.optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.lr)

        # Custom training loop
        for i in range(self.n_resampling_steps):

            # sampling
            t_int_points, S_int_points, t_ter_points, S_ter_points = self._sampler()

            t_int_points = tf.convert_to_tensor(t_int_points, dtype=tf.float32)
            S_int_points = tf.convert_to_tensor(S_int_points, dtype=tf.float32)
            t_ter_points = tf.convert_to_tensor(t_ter_points, dtype=tf.float32)
            S_ter_points = tf.convert_to_tensor(S_ter_points, dtype=tf.float32)

            # optimization steps
            for _ in range(self.n_optimization_steps):
                loss_value, L1, L3 = self._loss(t_int_points, S_int_points, t_ter_points, S_ter_points)
            print(float(loss_value), float(L1), float(L3), i)
            self.plotting()


    def plotting(self, save_fig = False):

        self.save_fig = save_fig
        # figure options
        plt.figure(figsize=(10, 20))

        # time points at which to compare prices
        time_steps = [self.t_low, self.T / 3, 2 * self.T / 3, 0.999 * self.T]

        S_plot = np.linspace(self.S_low, self.S_high, self.n_plot).reshape(-1, 1)

        total_predicted_price = np.zeros(len(time_steps) * len(S_plot))
        total_actual_price = np.zeros(len(time_steps) * len(S_plot))

        for i, curr_t in enumerate(time_steps):

            # subplot position
            plt.subplot(4, 1, i + 1)

            # analytical value at time t
            actual_price = self._BS_call(S_plot, curr_t)
            t_plot = curr_t * np.ones_like(S_plot)

            t_int_points = tf.convert_to_tensor(t_plot, dtype=tf.float32)
            S_int_points = tf.convert_to_tensor(S_plot, dtype=tf.float32)

            predicted_price = self.model_internal(t_int_points, S_int_points)

            plt.plot(S_plot, actual_price, color='r', label='Actual Price', linewidth=1, linestyle=':')
            plt.plot(S_plot, predicted_price, color='c', label='Predicted Price')

            total_actual_price[i * len(S_plot): (i + 1) * len(S_plot)] = actual_price[:, 0]
            total_predicted_price[i * len(S_plot): (i + 1) * len(S_plot)] = predicted_price[:, 0]

            # subplot options
            plt.ylim(ymin=0.0, ymax=self.K)
            plt.xlim(xmin=0.0, xmax=self.S_high)
            plt.xlabel(r"Stock Price", fontsize=15, labelpad=10)
            plt.ylabel(r"Option Price", fontsize=15, labelpad=20)
            plt.title(f"t = {curr_t : .2f}", fontsize=18, y=1.03)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)

            if i == 0:
                plt.legend(loc='upper left', prop={'size': 16})

        # adjust space between subplots
        plt.subplots_adjust(wspace=0.3, hspace=0.4)

        #print(mean_absolute_error(total_predicted_price, total_actual_price))
        print(r2_score(total_predicted_price, total_actual_price))

        if self.save_fig:
            plt.savefig(self.fig_name)


#set_all_seeds(101)
BS = BlackScholes_europeanCall_DGM_v2()
BS.train()
BS.plotting(save_fig=True)
K.clear_session()
