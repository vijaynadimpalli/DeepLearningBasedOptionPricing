from DGM_model import DGM

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

import tensorflow as tf
tf.compat.v1.disable_v2_behavior()

class BlackScholes_europeanCall_DGM:
    """
    This class calculates the Black Scholes European Call Option price using DGM model in Tensorflow 1.0
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
        self.n_resampling_steps = 100  # number of resampling steps
        self.n_optimization_steps = 10  # number of optimization steps before resampling

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


    def _loss(self, model, t_int_points, S_int_points, t_ter_points, S_ter_points):
        """
        Method computes loss function.
        """

        V = model(t_int_points, S_int_points)
        V_t = tf.gradients(ys=V, xs=t_int_points)[0]
        V_s = tf.gradients(ys=V, xs=S_int_points)[0]
        V_ss = tf.gradients(ys=V_s, xs=S_int_points)[0]
        diff_V = V_t + 0.5 * self.sigma ** 2 * S_int_points ** 2 * V_ss + self.r * S_int_points * V_s - self.r * V

        # Loss 1: calculate L2-norm of PDE differential operator
        L1 = tf.reduce_mean(input_tensor=tf.square(diff_V))

        # Loss 3: terminal condition loss
        calc_payoff = model(t_ter_points, S_ter_points)
        L3 = tf.reduce_mean(input_tensor=tf.square(calc_payoff - tf.nn.relu(S_ter_points - self.K))) # Calculates L2 norm between fitted payoff and target

        return L1, L3


    def train(self):


        model = DGM.DGMNet(self.n_nodes, self.n_layers, 1)

        self.t_int_points_tnsr = tf.compat.v1.placeholder(tf.float32, [None, 1])
        self.S_int_points_tnsr = tf.compat.v1.placeholder(tf.float32, [None, 1])
        self.t_ter_points_tnsr = tf.compat.v1.placeholder(tf.float32, [None, 1])
        self.S_ter_points_tnsr = tf.compat.v1.placeholder(tf.float32, [None, 1])

        # loss
        L1_tnsr, L3_tnsr = self._loss(model, self.t_int_points_tnsr, self.S_int_points_tnsr, self.t_ter_points_tnsr, self.S_ter_points_tnsr)
        loss_tnsr = L1_tnsr + L3_tnsr

        self.V = model(self.t_int_points_tnsr, self.S_int_points_tnsr)
        optimizer = tf.compat.v1.train.AdamOptimizer(self.lr).minimize(loss_tnsr)

        init_op = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.Session()
        self.sess.run(init_op)

        # Custom training loop
        for i in range(self.n_resampling_steps):

            # sampling
            t_int_points, S_int_points, t_ter_points, S_ter_points = self._sampler()

            # optimization steps
            for _ in range(self.n_optimization_steps):
                loss, L1, L3, _ = self.sess.run([loss_tnsr, L1_tnsr, L3_tnsr, optimizer],
                                           feed_dict={self.t_int_points_tnsr: t_int_points, self.S_int_points_tnsr: S_int_points,
                                                      self.t_ter_points_tnsr: t_ter_points, self.S_ter_points_tnsr: S_ter_points})

            print(loss, L1, L3, i)
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

            # Inference call on trained model
            predicted_price = self.sess.run(self.V,
                                          feed_dict={self.t_int_points_tnsr: t_plot, self.S_int_points_tnsr: S_plot})

            plt.plot(S_plot, actual_price, color='r', label='Actual Price', linewidth=1, linestyle=':')
            plt.plot(S_plot, predicted_price, color='c', label='Predicted Price')

            total_actual_price[i * len(S_plot): (i+1) * len(S_plot)] = actual_price[:, 0]
            total_predicted_price[i * len(S_plot): (i + 1) * len(S_plot)] = predicted_price[:, 0]


            plt.ylim(ymin=0.0, ymax=self.K)
            plt.xlim(xmin=0.0, xmax=self.S_high)
            plt.xlabel(r"Stock Price", fontsize=15, labelpad=10)
            plt.ylabel(r"Option Price", fontsize=15, labelpad=20)
            plt.title(f"t = {curr_t : .2f}", fontsize=18, y=1.03)
            plt.xticks(fontsize=13)
            plt.yticks(fontsize=13)
            plt.grid()

            if i == 0:
                plt.legend(loc='upper left', prop={'size': 16})


        plt.subplots_adjust(wspace=0.3,hspace=0.4)

        #print(mean_absolute_error(total_predicted_price, total_actual_price))
        print(r2_score(total_predicted_price, total_actual_price))

        if self.save_fig:
            plt.savefig(self.fig_name)

BS = BlackScholes_europeanCall_DGM()
BS.train()
BS.plotting(save_fig=True)