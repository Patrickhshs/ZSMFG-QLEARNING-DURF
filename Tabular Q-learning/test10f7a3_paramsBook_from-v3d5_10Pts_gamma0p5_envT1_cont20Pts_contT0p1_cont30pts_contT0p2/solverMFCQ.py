import numpy as np
import time
import itertools
import scipy
import scipy.stats as stats

import my_env




#### LOAD PREVIOUS
npzfile = np.load("test10f7a3_paramsBook_from-v3d5_10Pts_gamma0p5_envT1_cont20Pts_contT0p1_cont30pts_contT0p2/test10f7_results_iter10.npz")
Q_prev =             npzfile['Q']
# n_states_x_prev =    npzfile['n_states_x']
# n_steps_state_prev = npzfile['n_steps_state']
# n_steps_ctrl_prev =  npzfile['n_steps_ctrl']

# combi_mu_prev = itertools.product(np.linspace(0,n_steps_state_prev,n_steps_state_prev+1,dtype=int), repeat=n_states_x_prev) # all possible distributions in the discretization of the simplex
# distributions_unnorm_prev = np.asarray([el for el in combi_mu_prev])
# states_tmp_prev = distributions_unnorm_prev.copy()
# states_prev = states_tmp_prev[np.where(np.sum(states_tmp_prev, axis=1)==n_steps_state_prev)] / float(n_steps_state_prev)
# n_states_prev = np.shape(states_prev)[0]
# combi_ctrl_prev = itertools.product(np.linspace(0,1,n_steps_ctrl_prev+1), repeat=n_states_x_prev)#n_states_x) # all possible controls as functions of state_x
# controls_prev = np.asarray([el for el in combi_ctrl_prev]) #np.linspace(0,1,n_steps_ctrl+1)
# # print("controls = {}".format(controls))
# n_controls_prev = np.shape(controls_prev)[0]


# NEW


n_states_x = 4
n_steps_state = 30

combi_mu = itertools.product(np.linspace(0,n_steps_state,n_steps_state+1,dtype=int), repeat=n_states_x) #cartesian product; all possible distributions in the discretization of the simplex
distributions_unnorm = np.asarray([el for el in combi_mu])
print(distributions_unnorm)
states_tmp = distributions_unnorm.copy()
states = states_tmp[np.where(np.sum(states_tmp, axis=1)==n_steps_state)] / float(n_steps_state)#(5456,4)
print(np.shape(states))
n_states = np.shape(states)[0]
n_steps_ctrl = 1
combi_ctrl = itertools.product(np.linspace(0,1,n_steps_ctrl+1), repeat=n_states_x)#n_states_x) #cartesian product; all possible controls as functions of state_x
controls = np.asarray([el for el in combi_ctrl]) #np.linspace(0,1,n_steps_ctrl+1)
print("controls = {}".format(controls))
n_controls = np.shape(controls)[0]
print('MDP: n states = {}\nn controls = {}'.format(n_states, n_controls))
Q_old = np.zeros((n_states, n_controls))
# Q_old[:,11] = 0.01
# Q_new = np.zeros((n_states, n_controls))
print("Q shape = {}".format(np.shape(Q_old)))
lr = 1.0


# INITIALIZE WITH PREVIOUS MATRIX
# for i_mu in range(n_states):
#     mu = states[i_mu]
#     i_mu_proj = np.argmin(map(lambda mu2 : np.sum(np.abs(mu - mu2)), states_prev)) # proj on previous set of states
#     Q_old[i_mu] = Q_prev[i_mu_proj]
Q_old = Q_prev.copy()




iter_save = 5
env = my_env.MyEnvKFPCyberSecurity()

discount_gamma = 0.5 # for one unit of time
discount_beta = - np.log(discount_gamma)
discount = np.exp(-discount_beta * env.T)
print("discount = {}".format(discount))

def proj_W_index(mu):
    # print("W mu = {}".format(mu))
    #print("W map = {}".format(map(lambda mu2 : stats.wasserstein_distance(mu,mu2), states)))
    # return np.argmin(map(lambda mu2 : stats.wasserstein_distance(mu,mu2), states))
    return np.argmin(map(lambda mu2 : np.sum(np.abs(mu - mu2)), states))



N_episodes = 1000


def get_opt_ctrl(Q_table):
    return [np.argmax(Q_table[i_mu]) for i_mu in range(n_states)]

iters = []
Q_diff_sup = []
Q_diff_L2 = []
if __name__ == '__main__':
    for i in range(1, 1+N_episodes):
        print('\n\n======================================== Episode {}\n\n'.format(i))
        Q_new = (1-lr) * Q_old.copy()#First part of Q_new
        for i_mu in range(n_states):
            # print("i_mu = {}\n".format(i_mu))
            mu = states[i_mu]
            for i_alpha in range(n_controls):
                # print("i_alpha = {}\n".format(i_alpha))
                alpha = controls[i_alpha]
                mu_next, r_next = env.get_mu_and_reward(mu, alpha)
                i_mu_next = proj_W_index(mu_next)
                # print("mu = {},\t mu_next = {}, \t mu_next_proj = {}".format(mu, mu_next, states[i_mu_next]))
                Q_opt = np.max(Q_old[i_mu_next])
                Q_new[i_mu, i_alpha] += lr * (r_next + discount * Q_opt)
        # print("np.abs(Q_new - Q_old) = ", np.abs(Q_new - Q_old))
        iters.append(i)
        Q_diff_sup.append(np.max(np.abs(Q_new - Q_old)))
        print("***** sup|Q_new - Q_old| = {}".format(Q_diff_sup[-1]))
        Q_diff_L2.append(np.sqrt(np.sum(np.square(Q_new - Q_old))))
        print("***** L2|Q_new - Q_old| = {}\n".format(Q_diff_L2[-1]))
        # print("Q_new = {}".format(Q_new))
        opt_ctrls = get_opt_ctrl(Q_new)
        # print("***** opt_ctrls = {}".format(opt_ctrls))
        Q_old = Q_new.copy()
        if (i % iter_save == 0):
            np.savez("results_iter{}".format(i), Q=Q_new, n_states_x=n_states_x, n_steps_state=n_steps_state, n_steps_ctrl=n_steps_ctrl, iters=iters, Q_diff_sup=Q_diff_sup, Q_diff_L2=Q_diff_L2)
