import itertools
import numpy as np
import my_env

class MyTable():
    def __init__(self,n_steps_state=4,n_states_x=30,environment=0):
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

    def proj_W_index(self,mu):
            # print("W mu = {}".format(mu))
            #print("W map = {}".format(map(lambda mu2 : stats.wasserstein_distance(mu,mu2), states)))
            # return np.argmin(map(lambda mu2 : stats.wasserstein_distance(mu,mu2), states))
            return np.argmin(map(lambda mu2 : np.sum(np.abs(mu - mu2)), self.states))



        #N_episodes = 1000 delta t=0.01, truncate at horizon T=10


    def get_opt_ctrl(self,Q_table):
            return [np.argmax(Q_table[i_mu]) for i_mu in range(self.n_states)]