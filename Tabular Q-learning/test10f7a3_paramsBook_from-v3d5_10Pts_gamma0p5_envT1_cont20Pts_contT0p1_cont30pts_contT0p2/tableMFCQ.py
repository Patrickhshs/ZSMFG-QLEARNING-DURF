import itertools
import numpy as np
import my_env

class MyTable():
    def __init__(self,n_steps_state=4,n_states_x=30,environment=None):
        self.n_steps_state=n_steps_state
        self.n_states_x=n_states_x
        self.env=environment

    def init_states(self):
        combi_mu = itertools.product(np.linspace(0,self.n_steps_state,self.n_steps_state+1,dtype=int), repeat=self.n_states_x) #cartesian product; all possible distributions in the discretization of the simplex
        distributions_unnorm = np.asarray([el for el in combi_mu])
        states_tmp = distributions_unnorm.copy()
        self.states = states_tmp[np.where(np.sum(states_tmp, axis=1)==self.n_steps_state)] / float(self.n_steps_state)#shape:(5456,4)
        self.n_states = np.shape(self.states)[0]
        self.n_steps_ctrl = 1
    
    def init_ctrl(self):
        combi_ctrl = itertools.product(np.linspace(0,1,self.n_steps_ctrl+1), repeat=self.n_states_x)#n_states_x) #cartesian product; all possible controls as functions of state_x
        controls = np.asarray([el for el in combi_ctrl]) #np.linspace(0,1,n_steps_ctrl+1)
        print("controls = {}".format(controls))
        self.n_controls = np.shape(controls)[0]
        print('MDP: n states = {}\nn controls = {}'.format(self.n_states, self.n_controls))
        self.Q_old = np.zeros((self.n_states, self.n_controls))
        self.controls=controls
        # Q_old[:,11] = 0.01
        # Q_new = np.zeros((n_states, n_controls))
        print("Q shape = {}".format(np.shape(self.Q_old)))
        

    def proj_W_index(self,mu):#Get the distance of estimate new_mu(let's say) and the existed distribution, select the nearest state.
            

            # print("W mu = {}".format(mu))
            #print("W map = {}".format(map(lambda mu2 : stats.wasserstein_distance(mu,mu2), states)))
            # return np.argmin(map(lambda mu2 : stats.wasserstein_distance(mu,mu2), states))
            return np.argmin(map(lambda mu2 : np.sum(np.abs(mu - mu2)), self.states))



        #N_episodes = 1000 delta t=0.01, truncate at horizon T=10


    def get_opt_ctrl(self,Q_table):
            return [np.argmax(Q_table[i_mu]) for i_mu in range(self.n_states)]