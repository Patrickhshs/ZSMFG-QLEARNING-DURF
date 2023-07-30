import numpy as np
import itertools



class myQTable():
    
        def __init__(self,n_states_x=2,n_steps_state=10,environment=None):
            self.n_states_x=n_states_x
            self.n_steps_state=n_steps_state
            self.environment=environment
            self.n_steps_ctrl = 10

        def init_states(self):
            combi_mu = itertools.product(np.linspace(0,self.n_steps_state,self.n_steps_state+1,dtype=int), repeat=self.n_states_x) #cartesian product; all possible distributions in the discretization of the simplex
            distributions_unnorm = np.asarray([el for el in combi_mu])
            states_tmp = distributions_unnorm.copy()
            self.states = states_tmp[np.where(np.sum(states_tmp, axis=1) == self.n_steps_state)] / float(self.n_steps_state)#shape:(5456,4)
            self.n_states = np.shape(self.states)[0]
             # as we have 

        def init_ctrl(self):
            combi_ctrl = itertools.product(np.linspace(0,self.n_steps_ctrl,self.n_steps_ctrl), repeat=self.n_states_x)# n_states_x) # cartesian product; all possible controls as functions of state_x
            controls = np.asarray([el for el in combi_ctrl]) # np.linspace(0,1,n_steps_ctrl+1)
            print("controls = {}".format(controls))
            self.n_controls = np.shape(controls)[0]
            print('MDP: n states = {}\nn controls = {}'.format(self.n_states, self.n_controls))
            self.Q_old = np.zeros((self.n_states, self.n_controls ,self.n_controls)) # shape:(state,action_1,action_2)
            self.controls = controls
            # Q_old[:,11] = 0.01
            # Q_new = np.zeros((n_states, n_controls))v 
            print("Q shape = {}".format(np.shape(self.Q_old)))
        
        def proj_W_index(self,mu):
            
            return np.argmin(map(lambda mu2: np.sum(np.abs(mu - mu2)), self.states))






