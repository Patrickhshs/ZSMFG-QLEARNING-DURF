import numpy as np
#from battleGame.magent2.environments import battle_v4
from battleGame.magent2.gridworld import GridWorld

class myBattleGameEnv(object):

    def __init__(self,size,recovery_rate,dead_penalty=-10,attack_penalty=-1,attack_antagonist_reward=1,kill_antagonist_reward=10):
        self.size = size
        self.rec_rate = recovery_rate
        self.dead_penalty = dead_penalty
        self.att_penalty = attack_penalty
        self.att_reward = attack_antagonist_reward
        self.kill_reward = kill_antagonist_reward

        # To-do:
        # define \pi since \alpha ~ \pi
        # self.prob_go_left = 0.2
        # self.prob_go_right = 0.2
        # self.prob_go_up = 0.2
        # self.prob_go_down = 0.2
        # self.prob_attack = 0.2


        self.n_states = self.size*self.size 
        self.env = GridWorld('battle')


    def get_index(self,m,n):
        index=self.size*m+n
        
        return index

    def get_transition_matrix(self,mu_t,pi):

        trans_matrix=np.zeros((self.n_states,self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_states):
                trans_matrix[self.get_index(i,j),self.get_index(i,j)]

        # set diagonal with p_attack
        for i in range(self.n_states):
            trans_matrix[self.get_index(i,j),self.get_index(i,j)] = pi[-1]


    def get_q_t_withActions(self,mu_t,alpha_1,alpha_2):
        pass

    def get_mu_and_reward():
        pass

    