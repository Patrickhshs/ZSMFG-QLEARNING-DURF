import numpy as np
#from battleGame.magent2.environments import battle_v4
import nashpy as nash
from MAgent.python.magent.gridworld import GridWorld

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
        self.n_actions = self.env.get_action_space()



    def get_index(self,m,n):
        index=self.size*m+n
        
        return index

    # recalculate the transition_matrix because every time the agents deploys a different strategy pi
    def cal_transition_matrix(self,mu_t,pi):

        trans_matrix=np.zeros((self.n_states,self.n_states))
        for i in range(self.n_states):
            for j in range(self.n_states):
                trans_matrix[self.get_index(i,j),self.get_index(i,j)]

        # set diagonal with p_attack
        for i in range(self.n_states):
            trans_matrix[self.get_index(i,j),self.get_index(i,j)] = pi[-1]


    def get_q_t_withActions(self,mu_t,alpha_1,alpha_2):
        q_t = np.zeros((self.n_states,self.n_states))
        for iS in range(self.n_states):
            q_t[iS] = self.cal_transition_matrix(mu_t,alpha_1,alpha_2)


    # visit every action pair of player 1 and player 2 to get reward matrix
    def get_reward_mat(self):
        reward_mat=np.zeros(self.n_actions,self.n_actions)


    def get_next_mu(self):
        pass


    def get_nash_Q_value(self,Q_table,reward_matrix):
            # Zero sum case solver to get stage nash eq by lemke-Howson

            A = reward_matrix #reward of current agent
            B = -A # reward of antagonist 
            rps = nash.Game(A,B)
            #e = rps.support_enumeration()
            nash_pi =  rps.lemke_howson(initial_dropped_label=0)
            i_mu_next=self.compute_next_mu()
            nash_Q_value = np.mat(nash_pi[0],nash_pi[1]) * Q_table[i_mu_next]

            return nash_Q_value

    