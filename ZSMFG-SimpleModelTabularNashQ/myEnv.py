import numpy as np
import nashpy as nash


class my1dGridEnv(object):

    def __init__(self,size=10,p=4):
        self.size = size # Dimension of 1D world
        self.n_states = self.size 
        self.n_actions = 3 # [0,1,-1]
        self.epsilon = [None,None,None]
        self.action_space = [0,-1,1]
        self.c = 0.2 # proportion of the density of other population



    def get_index(self,m):

        return m-1

    # recalculate the transition_matrix because every time the agents deploys a different strategy pi
    def cal_transition_matrix(self,pi):

        # pi=[p_stay,p_left,p_right]
        trans_matrix=np.zeros((self.n_states,self.n_states))
        for i in range(1,self.n_states+1):
            if i==0:
                trans_matrix[self.get_index(i),self.get_index(i)]=pi[0]+self.epsilon[0]
                trans_matrix[self.get_index(i),-1]=pi[1]+self.epsilon[1]
                trans_matrix[self.get_index(i),self.get_index(i)+1]=pi[2]+self.epsilon[2]
            elif i==self.n_states-1:
                trans_matrix[self.get_index(i),self.get_index(i)]=pi[0]+self.epsilon[0]
                trans_matrix[self.get_index(i),self.get_index(i)-1]=pi[1]+self.epsilon[1]
                trans_matrix[self.get_index(i),0]=pi[2]+self.epsilon[2]
            else:
                trans_matrix[self.get_index(i),self.get_index(i)]=pi[0]+self.epsilon[0]
                trans_matrix[self.get_index(i),self.get_index(i)-1]=pi[1]+self.epsilon[1]
                trans_matrix[self.get_index(i),self.get_index(i)+1]=pi[2]+self.epsilon[2]
        
        return trans_matrix


    def get_q_t_withActions(self,mu_t,alpha_1,alpha_2):
        q_t = np.zeros((self.n_states,self.n_states))
        for iS in range(self.n_states):
            q_t[iS] = self.cal_transition_matrix(mu_t,alpha_1,alpha_2)

    def get_agent_level_reward(self,state,mu_of_other_population,action,agent1=True):
        if agent1:
            cost = -np.abs(state-self.size)-self.c*mu_of_other_population[state]-np.abs(action)
        else:
            cost = -np.abs(state-1)-self.c*mu_of_other_population[state]-np.abs(action)
        return cost
            


    # visit every action pair of player 1 and player 2 to get reward matrix
    def get_population_level_reward_mat(self,mu_1,mu_2, pi_1, pi_2):
        reward_mat_1=np.zeros(self.n_actions,self.n_actions)
        reward_mat_2=np.zeros(self.n_actions,self.n_actions)
        for i in range(self.n_actions):
            for l in range(self.n_actions):
                for s in range(self.n_states):
                    reward_mat_1[i,l]+=mu_1[s]*pi_1[i]*self.get_agent_level_reward(s,mu_2,self.action_space[l])
                    reward_mat_2[i,l]+=mu_2[s]*pi_2[l]*self.get_agent_level_reward(s,mu_1,self.action_space[l])
        
        return  reward_mat_1, reward_mat_2




    def get_next_mu(self,mu,nash_strategy):
        
        return np.mat(mu,self.cal_transition_matrix(nash_strategy))

        


    def get_nash_Q_value(self,Q_table,reward_mat_1, reward_mat_2):
            # Zero sum case solver to get stage nash eq by lemke-Howson

            rps = nash.Game(reward_mat_1,reward_mat_2)
            #e = rps.support_enumeration()
            nash_pi =  rps.lemke_howson(initial_dropped_label=0)
            i_mu_next=self.compute_next_mu()
            nash_Q_value = np.mat(nash_pi[0],nash_pi[1]) * Q_table[i_mu_next]

            return nash_Q_value