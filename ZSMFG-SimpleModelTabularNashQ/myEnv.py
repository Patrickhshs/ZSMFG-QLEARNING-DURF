import numpy as np
import nashpy as nash


class my1dGridEnv(object):

    def __init__(self,size=5,p=4):
        self.size = size # Dimension of 1D world
        self.n_states = self.size 
        self.n_actions = 3 # [0,1,-1]
        self.epsilon = [0.5,0.25,0.25]
        self.action_space = [0,-1,1]
        self.c = 0.2 # proportion of the density of other population
        self.T = 0.2



    def get_index(self,m):

        return m

    # recalculate the transition_matrix because every time the agents deploys a different strategy pi
    def cal_transition_matrix(self,pi):

        # pi=[p_stay,p_left,p_right]
        trans_matrix=np.zeros((self.n_states,self.n_states))
        for i in range(self.n_states):
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



    def get_agent_level_reward(self,state,mu_of_other_population,agent1=True):
        if agent1:
            cost = self.c*mu_of_other_population[state]

        return cost
            


    # visit every action pair of player 1 and player 2 to get reward matrix
    def get_population_level_reward(self,mu_1,mu_2):
        reward_1 = 0
        for s in range(self.n_states):
            reward_1 += mu_1[s]*self.get_agent_level_reward(s,mu_2)
            reward_2 += -mu_2[s]*self.get_agent_level_reward(s,mu_1)
        
        return  reward_1, reward_2

    def get_reward_mat(self,mu_1,mu_2):
        reward_mat_1=np.zeros((self.n_actions,self.n_actions))
        reward_mat_2=np.zeros((self.n_actions,self.n_actions))


        naive_action =[[1,0,0],[0,1,0],[0,0,1]]
        for i in range(len(naive_action)):
            for l in range(len(naive_action)):
                next_mu_1 = np.mat(mu_1,self.cal_transition_matrix(naive_action[i]))
                next_mu_2 = np.mat(mu_2,self.cal_transition_matrix(naive_action[l]))
                reward_1, reward_2 = self.get_population_level_reward(next_mu_1, next_mu_2)
                reward_mat_1[i][l] = reward_1
                reward_mat_2[i][l] = reward_2


        
        return  reward_mat_1, reward_mat_2




    def get_next_mu(self,mu,nash_strategy):
        
        return np.mat(mu,self.cal_transition_matrix(nash_strategy))

        


    def get_nash_Q_value(self,reward_mat_1, reward_mat_2):
            # Zero sum case solver to get stage nash eq by lemke-Howson

            rps = nash.Game(reward_mat_1,reward_mat_2)
            #e = rps.support_enumeration()
            nash_pi =  rps.lemke_howson(initial_dropped_label=0)
            

            return nash_pi[0], nash_pi[1]