from cmath import nan
import numpy as np
import nashpy as nash
from scipy.optimize import linprog


class my1dGridEnv(object):

    def __init__(self,size=3):
        self.size = size # Dimension of 1D world
        self.n_states = self.size 
        self.n_actions = 3 
        self.epsilon = [0.5,0.25,0.25] # [0,-1,1]
        self.action_space = [0,-1,1]
        self.c = 10 # proportion of the density of other population
        self.T = 0.2



    def get_index(self,m):

        return m

    # recalculate the transition_matrix because every time the agents deploys a different strategy pi
    def cal_transition_matrix(self,pi):

        # pi=[p_stay,p_left,p_right]
        trans_matrix=np.zeros((self.n_states,self.n_states))
        epsilon_matrix=np.zeros((self.n_states,self.n_states))

        for i in range(self.n_states):
            if i==0:
                trans_matrix[self.get_index(i),self.get_index(i)]=pi[0]
                epsilon_matrix[self.get_index(i),self.get_index(i)]=self.epsilon[0]
                trans_matrix[self.get_index(i),-1]=pi[1]
                epsilon_matrix[self.get_index(i),-1]=self.epsilon[1]
                trans_matrix[self.get_index(i),self.get_index(i)+1]=pi[2]
                epsilon_matrix[self.get_index(i),self.get_index(i)+1]=self.epsilon[2]
            elif i==self.n_states-1:
                trans_matrix[self.get_index(i),self.get_index(i)]=pi[0]
                epsilon_matrix[self.get_index(i),self.get_index(i)]=self.epsilon[0]
                trans_matrix[self.get_index(i),self.get_index(i)-1]=pi[1]
                epsilon_matrix[self.get_index(i),self.get_index(i)-1]=self.epsilon[1]
                trans_matrix[self.get_index(i),0]=pi[2]
                epsilon_matrix[self.get_index(i),0]=self.epsilon[2]
            else:
                trans_matrix[self.get_index(i),self.get_index(i)]=pi[0]
                epsilon_matrix[self.get_index(i),self.get_index(i)]=self.epsilon[0]
                trans_matrix[self.get_index(i),self.get_index(i)-1]=pi[1]
                epsilon_matrix[self.get_index(i),self.get_index(i)-1]=self.epsilon[1]
                trans_matrix[self.get_index(i),self.get_index(i)+1]=pi[2]
                epsilon_matrix[self.get_index(i),self.get_index(i)+1]=self.epsilon[2]
        
        return trans_matrix,epsilon_matrix





    # def get_agent_level_reward(self,state,mu_of_other_population,agent1=True):
    #     if agent1:
    #         cost = self.c*mu_of_other_population[state]

    #     return cost
            


    # visit every action pair of player 1 and player 2 to get reward matrix
    def get_population_level_reward(self,mu_1,mu_2):
        reward_1 = self.c*(np.dot(mu_1,mu_2.T))
        reward_2 = -self.c*(np.dot(mu_1,mu_2.T))
        
        
        return  reward_1, reward_2

    # def get_reward_mat(self,mu_1,mu_2,table):
    #     reward_mat_1 = np.zeros((table.n_controls,table.n_controls))
    #     reward_mat_2 = np.zeros((table.n_controls,table.n_controls))


    #     simplex_controls = table.controls
    #     for i in range(len(simplex_controls)):
    #         for l in range(len(simplex_controls)):
    #             next_mu_1 = self.get_next_mu(mu_1,simplex_controls[i])
    #             next_mu_2 = self.get_next_mu(mu_2,simplex_controls[l])
    #             i_next_1 = table.proj_W_index(next_mu_1)
    #             i_next_2 = table.proj_W_index(next_mu_2)

    #             reward_1, reward_2 = self.get_population_level_reward(table.states[i_next_1], table.states[i_next_2])
    #             reward_mat_1[i][l] = reward_1
    #             reward_mat_2[i][l] = reward_2


        
    #     return  reward_mat_1, reward_mat_2




    def get_next_mu(self,mu,strategy):
        transi_mat,epi_mat=self.cal_transition_matrix(strategy)
        P=np.dot(transi_mat,epi_mat)

        return np.dot(P,mu)

        


    def get_nash_Q_value(self,payoff_mat):
            # Zero sum case solver to get stage nash eq by lemke-Howson
            
            game = nash.Game(payoff_mat)
            #e = rps.support_enumeration()
            equi = game.lemke_howson_enumeration()
            for i in equi:
                if np.sum(i[0])>=0.99999:
                    pi_1=i[0]
                    return np.array(pi_1)

            #nash_pi =  rps.lemke_howson(initial_dropped_label=0)
            #equilibria = rps.lemke_howson_enumeration() 

            # return self.lemke_howson(reward_mat_1,reward_mat_2)

    # def linear_programming_duality(self,payoff_matrix_A, payoff_matrix_B):
    #     m, n = payoff_matrix_A.shape

    #     # Create the dual linear programming problems for both players
    #     c_A = np.ones(m)
    #     A_ub_A = -payoff_matrix_A.T
    #     b_ub_A = -np.ones(n)

    #     c_B = -np.ones(n)
    #     A_ub_B = payoff_matrix_B
    #     b_ub_B = np.ones(m)

    #     # Solve the dual linear programming problems using linprog
    #     res_A = linprog(c_A, A_ub=A_ub_A, b_ub=b_ub_A,method='revised simplex')
    #     res_B = linprog(c_B, A_ub=A_ub_B, b_ub=b_ub_B,method='revised simplex')
    #     print(res_B)
    #     # Extract the Nash equilibrium strategies from the results
    #     mixed_strategy_A = res_A.x / np.sum(res_A.x)
    #     mixed_strategy_B = res_B.x / np.sum(res_B.x)

    #     return mixed_strategy_A, mixed_strategy_B