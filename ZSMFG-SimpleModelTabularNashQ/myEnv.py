import numpy as np
import nashpy as nash


class my1dGridEnv(object):

    def __init__(self,size=4):
        self.size = size # Dimension of 1D world
        self.n_states = self.size 
        self.n_actions = 3 
        self.epsilon = [0.5,0.25,0.25] # [0,-1,1]
        self.action_space = [0,-1,1]
        self.c = 0.2 # proportion of the density of other population
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
        reward_1 = self.c*(np.dot(mu_1.T,mu_2))
        reward_2 = -self.c*(np.dot(mu_1.T,mu_2))
        
        
        return  reward_1, reward_2

    def get_reward_mat(self,mu_1,mu_2,table):
        reward_mat_1 = np.zeros((table.n_controls,table.n_controls))
        reward_mat_2 = np.zeros((table.n_controls,table.n_controls))


        simplex_controls = table.controls
        for i in range(len(simplex_controls)):
            for l in range(len(simplex_controls)):
                next_mu_1 = self.get_next_mu(mu_1,simplex_controls[i])
                next_mu_2 = self.get_next_mu(mu_2,simplex_controls[l])
                i_next_1 = table.proj_W_index(next_mu_1)
                i_next_2 = table.proj_W_index(next_mu_2)

                reward_1, reward_2 = self.get_population_level_reward(table.states[i_next_1], table.states[i_next_2])
                reward_mat_1[i][l] = reward_1
                reward_mat_2[i][l] = reward_2


        
        return  reward_mat_1, reward_mat_2




    def get_next_mu(self,mu,nash_strategy):
        transi_mat,epi_mat=self.cal_transition_matrix(nash_strategy)
        P=np.dot(transi_mat,epi_mat)

        return np.dot(P,mu)

        


    def get_nash_Q_value(self,reward_mat_1,reward_mat_2):
            # Zero sum case solver to get stage nash eq by lemke-Howson

            rps = nash.Game(reward_mat_1,reward_mat_2)
            #e = rps.support_enumeration()
            nash_pi =  rps.lemke_howson(initial_dropped_label=0)
            

            return nash_pi[0], nash_pi[1]
            # return self.lemke_howson(reward_mat_1,reward_mat_2)

    def lemke_howson(self,A, B):
        m, n = A.shape
        if m != n:
            raise ValueError("Input matrices A and B must be square matrices.")

        tableau_size = m + n + 1
        tableau = np.zeros((tableau_size, tableau_size))
        tableau[:-1, :-1] = np.vstack((np.hstack((B, -np.eye(n))), np.hstack((-A.T, np.eye(m)))))
        tableau[:-1, -1] = np.ones(m + n)
        basis = np.arange(n, n + m)

        try:
            while True:
                pivot_col = np.argmax(tableau[-1, :-1])
                if tableau[-1, pivot_col] <= 0:
                    break

                pivot_row = -1
                min_ratio = np.inf
                for i in range(tableau_size - 1):
                    if tableau[i, pivot_col] > 0 and tableau[i, -1] / tableau[i, pivot_col] < min_ratio:
                        min_ratio = tableau[i, -1] / tableau[i, pivot_col]
                        pivot_row = i

                if pivot_row == -1:
                    raise ValueError("No pivot row found")

                basis[pivot_row] = pivot_col
                tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]

                for i in range(tableau_size):
                    if i != pivot_row:
                        tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

            x = np.zeros(n)
            for i in range(tableau_size - 1):
                if basis[i] < n:
                    x[basis[i]] = tableau[i, -1]

            y = np.zeros(m)
            for i in range(tableau_size - 1):
                if basis[i] >= n:
                    y[basis[i] - n] = tableau[i, -1]

            return x, y

        except ValueError as e:
            print("Lemke-Howson algorithm failed to find a Nash equilibrium.")
            return None, None