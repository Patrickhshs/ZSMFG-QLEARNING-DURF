import numpy as np
import nashpy as nash
import pulp
from scipy.optimize import linprog
from numba import njit


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
        trans_matrix = np.zeros((self.n_states,self.n_states))
        epsilon_matrix = np.zeros((self.n_states,self.n_states))

        for i in range(self.n_states):
            if i==0:
                trans_matrix[self.get_index(i),self.get_index(i)] = pi[0]
                epsilon_matrix[self.get_index(i),self.get_index(i)] = self.epsilon[0]
                trans_matrix[self.get_index(i),-1] = pi[1]
                epsilon_matrix[self.get_index(i),-1] = self.epsilon[1]
                trans_matrix[self.get_index(i),self.get_index(i)+1] = pi[2]
                epsilon_matrix[self.get_index(i),self.get_index(i)+1] = self.epsilon[2]
            elif i==self.n_states-1:
                trans_matrix[self.get_index(i),self.get_index(i)] = pi[0]
                epsilon_matrix[self.get_index(i),self.get_index(i)] = self.epsilon[0]
                trans_matrix[self.get_index(i),self.get_index(i)-1] = pi[1]
                epsilon_matrix[self.get_index(i),self.get_index(i)-1] = self.epsilon[1]
                trans_matrix[self.get_index(i),0] = pi[2]
                epsilon_matrix[self.get_index(i),0] = self.epsilon[2]
            else:
                trans_matrix[self.get_index(i),self.get_index(i)] = pi[0]
                epsilon_matrix[self.get_index(i),self.get_index(i)] = self.epsilon[0]
                trans_matrix[self.get_index(i),self.get_index(i)-1] = pi[1]
                epsilon_matrix[self.get_index(i),self.get_index(i)-1] = self.epsilon[1]
                trans_matrix[self.get_index(i),self.get_index(i)+1] = pi[2]
                epsilon_matrix[self.get_index(i),self.get_index(i)+1] = self.epsilon[2]
        
        return trans_matrix, epsilon_matrix





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
        transi_mat,epi_mat = self.cal_transition_matrix(strategy)
        P = np.dot(transi_mat,epi_mat)

        return np.dot(P,mu)


    #@njit
    # def compute_equilibria(self,payoff_mat_1, payoff_mat_2):
    #     # Enumerate equilibria
    #     game = nash.Game(payoff_mat_1, payoff_mat_2)

    #     equilibria = list(game.support_enumeration())
        
    #     # Preallocate array
    #     pis = np.empty((len(equilibria), 2))
        
    #     # Loop to extract equilibria
    #     for i, pi in enumerate(equilibria):
    #         pis[i,0] = pi[0]
    #         pis[i,1] = pi[1]

            
    #     return pis
    # #@njit
    # def get_nash_Q_value(self,payoff_mat_1, payoff_mat_2):

    #     n_actions = payoff_mat_1.shape[0]

    #     # Use Numba compiled function
    #     pis = self.compute_equilibria(payoff_mat_1, payoff_mat_2)  

    #     # Vectorized validation
    #     valid = (pis[:,0].shape == (n_actions,) &
    #             pis[:,1].shape == (n_actions,) &
    #             ~np.any(np.isnan(pis[:,0]), axis=1) &
    #             ~np.any(np.isnan(pis[:,1]), axis=1))
                
    #     pi = pis[valid][0]
                
    #     if pi is None:
    #         pi = (np.ones(n_actions) / n_actions, 
    #             np.ones(n_actions) / n_actions)
                
    #     return pi[0],pi[1]
        


    def get_nash_Q_value(self,payoff_mat_1,payoff_mat_2,table):
            # Zero sum case solver to get stage nash eq by lemke-Howson
            
            game = nash.Game(payoff_mat_1, payoff_mat_2)

            #equilibria = game.support_enumeration()
            equilibria = game.lemke_howson_enumeration()
            # #equilibria = game.vertex_enumeration()
            
            pi = None
            for _pi in equilibria:
                if _pi[0].shape == (self.n_actions, ) and _pi[1].shape == (
                        self.n_actions, ):
                    if any(
                        np.isnan(
                            _pi[0])) is False and any(
                        np.isnan(
                            _pi[1])) is False:
                        pi = _pi
                        break

            if pi is None:
                # pi1 = np.repeat(
                #     1.0 / table.n_controls, table.n_controls)
                # pi2 = np.repeat(
                #     1.0 / table.n_controls, table.n_controls)
                strategy = np.random.uniform(0, 1, table.n_controls)
                pi1 = strategy / np.sum(strategy)
                strategy = np.random.uniform(0, 1, table.n_controls)
                pi2 = strategy / np.sum(strategy)

                pi = (pi1, pi2)

            return pi[0], pi[1]   
            # pi = game.lemke_howson(initial_dropped_label=0)
            
            # return pi[0],pi[1]

    def linear_programming_duality(self,payoff_matrix_A, payoff_matrix_B):
        m, n = payoff_matrix_A.shape

        # Create the dual linear programming problems for both players
        c_A = -np.ones(m)
        A_ub_A = payoff_matrix_A.T
        b_ub_A = np.ones(n)

        c_B = np.ones(n)
        A_ub_B = -payoff_matrix_B
        b_ub_B = -np.ones(m)

        # Solve the dual linear programming problems using linprog
        res_A = linprog(c_A, A_ub=A_ub_A, b_ub=b_ub_A)
        res_B = linprog(c_B, A_ub=A_ub_B, b_ub=b_ub_B)
        
        # Extract the Nash equilibrium strategies from the results
        mixed_strategy_A = res_A.x / np.sum(res_A.x)
        if res_B.success:
            mixed_strategy_B = res_B.x / np.sum(res_B.x)
            
        else:
            strategy = np.random.uniform(0, 1, n)
            mixed_strategy_B = strategy / np.sum(strategy)
            
        
        
        return mixed_strategy_A, mixed_strategy_B
    
    def compute_nash_equilibrium(self,A1, A2):

        # A1 and A2 are payoff matrices for player 1 and 2
        
        m, n = A1.shape # m strategies for player 1, n for player 2
        lp = pulp.LpProblem()

        x = pulp.LpVariable.dicts("x", range(m), lowBound=0, upBound=1, cat='Continuous')
        y = pulp.LpVariable.dicts("y", range(n), lowBound=0, upBound=1, cat='Continuous')
        x_eq = pulp.LpVariable.dicts("x_eq", range(m), lowBound=0, upBound=1, cat='Continuous')
        y_eq = pulp.LpVariable.dicts("y_eq", range(n), lowBound=0, upBound=1, cat='Continuous')
        # Equilibrium payoffs
        V1_eq = pulp.LpVariable("V1_eq", lowBound=0, cat='Continuous')
        V2_eq = pulp.LpVariable("V2_eq", lowBound=None, cat='Continuous')
        # Player 1 constraints
        lp += pulp.lpDot(x_eq, pulp.lpDot(A1, y_eq)) >= V1_eq, "player1_eq_cond"
        lp += pulp.lpDot(x, pulp.lpDot(A1, y_eq)) <= V1_eq, f"player1_cond" 
        # Player 2 constraints  
        lp += pulp.lpDot(x_eq, pulp.lpDot(A2, y_eq)) >= V2_eq, "player2_eq_cond"
        lp += pulp.lpDot(x_eq, pulp.lpDot(A2, y)) <= V2_eq, f"player2_cond"
        
        # sum should be 1 to be considered as  
        lp += pulp.lpSum([x_eq[i] for i in range(m)]) == 1, "sum_x_eq_unity"
        lp += pulp.lpSum([y_eq[j] for j in range(n)]) == 1, "sum_y_eq_unity"

        # Objective
        lp.setObjective(V1_eq + V2_eq)
        lp.solve()

        x_values = [x_eq[i].varValue for i in range(m)]
        y_values = [y_eq[j].varValue for j in range(n)]

        return x_values, y_values #, pulp.value(lp.objective)