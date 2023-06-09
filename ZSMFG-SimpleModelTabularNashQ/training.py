import time
import numpy as np

from myEnv import my1dGridEnv

from myTable import myQTable



npzfile = np.load("Tabular Q-learning/test10f7a3_paramsBook_from-v3d5_10Pts_gamma0p5_envT1_cont20Pts_contT0p1_cont30pts_contT0p2/test10f7_results_iter10.npz")
Q_prev =      npzfile['Q']


lr = 1.0

env = my1dGridEnv()
table = myQTable()
table.init_states() #initialize the states
table.init_ctrl() #initialize the controls

#Q_old should be table.Q_old if there is no previous Q-table

Q_old = table.Q_old
print(np.shape(Q_old))



iter_save = 5

discount_gamma = 0.5 # for one unit of time
discount_beta = - np.log(discount_gamma)
discount = np.exp(-discount_beta * env.T)
print("discount = {}".format(discount))
print(table.states[0])



N_episodes = 1000



iters = []
Q_diff_sup = []
Q_diff_L2 = []
if __name__ == '__main__':
    for i in range(1, 1+N_episodes):
        print('\n\n======================================== Episode {}\n\n'.format(i))
        Q_new = (1-lr) * Q_old.copy()#First part of Q_new
        for i_mu in range(table.n_states):
            # print("i_mu = {}\n".format(i_mu))

            mu_1 = table.states[i_mu]
            mu_2 = table.states[table.n_states-1-i_mu]
            for i_alpha_1 in range(table.n_controls):

                for i_alpha_2 in range(table.n_controls):

                    # print("i_alpha = {}\n".format(i_alpha))
                    alpha_1 = table.controls[i_alpha_1]
                    alpha_2 = table.controls[i_alpha_2]
                    trans_mat_1 = env.cal_transition_matrix(alpha_1)
                    next_mu_1 = np.inner(mu_1,trans_mat_1)
                    trans_mat_2 = env.cal_transition_matrix(alpha_2)
                    next_mu_2 = np.inner(mu_2,trans_mat_2)

                    
                    i_mu_1_next = table.proj_W_index(next_mu_1) # find its most nearest mu
                    i_mu_2_next = table.proj_W_index(next_mu_2) # find its most nearest mu

                    r_next_1, r_next_2 = env.get_population_level_reward(table.states[i_mu_1_next], table.states[i_mu_2_next])

                    # Nash Q
                    r_matrix_1,r_matrix_2 = env.get_reward_mat(table.states[i_mu_1_next],table.states[i_mu_2_next])

                    pi_1, pi_2 = env.get_nash_Q_value(r_matrix_1,r_matrix_2)
                    #print(Q_old[i_mu_1_next][i_alpha_1][i_alpha_2])
                    Q_nash = np.dot(pi_1,pi_2) * Q_old[i_mu_1_next][i_alpha_1][i_alpha_2]
                    print(r_next_1)
                    print("mu = {},\t mu_next = {}, \t mu_next_proj = {}".format(mu_1, next_mu_1, table.states[i_mu_1_next]))
                    
                    #update the New Q table
                    Q_new[i_mu][i_alpha_1][i_alpha_2] += lr * (r_next_1 + discount * Q_nash)

        # print("np.abs(Q_new - Q_old) = ", np.abs(Q_new - Q_old))

        # Calculate the Q_diff_sup and Q_diff_L2 to see if converges
        iters.append(i)
        Q_diff_sup.append(np.max(np.abs(Q_new - Q_old)))
        print("***** sup|Q_new - Q_old| = {}".format(Q_diff_sup[-1]))
        Q_diff_L2.append(np.sqrt(np.sum(np.square(Q_new - Q_old))))
        print("***** L2|Q_new - Q_old| = {}\n".format(Q_diff_L2[-1]))
        # print("Q_new = {}".format(Q_new))
        #opt_ctrls = table.get_opt_ctrl(Q_new)
        # print("***** opt_ctrls = {}".format(opt_ctrls))
        Q_old = Q_new.copy()
        if (i % iter_save == 0):
            np.savez("results_iter{}".format(i), Q=Q_new, n_states_x=table.n_states_x, n_steps_state=table.n_steps_state, n_steps_ctrl=table.n_steps_ctrl, iters=iters, Q_diff_sup=Q_diff_sup, Q_diff_L2=Q_diff_L2)