import numpy as np
from myTable import myQTable

from env import myBattleGameEnv

env=myBattleGameEnv
table=myQTable(n_states_x=9,n_steps_state=30,environment=env)

table.init_states()#initialize the states
table.init_ctrl()#initialize the controls

#Q_old should be table.Q_old if there is no previous Q-table

Q_old = table.init_states()#initialize the states
table.init_ctrl()#initialize the controls

#Q_old should be table.Q_old if there is no previous Q-table

Q_old = table.Q_old.copy()
print(np.shape(Q_old))



iter_save = 5

discount_gamma = 0.5 # for one unit of time
discount_beta = - np.log(discount_gamma)
discount = np.exp(-discount_beta * env.T)
print("discount = {}".format(discount))



N_episodes = 1000



iters = []
Q_diff_sup = []
Q_diff_L2 = []
print(np.shape(Q_old))



iter_save = 5

discount_gamma = 0.5 # for one unit of time
discount_beta = - np.log(discount_gamma)
discount = np.exp(-discount_beta * env.T)
print("discount = {}".format(discount))



N_episodes = 1000



iters = []
Q_diff_sup = []
Q_diff_L2 = []

lr=0.8



## Following is for Tabular Nash Q learning
if __name__ == "__main__":
    for i in range(1,1+N_episodes):
        print('\n============Episode {}\n'.format(i))
        Q_new = (1-lr)*Q_old.copy()
        for i_mu in range(n_states):
            mu = states[i_mu]
            for i_action_1 in range(n_actions):

                for i_action_2 in range(n_actions):

                    action_1 = actions[i_action]
                    action_2 = actions[...]
                    # To-Do:
                    # Define get_mu_and_reward
                    mu_next, r_next = env.get_mu_and_reward(mu,action_1,action_2)
                    i_mu_next = env.Proj_Mu_next(mu_next)
                    Nash_Q_value = table.get_nash_Q_value(Q_old,r_next)
                    Q_new[i_mu][i_action_1,i_action_2] += lr * (r_next + discount * Nash_Q_value)

            # iterates +1
            