import nashpy as nash
import numpy as np
import tqdm

class NashQPlayer():

    def __init__(self, env,tables,
                learning_rate = 0.5,
                iterations = 20,
                discount_factor = 0.7,
                decision_strategy = "epsilong-greedy",
                epsilon = 0.5,
                
                ):
        self.env=env
        self.lr = learning_rate
        self.max_itrs = iterations
        self.disct_fct = discount_factor
        self.decision_strategy = decision_strategy
        self.epsilon = epsilon
        self.table = tables
    
    def training(self):
        self.table.init_states() #initialize the states
        self.table.init_ctrl()

        for i in tqdm(range(self.max_itrs)):
            if self.decision_strategy == "random":

            if self.decision_strategy == "epsilon-greedy":

            if self.decision_strategy == "greedy":

    
    def get_best_policy(self,Q_1, Q_2,env):

