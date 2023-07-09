import os
import time
from .tabularNashQ import tabularNashQ

import numpy as np
import magent
from .env import myBattleGameEnv
from .battleFunctions import play,generate_map

env = magent.GridWorld("battle",map_size=3) #myBattleGameEnv

model = tabularNashQ

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__=="main":

    magent.utility.init__logger("battle")

    # init the directory to save render file
    env.set_render_dir(os.path.join(BASE_DIR,"examples/battle_model","build/render"))

    handles = env.get_handles()# I am not sure about what is the get handles 

    eval_obs = [None, None]
    names = ["battle"+"-l","battle"+"r"]

    savedir = 'save_model'
    print("view_space", env.get_view_space(handles[0]))
    print("feature_space", env.feature_space(handles[0]))

    n_round=1000
    start = time.time()
    for k in range(n_round):
        tic = time.time()
        loss, num, reward, value = play(env,) 




