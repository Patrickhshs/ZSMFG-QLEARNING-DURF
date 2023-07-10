import time
import numpy as np

from .myEnv import my1dGridEnv

from .myTable import myQTable



npzfile = np.load("Tabular Q-learning/test10f7a3_paramsBook_from-v3d5_10Pts_gamma0p5_envT1_cont20Pts_contT0p1_cont30pts_contT0p2/test10f7_results_iter10.npz")
Q_prev =             npzfile['Q']