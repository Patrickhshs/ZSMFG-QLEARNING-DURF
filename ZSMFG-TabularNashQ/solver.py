from myEnv import Env
import numpy as np

import nashpy as nash
import numpy as np

# 声明一个零和博弈
A = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
rps = nash.Game(A)

# 用两个博弈矩阵声明一个博弈
B = -A
rps = nash.Game(A, B)

# 计算Nash均衡
e = rps.support_enumeration()  # 返回一个迭代器
list(e)
e = rps.lemke_howson(initial_dropped_label=0)  # 返回一个纳什均衡