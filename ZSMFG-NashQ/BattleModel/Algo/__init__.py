from . import nash_q_learning

nashQ=q_learning.nashQ

def spawn_ai(algo_name,sess,env,handle,human_name,max_steps):
    if algo_name=="nashQ":
        model=nashQ(sess,human_name,handle,env,max_steps,memory_size=80000)
    # ...

    return model