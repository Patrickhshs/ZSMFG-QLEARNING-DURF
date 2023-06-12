import battle_v4 from magent2.environment 

myEnv=battle_v4.env(
    map_size=16,minimap_mode=False,step_reward=-0.005,dead_penalty=-0.1,
    attack_penalty=-0.1,
    attack_opponent_reward=0.2,max_cycles=1000,extra_features=False
)

if __name__=="__main__":
    myEnv.set_render_dir("render")
    
