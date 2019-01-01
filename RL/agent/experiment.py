import run
import default_params

def main():
    repeat = 3
    avg_rew = 0.0
    episodes = 1000
    sleep = 0.00
    options={
                'target':"LunarLander-v2",#Acrobot-v1 LunarLander-v2 Breakout-v0
                'episodes':episodes,
                'render':True,
                'plot':True,
                'monitor':False,
                'logging':1
            }
    params = default_params.get_default(options['target'])
    params.update(options)
    reward_list = []

    for i in range(repeat):
        reward, allparams, totrewlist, totrewavglist, greedyrewlist, reward_threshold = run.main(100,params,sleep)
        reward_list.append(reward)
        avg_rew+=reward
    avg_rew/=repeat
    print(allparams)
    print("avg_rew",avg_rew)
    print("rew list",reward_list)

if __name__ == '__main__':
    main()