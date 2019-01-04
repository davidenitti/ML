import run

def main():
    repeat = 5
    avg_rew = 0.0
    episodes = 100000
    sleep = 0.00
    params = run.getparams(episodes)
    reward_list = []

    for i in range(repeat):
        reward, allparams, totrewlist, totrewavglist, greedyrewlist, reward_threshold = run.main(100,params.copy(),sleep)
        reward_list.append(reward)
        avg_rew+=reward
    avg_rew/=repeat
    print(allparams)
    print("avg_rew",avg_rew)
    print("rew list",reward_list)

if __name__ == '__main__':
    main()