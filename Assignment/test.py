import gym


if __name__ == "__main__":

    env = gym.make('BipedalWalker-v3')
    for i_episode in range(20):
        observation = env.reset()
        total = 0
        done = False
        steps = 0
        while not done:
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            #print(reward)
            total += reward
            steps += 1
            if done:
                print(f'Episode finished: {total}, {steps}')
                total = 0
                steps = 0
                break
    env.close()



