import time

import GameEnv2
import pygame
import numpy as np

from ddqn_keras import DDQNAgent

from collections import deque
import random, math

TOTAL_GAMETIME = 100000
N_EPISODES = 1000
REPLACE_TARGET = 10

game = GameEnv2.RacingEnv()
game.fps = 60

GameTime = 0
GameHistory = []
renderFlag = True

ddqn_agent = DDQNAgent(alpha=0.0005, gamma=0.99, n_actions=5, epsilon=0.02, epsilon_end=0.01, epsilon_dec=0.999,
                       replace_target=REPLACE_TARGET, batch_size=64, input_dims=19, fname='ddqn_model.h5')

ddqn_agent.load_model()
ddqn_agent.update_network_parameters()

ddqn_scores = []
eps_history = []


def determine_motor_action(angle):
    # Define the motor action based on angle sectors
    if 0 <= angle < 45:
        return "Motor Action 1"
    elif 45 <= angle < 90:
        return "Motor Action 2"
    elif 90 <= angle < 135:
        return "Motor Action 3"
    # Add more conditions as needed
    else:
        return "Default Motor Action"


def run():
    scores = deque(maxlen=100)

    for e in range(N_EPISODES):
        # reset env
        game.reset()

        done = False
        score = 0
        counter = 0

        gtime = 0

        # first step
        observation_, reward, done = game.step(0)
        observation = np.array(observation_)

        while not done:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    return

            # new
            action = ddqn_agent.choose_action(observation)
            observation_, reward, done = game.step(action)
            observation_ = np.array(observation_)
            # Print the reward
            print(f"Step: {gtime}, Reward: {reward}")
            if reward == 0:
                counter += 1
                if counter > 100:
                    done = True
            else:
                counter = 0

            score += reward

            observation = observation_
            gtime += 1

            if gtime >= TOTAL_GAMETIME:
                done = True

            # Call the render function
            game.render(action)

# Run the simulation
run()
