import os
import ekinoksconstants
import numpy as np
import copy
import random
import torch
from collections import deque
import time
from tqdm import tqdm
from ekinokssos import SosEnv
from ekinoksAgent import Agent
from torch.utils.tensorboard import SummaryWriter
from ekinoksvisualizer import Visualizer

writer = SummaryWriter(flush_secs=5, log_dir = f'logs/{ekinoksconstants.MODEL_NAME}_{int(time.time())}') #log dosyasının oluşmasını sağlıyor

env.start()

p1InGameScore = 0
p2InGameScore = 0

done = False
while not done:

    while not done:
        gameEnd = env.getLegalMoves(env.board).count(1)
        if gameEnd == 0:
            done = True
            break 

        scored = False

        action = player1.makeMove()
        env.move(action[0], action[1])


        score = env.score[1]
        if p1InGameScore != score:
            scored = True
            p1InGameScore = score

        if not scored:
            break


    while not done:
        gameEnd = env.getLegalMoves(env.board).count(1)
        if gameEnd == 0:
            done = True
            break 

        scored = False

        action = player2.makeMove()
        env.move(action[0], action[1])


        score = env.score[2]
        if p2InGameScore != score:
            scored = True
            p2InGameScore = score

        if not scored:
            break

print(env.score)