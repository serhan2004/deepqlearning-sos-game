# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 18:59:15 2021

@author: Hp
"""

import os
import constants
import numpy as np
import copy
import random
import torch
from collections import deque
import time
from tqdm import tqdm
from sos import SosEnv
from sosAgent import Agent
from torch.utils.tensorboard import SummaryWriter
from visualizer import Visualizer

writer = SummaryWriter(flush_secs=5, log_dir = f'logs/{constants.MODEL_NAME}_{int(time.time())}') #log dosyasının oluşmasını sağlıyor



visualizer = Visualizer()
env = SosEnv()
visualizer.env = env
agent = Agent(env, (1,2), constants.LOAD_MODEL) #agent'ımızı oluşturuyor


#Loglayacağımız veriler
suma = 0
draw = 0
win1 = 0
win2 = 0

done=False
 

for episode in tqdm(range(1, constants.EPISODES + 1), ascii=True, unit='episodes'):

    env.start()
    MoveHistory1 = []
    MoveHistory2 = []

    done=False
    while not done:

        
        toAppend = [copy.deepcopy(env.board)]#önce şuan ki durumumuzu kayıt ediyoruz.
        action = agent.makeMove1(env.board) #yapılacak aksiyonun index'ini agent'dan alıyoruz.
        
        
        
        
        new_state, reward, done = env.move(action)#aksiyonun indexi'ini env'e vererek hareketi yapıyoruz ve oyunun yeni durumunu alıyoruz
        reward = -reward #X oyunu minimize etmeye çalışacağı için aldığı ödülü - olarak değiştiriyoruz.
        toAppend.append(action)
        toAppend.append(reward)
        MoveHistory1.append(toAppend)#agent'in yaptığı harekete karşı aldığı ödül'ler vs. eğitim için saklıyoruz

        

        if episode % constants.SHOW_EVERY == 0 and constants.IS_VISUALIZER_ON:#eğer görüntüleme açık ise oyunu her x oyunda bir görüntülüyoruz
            visualizer.show()
        
    
        ### Second Player ###
        if not done:
            toAppend=[copy.deepcopy(new_state)]

            action2 = agent.makeMove2(new_state) 
            _, reward, done = env.move(action2)
            toAppend.append(action2)
            toAppend.append(reward)

            MoveHistory2.append(toAppend)
            ## ^ ilk agent ile aynı, tek farkı reward'ı -'ye çekmiyoruz

            if done:#eğer oyun bittiyse, yani O kazandıysa (beraberlik durumunu sadece X yapabiliyor)
                state, action, _, = MoveHistory1.pop() 
                MoveHistory1.append([state, action, reward])
                #x'in oynamış olduğu son aksiyonun değeri 0 değil +1 olmuş oluyor, çünkü x'in hamlesinden sonra yapılan hamle yani o'nun hamlesi
                #x'e göre enviroment'ın bir parçası. Çünkü x'in yaptığı hareketten sonra her hangi bir müdahale şansı olmadı, yani son yaptığı aksiyon
                #x'in kayıp etmesine sebep oldu, X'in son aksiyonu için eklemiş olduğumuz ödülü değiştirmemiz lazım, üstteki işlem bunu yapmaktadır.
                suma += -1
                win2 += 1

        elif done and reward == 0:
            draw += 1 #eğer oyun bitti ve reward 0 ise draw demek

        else:
            #66 satırda anlatılan olayın aynısı, sadece O için
            state, action, _, = MoveHistory2.pop()
            MoveHistory2.append([state, action, reward])
            suma += 1
            win1 +=1
            
        if episode % constants.SHOW_EVERY == 0 and constants.IS_VISUALIZER_ON:
            visualizer.show()
    agent.train(MoveHistory1,MoveHistory2)
    
    if not episode % constants.AGGREGATE_STATS_EVERY or episode == 1:

        writer.add_scalar('sum', suma, episode)
        writer.add_scalar('epsilon', constants.epsilon, episode)
        writer.add_scalar('draw', draw, episode)
        writer.add_scalar('win1', win1, episode)
        writer.add_scalar('win2', win2, episode)
        #Yukarıda tensorboard'umuza aldığımız verileri giriyouz.
        if not constants.IS_TEST:#eğer test değil ise, modelimizi kayıt ediyoruz
            torch.save(agent.model.state_dict(), f'models/{constants.MODEL_NAME}_{win1}_{int(time.time())}.model')
        draw = 0
        suma = 0
        win2 = 0
        win1 = 0
        #verileri 0'lıyoruz üst üste sürekli artan bir grafik görmeyelim

    # epsilon değerini ayarlıyoruz.
    if constants.epsilon > constants.MIN_EPSILON:
        constants.epsilon *= constants.EPSILON_DECAY
        constants.epsilon = max(constants.MIN_EPSILON, constants.epsilon)












