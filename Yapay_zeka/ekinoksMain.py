

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



visualizer = Visualizer()
env = SosEnv()
visualizer.env = env
agent = Agent(env, (1,2), ekinoksconstants.LOAD_MODEL) #agent'ımızı oluşturuyor




done=False

p1InGameScore = 0
p2InGameScore = 0

for episode in tqdm(range(1, ekinoksconstants.EPISODES + 1), ascii=True, unit='episodes'):

    env.start()
    MoveHistory1 = []
    MoveHistory2 = []

    done=False
    while not done:   
        
        gameEnd = env.getLegalMoves(env.board).count(1)
        if gameEnd == 0:
            done = True
            break 
         
        
        
        toAppend = [copy.deepcopy(env.board)]#önce şuan ki durumumuzu kayıt ediyoruz.
        action = agent.makeMove() #yapılacak aksiyonun index'ini agent'dan alıyoruz.
        
        
        env.move(action[0],action[1])#aksiyonun indexi'ini env'e vererek hareketi yapıyoruz ve oyunun yeni durumunu alıyoruz

        reward , scored =agent.rewardalma(p1InGameScore,p2InGameScore  )
        
        #reward = -reward X oyunu minimize etmeye çalışacağı için aldığı ödülü - olarak değiştiriyoruz.
        toAppend.append(action)
        toAppend.append(reward)
        MoveHistory1.append(toAppend)#agent'in yaptığı harekete karşı aldığı ödül'ler vs. eğitim için saklıyoruz

        
        
        

        if episode % ekinoksconstants.SHOW_EVERY == 0 and ekinoksconstants.IS_VISUALIZER_ON:#eğer görüntüleme açık ise oyunu her x oyunda bir görüntülüyoruz
            visualizer.show()
        
    
        ### Second Player ###
        if not done:
            
           
            toAppend=[copy.deepcopy(env.board)]

            action2 = agent.makeMove() 
            
            env.move(action2[0],action[1])
            toAppend.append(action2)
            toAppend.append(reward)
            
            
            
            

            MoveHistory2.append(toAppend)
            
        
            
    

            if done:#eğer oyun bittiyse, yani O kazandıysa (beraberlik durumunu sadece X yapabiliyor)
                state, action, _, = MoveHistory1.pop() 
                MoveHistory1.append([state, action, reward])
                #x'in oynamış olduğu son aksiyonun değeri 0 değil +1 olmuş oluyor, çünkü x'in hamlesinden sonra yapılan hamle yani o'nun hamlesi
                #x'e göre enviroment'ın bir parçası. Çünkü x'in yaptığı hareketten sonra her hangi bir müdahale şansı olmadı, yani son yaptığı aksiyon
                #x'in kayıp etmesine sebep oldu, X'in son aksiyonu için eklemiş olduğumuz ödülü değiştirmemiz lazım, üstteki işlem bunu yapmaktadır.


        else:
            #66 satırda anlatılan olayın aynısı, sadece O için
            state, action, _, = MoveHistory2.pop()
            MoveHistory2.append([state, action, reward])


        if episode % ekinoksconstants.SHOW_EVERY == 0 and ekinoksconstants.IS_VISUALIZER_ON:
            visualizer.show()
    agent.train(MoveHistory1,MoveHistory2)
    
    if not episode % ekinoksconstants.AGGREGATE_STATS_EVERY or episode == 1:

        writer.add_scalar('epsilon', ekinoksconstants.epsilon, episode)

        #Yukarıda tensorboard'umuza aldığımız verileri giriyouz.
        if not ekinoksconstants.IS_TEST:#eğer test değil ise, modelimizi kayıt ediyoruz
            torch.save(agent.model.state_dict(), f'models/{ekinoksconstants.MODEL_NAME}_{int(time.time())}.model')
       
        #verileri 0'lıyoruz üst üste sürekli artan bir grafik görmeyelim

    # epsilon değerini ayarlıyoruz.
    if ekinoksconstants.epsilon > ekinoksconstants.MIN_EPSILON:
        ekinoksconstants.epsilon *= ekinoksconstants.EPSILON_DECAY
        ekinoksconstants.epsilon = max(ekinoksconstants.MIN_EPSILON, ekinoksconstants.epsilon)












