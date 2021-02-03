
import torch
from torch import nn
from torch.nn import MSELoss
import numpy as np
import ekinoksconstants
import random

class DDQN(nn.Module):
    def __init__(self):
        super().__init__()
        self.createModel()


    def createModel(self):
        self.layer1 = nn.Linear(36,128) #9 input alan ve bunu 36 output olarak verececk bir linear(Dense) layer
        self.layer2 = nn.Linear(128,128)#36 input 36 output veren bir layer
        self.layer3 = nn.Linear(128,72)#36 input 9 output veren bir layer
        #burada daha layerların bağlantıları sağlanmış durumda değil, bunu forward fonksiyonunda yapacağız.


    def forward(self, state): #bu fonksiyonun ismini değiştirmeyin, çalışması için adının forward olması lazım
        x = self.layer1(state) #state, yani inputumuzu layer'1 e vereceğimizi söylüyoruz
        x = torch.relu(x)#layer 1'den aldığımız outputları relu aktivaston fonksiyonundan geçiriyoruz

        x = self.layer2(x)#aktivasyon fonksiyonundan çıkan verileri layer2'ye atıyoruz
        x = torch.relu(x)#layer 2'den çıkanları relu fonksiyonundan geçiriyoruz

        x = self.layer3(x)#relu'den çıkanları layer3'e atıyoruz
        x = torch.tanh(x)#son olarak layer'e ten çıkan 9 verimizi tanh fonksiyonundan geçiriyoruz. ve çıkan sonucu dönüyoruz

        return x
    


class Agent():

    def __init__(self, env, player, load=True):
        self.player = player
        self.env = env

        self.model = DDQN()#modeli oluşturuyoruz
        self.target_model = DDQN()#modelimiz için tahminler yaparken, modelin stabil kalması için ekstra bir tane daha oluşturuyoruz
        
        if load:#eğer önceden eğittiğimiz ve üzerine eğitime devam edeceğimiz bir model var ise, bunu yüklüyoruz
            self.model.load_state_dict(torch.load(ekinoksconstants.MODEL_PATH))

        self.target_model.load_state_dict(self.model.state_dict())#model oluşturulurken değerler rastgele atılıyor, target model ve modelimizin
        #aynı değerlere sahip olması için target modelin ağırlıklarını model ile eşitliyoruz.
        self.target_model = self.target_model.eval()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=ekinoksconstants.LR) #optimizer olarak, scholastic gradient descent kullanıyoruz
        self.loss_function = MSELoss()#loss fonksiyonu olarak, mean square error kullanıyoruz.

        #loss fonksyionlar, optimizer'lar ve layer'lar hakkında ki seçeneklere ve nasıl çalıştıklarını
        #https://pytorch.org/docs/stable/nn.html adresinden bakabilirsiniz.

        self.target_update_counter = 0

        

        
        return outputs

    def convert_to_tensor(self, state):#oyun durumunu yapay zekaya vermek için hazırlıyor
 
        return torch.tensor(state, dtype=torch.float)
    
    def winner(self):
        score=self.env.score
        player1=score.get("1")
        player2=score.get("2")
        if self.done:
            if player1>player2:
                winner=1
            elif player1<player2:
                winner=2
            else:
                winner=0
        return winner
    
    def rewardalma(self,score1,score2):
        
        scored=False
        
        reward=0
        player1=self.env.score.get("1")
        player2=self.env.score.get("2")
        if score1==player1 and score2==player2:
            reward=0
        elif score1==player1 and score2 !=player2:
            reward=(-1)
            scored=True
        elif score1!=player1 and score2==player2:
            reward=1
            scored=True
        score1=player1
        score2=player2
        return reward,scored

    
    
    def rec(self,move):
        # piece S mi O mu belirleme işlemi yapıldı
        if move >35:
            index=move-36
            piece=-1
        else:
            index=move
            piece=1
        return (index,piece)
    


 
    def makeMove(self):
        
        board =self.env.board
        return board
        
        mask = self.env.getLegalMoves(board)
        q_values = []
        mask = mask*2
        
        for q, maskValue in zip(self.get_qs(board,self.model), mask):
            if maskValue == 1:
                q_values.append(q)
            else:
                if self.player ==1:
                    q_values.append(-999)
                else:
                    q_values.append(999)
        if self.player == 1:
            action = np.argmax(q_values)
        else:
            action = np.argmin(q_values)
    
            return action
       
            

        

    def piecedüzelt(self): #action=(5,-1)
        action=self.makeMove()
        a=action[0]
        b=action[1]
        if b==-1:
            action=(a+36,)
        return action
        


   
        