
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

    def __init__(self, env, player, load=False):
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

    def backpropagate(self, state, move_index, target_value):#eğitim işlemini nasıl gerçekleştireceğimiz
        
        self.optimizer.zero_grad()
        output = self.model(self.convert_to_tensor(state))#önce modelimizin tahminini alıyoruz,
        target = output.clone().detach()

        target[move_index] = target_value # daha sonra yaptığımız harekete denk gelen q value'yi değiştiriyoruz.

        loss = self.loss_function(output, target)#ilk alınan çıktı, ve değiştirdiğimiz çıktı arasında loss hesaplanıyor
        loss.backward()
        self.optimizer.step()
        return loss
        #bu loss'a göre optimizer hesaplama yapıp, gerekli ağırlıkları değiştiriyor.
        
        
    def get_qs(self, state, model):#verilen model ile, verilen oyun durumuna göre tahminleri veriyor
        
        inputs = self.convert_to_tensor(state)
        with torch.no_grad():
            outputs = model(inputs)
            
        
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
    


 
    def makeMove(self):#1 için hamle yapıyor
        
        board=self.env.board
        return board
        
        

        
        if self.env.turn == 1:
            
            if np.random.random() > ekinoksconstants.epsilon:#burada epsilon ile rastgele seçilen 0-1 arasında ki bir sayı karşılaştırılacak
#yapılacak hamlenin rastgele mi yoksa model tarafından mı yapılacağını belirliyor
                
        
                mask = self.env.getLegalMoves(board) #illegel hamleleri maskelemek için maske'yi env'den alıyoruz
         

                q_values=[]
    
                
                mask=mask*2
                for q, maskValue in zip(self.get_qs(board, self.model), mask):
                    
                    if maskValue == 1:#eğer maskeni değeri 1 ise, yani hamle legal bir hamle ise 
                        q_values.append(q) #q valuesini doğrudan listeye ekliyoruz
                    else:# eğer illegal bir hamle ise
                        q_values.append(-999)#-999 ekliyoruzki, maximum hamleyi seçerken illegal bir hamle seçmeyelim
                action = np.argmax(q_values)#maximum hamlenin index'ini seçiyoru
                
       
                action=self.rec(action)
   
                
            else:
            
                legalMoves = self.env.getLegalMoves(self.env.board)
                actionList = []
                for move in legalMoves:
                    actionList.append(ekinoksconstants.ACTION_LIST.index(move))
                action = random.choice(actionList)
                action=self.rec(action)

            return action
       
            
       
        else:
            if np.random.random() > ekinoksconstants.epsilon:#burada epsilon ile rastgele seçilen 0-1 arasında ki bir sayı karşılaştırılacak
#yapılacak hamlenin rastgele mi yoksa model tarafından mı yapılacağını belirliyoruz

        
                mask = self.env.getLegalMoves(board) #illegel hamleleri maskelemek için maske'yi env'den alıyoruz.
                
                q_values = []
                
              
                for q, maskValue in zip(self.get_qs(board, self.model), mask):
                    if maskValue == 1:#eğer maskeni değeri 1 ise, yani hamle legal bir hamle ise 
                        q_values.append(q) #q valuesini doğrudan listeye ekliyoruz
                    else:# eğer illegal bir hamle ise
                        q_values.append(-999)#-999 ekliyoruzki, maximum hamleyi seçerken illegal bir hamle seçmeyelim
                action = np.argmax(q_values)#maximum hamlenin index'ini seçiyoru
                action=self.rec(action)
                
            else:
                legalMoves = self.env.getLegalMoves(self.env.board)
                
                actionList = []
                for move in legalMoves:
                    actionList.append(ekinoksconstants.ACTION_LIST.index(move))
                action = random.choice(actionList)
                action=self.rec(action)
                
               
                
            return action
                

    def piecedüzelt(self): #action=(5,-1)
        action=self.makeMove()
        a=action[0]
        b=action[1]
        if b==-1:
            action=(a+36,)
        return action
        


    def train(self, MoveHistory1, MoveHistory2):#eğitim işleminin gerçekleştiği yer

        nextState, action, reward = MoveHistory2.pop()#pop'u kullanarak en son yapılan hamleyi alıyoruz.
        
        
        action=self.piecedüzelt()
        loss=self.backpropagate(nextState, action[0], reward)#en son hamlede ki ödül, son durum olduğundan kaynaklı
        #bu veriyi direk eğitiyoruz.

        for _ in range(len(MoveHistory2)):#2 nin  yaptığı tüm hamleler için eğitim yapacağız
            current_state, action, reward = MoveHistory2.pop()#hangi durumda, hangi aksiyonu aldığı ve buna karşılık aldığı ödülü alıyoruz.
            
            next_qs = self.get_qs(nextState, self.target_model)#yaptığı hamleden sonra ki yapabileceği hamlelerin q değer'lerini alıyoruz.
             
            mask = self.env.getLegalMoves(nextState)#en iyi q değerini bulmadan önce, yapamayacağı, yani illegal hamleleri çıkartıyoruz.
        
            qs_to_select = []
            for q, maskValue in zip(next_qs, mask):
                if maskValue == 0:
                    qs_to_select.append(q.item())


            next_q = max(qs_to_select)#legal hamleler arasından maximum q değerini seçiyoruz
            new_q  = reward + next_q * ekinoksconstants.DISCOUNT#yapmış olduğu aksiyonu, bu aksiyondan aldığı ödül, + sonra ki aksiyonlardan 
            #alacağı ödül şeklinde ayarlıyoruz. böylece modelimizin ileri görüşlülüğü oluyor. discount parametresini
            #değiştirerek sonraki hamleleri ne kadar önemsemesi gerektiğini ayarlıyoruz.

            self.backpropagate(current_state, action[0], new_q)#yeni hesapladığımız q ile eğitim gerçekleştiriyoruz.

            nextState = current_state

        #Aynı işlemleri X için yapıyoruz,
        #maximumu almak yerine minimumu alıyoruz.
        nextState, action, reward = MoveHistory1.pop()
        self.backpropagate(nextState, action[0], reward)
        for _ in range(len(MoveHistory1)):
            current_state, action, reward = MoveHistory1.pop()
            
            next_qs = self.get_qs(nextState, self.target_model)
            mask = self.env.getLegalMoves(nextState)
            qs_to_select = []
            for q, maskValue in zip(next_qs, mask):
                if maskValue == 0:
                    qs_to_select.append(q.item())
            

            new_q  = reward + next_q * ekinoksconstants.DISCOUNT

            self.backpropagate(current_state, action[0], new_q)

            nextState = current_state

        self.target_update_counter += 1
        if self.target_update_counter > ekinoksconstants.UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_update_counter = 0
        
        return loss
        