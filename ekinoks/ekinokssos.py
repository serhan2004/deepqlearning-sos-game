import ekinoksconstants
class SosEnv:

    def __init__(self, size=6):
        self.size = size

    def start(self):
        '''
        oyunu başlartır ve yeni boş bir oyun tahtası hazırlar.
        0 ilk oyuncuyu, 1 ikinci oyuncuyu temsil eder.
        tahta üzerinde, 0 boş, -1 S, 1 ise O yi temsil eder        
        '''

        self.board = self.createBoard()
        self.turn = 1
        self.score = {1:0, 2:0} #0 is player 1, 1 is player 2
        return self.board


    def createBoard(self):
        board = [0 for i in range(self.size**2)]
        return board


    def getLegalMoves(self, state):
        '''
        bu fonksiyon bir liste döner, listenin her bir indexi
        tahtadaki konumla eşleşir, eğer hamleyi yapmak mümkün ise
        1, mümkün değil ise 0 yazar, öğreniğin 3'e 3 bir tahtada
        tahta = [-1,1,-1,0,0,0,1,0,0]
        liste = [ 0,0,0 ,1,1,1,0,1,1] <-- döncek liste bu şekilde
        '''
        moveIndexes = [1 if i == 0 else 0 for i in state]

        return moveIndexes


    def move(self, moveIndex, piece):
        '''
        yapmak istediğiniz hamlenin index'ini ve koymak istediğiniz parçayı vermeniz gerekli,
        -1 --> S
         1 --> O
         6'ya 6'lık bir tahtada moveIndex 0 ile 36 arasındadır, 36 hariç.
         hamlenin legal olup olmadığı kontrol edilmemektedir. bu kontrolü sizin sağlamanız lazım.
        '''

        score = self.calculateScore(moveIndex, piece)
        
        self.score[self.turn] += score
        if not score:
            self.turn = 1 if self.turn == 2 else 2
        self.board[moveIndex] = piece    

    def calculateScore(self, moveIndex, piece):
        possible = [self.size-5, self.size-1, self.size, self.size+1]

        score = 0
        if piece == 1:
            if moveIndex % self.size == 0 or moveIndex % self.size == self.size-1:
                if self.size-5 in possible: possible.remove(self.size-5) 
                if self.size+1 in possible: possible.remove(self.size+1)
                if self.size-1 in possible: possible.remove(self.size-1) 

            if moveIndex in list(range(self.size)) or moveIndex in list(range(self.size**2-self.size,self.size**2)):
                if self.size in possible: possible.remove(self.size) 
                if self.size+1 in possible: possible.remove(self.size+1) 
                if self.size-1 in possible: possible.remove(self.size-1) 

            for p in possible:
                score += self.oScorer(moveIndex, p) 

        if piece == -1:
            for p in possible:
                score += self.sScorer(moveIndex, p)
                score += self.sScorer(moveIndex, -p)


        return score


    def oScorer(self, moveIndex, index):
        if moveIndex-index >= 0 and moveIndex+index <= self.size**2-1:
            if self.board[moveIndex-index] == -1 and self.board[moveIndex+index] == -1:
                return 1
        return 0

    def sScorer(self, moveIndex, index):
        if moveIndex+index >= 0 and moveIndex+index <= self.size**2-1:
            if moveIndex+index*2 >= 0 and moveIndex+index*2 <= self.size**2-1:
                if self.board[moveIndex+index] == 1 and self.board[moveIndex+index*2] == -1:
                    distance = abs(moveIndex%self.size - abs((moveIndex+index*2)%self.size))
                    if  distance == 2 or distance == 0:
                        return 1
        return 0



if __name__ == '__main__':
    env = SosEnv()

    env.start()


    env.move(0,-1)
    env.move(6, 1)


    print(env.score)
    print(env.board)
    env.move(12, -1)
    

    print(env.score)
    print(env.board)