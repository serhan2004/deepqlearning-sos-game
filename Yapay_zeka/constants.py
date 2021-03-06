from PIL import Image, ImageDraw


#Görüntüleme ile alakalı parametreler
SIZE = 1600
SPACING = SIZE//3

IS_VISUALIZER_ON = False
WAIT_TIME = 200 #0 girilir ise elle devam ettiriliyor
SHOW_EVERY = 10000


#Ödül/Ceza
WIN_REWARD = 1
LOSE_REWARD = -1
DRAW_REWARD = 0

# Exploration settings
epsilon = 1 # not a constant, going to be decayed
EPSILON_DECAY = 0.99999 #0.9996
MIN_EPSILON = 0.01


#Kayıt ile alakalı parametreler
MODEL_NAME = "3x36Linear-1E-0.1L-Tanh-D999996-PyTorch" #modelin hangi adla kayıt edileceği
AGGREGATE_STATS_EVERY = 1000  # episodes
LOAD_MODEL = False
IS_TEST = False
MODEL_PATH = "models/3x36Linear-1E-0.1L-Tanh-D999996_1_1610538057.model" # yüklenecek modelin adı

# Diğer parametreler
DISCOUNT = 1
UPDATE_TARGET_EVERY = 5
LR = 0.1
EPISODES = 2_000_000

ACTION_LIST=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,24,25,26,27,28,29,30,31,32,33,34,35]
"""
ACTION_LIST = [(0,0),(0,1),(0,2),(0,3),(0,4),(0,5),
               (1,0),(1,1),(1,2),(1,3),(1,4),(1,5),
               (2,0),(2,1),(2,2),(2,3),(2,4),(2,5),
               (3,0),(3,1),(3,2),(3,3),(3,4),(3,5),
               (4,0),(4,1),(4,2),(4,3),(4,4),(4,5),
               (5,0),(5,1),(5,2),(5,3),(5,4),(5,5),]

"""
    
    
    
    
    




#X ve O karakterlerinin oluşturulması.
pieces= {(0,0):"",(0,1):"s",(1,0):"o"}

for piece in pieces:
    image = Image.new("RGBA", (SPACING-100, SPACING-100))
    drawer = ImageDraw.Draw(image, 'RGBA')

    if pieces[piece] == 's':
        drawer.line([(0,0), (image.size[0], image.size[0])], fill=None, width=10)
        drawer.line([(0,image.size[0]), (image.size[0], 0)], fill=None, width=10)


    elif pieces[piece] == 'o':
        drawer.ellipse((0,0, image.size), outline ='white', width = 10)

    pieces[piece] = image