import cv2
import numpy as np
import cv2
from PIL import Image, ImageDraw
import ekinoksconstants
import copy

class Visualizer:


    def __init__(self):
        self.emptyBoard = self.drawEmptyBoard()
        self.env = None
        self.selected = False

    def drawEmptyBoard(self):
        size = ekinoksconstants.SIZE
        spacing = ekinoksconstants.SPACING

        image = Image.new("RGBA", (size, size))
        drawer = ImageDraw.Draw(image, 'RGBA')
        drawer.line([(spacing,0), (spacing, size)], fill=None, width=10)
        drawer.line([(spacing*2,0), (spacing*2, size)], fill=None, width=10)

        drawer.line([(0,spacing), (size, spacing)], fill=None, width=10)
        drawer.line([(0,spacing*2), (size, spacing*2)], fill=None, width=10)
        return image

        

    def drawPiecesOnBoard(self):
        boardImage = copy.deepcopy(self.emptyBoard)
        for x, row in enumerate(self.env.board):
            for y, square in enumerate(row):
                piece = ekinoksconstants.pieces[square]
                boardImage.paste(piece, (y*ekinoksconstants.SPACING+50, x*ekinoksconstants.SPACING+50), piece) 

        return boardImage


    def show(self, waitTime = ekinoksconstants.WAIT_TIME):
        self.open_cv_image = np.array(self.drawPiecesOnBoard())
        cv2.namedWindow('Board', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Board', 800,800)
        cv2.imshow('Board', self.open_cv_image)
        cv2.waitKey(waitTime)