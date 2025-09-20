import pygame as pg
import sys
from gui import start_menu
from animate import animate

def main():
    # Initialize parameters    
    N = 128 # grid dimension
    scale = 5 # scaling factor for surface (e.g. if N = 128, surface dim = 128*5, 128*5)    
    surface = pg.display.set_mode((N * scale, N * scale), pg.HWSURFACE | pg.DOUBLEBUF)
    clock = pg.time.Clock()
    font = pg.font.SysFont("Arial", 24)
    pg.display.set_caption("fluid-sim-py") # Window captioning.

    while True: # When application starts, show start menu
        choice = start_menu(N, scale, surface)
        animate(N, scale, surface, clock, font)
        
if __name__ == "__main__":
    pg.init()
    main()
    pg.quit()
    sys.exit()