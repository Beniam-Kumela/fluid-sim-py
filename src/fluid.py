import numpy as np
import pygame as pg
from solvers import diffuse, project, advect

class Fluid:
    def __init__(self, diffusion, viscosity, fade_rate, dt, dim, scale, surface, boundary_mask):
        '''
        Handles and updates fluid state, density field, and velocity field based on Jos Stam's "Stable Fluids" routine (see: https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf).

        Args:
            diffusion (float): diffusion coefficient for velocity field.
            viscosity (float): diffusion coefficient for density field.
            fade_rate (float): controls fluid dye color depletion.
            dt (float): time interval.
            dim (int): grid dimension.
            scale (int): screen scaling from underlying grid dimension.
            surface (pg.display.set_mode()): pygame Surface object
            boundary_mask (NxN matrix - bool): stores position of drawn boundaries.
        '''
        
        # Initialize fluid scalar properties.
        self.diff = diffusion
        self.visc = viscosity
        self.fade_rate = fade_rate
        self.dt = dt
        self.dim = dim
        self.scale = scale
        self.surface = surface
        self.boundary_mask = boundary_mask
        
        # Initialize fluid state arrays.
        self.s = np.zeros((self.dim, self.dim), dtype=float)
        self.density = np.zeros((self.dim, self.dim), dtype=float)
        self.Vx = np.zeros((self.dim, self.dim), dtype=float)
        self.Vy = np.zeros((self.dim, self.dim), dtype=float)
        self.Vx0 = np.zeros((self.dim, self.dim), dtype=float)
        self.Vy0 = np.zeros((self.dim, self.dim), dtype=float)
        self.rgba = np.ones((self.dim, self.dim, 4), dtype=np.uint8) * 255
    
    def step(self, boundary_mask=None):
        '''
        Define a single fluid solver step, comprised of several subroutines.

        Args:
            boundary_mask (NxN matrix - bool): stores position of drawn boundaries.
        '''

        # Diffuse x and y velocity components.
        diffuse(1, self.Vx0, self.Vx, self.visc, self.dt, self.dim, boundary_mask)
        diffuse(2, self.Vy0, self.Vy, self.visc, self.dt, self.dim, boundary_mask)

        # Enforce incompressibility condition.
        project(self.Vx0, self.Vy0, self.Vx, self.Vy, self.dim, boundary_mask)

        # Move velocities according to path traced by previous time step.
        advect(1, self.Vx, self.Vx0, self.Vx0, self.Vy0, self.dt, self.dim, boundary_mask)
        advect(2, self.Vy, self.Vy0, self.Vx0, self.Vy0, self.dt, self.dim, boundary_mask)

        # Enforce incompressibility once more.
        project(self.Vx, self.Vy, self.Vx0, self.Vy0, self.dim, boundary_mask)

        # Diffuse the dye.
        diffuse(0, self.s, self.density, self.diff, self.dt, self.dim, boundary_mask)

        # Move dye according to calculated velocities.
        advect(0, self.density, self.s, self.Vx, self.Vy, self.dt, self.dim, boundary_mask)
       
    def addDensity(self, x, y, amount):
        '''
        Adds fluid density at specified position.

        Args:
            x (int): x position.
            y (int): y position.
            amount (int): density amount to add.
        '''

        self.density[x, y] += amount
    
    def addVelocity(self, x, y, amountX, amountY):
        '''
        Adds fluid velocity at specified position.

        Args:
            x (int): x position.
            y (int): y position.
            amountX (int): x-component velocity amount to add.
            amountY (int): y-component velocity amount to add.
        '''

        self.Vx[x, y] += amountX
        self.Vy[x, y] += amountY
    
    def render(self):
        '''
        Renders fluid brightness (alpha in RGBA) based on fluid density field intensity.
        '''
        for i in range(self.dim):
            for j in range(self.dim):
                x = i * self.scale
                y = j * self.scale

                alpha = int(self.density[i, j] * 255 / self.scale)

                temp_surface = pg.Surface((self.scale, self.scale), pg.SRCALPHA)
                temp_surface.fill((255, 255, 255))
                temp_surface.set_alpha(alpha)
                
                self.surface.blit(temp_surface, (x, y))
         
    
    def fade(self):
        '''
        Fades density field by fade_rate.
        '''
        self.density *= self.fade_rate
        self.density = np.maximum(self.density, 0)
    
    def update_boundary_mask(self, dots):
        '''
        Updates boundary mask based on stored mouse position brush strokes.
         
        Args:
            dots (list(tuple)): (x, y) mouse position stored from each mouse stroke.
        '''
        self.boundary_mask.fill(0)
        for stroke in dots:
            for (x, y) in stroke:
                i = int(x / self.scale)
                j = int(y / self.scale)
                if 0 <= i < self.dim and 0 <= j < self.dim:
                    self.boundary_mask[i, j] = 1