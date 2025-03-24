import pygame as pg
import sys
import numpy as np
from numba import jit

class Fluid:
    # Initialize fluid properties and vectors
    def __init__(self, diffusion, viscosity, dt):
        
        self.size = N
        self.dt = dt
        self.diff = diffusion
        self.visc = viscosity
        
        self.s = np.zeros((N, N), dtype=float)
        self.density = np.zeros((N, N), dtype=float)
        
        self.Vx = np.zeros((N, N), dtype=float)
        self.Vy = np.zeros((N, N), dtype=float)
        
        self.Vx0 = np.zeros((N, N), dtype=float)
        self.Vy0 = np.zeros((N, N), dtype=float)
    
    # Simulate one step of the fluid solver
    def step(self):
        visc = self.visc
        diff = self.diff
        dt = self.dt
        Vx = self.Vx
        Vy = self.Vy
        Vx0 = self.Vx0
        Vy0 = self.Vy0
        s = self.s
        density = self.density

        diffuse(1, Vx0, Vx, visc, dt)
        diffuse(2, Vy0, Vy, visc, dt)

        project(Vx0, Vy0, Vx, Vy)

        advect(1, Vx, Vx0, Vx0, Vy0, dt)
        advect(2, Vy, Vy0, Vx0, Vy0, dt)

        project(Vx, Vy, Vx0, Vy0)

        diffuse(0, s, density, diff, dt)
        advect(0, density, s, Vx, Vy, dt)
       
    # Add amount to density
    def addDensity(self, x, y, amount):
        self.density[x, y] += amount
    
    # Add amount to velocity
    def addVelocity(self, x, y, amountX, amountY):
        
        self.Vx[x, y] += amountX
        self.Vy[x, y] += amountY
    
    # Render the fluid density in grayscale
    def render(self):
        for i in range(N):
            for j in range(N):
                x = i * SCALE
                y = j *SCALE
                d = self.density[i, j]
                color = (255, 255, 255)
                alpha = int(d*255/SCALE)
                rect_surface = pg.Surface((SCALE, SCALE), pg.SRCALPHA)
                rect_surface.fill(color)
                rect_surface.set_alpha(alpha)
                display_surface.blit(rect_surface, (x, y))
        
        # Draw the cylinder
        if choice == 'cylinder':
            pg.draw.circle(display_surface, (0, 0, 255), (N*SCALE//4, N*SCALE//2), N*SCALE//20, 0)
        
        # Draw the NACA 2412 Airfoil (https://en.wikipedia.org/wiki/NACA_airfoil)
        elif choice == 'airfoil':
            m, p, t = 0.02, 0.4, 0.12
            x = np.linspace(0, 1, 100)
            y_c = np.where(x < p,
                            m / p**2 * (2 * p * x - x**2),
                            m / (1 - p)**2 * (1 - 2 * p + 2 * p * x - x**2))
            y_t = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
            y_upper = y_c + y_t
            y_lower = y_c - y_t

            coords = np.vstack((np.hstack((x, x[::-1])), np.hstack((y_upper, y_lower[::-1])))).T
            
            width, height = display_surface.get_size()
            center_x = width // 8
            center_y = height // 2
            scale_x = width / 4
            scale_y = height / 4

            coords = [(int(x * scale_x + center_x), 
                    int(center_y - y * scale_y)) 
                    for x, y in coords]

            pg.draw.polygon(display_surface, (0, 0, 255), coords, 0)        
        
        pg.display.update()
    
    # Remove density at some fade rate so that the screen is not saturated
    def fade(self):
        fade_rate = 0.95
        for i in range(N):
            for j in range(N):
                self.density[i, j] *= fade_rate
                self.density[i, j] = max(self.density[i, j], 0)

# Function which diffuses the fluid by solving system of linear equations
@jit(nopython=True)        
def diffuse(b, x, x0, diff, dt):
    a = dt * diff * (N - 2) * (N - 2)
    lin_solve(b, x, x0, a, 1 + 6 * a)

# Linear equation solver using Gauss-Seidel method: https://en.wikipedia.org/wiki/Gauss%E2%80%93Seidel_method
@jit(nopython=True)
def lin_solve(b, x, x0, a, c):
        cRecip = 1.0 / c
        for k in range(iter):
            for j in range(1, N - 1):
                    for i in range(1, N - 1):
                        x[i, j] = (
                            x0[i, j]
                            + a * (
                                x[i + 1, j]
                                + x[i - 1, j]
                                + x[i, j + 1]
                                + x[i, j - 1]
                            )
                        ) * cRecip
            set_bnd(b, x)

# Function which maintains condition that fluid in each grid space box must remain constant
@jit(nopython=True)
def project(velocX, velocY, p, div):
    for j in range(1, N - 1):
        for i in range(1, N - 1):
            div[i, j] = -0.5 * (
                velocX[i + 1, j]
                - velocX[i - 1, j]
                + velocY[i, j + 1]
                - velocY[i, j - 1]
            ) / N
            p[i, j] = 0

    set_bnd(0, div)
    set_bnd(0, p)
    lin_solve(0, p, div, 1, 6)

    for j in range(1, N - 1):
        for i in range(1, N - 1):
            velocX[i, j] -= 0.5 * (p[i + 1, j] - p[i - 1, j]) * N
            velocY[i, j] -= 0.5 * (p[i, j + 1] - p[i, j - 1]) * N

    set_bnd(1, velocX)
    set_bnd(2, velocY)

# Function which looks at each cell using its previous and weighted neighbor average velocity to determine its current one.
@jit(nopython=True)
def advect(b, d, d0, velocX, velocY, dt):
    dtx = dt * (N - 2)
    dty = dt * (N - 2)
    
    Nfloat = float(N)
    
    for j in range(1, N - 1):
        jfloat = float(j)
        for i in range(1, N - 1):
            ifloat = float(i)
            
            tmp1 = dtx * velocX[i, j]
            tmp2 = dty * velocY[i, j]
            x = ifloat - tmp1
            y = jfloat - tmp2
            
            if x < 0.5:
                x = 0.5
            if x > Nfloat + 0.5:
                x = Nfloat + 0.5
            i0 = np.floor(x)
            i1 = i0 + 1.0
            
            if y < 0.5:
                y = 0.5
            if y > Nfloat + 0.5:
                y = Nfloat + 0.5
            j0 = np.floor(y)
            j1 = j0 + 1.0
            
            s1 = x - i0
            s0 = 1.0 - s1
            t1 = y - j0
            t0 = 1.0 - t1
            
            i0i = int(i0)
            i1i = int(i1)
            j0i = int(j0)
            j1i = int(j1)
            
            if(i0i < N and i1i < N and j0i < N and j1i < N):
                d[i, j] = (
                    s0 * (t0 * (d0[i0i, j0i]) +
                            (t1 * (d0[i0i, j1i]))) +
                    s1 * (t0 * (d0[i1i, j0i]) +
                            (t1 * (d0[i1i, j1i])))
                )
    
    set_bnd(b, d)

# Function which treats the outer layer of cells as a wall to keep fluid from leaking
@jit(nopython=True)
def set_bnd(b, x):
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            # Circle center and radius
            cx, cy = N // 4, N // 2
            r = N // 20

            # Check if current cell is within airfoil
            if choice == 'airfoil':
                if is_inside_airfoil(i, j):
                    if b == 1:
                        x[i, j] = -x[i, j]  # Reverse horizontal velocity
                    elif b == 2:
                        x[i, j] = -x[i, j]  # Reverse vertical velocity
                        
            if choice == 'cylinder':      
                # Check if current cell is within the circle
                if (i - cx) ** 2 + (j - cy) ** 2 < r ** 2:
                    if b == 1:
                        x[i, j] = -x[i, j]
                    elif b == 2:
                        x[i, j] = -x[i, j]

    # Apply boundary conditions on the edges
    for i in range(1, N - 1):
        x[i, 0] = -x[i, 1] if b == 2 else x[i, 1]
        x[i, N - 1] = -x[i, N - 2] if b == 2 else x[i, N - 2]

    for j in range(1, N - 1):
        x[0, j] = -x[1, j] if b == 1 else x[1, j]
        x[N - 1, j] = -x[N - 2, j] if b == 1 else x[N - 2, j]

    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1])
    x[0, N - 1] = 0.5 * (x[1, N - 1] + x[0, N - 2])
    x[N - 1, 0] = 0.5 * (x[N - 2, 0] + x[N - 1, 1])
    x[N - 1, N - 1] = 0.5 * (x[N - 2, N - 1] + x[N - 1, N - 2])

# Function to check if a point (i, j) is inside the airfoil
@jit(nopython=True)
def is_inside_airfoil(i, j):
    width, height = N, N
    center_x = width // 8
    center_y = height // 2
    scale_x = width / 4
    scale_y = height / 4
    
    # Convert grid coordinates to airfoil coordinates
    x_coord = (i - center_x) / scale_x
    y_coord = (center_y - j) / scale_y
    
    # Airfoil parameters
    m, p, t = 0.02, 0.4, 0.12
    y_c = m / p**2 * (2 * p * x_coord - x_coord**2) if x_coord < p else m / (1 - p)**2 * (1 - 2 * p + 2 * p * x_coord - x_coord**2)
    y_t = 5 * t * (0.2969 * np.sqrt(x_coord) - 0.1260 * x_coord - 0.3516 * x_coord**2 + 0.2843 * x_coord**3 - 0.1015 * x_coord**4)
    
    y_upper = y_c + y_t
    y_lower = y_c - y_t
    
    return y_lower <= y_coord <= y_upper

# This function defines what the start_menu looks like
def start_menu(): 
    
    # Load graphic for start_menu and scale appropriately
    image = pg.image.load('start_menu.png')
    scaled_image = pg.transform.scale(image, (N*SCALE, N*SCALE)) # Scale to display size
    
    # Define 3 buttons
    button_color = (0, 128, 255)
    button_hover_color = (0, 255, 128)
    
    button_laminar_rect = pg.Rect((N * SCALE // 3 - 50, N * SCALE // 2 - 120), (N * SCALE // 2, 50))
    button_cylinder_rect = pg.Rect((N * SCALE // 3 - 50, N * SCALE // 2 - 40), (N * SCALE // 2, 50))
    button_airfoil_rect = pg.Rect((N * SCALE // 3 - 50, N * SCALE // 2 + 40), (N * SCALE // 2, 50))
    button_sandbox_rect = pg.Rect((N * SCALE // 3 - 50, N * SCALE // 2 + 120), (N * SCALE // 2, 50))
    
    running = True
    while running:
        display_surface.blit(scaled_image, (0, 0))  # Fill background

        # Get mouse position
        mouse_pos = pg.mouse.get_pos()

        # Check if the mouse is over any button
        laminar_color = button_hover_color if button_laminar_rect.collidepoint(mouse_pos) else button_color
        cylinder_color = button_hover_color if button_cylinder_rect.collidepoint(mouse_pos) else button_color
        sandbox_color = button_hover_color if button_sandbox_rect.collidepoint(mouse_pos) else button_color
        airfoil_color = button_hover_color if button_airfoil_rect.collidepoint(mouse_pos) else button_color
        
        # Draw buttons
        pg.draw.rect(display_surface, laminar_color, button_laminar_rect)
        pg.draw.rect(display_surface, cylinder_color, button_cylinder_rect)
        pg.draw.rect(display_surface, sandbox_color, button_sandbox_rect)
        pg.draw.rect(display_surface, airfoil_color, button_airfoil_rect)

        # Button text
        font = pg.font.Font(None, 36)
        laminar_text = font.render("Laminar Flow", True, (255, 255, 255))
        cylinder_text = font.render("Flow past a Cylinder", True, (255, 255, 255))
        sandbox_text = font.render("Sandbox", True, (255, 255, 255))
        airfoil_text = font.render("Flow past an Airfoil", True, (255, 255, 255))
        display_surface.blit(laminar_text, laminar_text.get_rect(center=button_laminar_rect.center))
        display_surface.blit(cylinder_text, cylinder_text.get_rect(center=button_cylinder_rect.center))
        display_surface.blit(sandbox_text, sandbox_text.get_rect(center=button_sandbox_rect.center))
        display_surface.blit(airfoil_text, airfoil_text.get_rect(center=button_airfoil_rect.center))
        
        # Title text
        font = pg.font.SysFont('none', 45)
        description1 = font.render("Real-Time Fluid Dynamics Python Engine", True, (0, 0, 0))
        font = pg.font.Font(None, 36)
        description2 = font.render("Modes:", True, (0, 0, 0))

        display_surface.blit(description1, (display_surface.get_width() / 2 - description1.get_width() / 2, 90))
        display_surface.blit(description2, (display_surface.get_width() / 2 - description2.get_width() / 2, 150))
        
        # Event handling
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if button_laminar_rect.collidepoint(event.pos):
                        return 'laminar'  # Go to the basic simulation
                    elif button_cylinder_rect.collidepoint(event.pos):
                        return 'cylinder'  # Go to the advanced simulation
                    elif button_sandbox_rect.collidepoint(event.pos):
                        return 'sandbox'  # Go to the experimental simulation
                    elif button_airfoil_rect.collidepoint(event.pos):
                        return 'airfoil'  # Go to the airfoil simulation

        pg.display.flip()
        clock.tick(60)

def animate():    
    running = True
    x, y = 0, 0
    p_x, p_y = 0, 0
    fluid = Fluid(0, 0, 0.2)
    dragging = False  # Flag to track if the mouse is being dragged
    
    button_color = (0, 0, 0)
    button_hover_color = (0, 255, 128)
    button_rect = pg.Rect((N * SCALE - 110, 10), (100, 50))
    button_text_color = (255, 255, 255)
    font = pg.font.Font(None, 36)
    
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                sys.exit()
            
            if event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button                    
                    dragging = True
                    pos = pg.mouse.get_pos()
                    x, y = pos
            
            if event.type == pg.MOUSEBUTTONUP:
                if event.button == 1:  # Left mouse button
                    dragging = False
            
            if event.type == pg.MOUSEMOTION:
                if dragging:  # Update mouse position only if dragging
                    pos = pg.mouse.get_pos()
                    x, y = pos
        
        if dragging:
            # Convert the position to grid indices
            x_idx = int(x / SCALE)
            y_idx = int(y / SCALE)
            
            # Ensure indices are within bounds
            x_idx = max(0, min(x_idx, N - 1))
            y_idx = max(0, min(y_idx, N - 1))
            
            # Calculate velocity change
            amtX = (x - p_x) / SCALE
            amtY = (y - p_y) / SCALE
            
            # Add density and velocity to the fluid simulation
            fluid.addDensity(x_idx, y_idx, 128*2)
            fluid.addVelocity(x_idx, y_idx, amtX, amtY)
            
            # Update previous position
            p_x = x
            p_y = y
              
        if choice == 'laminar':
            velocity_magnitude = 0.1
            vx = velocity_magnitude * np.cos(np.radians(np.random.randint(265, 275)))
            vy = velocity_magnitude * np.sin(np.radians(np.random.randint(265, 275)))
            
            # Stationary density and velocity addition
            fluid.addDensity(N//2, N//2, N)
            fluid.addVelocity(N//2, N//2, vx, vy)        
        
        elif choice == 'airfoil':
            velocity_magnitude = 0.1
            vx = velocity_magnitude * np.cos(np.radians(0))
            vy = velocity_magnitude * np.sin(np.radians(0))
            
            # Stationary density and veloctiy addition
            fluid.addDensity(10, N//2, N)
            fluid.addVelocity(10, N//2, vx, vy)
            
            fluid.addDensity(10, N//2 + 4, N)
            fluid.addVelocity(10, N//2 + 4, vx, vy)
            
            fluid.addDensity(10, N//2 + 8, N)
            fluid.addVelocity(10, N//2 + 8, vx, vy)
            
            fluid.addDensity(10, N//2 + 12, N)
            fluid.addVelocity(10, N//2 + 12, vx, vy)
            
            fluid.addDensity(10, N//2 - 4, N)
            fluid.addVelocity(10, N//2 - 4, vx, vy)
            
            fluid.addDensity(10, N//2 - 8, N)
            fluid.addVelocity(10, N//2 - 8, vx, vy)
            
            fluid.addDensity(10, N//2 - 12, N)
            fluid.addVelocity(10, N//2 -12, vx, vy)
        
        elif choice == 'cylinder':  
            velocity_magnitude = 0.4
            vx = velocity_magnitude * np.cos(np.radians(0))
            vy = velocity_magnitude * np.sin(np.radians(0))
            
            # Stationary density and veloctiy addition
            fluid.addDensity(20, N//2, N*2)
            fluid.addVelocity(20, N//2, vx, vy)
            
            fluid.addDensity(20, N//2 + 4, N*2)
            fluid.addVelocity(20, N//2 + 4, vx, vy)
            
            fluid.addDensity(20, N//2 + 8, N*2)
            fluid.addVelocity(20, N//2 + 8, vx, vy)
            
            fluid.addDensity(20, N//2 + 12, N*2)
            fluid.addVelocity(20, N//2 + 12, vx, vy)
            
            fluid.addDensity(20, N//2 - 4, N*2)
            fluid.addVelocity(20, N//2 - 4, vx, vy)
            
            fluid.addDensity(20, N//2 - 8, N*2)
            fluid.addVelocity(20, N//2 - 8, vx, vy)
            
            fluid.addDensity(20, N//2 - 12, N*2)
            fluid.addVelocity(20, N//2 -12, vx, vy)
        
        elif choice == 'sandbox':
            velocity_magnitude = 0.1
            
        display_surface.fill((0, 0, 0)) 
                                   
        # Simulate fluid step
        fluid.step()
        
        # Render fluid simulation regardless of dragging state
        fluid.render()
        fluid.fade()
        
        pg.display.update()
        clock.tick(60)
         
def main():
    global display_surface, display_size, clock, N, iter, SCALE, choice
    
    # Display size
    N = 128
    SCALE = 5
    iter = 16
    
    # Generate high performace pygame display
    display_surface = pg.display.set_mode((N*SCALE, N*SCALE), pg.HWSURFACE | pg.DOUBLEBUF)
    pg.display.set_caption("Fluid Simulation")
    clock = pg.time.Clock()
    
    # Start by showing the main menu
    while True:
        choice = start_menu()
        animate()
        
if __name__ == "__main__":
    pg.init()
    main()
    pg.quit()
    sys.exit()