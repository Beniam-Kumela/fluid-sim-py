import pygame as pg
import sys
import numpy as np
import cv2
import pygame_gui
from fluid import Fluid

def get_grid_pos(mouse_pos, scale, N):
    '''
    Converts pygame mouse positions those that can be interpreted by the grid.

    Args:
        mouse_pos (tuple): x and y positions
        scale (int): screen scaling from underlying grid dimension.
        N (int): grid dimension
    '''
    x, y = mouse_pos
    x_idx = np.clip(int(x / scale), 0, N - 1)
    y_idx = np.clip(int(y / scale), 0, N - 1)
    return x_idx, y_idx

def animate(N, scale, surface, clock, font, diffusion=0, viscosity=0, dt=0.2, fade_rate=0.95):
    '''
    Handles all application animation routines which includes surface updates, FPS counter, and a responsive GUI.

    Args:
        N (int): grid dimension.
        scale (int): screen scaling from underlying grid dimension.
        surface (pg.display.set_mode()): pygame Surface object.
        clock (pg.time.Clock()): pygame Clock object.
        font (pg.font.SysFont): pygame Font object.
        diffusion (float): diffusion coefficient for velocity field.
        viscosity (float): diffusion coefficient for density field.
        dt (float): time interval.
        fade_rate (float): controls fluid dye color depletion.
    '''

    # Initialize GUI.
    manager = pygame_gui.UIManager((N*scale, N*scale))

    # Color slider and label for red line color value.
    line_color_r_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pg.Rect(10, 10, 200, 30),
        start_value=255,
        value_range=(0, 255),
        manager=manager
    )
    line_color_r_label = pygame_gui.elements.UILabel(
    relative_rect=pg.Rect(200, 15, 100, 20),
    text=f"R: {int(line_color_r_slider.get_current_value())}",
    manager=manager
    )
    
    # Color slider and label for green line color value.
    line_color_g_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pg.Rect(10, 50, 200, 30),
        start_value=255,
        value_range=(0, 255),
        manager=manager
    )
    line_color_g_label = pygame_gui.elements.UILabel(
    relative_rect=pg.Rect(200, 55, 100, 20),
    text=f"G: {int(line_color_g_slider.get_current_value())}",
    manager=manager
    )

    # Color slider and label for blue line color value.
    line_color_b_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pg.Rect(10, 90, 200, 30),
        start_value=255,
        value_range=(0, 255),
        manager=manager
    )
    line_color_b_label = pygame_gui.elements.UILabel(
    relative_rect=pg.Rect(200, 95, 100, 20),
    text=f"B: {int(line_color_b_slider.get_current_value())}",
    manager=manager
    )

    # Slider and label for line width adjustment.
    line_width_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pg.Rect(10, 130, 200, 30),
        start_value=5,
        value_range=(1, 50),
        manager=manager
    )
    line_width_label = pygame_gui.elements.UILabel(
        relative_rect=pg.Rect(210, 135, 100, 20),
        text=f"Width: {int(line_width_slider.get_current_value())}",
        manager=manager
    )

    # Slider and label for source fluid velocity adjustment.
    fluid_velocity_slider = pygame_gui.elements.UIHorizontalSlider(
        relative_rect=pg.Rect(10, 170, 200, 30),
        start_value=0.1,
        value_range=(0.01, 1.0),
        manager=manager
    )
    fluid_velocity_label = pygame_gui.elements.UILabel(
        relative_rect=pg.Rect(200, 175, 150, 20),
        text=f"Velocity: {fluid_velocity_slider.get_current_value():.2f}",
        manager=manager
    )

    # Buttons to start screen recording and go back to home menu.
    record_button = pygame_gui.elements.UIButton(
        relative_rect=pg.Rect(N*scale - 135, 90, 125, 30),
        text='Start Recording',
        manager=manager
    )
    back_button = pygame_gui.elements.UIButton(
        relative_rect=pg.Rect(N*scale - 110, 50, 100, 30),
        text='Start Menu',
        manager=manager
    )
    
    # Initialize starting pygame and video recording conditions.
    running = True
    dragging = False
    drawing = False
    recording = False
    current_stroke = None
    video_writer = None

    # Initialize fluid properties.
    fluid_sources = []
    dots = []
    x, y = 0, 0
    x0, y0 = 0, 0
    boundary_mask = np.zeros((N, N), dtype=bool)
    fluid = Fluid(diffusion, viscosity, fade_rate, dt, N, scale, surface, boundary_mask)
    
    while running:
        time_delta = clock.tick(60) / 1000.0
        for event in pg.event.get():
            
            if event.type == pg.QUIT: # Exit process if pygame app is quit.
                pg.quit()
                sys.exit()
            
            manager.process_events(event)

            
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == back_button: # Stop recording screen when back to home button pressed.
                    if video_writer is not None:
                        video_writer.release()
                        video_writer = None
                    return
                    
                if event.ui_element == record_button: 
                    recording = not recording
                    if recording: # Begin recording when starting recording pressed.
                        record_button.set_text('Stop Recording')
                        video_writer = cv2.VideoWriter(
                            'fluid_sim.mp4',
                            cv2.VideoWriter_fourcc(*'mp4v'),
                            60,
                            (N*scale, N*scale)
                        )
                    else: # Stop recording screen when stop recording pressed.
                        record_button.set_text('Start Recording')
                        video_writer.release()
                        video_writer = None

            
            if event.type == pg.MOUSEBUTTONDOWN:

                if event.button == 1: 
                    dragging = True
                    keys = pg.key.get_pressed()

                    if keys[pg.K_s]: # When left-click + S record mouse position as fluid source. 
                        x, y = pg.mouse.get_pos()
                        x_idx, y_idx = get_grid_pos((x, y), scale, N)

                        if (x_idx, y_idx) not in fluid_sources:
                            fluid_sources.append((x_idx, y_idx))
                
                if event.button == 3: # When right click, record mouse position as stroke.
                    drawing = True
                    x, y = pg.mouse.get_pos()
                    x_idx, y_idx = get_grid_pos((x, y), scale, N)
                    current_stroke = [(x, y)]
                    dots.append(current_stroke)
                    boundary_mask[x_idx, y_idx] = True

            if event.type == pg.MOUSEBUTTONUP: # When release mouse, reset all variables.

                if event.button == 1:
                    dragging = False

                if event.button == 3:
                    drawing = False
                    current_stroke = None
            
            if event.type == pg.MOUSEMOTION: 

                if dragging:
                    x, y = pg.mouse.get_pos()

                if drawing: # Append mouse positions as strokes if right click is also being pressed while moving. Update the boundary mask correspondingly. 
                    x, y = pg.mouse.get_pos()
                    x_idx, y_idx = get_grid_pos((x, y), scale, N)

                    if current_stroke is not None:
                        current_stroke.append((x, y))
                    
                    boundary_mask[x_idx, y_idx] = True

        if dragging: # When mouse moving while left click, add fluid density and velocity at current mouse position.
            x_idx, y_idx = get_grid_pos((x, y), scale, N)
            amtX = (x - x0) / scale
            amtY = (y - y0) / scale
            fluid.addDensity(x_idx, y_idx, amount=N*2)
            fluid.addVelocity(x_idx, y_idx, amtX, amtY)
            x0 = x
            y0 = y

        for (sx, sy) in fluid_sources: # Continuously update fluid density and velocity (pointing from left to right) from stored fluid source coordinates. 
            velocity = fluid_velocity_slider.get_current_value()
            angle = np.radians(np.random.randint(0, 5))
            vx = velocity * np.cos(angle)
            vy = velocity * np.sin (angle)
            fluid.addDensity(sx, sy, amount=N*2)
            fluid.addVelocity(sx, sy, vx, vy)

        # Main fluid update sequence.
        surface.fill((0, 0, 0))
        fluid.step(boundary_mask)
        fluid.render()
        fluid.fade()
        fluid.update_boundary_mask(dots)
        
        for stroke in dots: # Draw boundaries from recorded positions and chosen colors.
            color_r_val = int(line_color_r_slider.get_current_value())
            color_g_val = int(line_color_g_slider.get_current_value())
            color_b_val = int(line_color_b_slider.get_current_value())
            width_val = int(line_width_slider.get_current_value())

            if len(stroke) >= 2: # Draw lines if at least two positions in dots array.
                pg.draw.lines(surface, (color_r_val, color_g_val, color_b_val), False, stroke, width_val)
            
            if len(stroke) == 1: # Otherwise, draw a circle with radius equal to chosen line width.
                pg.draw.circle(surface, (color_r_val, color_g_val, color_b_val), stroke[0], width_val)
        
        if recording and video_writer is not None: # Store current video frame.
            frame = pg.surfarray.array3d(surface)
            frame = np.flipud(np.rot90(frame))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

        # Update label text with current slider values.
        line_color_r_label.set_text(f"R: {int(line_color_r_slider.get_current_value())}")
        line_color_g_label.set_text(f"G: {int(line_color_g_slider.get_current_value())}")
        line_color_b_label.set_text(f"B: {int(line_color_b_slider.get_current_value())}")
        line_width_label.set_text(f"Width: {int(line_width_slider.get_current_value())}")
        fluid_velocity_label.set_text(f"Velocity: {fluid_velocity_slider.get_current_value():.2f}")

        # Update GUI manager each time step and draw on the pygame Surface.
        manager.update(time_delta)
        manager.draw_ui(surface)

        # Update FPS counter.
        fps = int(clock.get_fps())
        fps_text = font.render(f"FPS: {fps}", True, (255, 255, 255))
        surface.blit(fps_text, (N*scale - 100, 10))

        pg.display.update()