import pygame as pg
import sys

def start_menu(N, scale, surface):
    '''
    Creates pygame application start menu.

    Args:
        N (int): grid dimension.
        scale (int): screen scaling from underlying grid dimension.
        surface (pg.display.set_mode()): pygame Surface object.
    ''' 

    # Display scaled background image.
    grid_scale = N * scale
    image = pg.image.load('../img/start_menu.png')
    scaled_image = pg.transform.scale(image, (grid_scale, grid_scale))
    
    # Define start button.
    button_color = (0, 128, 255)
    button_hover_color = (0, 255, 128)
    button_rect = pg.Rect((grid_scale // 3 - 50, grid_scale // 2 + 40), (grid_scale // 2, 50))
    
    running = True
    while running:
        surface.blit(scaled_image, (0, 0))  # Fill surface with background image.

        # Adjust button color according to whether mouse is hovering botton area.
        mouse_pos = pg.mouse.get_pos()
        color = button_hover_color if button_rect.collidepoint(mouse_pos) else button_color
        
        # Draw start button and corresponding text.
        pg.draw.rect(surface, color, button_rect)
        font = pg.font.Font(None, 36)
        text = font.render("Start", True, (255, 255, 255))
        surface.blit(text, text.get_rect(center=button_rect.center))
        
        # Draw title text.
        font = pg.font.SysFont('none', 45)
        description = font.render("Real-Time Fluid Dynamics Python Engine", True, (0, 0, 0))
        font = pg.font.Font(None, 36)
        surface.blit(description, (surface.get_width() / 2 - description.get_width() / 2, 150))
        
        for event in pg.event.get():
            if event.type == pg.QUIT: # Exit process if pygame app is quit.
                pg.quit()
                sys.exit()
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1: # If left click start button start fluid sim.
                    if button_rect.collidepoint(event.pos):
                        return ''

        pg.display.flip()