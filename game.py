import numpy as np
import matplotlib.pyplot as plt
from time import sleep

class GameOfLife:
    def __init__(self, size=1000, initial_size=20):
        self.size = size
        self.initial_size = initial_size
        self.grid = np.zeros((size, size), dtype=np.int8)
        
    def set_pattern(self, pattern):
        """Place un pattern au centre de la grille"""
        start_row = (self.size - pattern.shape[0]) // 2
        start_col = (self.size - pattern.shape[1]) // 2
        self.grid[start_row:start_row+pattern.shape[0], 
                 start_col:start_col+pattern.shape[1]] = pattern
        
    def compute_next_state(self):
        """Calculates the next state of the grid"""
        neighbors = np.zeros((self.size, self.size), dtype=np.int8)
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                neighbors += np.roll(np.roll(self.grid, i, axis=0), j, axis=1)
        
        survival = self.grid & ((neighbors == 2) | (neighbors == 3))
        birth = (~self.grid) & (neighbors == 3)
        self.grid = (survival | birth).astype(np.int8)
        
    def simulate(self, pattern=None):
        """Launches an interactive simulation with a given pattern"""
        if pattern is not None:
            self.set_pattern(pattern)
        
        # Configuration de la fenêtre
        plt.ion()
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Variables pour le zoom et le centre
        zoom_size = 100
        center_x = self.size // 2
        center_y = self.size // 2
        
        # Gestion des interactions
        def on_click(event):
            nonlocal center_x, center_y
            if event.inaxes:
                rel_x = event.xdata / zoom_size
                rel_y = event.ydata / zoom_size
                center_x = int(center_x - zoom_size/2 + rel_x * zoom_size)
                center_y = int(center_y - zoom_size/2 + rel_y * zoom_size)
        
        def on_scroll(event):
            nonlocal zoom_size
            if event.button == 'up':
                zoom_size = max(20, int(zoom_size * 0.8))
            else:
                zoom_size = min(self.size, int(zoom_size * 1.2))
                
        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('scroll_event', on_scroll)
        
        frame = 0
        # Animation
        while plt.get_fignums():  # Continue tant que la fenêtre est ouverte
            self.compute_next_state()
            frame += 1
            
            # Extraire la région visible
            half_zoom = zoom_size // 2
            visible = self.grid[center_x-half_zoom:center_x+half_zoom, 
                              center_y-half_zoom:center_y+half_zoom]
            
            ax.clear()
            ax.imshow(visible, cmap='binary')
            ax.set_title(f'Itération: {frame}\nCellules vivantes: {np.sum(self.grid)}')
            ax.grid(True, color='gray', alpha=0.3)
            plt.pause(0.1)

