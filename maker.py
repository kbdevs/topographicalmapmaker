import numpy as np
from PIL import Image, ImageFilter
from noise import pnoise2
import random
import time
from perlin_numpy import (
    generate_perlin_noise_2d, generate_fractal_noise_2d
)



def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return_value = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return return_value

def posterize(img, levels):
    start_time = time.time()
    
    """
    Posterize an image to a specified number of levels per channel
    """
    arr = np.array(img)
    scaled = arr * (levels - 1) / 255
    posterized = np.round(scaled) * (255 / (levels - 1))
    return_value = Image.fromarray(posterized.astype(np.uint8))
    print(f"posterize took {time.time()-start_time}s")
    return return_value

def colorize_edges(edge_image, background_color, line_color):
    start_time = time.time()
    """
    Convert edge image to colored version with specified background and line colors
    with anti-aliasing
    """
    if isinstance(background_color, str):
        background_color = hex_to_rgb(background_color)
    if isinstance(line_color, str):
        line_color = hex_to_rgb(line_color)
    
    # Convert edge image to numpy array
    arr = np.array(edge_image, dtype=float) / 255.0  # Normalize to 0-1
    
    # Create RGB array
    colored = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=float)
    
    # For each color channel, interpolate between background and line color
    for i in range(3):
        colored[:, :, i] = (1 - arr) * background_color[i] + arr * line_color[i]
    
    # Convert back to uint8
    colored = colored.clip(0, 255).astype(np.uint8)
    
    return_value = Image.fromarray(colored)

    print(f"colorize took {time.time()-start_time}s")
    return return_value


def generate_perlin_noise(width, height, scale=100.0, octaves=6, contrast=2.0, blur_radius=10, levels=6,
                         background_color="#1A0D21", line_color="#7B469A", edge_smoothing=1.0):
    start_time = time.time()
    
    # Create an empty array for the noise
    noise_map = np.zeros((height, width))
    
    # Random seed for different patterns each time
    seed = random.randint(0, 100)
    
    # Generate noise for each pixel
    for y in range(height):
        for x in range(width):
            noise_value = pnoise2(x/scale, 
                                y/scale, 
                                octaves=octaves,
                                persistence=0.5,
                                lacunarity=2.0,
                                repeatx=width,
                                repeaty=height,
                                base=seed)
            
            noise_map[y][x] = (noise_value + 1) * 128

    
    # Convert to float for contrast adjustment
    noise_map = noise_map.astype(float)
    
    # Adjust contrast
    mean = np.mean(noise_map)
    noise_map = (noise_map - mean) * contrast + mean
    
    # Clip values to 0-255 range
    noise_map = np.clip(noise_map, 0, 255)
    
    # Convert to uint8 for image creation
    noise_map = noise_map.astype(np.uint8)

    
    # Create image and apply Gaussian blur
    image = Image.fromarray(noise_map, mode='L')
    image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    print(f"perlin took {time.time()-start_time}s")
    
    # Apply posterization
    image = posterize(image, levels)
    image.save("noise.jpg")
    
    # Find edges
    edges = image.filter(ImageFilter.FIND_EDGES)
    
    # Apply anti-aliasing to edges
    edges = edges.filter(ImageFilter.GaussianBlur(radius=edge_smoothing))
    
    # Colorize the edges
    colored_image = colorize_edges(edges, background_color, line_color)
    
    # Save the final image
    colored_image.save('contour.png')
    return colored_image


image = generate_perlin_noise(
    3840, 2160, # resolution
    scale=150, # higher number more zoomed in 
    octaves=6, # Number of layers of noise (more = more detail)
    contrast=3.0, # inner lines amount i guess
    blur_radius=35, # shape simplicity/roundness
    levels=3, # how much detail total
    background_color="#1A0D21",  # Dark purple
    line_color="#7B469A",        # Lighter purple
    edge_smoothing=1.0           # Amount of edge smoothing
)