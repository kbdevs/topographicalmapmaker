import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import time
import cv2

def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def posterize(img, levels):
    """Posterize an image to a specified number of levels per channel"""
    arr = np.array(img, dtype=np.float32)
    scaled = arr * (levels - 1) / 255
    posterized = np.round(scaled) * (255 / (levels - 1))
    return Image.fromarray(np.clip(posterized, 0, 255).astype(np.uint8))


def colorize_edges(edge_image, background_color, line_color):
    """Convert edge image to colored version with specified background and line colors"""
    # Convert hex colors to RGB if needed
    if isinstance(background_color, str):
        background_color = hex_to_rgb(background_color)
    if isinstance(line_color, str):
        line_color = hex_to_rgb(line_color)

    # Convert to numpy arrays and reshape for broadcasting
    arr = np.array(edge_image, dtype=np.float32) / 255.0
    background = np.array(background_color, dtype=np.float32)
    line = np.array(line_color, dtype=np.float32)

    # Reshape arr for broadcasting
    arr = arr[..., np.newaxis]

    # Vectorized calculation using pre-allocated array
    result = np.empty((*arr.shape[:-1], 3), dtype=np.float32)
    np.multiply(1 - arr, background, out=result)
    np.add(result, arr * line, out=result)

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def generate_fractal_noise(width, height, octaves, scale):
    """Generate fractal noise using multiple octaves of filtered noise"""
    noise = np.zeros((height, width), dtype=np.float32)
    persistence = 0.5
    amplitude = 1.0
    max_amplitude = 0

    for octave in range(octaves):
        freq = 2 ** octave
        octave_width = max(1, int(width / (scale * freq)))
        octave_height = max(1, int(height / (scale * freq)))
        
        base = np.random.normal(0, 1, (octave_height, octave_width)).astype(np.float32)
        sigma = 1.0 / freq
        
        base = cv2.GaussianBlur(base, (0,0), sigma)
        
        resized = cv2.resize(base, (width, height), interpolation=cv2.INTER_CUBIC)
        noise += amplitude * resized
        max_amplitude += amplitude
        amplitude *= persistence
    
    noise = (noise / max_amplitude)
    noise = ((noise - noise.min()) / (noise.max() - noise.min()) * 255)
    return noise

def find_edges_and_smooth(image, edge_smoothing):
    """Find edges and apply smoothing"""
    edges = image.filter(ImageFilter.FIND_EDGES)
    if edge_smoothing > 0:
        edges = edges.filter(ImageFilter.GaussianBlur(radius=edge_smoothing))
    edges.save("edges.jpg")
    return edges

def generate_perlin_noise(width, height, scale=100.0, octaves=6, contrast=2.0, blur_radius=10, levels=6,
                         background_color="#1A0D21", line_color="#7B469A", edge_smoothing=1.0):
    """Generate noise-based contour map"""
    start_time = time.time()
    noise_map = generate_fractal_noise(width, height, octaves, scale)
    
    if contrast != 1.0:
        mean = np.mean(noise_map)
        noise_map = (noise_map - mean) * contrast + mean
        noise_map = np.clip(noise_map, 0, 255)
    
    image = Image.fromarray(noise_map.astype(np.uint8), mode='L')
    print(f"perlin took {time.time()-start_time}s")
    
    if blur_radius > 0:
        image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    start_time = time.time()
    image = posterize(image, levels)
    print(f"posterize took {time.time()-start_time}s")
    image.save("noise.jpg")
    
    start_time = time.time()
    edges = find_edges_and_smooth(image, edge_smoothing)
    print(f"find edges took {time.time()-start_time}s")
    
    colored_image = colorize_edges(edges, background_color, line_color)
    print(f"colorize took {time.time()-start_time}s")
    
    return colored_image

if __name__ == "__main__":
    # Image settings
    WIDTH = 3840
    HEIGHT = 2160
    OVERLAY = False
    AUTO_SCALE = True
    SCALE = 120

    # Perlin noise settings
    if AUTO_SCALE:
        SCALE = (WIDTH/3840)*105
    PERLIN_SETTINGS = {
        'scale': SCALE,        # Higher number = more zoomed in
        'octaves': 3,        # Number of layers of noise
        'contrast': 2.0,     # Inner lines amount 
        'blur_radius': 15,   # Shape simplicity/roundness
        'levels': 5,         # Posterize level
        # 'background_color': "#2B2231",  # Dark purple
        # 'line_color': "#FFB0E4",        # Light purple
        'background_color': "#1A0D21",  # Dark purple
        'line_color': "#7B469A",        # Light purple
        'edge_smoothing': 1.0           # Edge smoothing amount
    }
    
    # Circle overlay settings
    BLUR_RADIUS = (WIDTH/3840)*1400
    CIRCLE_SETTINGS = {
        'radius': WIDTH/4,           # Radius of circles
        'blur_radius': BLUR_RADIUS,         # Blur amount for circles
        'color': (255, 192, 203, 255)  # Pink color (R,G,B,A)
    }
    
    # Circle positions
    CIRCLES = [
        {'x': 0, 'y': 0},                    # Top left circle
        {'x': WIDTH, 'y': HEIGHT}  # Bottom right circle
    ]

    # Generate base perlin noise image
    lines = generate_perlin_noise(WIDTH, HEIGHT, **PERLIN_SETTINGS)

    if OVERLAY:
        # Create transparent overlay for circles
        overlay = Image.new('RGBA', (WIDTH, HEIGHT), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Draw circles
        for circle in CIRCLES:
            draw.ellipse([
                circle['x'] - CIRCLE_SETTINGS['radius'], 
                circle['y'] - CIRCLE_SETTINGS['radius'],
                circle['x'] + CIRCLE_SETTINGS['radius'], 
                circle['y'] + CIRCLE_SETTINGS['radius']
            ], fill=CIRCLE_SETTINGS['color'])

        # Apply blur to overlay
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=CIRCLE_SETTINGS['blur_radius']))

        # Composite images and save
        lines.paste(overlay, (0,0), mask=overlay)
    lines.save('contour.png')