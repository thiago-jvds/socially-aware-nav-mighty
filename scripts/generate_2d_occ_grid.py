import numpy as np
from PIL import Image
import os

# 200m long, 20m wide, at 0.1m/pixel resolution
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(script_dir, '../config/hunav_config/maps')
    os.makedirs(config_dir, exist_ok=True)

    width_px, height_px = 2000, 200
    map_data = np.full((height_px, width_px), 254, dtype=np.uint8) # 254 = Free space (white)

    # In the world file, walls are at y = -4.1 and y = 4.1, and are 2m thick
    # At 0.1 res, y=0 is pixel 100. y=4.1 is pixel 141. Thickness 2m = 20 pixels.
    # Let's draw the black lines (0 = obstacle)
    map_data[131:151, :] = 0 # Positive wall
    map_data[49:69, :] = 0   # Negative wall

    # Save the PGM image
    img = Image.fromarray(map_data)
    img.save(os.path.join(config_dir, 'corridor_map.pgm'))

    # Create the YAML file
    yaml_content = """image: config/hunav_config/maps/corridor_map.pgm
    resolution: 0.1
    origin: [-100.0, -10.0, 0.0]
    negate: 0
    occupied_thresh: 0.65
    free_thresh: 0.196
    """

    with open(os.path.join(config_dir, 'corridor_map.yaml'), 'w') as f:
        f.write(yaml_content)

    print("Map generated successfully!")

if __name__ == "__main__":
    main()