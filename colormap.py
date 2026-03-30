import numpy as np
import cv2

def get_distinct_colormap(n):
    """
    Generate a distinct colormap
     with `n` colors. The first color is black (for 0),
    and the rest are distinct random colors.
    """
    # Generate random colors for the components (excluding black)
    np.random.seed(42)  # For reproducibility
    colors = np.random.randint(0, 256, size=(n - 1, 3))  # n-1 distinct colors
    
    # Add black as the first color (for index 0)
    colormap = np.vstack([[0, 0, 0], colors])
    
    return colormap

def apply_colormap_to_component(comp):
    """
    Apply the distinct colormap to the component image.
    """
    # Get the maximum label (excluding 0) to define how many colors are needed
    num_colors = np.max(comp) + 1
    
    # Get the colormap
    colormap = get_distinct_colormap(num_colors)
    
    # Create an empty image to store the colored result
    colored_image = np.zeros((comp.shape[0], comp.shape[1], 3), dtype=np.uint8)
    
    # Map each component to its corresponding color
    for i in range(comp.shape[0]):
        for j in range(comp.shape[1]):
            labeled_value = comp[i, j]
            colored_image[i, j] = colormap[labeled_value]
    
    return colored_image

def show_component_with_cv2(comp):
    """
    Display the component image using OpenCV.
    """
    colored_image = apply_colormap_to_component(comp)
    
    # Show the image using OpenCV
    cv2.imshow('Component', colored_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
