# preprocess.py
import cv2
import numpy as np

def mask_image(img, left_percent, right_percent):
    """Masks the image to keep only the center strip."""
    height, width = img.shape
    start_col = int((left_percent / 100) * width)
    end_col = int((right_percent / 100) * width)
    masked = np.zeros_like(img)
    masked[:, start_col:end_col] = img[:, start_col:end_col]
    return masked

def segment_and_threshold(img, num_segments, threshold_percent):
    """Splits the image into segments and thresholds them."""
    height, width = img.shape
    segment_height = height // num_segments
    thresholded_segments = []

    for i in range(num_segments):
        segment = img[i * segment_height:(i + 1) * segment_height, :]
        threshold_value = np.percentile(segment, 100 - threshold_percent)
        thresholded_segment = np.where(segment >= threshold_value, 255, 0).astype(np.uint8)
        thresholded_segments.append(thresholded_segment)

    return np.concatenate(thresholded_segments, axis=0)

def sharpen_image(img):
    """Sharpens the image using a Laplacian kernel."""
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sharpened = np.uint8(np.absolute(laplacian))
    return img + sharpened

def apply_power_rule(image, gamma=1.0):
    """
    Apply power-law (gamma) transformation to an image.

    Args:
        image (numpy.ndarray): Grayscale image as a NumPy array.
        gamma (float): Gamma value for the transformation.

    Returns:
        numpy.ndarray: Gamma-corrected image.
    """
    # Normalize the image to range [0, 1]
    normalized_img = image / 255.0
    
    # Apply power law transformation
    gamma_corrected = np.power(normalized_img, gamma)
    
    # Scale back to range [0, 255]
    gamma_corrected = np.uint8(gamma_corrected * 255)
    
    return gamma_corrected

def apply_average_filter(image, kernel_size=3):
    """
    Apply a 3×3 average filter to an image.

    Args:
        image (numpy.ndarray): Grayscale image as a NumPy array.

    Returns:
        numpy.ndarray: Filtered image.
    """
    # Define a 3×3 kernel for averaging
    kernel = np.ones((kernel_size, kernel_size), np.float32) / 9

    # Apply the filter using cv2.filter2D
    filtered_img = cv2.filter2D(image, -1, kernel)

    return filtered_img


def preprocess_image_with_filter(image_path, gamma=1.0, mask_left=30, mask_right=70):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Apply power rule transformation (optional)
    img_power = apply_power_rule(img, gamma=gamma)

    # Apply the 3×3 average filter
    img_filtered = apply_average_filter(img_power)

    # Proceed with further processing (cropping, thresholding, etc.)
    height, width = img_filtered.shape
    start_col = int(mask_left/100 * width)
    end_col = int(mask_right/100 * width)
    img_cropped = np.zeros_like(img_filtered)
    img_cropped[:, start_col:end_col] = img_filtered[:, start_col:end_col]

    return img_cropped





