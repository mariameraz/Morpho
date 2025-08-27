#################################################################################################
# Load modules
#################################################################################################

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
from sklearn.neighbors import NearestNeighbors
from scipy.stats import circstd
from scipy.spatial import ConvexHull
import math
import os
from ..utils.common_functions import load_img, detect_label, detect_img_name, plot_img, pdf_to_img, is_contour_valid
from .functions import AnnotatedImage
from .. import valid_extensions
import time
import pandas as pd 
from tqdm import tqdm
import psutil
from scipy.spatial.distance import cdist
from scipy.stats import circmean
from scipy.optimize import linear_sum_assignment


#################################################################################################
# Determinate inner pericarp area
#################################################################################################
def inner_pericarp_area(locules, contours, px_per_cm_width, px_per_cm_length, img=None,
                        draw_inner_pericarp=False, 
                        use_ellipse=False, epsilon=0.0001, rel_tol=1e-6,
                        contour_thickness=2, contour_color=(0, 240, 240)):
    """
    Calculates and visualizes the inner pericarp area (enclosing locules) using either ellipse fitting 
    or convex hull approximation. Returns the annotated image and calculated area in square pixels.

    Args:
        REQUIRED:
        - locules (List[int]): Indices of contours in `contours` that correspond to fruit locules.
        - contours (List[numpy.ndarray]): Detected contours (as returned by cv2.findContours()).
         - px_per_cm_width (float): Pixels per centimeter along the shorter side of the image (width).
        - px_per_cm_length (float): Pixels per centimeter along the longer side of the image (length).

        OPTIONAL:
        - img (numpy.ndarray): Input BGR image (uint8) where contours will be drawn (if draw_inner_pericarp=True).
        - draw_inner_pericarp (bool): If True, draws the contour on `img` (default: False).
        - use_ellipse (bool): If True, uses ellipse fitting; otherwise uses convex hull (default: False).
        - epsilon (float): Smoothing factor as percentage of arc length (range: [0, 1], default: 0.0001).
        - rel_tol (float): Relative tolerance for isotropy determination (default: 1e-6).
        - contour_thickness (int): Thickness of drawn contours in pixels (default: 2).
        - contour_color (Tuple[int, int, int]): BGR color for contours (default: cyan (0, 240, 240)).

    Returns:
            - img: Input image with drawn contours (if draw_inner_pericarp=True), otherwise unchanged.
            - area_cm2 (float): Calculated area in square centimeters.

    Raises:
        - ValueError: If `epsilon` is outside [0, 1] or `contours`/`loculi` indices are invalid.
        - cv2.error: If OpenCV operations fail (e.g., insufficient points for ellipse fitting).

    Notes:
        - For ellipse fitting: Requires ≥5 contour points (returns area=0 if insufficient).
        - Convex hull: More stable for irregular shapes but may overestimate area.
        - Smoothing (epsilon): Lower values preserve detail; higher values simplify the contour.
        - Color convention: Uses BGR (OpenCV standard) for `contour_color`.
        - Area conversion: Handles both isotropic and anisotropic pixel scaling.
    """

    # Validación: si draw_inner_pericarp es True, img no puede ser None
    if draw_inner_pericarp and img is None:
        raise ValueError("img cannot be None when draw_inner_pericarp=True")

    if len(locules) == 0:
        return img, 0, 0  # Returns unchanged image and area == 0 for both px and cm
        
    all_points = np.vstack([contours[i] for i in locules])
    area_px = 0  # Area in square pixels
    
    if use_ellipse:
        # Adjust ellipse
        if all_points.shape[0] >= 5:  # Require at least 5 points for fitEllipse
            ellipse = cv2.fitEllipse(all_points.astype(np.float32)) # Convert points to float32 (required by fitEllipse) and calculates the best-fit ellipse
            if draw_inner_pericarp:
                cv2.ellipse(img, ellipse, contour_color, contour_thickness) # Draw the ellipse on the image
            
            a, b = ellipse[1][0]/2, ellipse[1][1]/2 # Get semi-major (a) and semi-minor (b) axes.
            area_px = np.pi * a * b # Calculate the area of the ellipse in pixels

    else:
        hull = cv2.convexHull(all_points) # Calculates the convex hull of all points
        epsilon_val = epsilon * cv2.arcLength(hull, True) # Smooth the contour multiplying its perimeter by epsilon
        smoothed_hull = cv2.approxPolyDP(hull, epsilon_val, True) # Approximates contour with smoothed polygon
        if draw_inner_pericarp:
            cv2.drawContours(img, [smoothed_hull], -1, contour_color, contour_thickness) # Draw the smoothed contour
        area_px = cv2.contourArea(smoothed_hull) # Calculate the area of the polygon in pixels
    
    # Convert area from pixels to cm²
    if math.isclose(px_per_cm_width, px_per_cm_length, rel_tol=rel_tol):
        # Isotropic pixels: area conversion is straightforward
        px_per_cm = (px_per_cm_width + px_per_cm_length) / 2
        area_cm2 = area_px / (px_per_cm ** 2)
    else:
        # Anisotropic pixels: area conversion considers different scaling factors
        area_cm2 = area_px / (px_per_cm_width * px_per_cm_length)
    
    return area_cm2

#################################################################################################
# Determinate rotated bounding box around fruits
#################################################################################################

def rotate_box(contour, px_per_cm_width, px_per_cm_length, img = None, draw_box=False, 
               box_color=(255, 180, 0), box_thickness=3):
    """
    Calculates the rotated bounding box of a contour and its dimensions in pixels and centimeters.
    Optionally draws the bounding box on the input image.

    Args:
        REQUIRED:
        - contour (numpy.ndarray): Contour of the object (e.g., fruit) as returned by cv2.findContours().
        - px_per_cm_width (float): Pixels per centimeter along the shorter side of the image (width).
        - px_per_cm_length (float): Pixels per centimeter along the longer side of the image (length).

        OPTIONAL:
        - img (numpy.ndarray): BGR image where the bounding box will be drawn (if draw_box=True).
        - draw_box (bool): If True, draws the bounding box on `img` (default: True).
        - box_color (Tuple[int,int,int]): BGR color for the bounding box (default: light blue (255, 180, 0)).
        - box_thickness (int): Thickness of the bounding box lines in pixels (default: 3).

    Returns:
        Tuple[float, float] containing:
            - box_height_cm: Height in centimeters (calculated using average resolution).
            - box_width_cm: Width in centimeters (calculated using average resolution).
            
    Notes:
        - The bounding box is axis-independent (rotated to fit the contour tightly).
        - Dimensions are converted to cm using the average of `px_per_cm_width` and `px_per_cm_length`.
        - The "height" is always the longer side, and "width" the shorter side, regardless of orientation.
    """

    if draw_box and img is None:
        raise ValueError("img cannot be None when draw_box=True")
    
    rotated_rect = cv2.minAreaRect(contour) # Computes the smallest rotated rectangle that encloses a contour (fruit)
    (center, (width_px, height_px), angle) = rotated_rect # Obtain the width and height in pixels of the rectangle computed
    # The center coordinates and the rotation angle are stored in rotated_rect
    
    box_points = cv2.boxPoints(rotated_rect) # Convert the rotated box into its 4 corner points
    box_points = np.int0(box_points) # Round the points to integer values for drawing purposes
    
    box_length_px = max(width_px, height_px) # Determinate the height (maximum value)
    box_width_px = min(width_px, height_px) # Determinate the width value (minimun value)
    

    if math.isclose(px_per_cm_width, px_per_cm_length, rel_tol=1e-6):
        px_per_cm_avg = (px_per_cm_width + px_per_cm_length) / 2
        box_length_cm = box_length_px / px_per_cm_avg
        box_width_cm = box_width_px / px_per_cm_avg
    else:
        box_length_cm = box_length_px / px_per_cm_width  # Major axis
        box_width_cm = box_width_px / px_per_cm_length   # Minor axis
    
    if draw_box: 
        cv2.drawContours(img, [box_points], 0, box_color, box_thickness) # Draw the rotated box on the image as a light blue rectangle
    
    return box_length_cm, box_width_cm


#################################################################################################
# Create fruit mask
#################################################################################################

def create_mask(
    img_hsv,lower_hsv=None, upper_hsv=None,
    n_iteration=1, n_kernel=7, kernel_open = None,
    kernel_close = None, canny_min=30, canny_max=100,
    plot=True, plot_size=(20,10), fig_axis = False,
):
    """
    Creates a binary mask to segment objects from an HSV image using color thresholding, morphological operations and edge detection
    
    Arguments:
    
    REQUIRED:
        - img_hsv (numpy.ndarray): Image in HSV format.

    OPTIONAL:
        - lower_hsv (Tuple[int, int, int]): Lower bound for HSV background detection (default: (0,0,0)).
        - upper_hsv (Tuple[int, int, int]): Upper bound for HSV background detection default: (180,255,30).
        - n_iteration (int): Number of iterations for morphological operations.
        - n_kernel (int): Kernel size (odd) for morphological ops when kernel_open/kernel_close are None (default: 7).
        - kernel_open (int): Custom kernel size for opening (overrides n_kernel if set).
        - kernel_close (int): Custom kernel size for closing (overrides n_kernel if set).
        - canny_min (int): First threshold for Canny edge detection.
        - canny_max (int): Second threshold for Canny edge detection.
        - plot (numpy.ndarray): Whether to plot the resulting mask as a binary image.
        - figsize (Tuple[int, int]): Figure size for plotting.
        
    Returns:
        - Binary mask as 2D numpy array (numpy.dnarray)
    
    Raises:
        - ValueError: If parameters are invalid
        - TypeError: If input types are incorrect
        - RuntimeError: If image processing fails
    """
    try:
        # Input validation
        if not isinstance(img_hsv, np.ndarray):
            raise TypeError("Input image must be a numpy array")
            
        if img_hsv.ndim != 3 or img_hsv.shape[2] != 3:
            raise ValueError("Image must be in HSV format (3 channels)")
            
        if not isinstance(n_iteration, int) or n_iteration < 1:
            raise ValueError("n_iteration must be a positive integer")
            
        if not isinstance(n_kernel, int) or n_kernel < 1 or n_kernel % 2 == 0:
            raise ValueError("n_kernel must be a positive odd integer")
            
        if img_hsv.dtype != np.uint8:
            raise ValueError("HSV image must be uint8 type (0-180 for H, 0-255 for S/V)")
    
        # Set default HSV values for black/dark backgrounds if not provided
        if lower_hsv is None:
            lower_hsv = np.array([0, 0, 0], dtype=np.uint8)
        elif isinstance(lower_hsv, list):
            lower_hsv = np.array(lower_hsv, dtype=np.uint8)
            
        if upper_hsv is None:
            upper_hsv = np.array([180, 255, 30], dtype=np.uint8)
        elif isinstance(upper_hsv, list):
            upper_hsv = np.array(upper_hsv, dtype=np.uint8)

        # Validate HSV bounds
        if not isinstance(lower_hsv, np.ndarray) or lower_hsv.shape != (3,):
            raise ValueError("lower_hsv must be a numpy array with shape (3,)")
        if not isinstance(upper_hsv, np.ndarray) or upper_hsv.shape != (3,):
            raise ValueError("upper_hsv must be a numpy array with shape (3,)")
            
        if (lower_hsv > upper_hsv).any():
            raise ValueError("All values in lower_hsv must be <= corresponding values in upper_hsv")

        
        mask_background = cv2.inRange(img_hsv, lower_hsv, upper_hsv) # Create binary mask where [lower_hsv, upper_hsv] are white (255) (background) and others black (0) (fruits/label)
        if mask_background is None:
            raise RuntimeError("Failed to create initial mask")

        mask_inverted = cv2.bitwise_not(mask_background) # Invert the binary mask to focus on foreground objects (fruits/label)
        
        kernel_open = kernel_open if kernel_open is not None else n_kernel
        kernel_close = kernel_close if kernel_close is not None else n_kernel

        kernel_o = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_open, kernel_open)) # Creates an elliptical kernel for morphological operations
        kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_close, kernel_close)) 

        mask_open = cv2.morphologyEx(mask_inverted, cv2.MORPH_OPEN, kernel_o, iterations=n_iteration) # Opening (erosion followed by dilation) to remove small noise
        mask_closed = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_c, iterations=n_iteration) # Closing (dilation followed by erosion) to fill small holes
        
        blurred = cv2.GaussianBlur(mask_closed, (n_kernel, n_kernel), 0) # Applies Gaussian blur to smooth edges
        edges = cv2.Canny(blurred, canny_min, canny_max) # Detects edges using the Canny algorithm
        
        final_mask = cv2.bitwise_or(mask_closed, edges) # Combines the closed mask with edges to refine boundaries

        if plot:# Displays the final mask with/without axes based on the `axis` parameter
            plot_img(final_mask, 
                     fig_axis=fig_axis, 
                     plot_size=plot_size, 
                     metadata = False, gray = True)

        return final_mask
        
    except cv2.error as e:
        raise RuntimeError(f"OpenCV error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {str(e)}")

#################################################################################################
# Calculate pixels per centimeter 
#################################################################################################

def px_per_cm(img, size = 'letter_ansi',  width_cm = None, length_cm = None):
    """
    Calculates pixel density (pixels/cm) from an image and physical dimensions.
    Always treats the longest physical dimension as height and shortest as width,
    regardless of image orientation (portrait/landscape).

    Arguments:

    REQUIRED:
        - img (numpy.ndarray): Input image (2D grayscale or 3D BGR array).

    OPTIONAL:
        - size (str): Predefined physical size ('letter_ansi', 'legal_ansi', 'a4_iso', 'a3_iso') (default: 'letter_ansi').
        - width_cm (float): Custom physical width in cm (short side, overrides size if provided with height_cm).
        - length_cm (float): Custom physical length in cm (long side, overrides size if provided with width_cm).

    Returns:
        Tuple[float, float, float, float] containing:
            - pixels_per_cm_x: Pixel density for horizontal dimension (short side) (float)
            - pixels_per_cm_y: Pixel density for vertical dimension (long side) (float)
            - used_width_cm: Physical width used (short dimension in cm) (float)
            - used_length_cm: Physical length used (long dimension in cm) (float)

    Raises:
        - ValueError: If parameters are invalid (length < width, negative values, etc.)
        - TypeError: If input types are incorrect
        - RuntimeError: If calculation fails

    Note:
        - Physical dimensions are automatically converted to (long_side, short_side) format
        - Image dimensions are matched to physical proportions (longest image side to longest paper side)
        - For custom dimensions: length_cm must be >= width_cm (long side must be length)
    """
    try:
        # Input validation
        if not isinstance(img, np.ndarray):
            raise TypeError("Input must be a numpy array")
            
        if img.ndim not in [2, 3]:
            raise ValueError("Image must be 2D (grayscale) or 3D (color)")
            
        if size not in ['letter_ansi', 'legal_ansi', 'a4_iso', 'a3_iso'] and (width_cm is None or length_cm is None):
            raise ValueError("Provide either valid physical size or custom dimensions")
            
        if width_cm is not None and (not isinstance(width_cm, (int, float)) or width_cm <= 0):
            raise ValueError("width_cm must be positive")
            
        if length_cm is not None and (not isinstance(length_cm, (int, float)) or length_cm <= 0):
            raise ValueError("length_cm must be positive")

        # Scanner paper sizes (stored as width x height where height >= width)
        paper_sizes = {
            'letter_ansi': (21.6, 27.9),   # Short x Long
            'legal_ansi': (21.59, 35.56),
            'a4_iso': (21.0, 29.7),
            'a3_iso': (29.7, 42.0)
        }

        # Get physical dimensions (force long side as height)
        if width_cm is not None and length_cm is not None:
            if width_cm > length_cm:
                raise ValueError('!! Error: height_cm < width_cm. Height must be the longest paper dimension.')
            used_length_cm, used_width_cm = max(width_cm, length_cm), min(width_cm, length_cm)
        else:
            used_length_cm, used_width_cm = max(paper_sizes[size]), min(paper_sizes[size])

        # Get image dimensions (long side as height)
        img_height_px, img_width_px = max(img.shape[:2]), min(img.shape[:2])

        # Calculate density
        px_per_cm_width = img_width_px / used_width_cm  # Short side
        px_per_cm_length = img_height_px / used_length_cm  # Long side

        return px_per_cm_width, px_per_cm_length, used_width_cm, used_length_cm
        
    except Exception as e:
        raise RuntimeError(f"Calculation error: {str(e)}")
    
#################################################################################################
# Detect fruit contours in a binary mask
#################################################################################################

def find_fruits(
    binary_mask,
    min_locule_area = 50,
    min_locules_per_fruit = 1,
    min_circularity = 0.4,
    max_circularity = 1.0,
    min_aspect_ratio = 0.3,
    max_aspect_ratio = 3.0,
    rescale_factor = None,
    contour_approximation = cv2.CHAIN_APPROX_SIMPLE,
    contour_filters = None):
    """
    Detects fruit contours in a binary mask using morphological filtering criteria and returns 
    a mapping of fruits to their internal cavities (locules).

    Args:
        REQUIRED:
            - binary_mask (np.ndarray): Binary image where white represents objects (fruits) and black background (uint8).
        
        OPTIONAL:
            - min_locule_area (int): Minimum pixel area for a locule to be considered valid (default: 50).
            - min_locules_per_fruit (int): Minimum number of locules required to classify as fruit (default: 1).
            - min_circularity (float): Minimum circularity threshold (0-1, 1=perfect circle) (default: 0.4).
            - max_circularity (float): Maximum circularity threshold (default: 1.0).
            - min_aspect_ratio (float): Minimum width/height ratio for valid contours (default: 0.3).
            - max_aspect_ratio (float): Maximum width/height ratio (default: 3.0).
            - rescale_factor (float): Scaling factor (0.0-1.0) for faster processing (None=no rescaling).
            - contour_approximation: OpenCV contour approximation method (default: CHAIN_APPROX_SIMPLE).
            - contour_filters (Dict): Dictionary to override default filter values.

    Returns:
        Tuple[List[np.ndarray], Dict[int, List[int]]] containing:
            - contours: List of all detected contours (in original coordinates)
            - fruit_locules_map: Dictionary mapping fruit indices to lists of locule indices

    Raises:
        ValueError: If input parameters are invalid
        cv2.error: If OpenCV contour detection fails
    """
    # Validate rescale_factor
    if rescale_factor is not None and not (0 < rescale_factor <= 1):
        raise ValueError('rescale_factor must be between 0 and 1')

    # Store original dimensions for later rescaling
    original_shape = binary_mask.shape[:2] if rescale_factor is not None else None

    # Conditional image resizing
    if rescale_factor is not None and rescale_factor < 1: # Check that rescale_factor is a value between 0 and 1
        new_size = (int(binary_mask.shape[1] * rescale_factor), 
                   int(binary_mask.shape[0] * rescale_factor))
        resized_mask = cv2.resize(binary_mask, new_size, interpolation=cv2.INTER_NEAREST)
        min_locule_area = int(min_locule_area * (rescale_factor ** 2))
    else:
        resized_mask = binary_mask.copy()

    # Configure filters with validation
    default_filters = {
        'min_area': min_locule_area, 
        'min_circularity': min_circularity,
        'max_circularity': max_circularity,
        'min_aspect_ratio': min_aspect_ratio,
        'max_aspect_ratio': max_aspect_ratio
    }
    
    if contour_filters:
        invalid_keys = set(contour_filters.keys()) - set(default_filters.keys())
        if invalid_keys:
            raise ValueError(f"Invalid filter keys: {invalid_keys}. Valid keys are: {list(default_filters.keys())}")
    
    filters = {**default_filters, **(contour_filters or {})}

    # Input validation
    if not isinstance(resized_mask, np.ndarray) or resized_mask.dtype != np.uint8:
        raise ValueError("Input mask must be uint8 numpy array")
    
    if any(v <= 0 for v in [min_locule_area, *filters.values()]):
        raise ValueError("All parameters must be positive values")

    # Contour detection
    contours, hierarchy = cv2.findContours(
        resized_mask, 
        cv2.RETR_TREE,
        contour_approximation
    )
    
    if not contours or hierarchy is None:
        return [], {}

    hierarchy = hierarchy[0]  # Simplify hierarchy structure

    # Process contours and build fruit-locules mapping
    fruit_locules_map = {}
    for i, contour in enumerate(contours):
        # Check if contour is top-level (fruit candidate) and passes filters
        if hierarchy[i][3] == -1 and is_contour_valid(contour, filters):
            # Find all valid child contours (locules)
            locules = [
                j for j, h in enumerate(hierarchy)
                if h[3] == i and  # Is direct child
                cv2.contourArea(contours[j]) >= filters['min_area']
            ]
            
            # Only register as fruit if minimum locules count is met
            if len(locules) >= min_locules_per_fruit:
                fruit_locules_map[i] = locules

    # Rescale contours back to original coordinates if needed
    if rescale_factor is not None and rescale_factor < 1:
        scale_x = original_shape[1] / resized_mask.shape[1]
        scale_y = original_shape[0] / resized_mask.shape[0]
        
        rescaled_contours = [
            (contour.astype(np.float32) * np.array([scale_x, scale_y])).astype(np.int32)
            for contour in contours
        ]
        contours = rescaled_contours
            
    return contours, fruit_locules_map

#################################################################################################
# Merge close locules
#################################################################################################

def merge_locules(locules_indices, contours, min_distance=0, max_distance=50, min_area=10):
    """
    Merge fragmented locule contours that are close to each other into single contours. 
    It helps consolidate small, nearby fragments into larger, more meaningful shapes.
    
    Args:
        REQUIRED:
            - locules_indices (List[]): List of indices pointing to locule contours in the contour list
            - contours (List[np.ndarray]): List of all detected contours (including non-locules)
        OPTIONAL:
            - min_distance (int): Minimun allowed distance between locules to consider them part of the same structure (default: 50 pixels)
            - max_distance (int): Maximum allowed distance between locules to consider them part of the same structure (default: 50 pixels)
            - min_area (int): Minimum area a contour must have to be considered valid (default: 10 pixels^2). Smaller contours are discarded as noise
            
    Returns:
        result_locules (List): Merged contours (locules that were close enough have been combined)
    """
    if not locules_indices: # Check if there are no locules indices provided
        return [] # Returns empty list if no locules exist
    
    # Filter out small contours (like noise) by keeping only those with area > min_area
    valid_locules = [i for i in locules_indices if cv2.contourArea(contours[i]) > min_area]
    
    # Check if no valid locules remain after filtering
    if not valid_locules:
        return [] # Returns empty list if all locules were removed
    
    merged = [False] * len(valid_locules) # Create a list to track which locules have been merged (avoid processing merged contours)
    result_locules = [] # Initialize empty list to store new merged contours
    
    for i in range(len(valid_locules)): # Process each valid locule
        if not merged[i]: # Only process locules that have not been merged
            current_idx = valid_locules[i] # Get the current locule contour
            current_contour = contours[current_idx]
            merged[i] = True # Store the current contour and mark it as merged
            to_merge = [current_contour] # to_merge will hold all nearby contours to be combined
            
            # Compute the centroid of the current locule (for reference, but not used for merging)
            #M = cv2.moments(current_contour)
            #if M["m00"] == 0:
            #    cx, cy = current_contour[0][0][0], current_contour[0][0][1]
            #else:
            #    cx = int(M["m10"] / M["m00"])
            #    cy = int(M["m01"] / M["m00"])
            
            # Search for nearby locules using contour-to-contour distance
            for j in range(i+1, len(valid_locules)):
                if not merged[j]:
                    other_idx = valid_locules[j]
                    other_contour = contours[other_idx]
                    
                    # Compute the minimal distance between current_contour and other_contour
                    min_dist = float('inf')
                    for point in other_contour[::2, 0, :]:  # Check 1 of each two points in the other contour
                        dist = cv2.pointPolygonTest(current_contour, (float(point[0]), float(point[1])), True)
                        if dist < min_dist:
                            min_dist = dist
                            if min_dist <= 0:  # If contours overlap or touch, exit early
                                break
                    
                    # If minimal distance is within the threshold, merge
                    if min_distance < abs(min_dist) < max_distance:
                        to_merge.append(other_contour)
                        merged[j] = True
            
            # Combine nearby locules into a single contour
            if len(to_merge) > 1: # If multiple locules were merged
                merged_contour = np.vstack(to_merge) # Stack their points into a single array (np.vstack)
                epsilon = 0.001 * cv2.arcLength(merged_contour, True)  # 2% del perímetro
                merged_loculus = cv2.approxPolyDP(merged_contour, epsilon, True) # Compute a contour approximation form a smooth, merged shape (cv2.approxPolyDP)
                #merged_loculus = cv2.convexHull(merged_contour) # Compute the convex hull to form a smooth, merged shape (cv2.convexHull)
                result_locules.append(merged_loculus)
            else:
                result_locules.append(current_contour) # If no merging occurred, keep the original contour
    
    return result_locules # Return a list of merged contours


#################################################################################################
# Precalculates and stores geometric data
#################################################################################################

def calculate_fruit_centroids(contours):
    """
    Calculates the centroid (cx, cy) for each contour in the list.

    Args:
        contours (List[np.ndarray]): List of contours (OpenCV format).

    Returns:
        List[Tuple[int, int] | None]: A list containing centroid coordinates (cx, cy)
                                      for each contour index. If the contour has zero area,
                                      returns None for that position.
    """
    centroids = []
    for cnt in contours:
        M = cv2.moments(cnt)  # Compute moments for the contour
        if M["m00"] != 0:  # Avoid division by zero
            cx = int(M["m10"] / M["m00"])  # X coordinate of the centroid
            cy = int(M["m01"] / M["m00"])  # Y coordinate of the centroid
            centroids.append((cx, cy))
        else:
            centroids.append(None)  # No centroid if contour area is zero
    return centroids


#################################################################################################
# Precalculates and stores geometric data
#################################################################################################

def precalculate_locules_data(contours, locules, centroid):
    """ 
    Precalculates and stores geometric data about locules from image contours to optimize firther processing.

    Args:

        REQUIRED:
            - contours (List[np.ndarray]): List of contous points (OpenCV format).
            - locules (List[int]): Indices of contours that represent locules.
            - centroid (Tuple[int,int]): Reference centroid as a tuple (x,y).
 
    Returns:
        - locules_data (List[Dict]): A list of dictionaries, each containg:
            - 'contour_id' (int): Contour identifier.
            - 'centroid' (Tuple[int, int]): (x,y) coordinates of the locule's centroid.
            - 'area' (float): Area of the locule in pixels.
            - 'perimeter' (float): Perimeter of the locule in pixels.
            - 'contour' (np.ndarray): Original contour points.
            - 'polar_coords' (Tuple[float, float]): Pair of (angle_in_radians, radius) calculated relative to the reference centroid.
            - 'circularity' (float): Circularity of the locule contour in a range from 0 to 1, where 1 indicates a perfect circle and 0 a extreme imperfect shape.

    Notes:
        - Uses OpenCV moments for centroid calculation.
        - Skips contours with zero area (m00 = 0)
    """
    locules_data = [] # Initialize an empty list to store calculated data for each locule

    for locule in locules: # For each locule:
        M = cv2.moments(contours[locule]) # Calculate moments for the contour at index locule 
    
        if M["m00"] == 0: # If contour area is 0
            continue # skip
        
        ## Cartesian coordinates
        cx = int(M["m10"] / M["m00"]) # Calculate x-coordinates of centroid
        cy = int(M["m01"] / M["m00"]) # Calculate y-coordinates of centroid

        area = cv2.contourArea(contours[locule]) # Calculate the area of the locule 
        perimeter = cv2.arcLength(contours[locule], True) # Calculate the perimeter of the locule (True = Closed contour)


        ## Polar coordinates
        dx, dy = cx - centroid[0], cy - centroid[1] # Calculate the displacement vector from the reference centroid to the locule's centroid
        angle = math.atan2(dy, dx) % (2 * np.pi) # Calculate the angle (in radians) of this vector from the positive x-axis in [0,2pi] range.
        radius = math.hypot(dx, dy) # Calculate the Eucladian distance between the displacement vector

        ## Get perfect angles

        locules_data.append({ # Store calculated data in a dictionary
            'contour_id': locule, # Store the locule contour id
            'centroid': (cx, cy), # Centroid coordinates
            'area': area, # Area measurement
            'perimeter': perimeter, # Perimeter measurement
            'contour': contours[locule], # Original contour points
            'polar_coord': (angle, radius), # Angle and radius calculated relative to the reference centroid
            'circularity': (4 * np.pi * area) / (perimeter ** 2),  # 1.0 = perfect circle
            #'hu_moments': cv2.HuMoments(M).flatten()  # Hu invariant moments (7 values)
        })

    return locules_data

#################################################################################################
# Angular locule symmetry
#################################################################################################

def angular_symmetry(locules_data, num_shifts=500):
    """
    Calculate angular symmetry by comparing actual locule angles with the most symmetrical arrangement.

    Args:
        REQUIRED:
            - locules_data (List[Dict]): List of dictionaries, each containing at least the 'polar_coord'
              of a locule, where 'polar_coord'[0] is the angle in radians from the reference centroid.
        OPTIONAL:
            - num_shifts (int): Number of angular shifts to test when trying to align the ideal angles
              to the observed angles (default = 1000).

    Returns:
        float: Normalized angular error in range [0, 1]:
               - 0.0  → perfect angular symmetry.
               - 1.0  → maximum possible angular deviation for given number of locules.
               - nan  → undefined if fewer than 2 locules.
    """
    if len(locules_data) < 2:  # If fewer than 2 locules, angular symmetry is undefined
        return np.nan

    angles = np.array([d['polar_coord'][0] for d in locules_data]) % (2 * np.pi) # Extract angles (in radians) for each locule, normalized to [0, 2π)
    n = len(angles)  # Total number of locules

    
    mean_angle = circmean(angles) # Center angles around their circular mean 
    angles_centered = (angles - mean_angle) % (2 * np.pi)

    ideal_angles = np.linspace(0, 2*np.pi, n, endpoint=False) # Define the ideal angles for a perfectly symmetric arrangement

    # Initialize best alignment search
    best_error = np.inf
    best_shift = None

    
    for shift in np.linspace(0, 2*np.pi, num_shifts, endpoint=False): # Test multiple rotational shifts to find best alignment with minimal angular deviation
        shifted_ideal = (ideal_angles + shift) % (2*np.pi)

        diff = np.abs(angles_centered[:, None] - shifted_ideal[None, :])  # Compute angular differences, considering wrap-around at 2π
        cost_matrix = np.minimum(diff, 2*np.pi - diff)
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix) # Find optimal assignment of observed to ideal angles using Hungarian algorithm
        angle_error = cost_matrix[row_ind, col_ind].mean()
        
        if angle_error < best_error: # Keep the best shift (smallest mean angular error)
            best_error = angle_error
            #best_shift = shift

    # Maximum possible mean angular error for given number of locules
    #max_angle_error = np.pi / n

    # Debug prints (can be commented out in production)
    #print(f"Best angle error (rad): {best_error}")
    #print(f"Max angle error (rad): {max_angle_error}")
    #print(f"Best shift (rad): {best_shift}")

    # Normalize error to range [0, 1]
    #angle_error_norm = min(best_error / max_angle_error, 1.0)
    return best_error



#################################################################################################
# Radial locules symmetry
#################################################################################################

def radial_symmetry(locules_data):
    """
    Calculate radial symmetry using coefficient of variation (CV) of distances.
    Args:
        REQUIRED:
            - locules_data (List[Dict]): List of dictionaries, where each dictionary contains the centroid coordinates (x,y) of a locule and precalculated 'polar_coordinates'.


    Returns:
        float: CV of distances (0 = perfect symmetry, nan = undefined).
    """
    if len(locules_data) < 2: # If there is fewer than 2 locules, symettry is undefined (no symmetry) 
        return np.nan

    radii = [data['polar_coord'][1] for data in locules_data] # Extract precalculated radii for each locule's data
    
    return np.std(radii) / np.mean(radii) if np.mean(radii) > 0 else 0.0 # Compute coefficient of variation (CV = standard deviation / mean)


#################################################################################################
# Rotational symmetry 
#################################################################################################

def rotational_symmetry(locules_data, angle_error=None, angle_weight=0.5, radius_weight=0.5, min_radius_threshold=0.1):
    """
    Calculates rotational symmetry for a fruit using both angular and radial asymmetry.
    0 = perfect symmetry, 1 = maximum asymmetry.
    Optionally accepts a precomputed angular error to avoid recalculation.

    Args:
        REQUIRED:
            - locules_data (List[Dict]): Each dict contains 'polar_coord' = (angle, radius)
        OPTIONAL:
            - angle_error (float, optional): Precomputed angular error (0-1). If None, it is calculated internally.
            - angle_weight (float): Weight of angular error in combined metric (default=0.5)
            - radius_weight (float): Weight of radial error in combined metric (default=0.5)
            - min_radius_threshold (float): Ignore locules with radius < fraction of mean (default=0.1)

    Returns:
        float: Combined rotational symmetry metric in [0,1], or np.nan if undefined.
    """

    if len(locules_data) < 2: # Check for minimum number of locules
        return np.nan  # Cannot define symmetry with fewer than 2 locules

    # Extract and normalize radial distances
    radii = np.array([d['polar_coord'][1] for d in locules_data]) # Extract radius for each locule
    radii_normalized = radii / np.mean(radii) # Normalize by mean radius for comparability
    valid_mask = radii_normalized >= min_radius_threshold # Ignore very small locules (likely noise)
    radii_normalized = radii_normalized[valid_mask] # Keep only valid radii

    if len(radii_normalized) < 2: # Check if enough locules remain after filtering    
        return np.nan  # Symmetry undefined if too few valid locules remain

    # Calculate radial error using Median Absolute Deviation (MAD)
    median_abs_dev = np.median(np.abs(radii_normalized - 1.0)) # Typical deviation from mean radius
    radius_error_norm = np.tanh(median_abs_dev / 0.6745) # Normalize radial error to ~[0,1), robust to outliers
    
    if angle_error is None:
        angle_error = angular_symmetry(locules_data) # Compute angular asymmetry

    total_weight = angle_weight + radius_weight 
    combined_error = (angle_weight * angle_error + radius_weight * radius_error_norm) / total_weight # Combine angular and radial errors (weighted average)

    return np.clip(combined_error, 0.0, 1.0) # Ensure combined error is within [0,1]



#################################################################################################
# Calculate minor axis (fruit width approximation)
#################################################################################################


def calculate_axes(fruit_contour, px_per_cm_width, px_per_cm_length, rel_tol=1e-6, 
                   img=None, draw_axes=False, 
                   major_axis_color=(0, 255, 0), minor_axis_color=(255, 0, 0), 
                   axis_thickness=2): 
    """
    Calculate the minor and major axes of a fruit's contour in centimeters, accounting for 
    pixel scaling differences along X and Y axes. Optionally draw the axes on an image.

    The major axis is found as the maximum distance between hull vertex pairs in an appropriate
    coordinate space (pixels if isotropic, centimeters if anisotropic). The minor axis is defined
    as the maximum thickness measured perpendicular to the major axis.

    Args:
        REQUIRED:
        - fruit_contour (np.ndarray): Nx2 or Nx1x2 array of contour points.
        - px_per_cm_width (float): Pixel-to-cm conversion factor for X-axis.
        - px_per_cm_length (float): Pixel-to-cm conversion factor for Y-axis.
        
        OPTIONAL:
        - rel_tol (float): Relative tolerance to determine isotropy.
        - annotated_img (np.ndarray): Image where axes will be drawn if draw_axes=True.
        - draw_axes (bool): Whether to draw the axes on the annotated_img.
        - major_axis_color (tuple): BGR color for major axis (default: green).
        - minor_axis_color (tuple): BGR color for minor axis (default: blue).
        - axis_thickness (int): Thickness of axis lines in pixels.

    Returns:
        tuple:
            - (float, np.ndarray, np.ndarray): (major_axis_cm, p1_px, p2_px)
            - (float, np.ndarray, np.ndarray): (minor_axis_cm, p_min_px, p_max_px)
    """

    if px_per_cm_width <= 0 or px_per_cm_length <= 0:
        raise ValueError("Conversion factors (px_per_cm_width and px_per_cm_length) must be positive")

    # Reshape and convert contour to float
    points_px = fruit_contour.reshape(-1, 2).astype(np.float32)

    n = points_px.shape[0]
    if n < 2: 
        return (0.0, None, None), (0.0, None, None)
    
    isotropic = math.isclose(px_per_cm_width, px_per_cm_length, rel_tol=rel_tol)

    # For major axis search
    if isotropic:
        px_per_cm = (px_per_cm_width + px_per_cm_length) * 0.5
        coord_for_major = points_px  # still in px
    else:
        coord_for_major = np.column_stack([
            points_px[:, 0] / px_per_cm_width,
            points_px[:, 1] / px_per_cm_length
        ])

    ## Major axis
    if n >= 3:
        verts = ConvexHull(coord_for_major).vertices
    else:
        verts = np.arange(n)

    max_dist = 0.0
    point1_idx = point2_idx = verts[0]

    for a in range(len(verts)):
        i = verts[a]
        for b in range(a + 1, len(verts)):
            j = verts[b]
            d = np.linalg.norm(coord_for_major[i] - coord_for_major[j])
            if d > max_dist:
                max_dist = d
                point1_idx, point2_idx = i, j

    if max_dist == 0:
        return (0.0, None, None), (0.0, None, None)

    # Major axis length in cm
    major_axis_cm = (max_dist / px_per_cm) if isotropic else float(max_dist)

    # Major axis endpoints in original px space
    p1_px = points_px[point1_idx]
    p2_px = points_px[point2_idx]

    ## Minor axis
    major_vec = coord_for_major[point2_idx] - coord_for_major[point1_idx]
    major_norm = np.linalg.norm(major_vec)

    if major_norm < 1e-10:
        return (major_axis_cm, p1_px, p2_px), (0.0, None, None)

    perp_unit = np.array([-major_vec[1], major_vec[0]], dtype=np.float32) / major_norm
    proj = np.dot(coord_for_major - coord_for_major[point1_idx], perp_unit)

    if isotropic:
        minor_axis_cm = (proj.max() - proj.min()) / px_per_cm
    else:
        minor_axis_cm = float(proj.max() - proj.min())

    # Minor axis endpoints in original px space (indices from same point set)
    idx_min = int(np.argmin(proj))
    idx_max = int(np.argmax(proj))
    p_min_px = points_px[idx_min]
    p_max_px = points_px[idx_max]

    # Draw axes if requested
    if draw_axes and img is not None:
        # Draw major axis (green)
        if p1_px is not None and p2_px is not None:
            cv2.line(img, 
                    tuple(p1_px.astype(int)), 
                    tuple(p2_px.astype(int)), 
                    major_axis_color, axis_thickness)
        
        # Draw minor axis (blue)
        if p_min_px is not None and p_max_px is not None:
            cv2.line(img, 
                    tuple(p_min_px.astype(int)), 
                    tuple(p_max_px.astype(int)), 
                    minor_axis_color, axis_thickness)

    return major_axis_cm, minor_axis_cm

######

def get_fruit_contour(contours, fruit_id, contour_mode = 'raw', epsilon_hull = 0.0001):
    # Extract contour points for each fruit
    fruit_contour = contours[fruit_id]
    
    epsilon_hull = float(epsilon_hull)

    # Reshape contours if needed (otherwise, original (raw) contour is used):
    if contour_mode == 'hull':
        fruit_contour = cv2.convexHull(fruit_contour) # Hull approximation 
            
    elif contour_mode == 'approx': # Apploximate polygon 
        peri = cv2.arcLength(fruit_contour, True)
        epsilon = max(1.0, epsilon_hull  * peri)
        fruit_contour = cv2.approxPolyDP(fruit_contour, epsilon, True)
                
    elif contour_mode == 'ellipse': # Fit an ellipse around the original contour
        if len(fruit_contour) >= 5:
            ellipse = cv2.fitEllipse(fruit_contour)
            fruit_contour = cv2.ellipse2Poly(
                (int(ellipse[0][0]), int(ellipse[0][1])),
                (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                int(ellipse[2]), 0, 360, 2)
            fruit_contour = fruit_contour.reshape(-1, 1, 2)

    elif contour_mode == 'raw': # Original contour
        fruit_contour = contours[fruit_id]

    
    return fruit_contour



#### 

def get_fruit_morphology(contour, px_per_cm_width, px_per_cm_length, rel_tol=1e-6):
    """
    Calculate comprehensive fruit morphology metrics.
    
    Args:
        contour (np.ndarray): Fruit contour points
        px_per_cm_width (float): Pixel to cm conversion for X-axis
        px_per_cm_length (float): Pixel to cm conversion for Y-axis
        rel_tol (float): Relative tolerance for isotropy determination
        
    Returns:
        dict: Dictionary containing fruit morphology metrics
    """
    # Calculate area and perimeter in pixels
    area_px = cv2.contourArea(contour)
    perimeter_px = cv2.arcLength(contour, True)
    
    if area_px <= 0 or perimeter_px <= 0:
        return {
            'fruit_area_px': np.nan,
            'fruit_area_cm2': np.nan,
            'fruit_perimeter_px': np.nan,
            'fruit_perimeter_cm': np.nan,
            'fruit_circularity': np.nan,
            'fruit_solidity': np.nan,
            'fruit_compactness': np.nan,
            'fruit_convex_hull_area_px': np.nan
        }
    
    # Convert to cm based on isotropy
    if math.isclose(px_per_cm_width, px_per_cm_length, rel_tol=rel_tol):
        px_per_cm = (px_per_cm_width + px_per_cm_length) / 2
        area_cm2 = area_px / (px_per_cm ** 2)
        perimeter_cm = perimeter_px / px_per_cm
    else:
        area_cm2 = area_px / (px_per_cm_width * px_per_cm_length)
        perimeter_cm = perimeter_px / ((px_per_cm_width + px_per_cm_length) / 2)
    
    # Calculate shape metrics
    circularity = (4 * np.pi * area_px) / (perimeter_px ** 2) if perimeter_px > 0 else np.nan
    
    
    # Solidity (area / convex hull area)
    hull = cv2.convexHull(contour)
    hull_area_px = cv2.contourArea(hull)
    solidity = area_px / hull_area_px if hull_area_px > 0 else np.nan
    
    # Compactness (perimeter² / area)
    compactness = (perimeter_px ** 2) / area_px if area_px > 0 else np.nan
    
    return {
        'fruit_area_px': float(area_px),
        'fruit_area_cm2': float(area_cm2),
        'fruit_perimeter_px': float(perimeter_px),
        'fruit_perimeter_cm': float(perimeter_cm),
        'fruit_circularity': float(circularity),
        'fruit_solidity': float(solidity),
        'fruit_compactness': float(compactness),
        'fruit_convex_hull_area_px': float(hull_area_px)
    }

#########

def analyze_fruits(
                    # REQUIRED
                    img, # Image (BGR)
                    contours, # List with all valid contours
                    fruit_locus_map, # Dictionary with the information about the locule ids per each fruit
                    px_per_cm_width, # Pixel density per cm for the shortest side of the image (width)
                    px_per_cm_length, # Pixel density per cm for the largest side of the image (length)
                    img_name, # Image name
                    label_text, # Label text

                    # OPTIONAL 
                    label_id=None, # Label contour ID
                    contour_mode='raw', # Contour shape 
                    epsilon_hull=0.001, # Epsilon value for polygon adjustment around the fruit contour
                    min_locule_area=300, # Minimum area in pixels necessary to declare a locule as a valid contour
                    max_locule_area=None, # Maximum area in pixels necessary to declare a locule as a valid contour
                    max_dist=30, # Maximum distance to merge two locules
                    min_dist=2, # Minimum distance to merge two locules
                    
                    # Symmetry:
                    num_shifts = 500, # Number of angular shifts to test when trying to align to ideal angles (angular symmetry())
                    angle_weight = 0.5, # Weight of angular error in combined metric (rotational symmetry())
                    radius_weight = 0.5, # Weight of radius error in combined metric (rotational symmetry())
                    min_radius_threshold = 0.1, # Ignore locules with radius < fraction of mean
                    
                    # Inner pericarp
                    use_ellipse=False, 
                    rel_tol = 1e-6,

                    # Stamps
                    stamp=False,

                    # Plot
                    plot=True, 
                    plot_size=(20,10),
                    font_scale=1, 
                    font_thickness=2, 
                    text_color=(0,0,0), 
                    bg_color=(255,255,255),
                    padding=15, 
                    line_spacing=15, 
                    path=None, 
                    fig_axis=True, 
                    title_fontsize=20, 
                    title_location='center',

                    ):
    
    """
    Analyze fruit contours and their locules to extract morphological, pericarp, and locule features. 
    Optionally annotate the results on the input image.

    The function processes each fruit contour, estimates its major/minor axes, area, perimeter, 
    and shape descriptors. It also identifies locules, computes their size and circularity, 
    and evaluates symmetry metrics. An annotated image with contours, centroids, axes, and 
    text labels can be optionally returned for visualization.

    Args:
        REQUIRED:
        - img (np.ndarray): Input image in BGR format.
        - contours (list): List of all valid contours detected in the image.
        - fruit_locus_map (dict): Dictionary mapping fruit IDs to corresponding locule indices.
        - px_per_cm_width (float): Pixel density per cm for the shortest side of the image (width).
        - px_per_cm_length (float): Pixel density per cm for the longest side of the image (length).
        - img_name (str): Name of the analyzed image.
        - label_text (str): Label associated with the image (e.g., treatment, cultivar).

        OPTIONAL:
        - label_id (int): Contour ID corresponding to the label region, excluded from fruit analysis.
        - contour_mode (str): Contour adjustment method ('raw' or polygonal).
        - epsilon_hull (float): Epsilon for polygonal approximation of fruit contour.
        - min_locule_area (int): Minimum area in pixels required for a contour to be considered a locule.
        - max_locule_area (int): Maximum area in pixels for a contour to be considered a locule.
        - max_dist (int): Maximum distance in pixels to merge two locules.
        - min_dist (int): Minimum distance in pixels to merge two locules.

        Symmetry parameters:
        - num_shifts (int): Number of angular shifts tested for angular symmetry alignment.
        - angle_weight (float): Weight of angular error in rotational symmetry.
        - radius_weight (float): Weight of radius error in rotational symmetry.
        - min_radius_threshold (float): Ignore locules with radius smaller than this fraction of the mean.

        Inner pericarp:
        - use_ellipse (bool): Whether to fit ellipses instead of raw contours for pericarp.
        - rel_tol (float): Relative tolerance to determine isotropy for pixel scaling.

        Stamps:
        - stamp (bool): If True, invert the input image colors (used for stamp images).

        Plotting and annotation:
        - plot (bool): If True, display the annotated image at the end of processing.
        - plot_size (tuple): Figure size for the annotated image.
        - font_scale (int): Font scaling for annotations.
        - font_thickness (int): Font thickness for annotations.
        - text_color (tuple): BGR text color for annotations.
        - bg_color (tuple): BGR background color for text boxes.
        - padding (int): Padding around text annotations.
        - line_spacing (int): Spacing between text lines.
        - path (str): Optional path where annotated results will be stored.
        - fig_axis (bool): If True, show axis around the plotted image.
        - title_fontsize (int): Font size of the figure title.
        - title_location (str): Title location in the figure.

    Returns:
        AnnotatedImage:
            - annotated_img (np.ndarray): Annotated image with drawn contours, centroids, axes, and labels.
            - results (list[dict]): List of dictionaries with morphological and locule metrics per fruit.
              Each dictionary includes:
                * major_axis_cm, minor_axis_cm
                * fruit area, perimeter, circularity, solidity, compactness
                * pericarp metrics (inner/outer area, thickness)
                * locule metrics (area, circularity, density, ratios)
                * symmetry metrics (angular, radial, rotational)
                * rotated box dimensions and area
    """

    # Create an img copy for annotation
    annotated_img = img.copy()
    if stamp is True:
        annotated_img = cv2.bitwise_not(annotated_img)

    # Validations for pixel scales
    if px_per_cm_width <= 0 or px_per_cm_length <= 0:
        raise ValueError("Conversion factors (px_per_cm_width and px_per_cm_length) must be positive")
    if px_per_cm_length <= px_per_cm_width:
        raise ValueError("Conversion factor for Y-axis (px_per_cm_length) must be greater than X-axis (px_per_cm_width)")

    results = []  # Store results
    sequential_id = 1

    # Precompute all fruit centroids
    fruit_centroids = calculate_fruit_centroids(contours)

    for fruit_id, locules in fruit_locus_map.items():
        
        # Exclude label contour
        if fruit_id == label_id:
            continue
        try:
            ################################
            ## PREPARE FRUIT DATA         ##
            ################################
            
            # Get fruit contour
            fruit_contour = get_fruit_contour(
                fruit_id=fruit_id,
                contours=contours,
                contour_mode=contour_mode,
                epsilon_hull=epsilon_hull
            )

            # Draw fruit_contour:
            cv2.drawContours(annotated_img, [fruit_contour], -1, (0, 255, 0), 2)  

            # Get precomputed centroid
            fruit_centroid = fruit_centroids[fruit_id]

            # Draw fruit centroid. Skip if centroid is invalid.
            if fruit_centroid:
                cx, cy = map(int, fruit_centroid)
                cv2.circle(annotated_img, (cx, cy), 15, (255, 255, 51), -1)  # Blue dot (RGB)
            else:
                continue

            ###############################################
            ## Calculating length and width of the fruit ##
            ###############################################
            # Major/Minor axes
            major_axis_cm, minor_axis_cm = calculate_axes(
                fruit_contour, 
                px_per_cm_width, 
                px_per_cm_length, 
                rel_tol=rel_tol, 
                img=annotated_img, 
                draw_axes=True
            )

            # Rotated box
            box_length_cm, box_width_cm = rotate_box(
                fruit_contour, 
                px_per_cm_width, 
                px_per_cm_length, 
                img=annotated_img,
                draw_box=True
            )
            
            # Extract fruit morphology metrics
            fruit_morphology = get_fruit_morphology(
                contour=fruit_contour,
                px_per_cm_width=px_per_cm_width,
                px_per_cm_length=px_per_cm_length,
                rel_tol=rel_tol
            )

            # Extract morphology metrics for the fruit
            fruit_area_cm2 = fruit_morphology['fruit_area_cm2']
            fruit_perimeter_cm = fruit_morphology['fruit_perimeter_cm']
            fruit_circularity = fruit_morphology['fruit_circularity']
            fruit_solidity = fruit_morphology['fruit_solidity']
            fruit_compactness = fruit_morphology['fruit_compactness']

            # Inner pericarp
            inner_pericarp_area_cm2 = inner_pericarp_area(
                locules=locules,
                contours=contours,
                px_per_cm_width=px_per_cm_width,
                px_per_cm_length=px_per_cm_length,
                img=annotated_img,
                draw_inner_pericarp=True, # Draw inner pericarp 
                use_ellipse=use_ellipse, 
                epsilon=epsilon_hull, 
                rel_tol=rel_tol,
            )

            # Estimating pericarp thickness

            # Get the fruit contour modeled as an ellipse
            if contour_mode == 'ellipse':
                ellipse_fruit_contour = fruit_contour
            else:
                    ellipse_fruit_contour = get_fruit_contour(
                    fruit_id=fruit_id,
                    contours=contours,
                    contour_mode='ellipse',  # <-- FORCING ellipse mode here
                    epsilon_hull=epsilon_hull
                )
    
            # Calculate the area of the external ellipse from the polygonal contour
            fruit_area_ellipse_px = cv2.contourArea(ellipse_fruit_contour)
            
            # Convert area to cm2
            if math.isclose(px_per_cm_width, px_per_cm_length, rel_tol=rel_tol):
                px_per_cm = (px_per_cm_width + px_per_cm_length) / 2
                fruit_area_ellipse_cm2 = fruit_area_ellipse_px / (px_per_cm ** 2)
            else:
                fruit_area_ellipse_cm2 = fruit_area_ellipse_px / (px_per_cm_width * px_per_cm_length)

            # Calculate internal area using ellipse (for pericarp)
            if use_ellipse:
                inner_pericarp_area_ellipse_cm2 = inner_pericarp_area_cm2
            else:
                inner_pericarp_area_ellipse_cm2 = inner_pericarp_area(
                    locules=locules,
                    contours=contours,
                    px_per_cm_width=px_per_cm_width,
                    px_per_cm_length=px_per_cm_length,
                    img=annotated_img,
                    draw_inner_pericarp=False,  # Don't draw now, we only need the area
                    use_ellipse=True,           # <-- FORCING ellipse usage here
                    epsilon=epsilon_hull,
                    rel_tol=rel_tol,
                )

            # 3. Calculate pericarp thickness with ellipses
            if inner_pericarp_area_ellipse_cm2 > 0 and fruit_area_ellipse_cm2 > 0:
                # Calculate equivalent radii from the areas of the ellipses
                inner_radius_ellipse = math.sqrt(inner_pericarp_area_ellipse_cm2 / math.pi)
                outer_radius_ellipse = math.sqrt(fruit_area_ellipse_cm2 / math.pi)
                avg_pericarp_thickness_cm = outer_radius_ellipse - inner_radius_ellipse
            else:
                avg_pericarp_thickness_cm = np.nan


            # Calculating fruit aspect ratio 
            if box_length_cm > 0 and box_width_cm > 0:
                fruit_aspect_ratio = float(box_width_cm / box_length_cm)  # 1.0 ~ circular; <1.0 more elongated (oval)
            else: 
                fruit_aspect_ratio = np.nan

            ################################
            ## PREPARE LOCULE DATA        ##
            ################################

            # Precalculate locule data for the fruit and filter small locules
            locules_data = [data for data in precalculate_locules_data(contours, locules, fruit_centroid)
                            if data['area'] >= min_locule_area and 
                            (max_locule_area is None or data['area'] <= max_locule_area)
            ]

            # Get filtered locules indices
            filtered_indices = [data['contour_id'] for data in locules_data]  

            # Merge close locules
            merged_locules_contours = merge_locules(
                locules_indices=filtered_indices, # Only filtered indices 
                contours=contours, # Contour list
                max_distance=max_dist,
                min_distance=min_dist
            ) or []

            for loculus_contour in merged_locules_contours:
                cv2.drawContours(annotated_img, [loculus_contour], -1, (255, 0, 255), 2)
            
            ###############################################
            ## Calculating locule area and circularity   ##
            ###############################################            
            if locules_data:
                # Convert pixels to cm^2 for each locule area value
                if math.isclose(px_per_cm_width, px_per_cm_length, rel_tol=1e-6):
                    px_area_cm = 1 / (px_per_cm_width ** 2) # Isotropic pixels (squared)
                else:
                    px_area_cm = (1 / px_per_cm_width) * (1 / px_per_cm_length) # Anisotropic pixels (rectangular)

                locule_areas_cm = [data['area'] * px_area_cm for data in locules_data] # Convert pixels to cm^2 for the locule area
                
                # Calculate the locule mean area (cm^2) and its standard deviation
                mean_locule_area = float(np.mean(locule_areas_cm))
                std_locule_area = float(np.std(locule_areas_cm)) # Standard deviation (cm^2)
                cv_locule_area = float(std_locule_area / mean_locule_area) if mean_locule_area > 0 else np.nan # Coefficient of Variation
                n_locules = len(locules_data)

                # Calculate the circularity for each locule (0-1)
                locule_circularities = [
                    (4 * np.pi * data['area']) / (data['perimeter']**2 + 1e-6)
                    for data in locules_data
                ]

                # Calculate the locule mean circularity
                mean_locule_circularity = float(np.mean(locule_circularities))
                std_locule_circularity = float(np.std(locule_circularities))  
                cv_locule_circularity = float(std_locule_circularity / mean_locule_circularity) if mean_locule_circularity > 0 else np.nan

                # Draw locule centroids:
                for locule_data in locules_data:
                    cx, cy = locule_data['centroid']
                    cv2.circle(annotated_img, (int(cx), int(cy)), 7, (0, 255, 255), -1)

            else: # If no locules:
                locule_areas_cm = []
                mean_locule_area = std_locule_area = cv_locule_area = np.nan
                locule_circularities = []
                mean_locule_circularity = std_locule_circularity = cv_locule_circularity = np.nan
                n_locules = 0

            #################################
            ## Calculating locule symmetry ##
            #################################
            angular_symmetry_score = angular_symmetry(locules_data, num_shifts=num_shifts)
            radial_symmetry_score = radial_symmetry(locules_data)
            rotational_symmetry_score = rotational_symmetry(
                locules_data, angle_error=None, angle_weight=angle_weight, 
                radius_weight=radius_weight, min_radius_threshold=min_radius_threshold
            )

            # Additional locule metrics
            locules_density = n_locules / fruit_area_cm2 if fruit_area_cm2 > 0 else 0
            inner_area_ratio = inner_pericarp_area_cm2 / fruit_area_cm2 if fruit_area_cm2 > 0 else 0
            locule_area_ratio = max(locule_areas_cm) / min(locule_areas_cm) if locule_areas_cm and min(locule_areas_cm) > 0 else 0
            total_locule_area_cm2 = sum(locule_areas_cm) if locule_areas_cm else 0
            locule_area_percentage = (total_locule_area_cm2 / fruit_area_cm2) * 100 if fruit_area_cm2 > 0 else 0
            locule_packing_efficiency = (total_locule_area_cm2 / inner_pericarp_area_cm2) * 100 if inner_pericarp_area_cm2 > 0 else 0

            # Text annotation set up
            x, y, w, h = cv2.boundingRect(fruit_contour)
            text = f"id {sequential_id}: \n{n_locules} loc"
            

            (size_w, size_h), baseline = cv2.getTextSize("Test", cv2.FONT_HERSHEY_SIMPLEX, 
                                                         font_scale, font_thickness)

            single_line_height = size_h

            total_height = (single_line_height * 2) + line_spacing
            text_x = max(10, x)
            text_y = max(total_height + 15, y - 15)
            text_width = max([
                cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0][0] 
                for line in text.split('\n')
            ])
            
            # Draw a semi-translucid text box
            text_bg_layer = annotated_img.copy()
            cv2.rectangle(
                text_bg_layer,
                (text_x - padding, text_y - total_height - padding),
                (text_x + text_width + padding, text_y + padding),
                bg_color, -1
            )
            cv2.addWeighted(text_bg_layer, 0.7, annotated_img, 0.3, 0, annotated_img)
            
            # Text annotation
            for i, line in enumerate(text.split('\n')):
                y_offset = text_y - (total_height - single_line_height) + (i * (single_line_height + line_spacing))
                cv2.putText(
                    annotated_img, line,
                    (text_x, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                    text_color, font_thickness,
                    cv2.LINE_AA
                )

            # Save results
            result = {
                # Identification
                'image_name': img_name,
                'label': label_text, 
                'fruit_id': sequential_id,
                
                # Fruit - Basic Metrics
                'n_locules': n_locules,
                'major_axis_cm': major_axis_cm,  
                'minor_axis_cm': minor_axis_cm,   
                'fruit_area_cm2': fruit_area_cm2,
                'fruit_perimeter_cm': fruit_perimeter_cm,
                'fruit_circularity': fruit_circularity,
                'fruit_aspect_ratio': fruit_aspect_ratio,
                'fruit_solidity': fruit_solidity,  
                'fruit_compactness': fruit_compactness,  

                # Fruit - Rotated Box
                'box_length_cm': box_length_cm,
                'box_width_cm': box_width_cm,
                #'rotated_box_area_cm2': box_length_cm * box_width_cm
                'compactness_index': fruit_area_cm2 / (box_length_cm * box_width_cm) if box_length_cm > 0 and box_width_cm > 0 else np.nan,
                
                # Pericarp
                'inner_pericarp_area_cm2': inner_pericarp_area_cm2,
                'outer_pericarp_area_cm2': fruit_area_cm2 - inner_pericarp_area_cm2,
                'avg_pericarp_thickness_cm': avg_pericarp_thickness_cm,
                
                # Locules - Area
                'mean_locule_area_cm2': mean_locule_area,
                'std_locule_area_cm2': std_locule_area,
                'total_locule_area_cm2': total_locule_area_cm2,
                'cv_locule_area': cv_locule_area,
                
                # Locules - Circularity
                'mean_locule_circularity': mean_locule_circularity,
                'std_locule_circularity': std_locule_circularity,
                'cv_locule_circularity': cv_locule_circularity,
                
                # Locules - Symmetry
                'angular_symmetry': angular_symmetry_score,
                'radial_symmetry': radial_symmetry_score,
                'rotational_symmetry': rotational_symmetry_score,
                
                # Locules - Spatial Distribution
                'locules_density': locules_density,
                'inner_area_ratio': inner_area_ratio,
                'locule_area_ratio': locule_area_ratio,
                'locule_area_percentage': locule_area_percentage,
                'locule_packing_efficiency': locule_packing_efficiency

            }
            
            results.append(result)
            sequential_id += 1

        except Exception as e:
            print(f"Error processing fruit {fruit_id}: {e}")
            continue

    if plot is True:
        # Mostrar resultados si está habilitado
        plot_img(
            img=annotated_img, # Annotated image
            fig_axis=fig_axis, # Include (or not) plot axis
            plot_size=plot_size, # Define plot dimension
            label_text=label_text, # Indicate label text (if any)
            img_name=img_name, # Indicate image name for visualization purposes
            title_fontsize=title_fontsize,  
            title_location=title_location
        )
        
    return AnnotatedImage(annotated_img, results, image_path=path)



####



############### 

# Analyzing multiple photos in a folder

def processing_images(path_input, output_dir=None, canny_min=300, canny_max=100, n_kernel=7, 
                     max_merge_dist=50, min_merge_dist=2, n_iterations=1, 
                     lower_black=None, upper_black=None, min_loculi_count=1, 
                     contour_approx: int=cv2.CHAIN_APPROX_SIMPLE, min_locule_area=500, 
                     use_ellipse=False, line_spacing=15, min_circularity=0.3, 
                     max_circularity=1.2, epsilon_hull=0.005, contour_mode='raw',
                     stamps=False, width_cm=None, length_cm=None, 
                     size_dimension='letter_ansi', min_aspect_ratio = 0.3, max_aspect_ratio = 3, 
                     rescale_factor = None,
                     language = ['es','en'], min_area_label = 500, blur_label = (11,11), 
                     min_canny_label = 0, max_canny_label = 150):

    # 1. Configuración del directorio de salida
    if output_dir is None:
        output_dir = os.path.join(path_input, "Results")
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Contadores de estadísticas
    processed_count = 0
    skipped_no_contours = 0
    skipped_errors = 0

    # 3. Seguimiento de tiempo y uso de memoria
    start_time = time.time()
    process = psutil.Process()

    # Listar archivos válidos
    file_list = [f for f in os.listdir(path_input) 
                if os.path.splitext(f)[1].lower() in valid_extensions]

    # Listas para almacenar resultados
    all_results = []
    errors_report = []

    print("Traitly running ⋆✧｡٩(ˊᗜˋ )و✧*｡   ")


    for filename in tqdm(file_list, desc='Processing images', unit='image'):
        ext = os.path.splitext(filename)[1].lower()

        if ext in valid_extensions:
            path_image = os.path.join(path_input, filename)
            
            try:
                # Carga y preprocesamiento de imagen
                img = load_img(path_image, plot=False)
                if img is None:
                    print(f"Error: No se pudo cargar la imagen: {filename}")
                    errors_report.append({'filename': filename, 'status': 'Error loading image'})
                    skipped_errors += 1
                    continue

                if stamps:
                    img = cv2.bitwise_not(img)

                # Procesamiento de la imagen
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                final_mask = create_mask(img_hsv,
                                      plot=False,
                                      n_kernel=n_kernel,
                                      canny_max=canny_min,
                                      canny_min=canny_max,
                                      n_iteration=n_iterations,
                                      lower_hsv=lower_black,
                                      upper_hsv=upper_black)
                
                # Detección de frutos
                contours, fruit_locus_map = find_fruits(final_mask,
                                                     min_locules_per_fruit=min_loculi_count,
                                                     min_locule_area=min_locule_area,
                                                     contour_approximation=contour_approx,
                                                     min_circularity=min_circularity,
                                                     max_circularity=max_circularity,
                                                     min_aspect_ratio = min_aspect_ratio,
                                                     max_aspect_ratio = max_aspect_ratio, 
                                                     contour_filters = None,
                                                     rescale_factor = rescale_factor)
                
                if not contours:
                    errors_report.append({'filename': filename, 'status': 'No contours found'})
                    skipped_no_contours += 1
                    continue

                # Análisis morfométrico
                label_text, label_coord = detect_label(img, contours, fruit_locus_map, language = language, 
                                          min_area_label = min_area_label, blur_label = blur_label, min_canny_label = min_canny_label,
                                        max_canny_label = max_canny_label)
                
                img_name = detect_img_name(path_image)
                px_per_cm_width, px_per_cm_length, _, _ = px_per_cm(img,
                                                             width_cm=width_cm,
                                                             length_cm=length_cm,
                                                             size=size_dimension)

                results = analyze_fruits(
                    img=img,
                    contours=contours,
                    max_dist=max_merge_dist,
                    min_dist=min_merge_dist,
                    min_locule_area=min_locule_area,
                    fruit_locus_map=fruit_locus_map,
                    px_per_cm_width=px_per_cm_width,
                    px_per_cm_length=px_per_cm_length,
                    img_name=img_name,
                    label_text=label_text,
                    use_ellipse=use_ellipse,
                    plot=False,
                    line_spacing=line_spacing,
                    stamp=stamps,
                    contour_mode=contour_mode,
                    epsilon_hull=epsilon_hull
                )

                # CORRECCIÓN PRINCIPAL: Usar results.results para iterar
                current_results = results.results  # Esta es la lista de diccionarios
                
                if not current_results:
                    errors_report.append({'filename': filename, 'status': 'No valid fruits detected'})
                    skipped_no_contours += 1
                    continue

                # Guardar imagen anotada con verificación de atributo
                try:
                    annotated_filename = f"annotated_{filename}"
                    annotated_path = os.path.join(output_dir, annotated_filename)
                    
                    # Verificar si existe el atributo rgb_image o rgb_img
                    if hasattr(results, 'rgb_image'):
                        img_to_save = results.rgb_image
                    elif hasattr(results, 'rgb_img'):
                        img_to_save = results.rgb_img
                    else:
                        raise AttributeError("No se encontró atributo rgb_image o rgb_img")
                    
                    cv2.imwrite(annotated_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
                    
                except Exception as e:
                    print(f"Error al guardar imagen {filename}: {str(e)}")
                    errors_report.append({'filename': filename, 'status': f'Error saving image: {str(e)}'})
                    skipped_errors += 1
                    continue

                # Procesar resultados
                df = pd.DataFrame(current_results)
                all_results.append(df)
                
                # Verificar lóculos (usando current_results en lugar de results)
                if any(r.get('n_locules', 0) == 0 for r in current_results):
                    errors_report.append({'filename': filename, 'status': 'No locules detected'})
                else:
                    errors_report.append({'filename': filename, 'status': 'Successfully processed'})
                    processed_count += 1

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                errors_report.append({'filename': filename, 'status': f'Processing error: {str(e)}'})
                skipped_errors += 1

    # Guardar resultados finales
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        csv_path = os.path.join(output_dir, "all_results.csv")
        df_all.to_csv(csv_path, index=False)

    if errors_report:
        df_errors = pd.DataFrame(errors_report)
        error_csv_path = os.path.join(output_dir, "error_report.csv")
        df_errors.to_csv(error_csv_path, index=False)

    # Estadísticas finales
    end_time = time.time()
    elapsed_min = (end_time - start_time) / 60
    ram_used_gb = (process.memory_info().rss / 1024**2) / 1024

    print("   ( ദ്ദി ˙ᗜ˙ ) ✧   Processing completed successfully!")
    print("══════════════════════════════════════════════════════════════════════")
    print(f"🕒 Total time:      {elapsed_min:.2f} minutes")
    print(f" 💾 RAM used:        {ram_used_gb:.2f} GB")
    print(f"🖼️ Images processed: {processed_count}/{len(file_list)}")
    print(f" 📁 Output folder:   {output_dir}")
    print(f"⚠️  Errors:          {skipped_errors} skipped")



