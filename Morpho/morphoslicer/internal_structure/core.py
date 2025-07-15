import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from sklearn.neighbors import NearestNeighbors
from scipy.stats import circstd
from scipy.spatial import ConvexHull
import math
from pdf2image import convert_from_path
import os
from ..utils.common_functions import load_image, detect_label, detect_img_name
from .functions import AnnotatedImage
from .. import valid_extensions
import time
import pandas as pd 
from tqdm import tqdm
import psutil

#########
# Morphological functions

def inner_pericarp_area(annotated_img, loculi, contours, use_ellipse=False):
    """
    Dibuja el pericarpio interno (contorno alrededor de los lóculos) y calcula su área.
    
    Args:
        annotated_img: Imagen donde se dibujará el contorno
        loculi: Lista de índices de contornos que representan los lóculos
        contours: Lista de todos los contornos detectados
        use_ellipse: Booleano para usar ajuste de elipse (True) o convex hull (False)
        
    Returns:
        tuple: (imagen con el contorno dibujado, área del pericarpio interno en píxeles)
    """
    if len(loculi) == 0:
        return annotated_img, 0
    
    # Apilar todos los puntos de los lóculos
    all_points = np.vstack([contours[i] for i in loculi])
    area = 0
    
    if use_ellipse:
        # Ajuste de elipse
        if all_points.shape[0] >= 5:  # Requiere al menos 5 puntos para fitEllipse
            ellipse = cv2.fitEllipse(all_points.astype(np.float32))
            cv2.ellipse(annotated_img, ellipse, (0, 255, 255), 2)
            # Calcular área de la elipse
            a, b = ellipse[1][0]/2, ellipse[1][1]/2
            area = np.pi * a * b
    else:
        # Envoltura convexa como perímetro interno
        hull = cv2.convexHull(all_points)
        # Suavizar el contorno
        epsilon = 0.0001 * cv2.arcLength(hull, True)
        smoothed_hull = cv2.approxPolyDP(hull, epsilon, True)
        cv2.drawContours(annotated_img, [smoothed_hull], -1, (0, 255, 255), 2)
        # Calcular área del polígono suavizado
        area = cv2.contourArea(smoothed_hull)
    
    return annotated_img, area


def rotate_box(img, contour, px_per_cm_x, px_per_cm_y, draw_on_image=True):
    """
    Calcula el rotated bounding box de un contorno y sus dimensiones en cm.
    
    Args:
        img: Imagen donde se dibujará (si draw_on_image=True)
        contour: Contorno del fruto
        px_per_cm_x: Píxeles por cm en eje X
        px_per_cm_y: Píxeles por cm en eje Y
        draw_on_image: Si True, dibuja el bbox y las medidas
        
    Returns:
        tuple: (largo_px, ancho_px, largo_cm, ancho_cm, rotated_rect)
    """
    # Calcular el rectángulo rotado mínimo
    rotated_rect = cv2.minAreaRect(contour)
    (center, (width_px, height_px), angle) = rotated_rect
    
    # Obtener los 4 puntos del rectángulo rotado
    box_points = cv2.boxPoints(rotated_rect)
    box_points = np.int0(box_points)
    
    # Determinar largo y ancho (el mayor valor es el largo)
    largo_px = max(width_px, height_px)
    ancho_px = min(width_px, height_px)
    
    # Convertir a centímetros usando el promedio de las resoluciones X e Y
    px_per_cm_avg = (px_per_cm_x + px_per_cm_y) / 2
    largo_cm = largo_px / px_per_cm_avg
    ancho_cm = ancho_px / px_per_cm_avg
    
    if draw_on_image:
        cv2.drawContours(img, [box_points], 0, (255,180, 0), 3)
    
    return largo_px, ancho_px, largo_cm, ancho_cm, rotated_rect


####
def create_mask(
    img_hsv,
    lower_black=None,  # Cambiado a None para manejar el valor por defecto después
    upper_black=None,  # Cambiado a None para manejar el valor por defecto después
    n_iteration=1,
    n_kernel=7,
    canny_min=30,
    canny_max=100,
    plot=True,
    figsize=(20,10),
    axis = False
):
    """
    Creates a mask to segment objects in an HSV image with customizable parameters.
    
    Parameters:
    - img_hsv: Image in HSV format (3D numpy array)
    - lower_black: Lower bound for HSV background detection (numpy array or list)
    - upper_black: Upper bound for HSV background detection (numpy array or list)
    - n_iteration: Number of iterations for morphological operations
    - n_kernel: Kernel size for morphological operations (must be odd)
    - canny_min: First threshold for Canny edge detection
    - canny_max: Second threshold for Canny edge detection
    - plot: Whether to plot the resulting mask
    - figsize: Figure size for plotting
    
    Returns:
    - Binary mask as 2D numpy array
    
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
    
        # Set default values if not provided
        if lower_black is None:
            lower_black = np.array([0, 0, 0], dtype=np.uint8)
        elif isinstance(lower_black, list):
            lower_black = np.array(lower_black, dtype=np.uint8)
            
        if upper_black is None:
            upper_black = np.array([180, 255, 30], dtype=np.uint8)
        elif isinstance(upper_black, list):
            upper_black = np.array(upper_black, dtype=np.uint8)

        # Validate HSV bounds
        if not isinstance(lower_black, np.ndarray) or lower_black.shape != (3,):
            raise ValueError("lower_black must be a numpy array with shape (3,)")
        if not isinstance(upper_black, np.ndarray) or upper_black.shape != (3,):
            raise ValueError("upper_black must be a numpy array with shape (3,)")
            
        if (lower_black > upper_black).any():
            raise ValueError("All values in lower_black must be <= corresponding values in upper_black")

        # Rest of the function remains the same...
        mask_background = cv2.inRange(img_hsv, lower_black, upper_black)
        if mask_background is None:
            raise RuntimeError("Failed to create initial mask")

        mask_inverted = cv2.bitwise_not(mask_background)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n_kernel, n_kernel))
        mask_open = cv2.morphologyEx(mask_inverted, cv2.MORPH_OPEN, kernel, iterations=n_iteration)
        mask_closed = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel, iterations=n_iteration)
        
        blurred = cv2.GaussianBlur(mask_closed, (n_kernel, n_kernel), 0)
        edges = cv2.Canny(blurred, canny_min, canny_max)
        
        final_mask = cv2.bitwise_or(mask_closed, edges)

        if plot:
            if axis == False:
                plt.figure(figsize=figsize)
                plt.imshow(final_mask, cmap='gray')
                plt.axis('off')
                plt.show()
            else:
                plt.figure(figsize=figsize)
                plt.imshow(final_mask, cmap='gray')
                plt.show()

        return final_mask
        
    except cv2.error as e:
        raise RuntimeError(f"OpenCV error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Unexpected error: {str(e)}")

    ######

def pixels_per_cm(img, size='letter_ansi', width_cm=None, height_cm=None):
    """
    Calculate pixels per centimeter for a given image and paper size.
    
    Parameters:
    - img: Input image (numpy array)
    - size: Predefined paper size ('letter_ansi', 'legal_ansi', 'a4_iso', 'a3_iso')
    - width_cm: Custom width in cm (overrides size if provided)
    - height_cm: Custom height in cm (overrides size if provided)
    
    Returns:
    - Tuple of (pixels_per_cm_x, pixels_per_cm_y, width_cm, height_cm)
    
    Raises:
    - ValueError: For invalid inputs
    """
    try:
        # Validación de la imagen
        if not isinstance(img, np.ndarray) or img.ndim not in [2, 3]:
            raise ValueError("Input must be a valid 2D or 3D numpy array image")
            
        # Obtener dimensiones de la imagen
        img_height_px, img_width_px = img.shape[:2]
        
        # Validación de dimensiones
        if img_width_px <= 0 or img_height_px <= 0:
            raise ValueError("Image dimensions must be positive")

        # Asignar dimensiones según tamaño predefinido o valores personalizados
        size_dimensions = {
            'letter_ansi': (27.85, 21.8),
            'legal_ansi': (21.59, 35.56),
            'a4_iso': (21.0, 29.7),
            'a3_iso': (29.7, 42.0)
        }

        if width_cm is not None and height_cm is not None:
            # Usar dimensiones personalizadas si se proporcionan
            if width_cm <= 0 or height_cm <= 0:
                raise ValueError("Dimensions must be positive values")
            used_width_cm = width_cm
            used_height_cm = height_cm
        elif size in size_dimensions:
            # Usar tamaño predefinido
            used_width_cm, used_height_cm = size_dimensions[size]
        else:
            raise ValueError("Invalid size parameter or missing dimensions")

        # Calcular píxeles por centímetro
        pixels_per_cm_x = img_width_px / used_width_cm
        pixels_per_cm_y = img_height_px / used_height_cm

        return pixels_per_cm_x, pixels_per_cm_y, used_width_cm, used_height_cm

    except Exception as e:
        raise RuntimeError(f"Error calculating pixels per cm: {str(e)}")
    

#####



def find_fruit_loculi(
    mask: np.ndarray,
    min_loculus_area: int = 50,
    min_loculi_count: int = 1,
    is_contour_bad_func=None
) -> Dict[int, List[int]]:
    """
    Identifica frutas y sus lóculos (segmentos internos) a partir de una máscara binaria.
    
    Args:
        mask: Máscara binaria donde los objetos blancos representan frutas
        min_loculus_area: Área mínima (en píxeles) para considerar un lóculo válido
        min_loculi_count: Número mínimo de lóculos para considerar una fruta válida
        is_contour_bad_func: Función opcional para filtrar contornos malformados
        
    Returns:
        Diccionario donde las claves son índices de frutas y los valores son listas 
        de índices de lóculos asociados a cada fruta
        
    Raises:
        ValueError: Si la máscara no es binaria o los parámetros son inválidos
    """
    # Validación de entrada
    if len(mask.shape) != 2 or mask.dtype != np.uint8:
        raise ValueError("La máscara debe ser una imagen binaria (1 canal, tipo uint8)")
    
    if min_loculus_area <= 0 or min_loculi_count < 0:
        raise ValueError("Los parámetros de área y conteo deben ser positivos")

    # Encontrar contornos
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if hierarchy is None:
        return {}

    hierarchy = hierarchy[0]  # Simplificar estructura
    
    # Filtrar frutas (contornos externos)
    fruit_indices = [
        i for i in range(len(contours))
        if hierarchy[i][3] == -1 and  # Contorno padre es -1 (externo)
           (is_contour_bad_func is None or not is_contour_bad_func(contours[i]))
    ]
    
    # Mapear frutas a sus lóculos
    fruit_loculi_map = {}
    for fruit_idx in fruit_indices:
        loculi = [
            j for j in range(len(contours))
            if hierarchy[j][3] == fruit_idx  # Lóculos son hijos directos
        ]
        
        # Filtrar lóculos por área mínima
        filtered_loculi = [
            loculus for loculus in loculi
            if cv2.contourArea(contours[loculus]) >= min_loculus_area
        ]
        
        # Solo incluir frutas con suficientes lóculos
        if len(filtered_loculi) >= min_loculi_count:
            fruit_loculi_map[fruit_idx] = filtered_loculi
    
    return contours, fruit_loculi_map




def find_fruits(
    binary_mask: np.ndarray,
    min_loculus_area = 50,
    min_loculi_per_fruit = 1,
    min_circularity = 0.4,
    max_circularity = 1.1,
    min_aspect_ratio=0.3,
    max_aspect_ratio= 3.0,
    contour_approximation: int = cv2.CHAIN_APPROX_SIMPLE,
    contour_filters: Optional[Dict] = None
) -> Dict[int, List[int]]:
    """
    Detecta frutas y sus lóculos en una máscara binaria con filtrado avanzado de contornos.
    
    Args:
        binary_mask: Máscara binaria (blanco=objetos, negro=fondo)
        min_loculus_area: Área mínima en píxeles para lóculos
        min_loculi_per_fruit: Mínimo de lóculos para considerar fruta
        contour_approximation: Método de aproximación de contornos
        contour_filters: {
            'min_area': 50,
            'min_circularity': 0.2,
            'max_circularity': 1.1,
            'min_aspect_ratio': 0.3,
            'max_aspect_ratio': 3.0
        }
        
    Returns:
        {índice_fruta: [índices_lóculos], ...}
        
    Raises:
        ValueError: Si parámetros son inválidos
    """
    # 1. Configuración de filtros por defecto
    default_filters = {
        'min_area': min_loculus_area,
        'min_circularity': min_circularity,
        'max_circularity': max_circularity,
        'min_aspect_ratio': min_aspect_ratio,
        'max_aspect_ratio': max_aspect_ratio
    }
    
    filters = {**default_filters, **(contour_filters or {})}
    
    # 2. Validación de parámetros
    if not isinstance(binary_mask, np.ndarray) or binary_mask.dtype != np.uint8:
        raise ValueError("Máscara debe ser numpy array uint8")
    
    if any(v <= 0 for v in [min_loculus_area, *filters.values()]):
        raise ValueError("Parámetros deben ser positivos")

    # 3. Detección de contornos
    contours, hierarchy = cv2.findContours(
        binary_mask, 
        cv2.RETR_TREE, 
        contour_approximation
    )
    
    if not contours or hierarchy is None:
        return {}

    hierarchy = hierarchy[0]

    # 4. Función de filtrado integrada con todos los parámetros
    def is_contour_valid(contour: np.ndarray) -> bool:
        """Evalúa múltiples criterios geométricos"""
        area = cv2.contourArea(contour)
        if area < filters['min_area']:
            return False
            
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return False
            
        # Cálculo de métricas
        circularity = 4 * np.pi * area / (perimeter ** 2)
        _, (w, h), _ = cv2.minAreaRect(contour)
        aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
        
        # Evaluación combinada
        return all([
            filters['min_circularity'] <= circularity <= filters['max_circularity'],
            filters['min_aspect_ratio'] <= aspect_ratio <= filters['max_aspect_ratio']
        ])

    # 5. Procesamiento de contornos
    fruit_loculi_map = {}
    for i, contour in enumerate(contours):
        if hierarchy[i][3] == -1 and is_contour_valid(contour):  # Frutas válidas
            loculi = [
                j for j in range(len(contours))
                if hierarchy[j][3] == i and 
                   cv2.contourArea(contours[j]) >= min_loculus_area
            ]
            
            if len(loculi) >= min_loculi_per_fruit:
                fruit_loculi_map[i] = loculi
    
    return contours, fruit_loculi_map

def merge_loculi(loculi_indices, contours, max_distance=50, min_area=2):
    """
    Fusiona lóculos que están fragmentados en múltiples contornos cercanos.
    
    Args:
        loculi_indices: Lista de índices de los lóculos en la lista de contornos
        contours: Lista completa de todos los contornos
        max_distance: Distancia máxima entre lóculos para considerarlos fragmentos del mismo
        min_area: Área mínima para considerar un lóculo (elimina ruido pequeño)
        
    Returns:
        Lista de contornos fusionados (los lóculos ya unidos)
    """
    if not loculi_indices:
        return []
    
    # Filtrar lóculos muy pequeños (probablemente ruido)
    valid_loculi = [i for i in loculi_indices if cv2.contourArea(contours[i]) > min_area]
    
    if not valid_loculi:
        return []
    
    # Lista para marcar lóculos ya fusionados
    merged = [False] * len(valid_loculi)
    result_loculi = []
    
    for i in range(len(valid_loculi)):
        if not merged[i]:
            current_idx = valid_loculi[i]
            current_contour = contours[current_idx]
            merged[i] = True
            to_merge = [current_contour]
            
            # Obtener centroide del lóculo actual
            M = cv2.moments(current_contour)
            if M["m00"] == 0:
                cx, cy = current_contour[0][0][0], current_contour[0][0][1]
            else:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            
            # Buscar lóculos cercanos
            for j in range(i+1, len(valid_loculi)):
                if not merged[j]:
                    other_idx = valid_loculi[j]
                    other_contour = contours[other_idx]
                    
                    M_j = cv2.moments(other_contour)
                    if M_j["m00"] == 0:
                        cx_j, cy_j = other_contour[0][0][0], other_contour[0][0][1]
                    else:
                        cx_j = int(M_j["m10"] / M_j["m00"])
                        cy_j = int(M_j["m01"] / M_j["m00"])
                    
                    # Calcular distancia entre centroides
                    dist = np.sqrt((cx - cx_j)**2 + (cy - cy_j)**2)
                    
                    if dist < max_distance:
                        to_merge.append(other_contour)
                        merged[j] = True
            
            # Fusionar los lóculos cercanos
            if len(to_merge) > 1:
                merged_contour = np.vstack(to_merge)
                # Usamos convexHull para obtener una forma suave
                merged_loculus = cv2.convexHull(merged_contour)
                result_loculi.append(merged_loculus)
            else:
                result_loculi.append(current_contour)
    
    return result_loculi


####
# Funciones auxiliares mejoradas
def precalculate_loculi_data(contours, loculi):
    """Precalcula y almacena datos de lóculos para optimización"""
    loculi_data = []
    for loculus in loculi:
        M = cv2.moments(contours[loculus])
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            area = cv2.contourArea(contours[loculus])
            perimeter = cv2.arcLength(contours[loculus], True)
            loculi_data.append({
                'centroid': (cx, cy),
                'area': area,
                'perimeter': perimeter,
                'contour': contours[loculus]
            })
    return loculi_data

def angular_symmetry_from_data(loculi_data, centroid):
    angles = []
    for data in loculi_data:
        cx, cy = data['centroid']
        dx, dy = cx - centroid[0], cy - centroid[1]
        angle = math.atan2(dy, dx)
        angles.append(angle)
    
    if len(angles) < 2:
        return 0.0
    
    if len(np.unique(angles)) == 1:
        return 0.0
    
    try:
        return circstd(angles)
    except:
        return float('inf')

def radial_symmetry_from_data(loculi_data, centroid):
    distances = []
    for data in loculi_data:
        cx, cy = data['centroid']
        distance = math.sqrt((cx - centroid[0])**2 + (cy - centroid[1])**2)
        distances.append(distance)
    return np.std(distances) if distances else 0.0

def spatial_symmetry_from_data(loculi_data):
    if len(loculi_data) < 2:
        return 0.0
    
    positions = [data['centroid'] for data in loculi_data]
    nbrs = NearestNeighbors(n_neighbors=2).fit(positions)
    distances, _ = nbrs.kneighbors(positions)
    mean_distance = np.mean(distances[:, 1])
    std_distance = np.std(distances[:, 1])
    return std_distance / mean_distance if mean_distance > 0 else 0.0

def calculate_minor_axis(fruit_contour, px_per_cm_x, px_per_cm_y, point1, point2, img_name = None, fruit_id = None):
    points = fruit_contour.reshape(-1, 2).astype(np.float32)
    
    # Método actual (proyección perpendicular)
    line_vector = point2 - point1
    perpendicular_unit_vector = np.array([-line_vector[1], line_vector[0]])
    perpendicular_unit_vector /= np.linalg.norm(perpendicular_unit_vector)
    
    projections = [np.dot(point - point1, perpendicular_unit_vector) for point in points]
    min_projection, max_projection = min(projections), max(projections)
    min_diameter_px = max_projection - min_projection
    
    # Método alternativo con elipse (para comparación)
    if len(points) >= 5:
        ellipse = cv2.fitEllipse(points)
        ellipse_minor = min(ellipse[1])
        discrepancy = abs(min_diameter_px - ellipse_minor) / ((min_diameter_px + ellipse_minor)/2)
        if discrepancy > 0.2:
            warning_msg = f"Warning: Discrepancy on minor axis ({discrepancy*100:.1f}%)"
            if img_name:
                warning_msg += f' in: {img_name} (fruit ID: {fruit_id})'

            print(warning_msg)
    
    return min_diameter_px / ((px_per_cm_x + px_per_cm_y) / 2)


######################################

def analyze_fruits(img, contours, fruit_locus_map, px_per_cm_x, px_per_cm_y, 
                  img_name, label_text, use_ellipse_loculi=False, use_ellipse_fruit=False,
                  max_dist=30, min_dist=2, plot=True, figsize=(20,10),
                  font_scale=1, font_thickness=2, text_color=(0,0,0), 
                  bg_color=(255,255,255), padding=15, line_spacing=15, min_locule_area=300,
                  max_locule_area = None, path = None,
                  stamp=False, contour_mode = 'raw', epsilon_hull = '0.005', fig_axis = True,
                  fig_title_fontsize = 20, fig_title_loc = 'center'):
    """
    Analiza características morfométricas de frutos y sus lóculos
    
    Parámetros:
    -----------
    img : ndarray
        Imagen original en formato BGR
    contours : list
        Lista de contornos detectados
    fruit_locus_map : dict
        Diccionario que mapea índices de frutos a listas de índices de lóculos
    px_per_cm_x : float
        Píxeles por centímetro en el eje X
    px_per_cm_y : float
        Píxeles por centímetro en el eje Y
    img_name : str
        Nombre de la imagen para identificación
    label_text : str
        Etiqueta de clasificación del fruto
    use_ellipse_loculi : bool, optional
        Si True, usa elipses para aproximar los lóculos (default: True)
    use_ellipse_fruit : bool, optional
        Si True, usa elipses para aproximar los frutos (default: False)
    max_dist : int, optional
        Distancia máxima para fusionar lóculos (default: 50)
    min_dist : int, optional
        Distancia mínima para considerar lóculos (default: 2)
    plot : bool, optional
        Si True, muestra los resultados gráficos (default: True)
    figsize : tuple, optional
        Tamaño de la figura para plotting (default: (20,10))
    font_scale : float, optional
        Tamaño de fuente para anotaciones (default: 3.5)
    font_thickness : int, optional
        Grosor de fuente para anotaciones (default: 8)
    text_color : tuple, optional
        Color del texto (BGR) (default: (0,0,0))
    bg_color : tuple, optional
        Color de fondo (BGR) (default: (255,255,255))
    padding : int, optional
        Espaciado para anotaciones (default: 15)
    line_spacing : int, optional
        Espacio entre líneas de texto (default: 20)
    
    Retorna:
    --------
    tuple: (results, annotated_img)
        results: Lista de diccionarios con métricas para cada fruto
        annotated_img: Imagen con anotaciones visuales
    """
    
    # 1. Copia de la imagen original para anotaciones
    annotated_img = img.copy()
    if stamp == True:
        annotated_img = cv2.bitwise_not(annotated_img)   


    results = []
    
    # 2. Inicializar ID secuencial
    sequential_id = 1

    for fruit_id, loculi in fruit_locus_map.items():
        try:

            # Si es 'raw', no se modifica
            fruit_contour = contours[fruit_id]
            
            if contour_mode == 'hull':
                fruit_contour = cv2.convexHull(fruit_contour)
            elif contour_mode == 'approx':
                peri = cv2.arcLength(fruit_contour, True)
                epsilon = max(1.0, epsilon_hull  * peri)
                fruit_contour = cv2.approxPolyDP(fruit_contour, epsilon, True)
            elif contour_mode == 'ellipse':
                if len(fruit_contour) >= 5:
                    ellipse = cv2.fitEllipse(fruit_contour)
                    fruit_contour = cv2.ellipse2Poly(
                        (int(ellipse[0][0]), int(ellipse[0][1])),
                        (int(ellipse[1][0] / 2), int(ellipse[1][1] / 2)),
                        int(ellipse[2]), 0, 360, 2
                    )
                    fruit_contour = fruit_contour.reshape(-1, 1, 2)


            # Precalcular datos de lóculos (optimización)
            loculi_data = precalculate_loculi_data(contours, loculi)
            
            # =============================================
            # NUEVO: Filtrar lóculos pequeños (si min_locule_area > 0)
            if min_locule_area > 0 or max_locule_area is not None:
                loculi = [
                    loculus for loculus in loculi
                    if cv2.contourArea(contours[loculus]) >= min_locule_area and
                    (max_locule_area is None or cv2.contourArea(contours[loculus]) <= max_locule_area)
                ]
                loculi_data = [
                    data for data in loculi_data
                    if data['area'] >= min_locule_area and
                    (max_locule_area is None or data['area'] <= max_locule_area)
                ]
            # =============================================
            
            # Fusionar lóculos usando parámetros configurables
            merged_loculi_contours = merge_loculi(loculi, contours, 
                                                max_distance=max_dist, 
                                                min_area=min_dist)
            n_locules = len(merged_loculi_contours)

            # Calcular áreas de lóculos usando datos precalculados
            locule_areas = [data['area'] / (px_per_cm_x * px_per_cm_y) for data in loculi_data] if loculi_data else []
            mean_locule_area = np.mean(locule_areas) if locule_areas else 0.0
            std_locule_area = np.std(locule_areas) if locule_areas else 0.0

            # Calcular circularidad promedio de los lóculos
            locule_circularities = [
                4 * np.pi * area / (data['perimeter'] / ((px_per_cm_x + px_per_cm_y) / 2))**2 
                if data['perimeter'] > 0 else 0 
                for data, area in zip(loculi_data, locule_areas)
            ] if loculi_data else []
            
            mean_locule_circularity = np.mean(locule_circularities) if locule_circularities else 0.0
            std_locule_circularity = np.std(locule_circularities) if locule_circularities else 0.0

            # ------------------------------------------------
            # Cálculo de ejes mayor y menor
            # ------------------------------------------------
            points = fruit_contour.reshape(-1, 2).astype(np.float32)
            hull = ConvexHull(points)
            max_diameter_px = 0
            point1 = point2 = None
            
            # Buscar los dos puntos más distantes
            for i in hull.vertices:
                for j in hull.vertices:
                    dist = np.linalg.norm(points[i] - points[j])
                    if dist > max_diameter_px:
                        max_diameter_px = dist
                        point1, point2 = points[i], points[j]

            # Convertir el diámetro a centímetros
            max_diameter_cm = max_diameter_px / ((px_per_cm_x + px_per_cm_y) / 2)
            
            # Calcular eje menor mejorado con verificación
            min_diameter_cm = calculate_minor_axis(fruit_contour, px_per_cm_x, px_per_cm_y, point1, point2, img_name = img_name, fruit_id = sequential_id)

            # Aproximación por elipse (si está habilitada)
            if use_ellipse_fruit and len(points) >= 5:
                ellipse = cv2.fitEllipse(points)
                a = max(ellipse[1]) / 2
                b = min(ellipse[1]) / 2
                largo_px = a * 2
                ancho_px = b * 2
                largo_cm = largo_px / ((px_per_cm_x + px_per_cm_y) / 2)
                ancho_cm = ancho_px / ((px_per_cm_x + px_per_cm_y) / 2)
                cv2.ellipse(annotated_img, ellipse, (255, 255, 0), 2)
            else:
                # Método alternativo con rotated rectangle
                largo_px, ancho_px, largo_cm, ancho_cm, rotated_rect = rotate_box(
                    annotated_img, fruit_contour, px_per_cm_x, px_per_cm_y)
                box_height_cm = largo_cm
                box_width_cm = ancho_cm

            # Calcular centroide del fruto
            M_fruit = cv2.moments(fruit_contour)
            if M_fruit["m00"] != 0:
                fruit_centroid = (int(M_fruit["m10"] / M_fruit["m00"]), int(M_fruit["m01"] / M_fruit["m00"]))
            else:
                fruit_centroid = (0, 0)

            # === DIBUJAR CENTROIDE DEL FRUTO ===

            cv2.circle(annotated_img, fruit_centroid, 15, (255, 255, 51), -1)  # Azul (BGR)

            # === DIBUJAR CENTROIDES DE LOS LÓCULOS ===
            #for data in loculi_data:
            #    cx, cy = data['centroid']
            #    cv2.circle(annotated_img, (int(cx), int(cy)), 7, (0, 255, 255), -1)  # Amarillo

            # Calcular simetrías con datos precalculados
            spatial_symmetry = spatial_symmetry_from_data(loculi_data)
            radial_symmetry = radial_symmetry_from_data(loculi_data, fruit_centroid)
            angular_symmetry = angular_symmetry_from_data(loculi_data, fruit_centroid)

            # Dibujar eje menor
            if point1 is not None and point2 is not None:
                line_vector = point2 - point1
                perpendicular_unit_vector = np.array([-line_vector[1], line_vector[0]])
                perpendicular_unit_vector /= np.linalg.norm(perpendicular_unit_vector)
                
                projections = [np.dot(point - point1, perpendicular_unit_vector) for point in points]
                min_idx = np.argmin(projections)
                max_idx = np.argmax(projections)
                
                cv2.line(
                    annotated_img,
                    tuple(points[min_idx].astype(int)),
                    tuple(points[max_idx].astype(int)),
                    (255, 0, 0), 2
                )

            # Calcular métricas principales
            fruit_perimeter_px = cv2.arcLength(fruit_contour, True)
            fruit_perimeter_cm = fruit_perimeter_px / ((px_per_cm_x + px_per_cm_y) / 2)
            fruit_area_cm2 = cv2.contourArea(fruit_contour) / (px_per_cm_x * px_per_cm_y)

            # Calcular área del pericarpio interno (configurable)
            inner_pericarp_area_px = 0
            if len(merged_loculi_contours) > 0:
                all_points = np.vstack(merged_loculi_contours)
                
                if use_ellipse_loculi and all_points.shape[0] >= 5:
                    ellipse = cv2.fitEllipse(all_points.astype(np.float32))
                    cv2.ellipse(annotated_img, ellipse, (0, 255, 255), 2)
                    a, b = ellipse[1][0]/2, ellipse[1][1]/2
                    inner_pericarp_area_px = np.pi * a * b
                else:
                    hull = cv2.convexHull(all_points)
                    epsilon = 0.0001 * cv2.arcLength(hull, True)
                    smoothed_hull = cv2.approxPolyDP(hull, epsilon, True)
                    cv2.drawContours(annotated_img, [smoothed_hull], -1, (0, 255, 255), 2)
                    inner_pericarp_area_px = cv2.contourArea(smoothed_hull)
                    
            inner_pericarp_area_cm2 = float(inner_pericarp_area_px) / (px_per_cm_x * px_per_cm_y) if inner_pericarp_area_px else 0.0

            # Calcular métricas de forma
            fruit_circularity = 4 * np.pi * fruit_area_cm2 / (fruit_perimeter_cm ** 2) if fruit_perimeter_cm > 0 else 0
            fruit_aspect_ratio = max_diameter_cm / min_diameter_cm if min_diameter_cm > 0 else 0
            hull = cv2.convexHull(fruit_contour)
            hull_area = cv2.contourArea(hull) / (px_per_cm_x * px_per_cm_y)
            solidity = fruit_area_cm2 / hull_area if hull_area > 0 else 0

            # Métricas adicionales
            locules_density = n_locules / fruit_area_cm2 if fruit_area_cm2 > 0 else 0
            inner_area_ratio = inner_pericarp_area_cm2 / fruit_area_cm2 if fruit_area_cm2 > 0 else 0
            locule_area_ratio = max(locule_areas) / min(locule_areas) if locule_areas and min(locule_areas) > 0 else 0
            compactness = fruit_perimeter_cm ** 2 / (4 * np.pi * fruit_area_cm2) if fruit_area_cm2 > 0 else 0

            # ------------------------------------------------
            # Dibujar contornos y anotaciones
            # ------------------------------------------------
            # ------------------------------------------------
            # Dibujar contornos y anotaciones (versión ajustada)
            # ------------------------------------------------
            # Dibujar el contorno del fruto (ajustado según contour_mode)
            cv2.drawContours(annotated_img, [fruit_contour], -1, (0, 255, 0), 2)  # Cambio clave aquí

            # Dibujar lóculos fusionados
            for loculus_contour in merged_loculi_contours:
                cv2.drawContours(annotated_img, [loculus_contour], -1, (255, 0, 255), 2)

            # Dibujar línea de medición (si existe)
            if point1 is not None and point2 is not None:
                cv2.line(annotated_img, 
                        tuple(point1.astype(int)), 
                        tuple(point2.astype(int)), 
                        (0, 0, 255), 3)

            # Añadir anotaciones de texto con parámetros configurables
            x, y, w, h = cv2.boundingRect(fruit_contour)
            text = f"id {sequential_id}: \n{n_locules} loc"
            
            (single_line_width, single_line_height), baseline = cv2.getTextSize("Test", cv2.FONT_HERSHEY_SIMPLEX, 
                                                                              font_scale, font_thickness)
            total_height = (single_line_height * 2) + line_spacing
            text_x = max(10, x)
            text_y = max(total_height + 15, y - 15)
            text_width = max([cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0][0] 
                            for line in text.split('\n')])
            
            # Dibujar fondo semitransparente
            text_bg_layer = annotated_img.copy()
            cv2.rectangle(text_bg_layer,
                        (text_x - padding, text_y - total_height - padding),
                        (text_x + text_width + padding, text_y + padding),
                        bg_color, -1)
            cv2.addWeighted(text_bg_layer, 0.7, annotated_img, 0.3, 0, annotated_img)
            
            # Dibujar texto con parámetros configurables
            for i, line in enumerate(text.split('\n')):
                y_offset = text_y - (total_height - single_line_height) + (i * (single_line_height + line_spacing))
                cv2.putText(annotated_img, line,
                          (text_x, y_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                          text_color, font_thickness,
                          cv2.LINE_AA)

            # Almacenar resultados
            result = {
                'image_name': img_name,
                'label': label_text, 
                'fruit_id': sequential_id,
                'n_locules': n_locules,
                'major_axis_cm': max_diameter_cm,
                'minor_axis_cm': min_diameter_cm,
                'fruit_area_cm2': fruit_area_cm2,
                'inner_pericarp_area_cm2': inner_pericarp_area_cm2,
                'outer_pericarp_area_cm2': fruit_area_cm2 - inner_pericarp_area_cm2,
                'fruit_perimeter_cm': fruit_perimeter_cm,
                'mean_locule_area_cm2': mean_locule_area,
                'std_locule_area_cm2': std_locule_area,
                'fruit_spatial_symmetry': spatial_symmetry,
                'fruit_radial_symmetry': radial_symmetry,
                'fruit_angular_symmetry': angular_symmetry,
                'fruit_circularity': fruit_circularity,
                'fruit_aspect_ratio': fruit_aspect_ratio,
                'fruit_solidity': solidity,
                'locules_density': locules_density,
                'inner_area_ratio': inner_area_ratio,
                'locule_area_ratio': locule_area_ratio,
                'fruit_compactness': compactness,
                'mean_locule_circularity': mean_locule_circularity,
                'std_locule_circularity': std_locule_circularity
            }
            
            # Añadir métricas específicas según método usado
            if use_ellipse_fruit:
                result.update({
                    'ellipse_height_cm': largo_cm,
                    'ellipse_width_cm': ancho_cm
                })
            else:
                result.update({
                    'box_height_cm': box_height_cm,
                    'box_width_cm': box_width_cm
                })
                
            results.append(result)
            sequential_id += 1

        except Exception as e:
            print(f"Error procesando fruto {fruit_id}: {str(e)}")
            continue


    # Mostrar resultados si está habilitado
    if plot:
        if fig_axis == False:
            plt.figure(figsize=figsize)
            plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)), 
            plt.title(f"{img_name}: {label_text}", fontsize = fig_title_fontsize, loc=fig_title_loc)
            plt.tight_layout()
            plt.axis('off')
            plt.show()
        else:
            plt.figure(figsize=figsize)
            plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)), 
            plt.title(f"{img_name}: {label_text}", fontsize = fig_title_fontsize, loc=fig_title_loc)
            plt.tight_layout()
            plt.show()
   
    
    return AnnotatedImage(annotated_img, results, image_path = path)


#########

def pdf_to_img(path_pdf, dpi = 600, path_img = None, n_threads = None, output_message = True):
    # Verify PDF extension first
    pdf_basename = os.path.basename(path_pdf)
    pdf_name, ext = os.path.splitext(pdf_basename)
    valid_extension = '.pdf'

    if ext.lower() not in valid_extension:
        print("Error: Input is not a PDF file")
    else:
        try: 
            if n_threads is None:
                n_threads = 1
            
            images = convert_from_path(
                path_pdf,
                dpi = dpi,
                thread_count = n_threads
            )
        
            if path_img is None or not os.path.exists(path_img):
                    dirname = os.path.dirname(path_pdf)
                    path_res = os.path.join(dirname, "images_from_pdf")
                    os.makedirs(path_res, exist_ok=True)
                    path_img = path_res  # Update path_img to the new directory

            # Save images
            for i, image in enumerate(images):
                img_name = f"{pdf_name}_page{i+1}.jpg"
                output_path = os.path.join(path_img, img_name)
                image.save(output_path, 'JPEG')
            
            if output_message:
                print(f"{len(images)} images saved in: {path_res}")

        except Exception as e:
            print(f'An unexpected error ocurred: {str(e)}')

############### 

# Analyzing multiple photos in a folder

def processing_images(path_input, output_dir=None, canny_min=300, canny_max=100, n_kernel=7, 
                     max_merge_dist=50, min_merge_dist=2, padding=10, n_iterations=1, 
                     lower_black=None, upper_black=None, min_loculi_count=1, 
                     contour_approx: int=cv2.CHAIN_APPROX_SIMPLE, min_locule_area=500, 
                     ellipse_loculi=False, ellipse_fruit=False, font_scale=4, 
                     font_thickness=4, line_spacing=15, min_circularity=0.3, 
                     max_circularity=1.2, epsilon_hull='0.005', contour_mode='raw',
                     stamps=False, plot=False, width_cm=None, height_cm=None, 
                     size_dimension='letter_ansi'):

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

    print("╔═════════════════════════╗")
    print("║    MorphoSlicer running ⋆✧｡٩(ˊᗜˋ )و✧*｡   ║")
    print("╚═════════════════════════╝")

    for filename in tqdm(file_list, desc='Processing images', unit='image'):
        ext = os.path.splitext(filename)[1].lower()

        if ext in valid_extensions:
            path_image = os.path.join(path_input, filename)
            
            try:
                # Carga y preprocesamiento de imagen
                img = load_image(path_image, plot=False)
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
                                      plot=plot,
                                      n_kernel=n_kernel,
                                      canny_max=canny_min,
                                      canny_min=canny_max,
                                      n_iteration=n_iterations,
                                      lower_black=lower_black,
                                      upper_black=upper_black)
                
                # Detección de frutos
                contours, fruit_locus_map = find_fruits(final_mask,
                                                     min_loculi_per_fruit=min_loculi_count,
                                                     min_loculus_area=min_locule_area,
                                                     contour_approximation=contour_approx,
                                                     min_circularity=min_circularity,
                                                     max_circularity=max_circularity)
                
                if not contours:
                    errors_report.append({'filename': filename, 'status': 'No contours found'})
                    skipped_no_contours += 1
                    continue

                # Análisis morfométrico
                label_text = detect_label(img, contours, fruit_locus_map)
                img_name = detect_img_name(path_image)
                px_per_cm_x, px_per_cm_y, _, _ = pixels_per_cm(img,
                                                             width_cm=width_cm,
                                                             height_cm=height_cm,
                                                             size=size_dimension)

                results = analyze_fruits(
                    img=img,
                    contours=contours,
                    max_dist=max_merge_dist,
                    min_dist=min_merge_dist,
                    min_locule_area=min_locule_area,
                    fruit_locus_map=fruit_locus_map,
                    px_per_cm_x=px_per_cm_x,
                    px_per_cm_y=px_per_cm_y,
                    img_name=img_name,
                    label_text=label_text,
                    use_ellipse_loculi=ellipse_loculi,
                    use_ellipse_fruit=ellipse_fruit,
                    plot=plot,
                    font_scale=font_scale,
                    font_thickness=font_thickness,
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

    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║   ( ദ്ദി ˙ᗜ˙ ) ✧   Processing completed successfully!")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║ 🕒 Total time:      {elapsed_min:.2f} minutes")
    print(f"║ 💾 RAM used:        {ram_used_gb:.2f} GB")
    print(f"║ 🖼️ Images processed: {processed_count}/{len(file_list)}")
    print(f"║ 📁 Output folder:   {output_dir}")
    print(f"║ ⚠️  Errors:          {skipped_errors} skipped")
    print("╚══════════════════════════════════════════════════════════════════════╝")