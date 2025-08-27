import cv2
import os
import numpy as np
import easyocr
import warnings
import matplotlib.pyplot as plt
from pdf2image import convert_from_path

valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}

##############################################################################
# Load an image
##############################################################################

def load_img(path, plot=True, plot_size=(20, 10), fig_axis=True):
    """Load a BGR image from file and validate its format.
    
    Args:
        path (str): Full path to the image file.
        plot (bool, optional): Whether to display the image after loading. 
                              Defaults to True.
        plot_size (tuple, optional): Figure size for display in inches (width, height). 
                                    Defaults to (20, 10).
        fig_axis (bool, optional): Whether to show axis when plotting. 
                                  Defaults to True.

    Returns:
        numpy.ndarray or None: Loaded image as a numpy array in BGR format, 
                              or None if loading failed.

    Raises:
        ValueError: If the file extension is not valid or image cannot be loaded.

    """
    try:
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}  # BGR only
        ext = os.path.splitext(path)[1].lower()  # Obtain image extension

        if ext not in valid_extensions: 
            raise ValueError("Image format not valid (expected .jpg/.jpeg/.png/.tiff)")

        img = cv2.imread(path)  # Read image in BGR format
        if img is None:
            raise ValueError(f"Cannot load image: {os.path.basename(path)}")
        
        if plot:
            plot_img(img, metadata=False, fig_axis=fig_axis, plot_size=plot_size)

        return img

    except Exception as e:
        print(f"Error loading image: {e}")
        return None

##############################################################################
# Evaluate if a contour is valid using geometric thresholds
##############################################################################

def is_contour_valid(contour, filters=None):
    """Evaluates if a contour meets all geometric criteria.
    
    Args:
        REQUIRED:
        - contour (np.ndarray): Input contour to evaluate (shape: [N, 1, 2]).
        
        OPTIONAL:
        - filters (Dict): Dictionary containing filter thresholds with keys:
            - min_area (int): Minimum area threshold (default: 300)
            - min_circularity (float): Minimum circularity threshold (default: 0.7)
            - max_circularity (float): Maximum circularity threshold (default: 1.0)
            - min_aspect_ratio (float): Minimum aspect ratio threshold (default: 0.8)
            - max_aspect_ratio (float): Maximum aspect ratio threshold (default: 1.0)
            
    Returns:
        bool: True if contour passes all filters, False otherwise
    """
    # Valores por defecto
    default_filters = {
        'min_area': 300,
        'min_circularity': 0.6,
        'max_circularity': 1.0,
        'min_aspect_ratio': 0.4,
        'max_aspect_ratio': 1.0
    }
    
    # Combinar filtros proporcionados con valores por defecto
    if filters is None:
        filters = default_filters
    else:
        # Actualizar solo las claves proporcionadas, mantener las demás por defecto
        filters = {**default_filters, **filters}
    
    area = cv2.contourArea(contour)
    if area < filters['min_area']:
        return False
        
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
        
    circularity = (4 * np.pi * area) / (perimeter ** 2) 
    _, (w, h), _ = cv2.minAreaRect(contour)
    aspect_ratio = min(w, h) / max(w, h) if max(w, h) > 0 else 0
    
    return (filters['min_circularity'] <= circularity <= filters['max_circularity'] and
            filters['min_aspect_ratio'] <= aspect_ratio <= filters['max_aspect_ratio'])


#################################################################################################
# Detect label text
#################################################################################################

def detect_label(img, contours, filtered_fruit_locus_map, language = ['es','en'], min_area_label = 500, blur_label = (11,11), min_canny_label = 0, max_canny_label = 150):
    try:
        if language:
            reader = easyocr.Reader(language)
    except: 
        reader = easyocr.Reader(['en'])
        
    label_contours = []
    filtered_contours = []

    label_text = "No label included/detected"
    label_coordinates = None  

    for contour in contours:
        contour = np.asarray(contour, dtype=np.float32)

        try:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            area = cv2.contourArea(contour)

            if area > min_area_label and len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                label_region = img[y:y+h, x:x+w]

                gray = cv2.cvtColor(label_region, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, blur_label, 0)
                edges = cv2.Canny(blur, min_canny_label, max_canny_label)
                _, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                results = reader.readtext(thresh)

                if results:
                    label_text = " ".join([result[1] for result in results])
                    label_coordinates = contour.tolist()  
                    label_contours.append(contour)
                    return label_text, label_coordinates  
            filtered_contours.append(contour)
        except Exception as e:
            print(f"Error procesando contorno: {e}")
            continue

    return label_text, label_coordinates 


#################################################################################################
# Detect image name
#################################################################################################

def detect_img_name(path_image):
    try:
        if not isinstance(path_image, str):
            raise TypeError('Path input should be of type str')
        
        extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}

        filename = os.path.basename(path_image)
        name, ext = os.path.splitext(filename)

        if ext.lower() not in extensions:
            warnings.warn("Warning: File extension is not a valid image format.")
        return name
    
    except Exception as e:
        print(f"Error: {e}")
        return None
    


#################################################################################################
# Plotting image on screen
#################################################################################################

def plot_img(img, fig_axis=False, plot_size=(10, 10), label_text='None', 
             img_name='None', title_fontsize=12, title_location='center', 
             metadata=True, gray = False):
    """
    Plots an image with customizable display options.
    
    Args:
        img (numpy.ndarray): Input image in BGR format
        fig_axis (bool): Whether to show axis (default: False)
        plot_size (tuple): Figure size (width, height) in inches (default: (10, 10))
        label_text (str): Text label for the title (default: 'None')
        img_name (str): Image name for the title (default: 'None')
        title_fontsize (int): Font size for the title (default: 12)
        title_location (str): Title location ('left', 'center', 'right') (default: 'center')
        metadata (bool): If True, suppresses title display (default: False)
    """
    plt.figure(figsize=plot_size)
    if gray:
        plt.imshow(img, cmap='gray') 
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if metadata:
        plt.title(f"{img_name}: {label_text}", fontsize=title_fontsize, loc=title_location)
    
    plt.tight_layout()
    
    if not fig_axis:
        plt.axis('off')
    
    plt.show()


#################################################################################################
# Converting PDF pages to JPEG images
#################################################################################################

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




def validate_dir(path):
    """
    Ensure the directory exists and return the absolute path.
    
    Args:
        path (str): File path to check
        
    Returns:
        str: Absolute path with ensured directory existence
    """
    # Convert to absolute path and expand user directory (e.g., ~/file.txt)
    abs_path = os.path.abspath(os.path.expanduser(path))
    
    # Extract directory portion from the absolute path
    dir_path = os.path.dirname(abs_path)
    
    # Create directory hierarchy if it doesn't exist and path contains directories
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)
    
    return abs_path