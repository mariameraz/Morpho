import cv2
import os
import numpy as np
import easyocr
import warnings
import matplotlib.pyplot as plt


valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif'}

def load_image(path, plot = True, figsize = (20,10), axis = True, title = None, title_location = None, title_fontsize = None):
    """Load an sBGR image and validate if its format is valid (.jpg/.jpeg/.png/.tiff).


    Args: 
        path: Full path to the image file (str).

    Returns:
        img: Loaded image as a numpy array (BGR) or None if loading failed.

    Raises:
        ValueError: If the file extension is not valid. 
    """
    try:
        valid_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'} # sRGB only
        ext = os.path.splitext(path)[1].lower() # Obtain image extension

        if ext not in valid_extensions: 
            raise ValueError("Image format not valid (expected .jpg/.jpeg/.png/.tiff)")

        img = cv2.imread(path) # Read image
        if img is None:
            raise ValueError(f"Cannot load image: {os.path.basename(path)}")
        
        if plot == True:
            if axis == False:
                plt.figure(figsize=figsize)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title(title, loc=title_location, fontsize = title_fontsize)
                plt.show()
            else:
                plt.figure(figsize=figsize)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(title, loc=title_location, fontsize=title_fontsize)
                plt.show()



        return img

    except Exception as e:
        print(e)
        return None


def is_contour_bad(contour, min_area=50, min_circularity=0.2, max_circularity=1.1):
    """Filtra contornos por área y circularidad"""
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return True
    circularity = 4 * np.pi * (area / (perimeter * perimeter))
    
    return area < min_area or not (min_circularity <= circularity <= max_circularity)


def detect_label(img, contours, filtered_fruit_locus_map):
    try:
        reader = easyocr.Reader(['es', 'en'])
    except: 
        reader = easyocr.Reader(['en'])
        
    label_contours = []
    filtered_contours = []

    label_text = "No label included/detected"  # Valor por defecto
    for contour in contours:
        contour = np.asarray(contour, dtype=np.float32)

        try:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            area = cv2.contourArea(contour)

            # Buscar etiquetas (contornos rectangulares grandes)
            if area > 500 and len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                label_region = img[y:y+h, x:x+w]
                
                # Añadir preprocesamiento para mejorar OCR
                gray = cv2.cvtColor(label_region, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (11, 11), 0)
                edges = cv2.Canny(blur, 0, 150)
                _, thresh = cv2.threshold(edges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                results = reader.readtext(thresh)  # Procesar imagen preprocesada
                
                if results:
                    label_text = " ".join([result[1] for result in results])
                    label_contours.append(contour)
                    return label_text  # Retornar inmediatamente si encontramos una etiqueta

            filtered_contours.append(contour)
        except Exception as e:
            print(f"Error procesando contorno: {e}")
            continue

    return label_text

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
    

##########

