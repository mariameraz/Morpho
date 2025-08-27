import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Tuple, Union
import numpy as np
import morphoslicer.internal_structure.core as morpho
from ..utils import common_functions as ms
from tqdm import tqdm
import psutil
from .. import valid_extensions
import time
from pdf2image import convert_from_path

class AnnotatedImage:
    """
    Handles annotated images and results management.
    Stores analysis results and provides saving functionality.
    """
    
    def __init__(self, cv2_image: np.ndarray, results: list = None, image_path: Optional[str] = None):
        self.rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        self.results = results if results else []   
        self.table = self.results                  
        self.image_path = image_path  

    def _ensure_dir_exists(self, path: str) -> str:
        """
        Ensure the directory exists and return the absolute path.
        
        Args:
            path (str): File path to check
            
        Returns:
            str: Absolute path with ensured directory existence
        """
        abs_path = os.path.abspath(os.path.expanduser(path))
        dir_path = os.path.dirname(abs_path)
        
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        
        return abs_path

    def save_img(self, path: Optional[str] = None, format: Optional[str] = None, 
                 dpi: int = 300, output_message: bool = True, **kwargs):
        """
        Save the image in the same directory as the original image.
        
        Args:
            path (str, optional): Output path. If None, generated automatically.
            format (str, optional): Image format. Defaults to extension inference.
            dpi (int): Resolution for raster formats.
            output_message (bool): Whether to show confirmation message.
        """
        try:
            if path is None:
                if not self.image_path:
                    raise ValueError("No path provided and no original image reference available")
                
                original_dir = os.path.dirname(self.image_path)
                base_name = os.path.splitext(os.path.basename(self.image_path))[0]
                ext = format.lower() if format else 'jpg'
                path = os.path.join(original_dir, f"{base_name}_annotated.{ext}")

            full_path = self._ensure_dir_exists(path)
            format = format or os.path.splitext(full_path)[1][1:].lower()
            
            # Create temporary figure for saving
            temp_fig, temp_ax = plt.subplots(figsize=(10, 10))
            temp_ax.imshow(self.rgb_image)
            temp_ax.axis('off')
            temp_ax.set_position([0, 0, 1, 1])  # Remove margins
            
            plt.savefig(full_path, format=format, dpi=dpi, 
                       bbox_inches='tight', pad_inches=0, **kwargs)
            plt.close(temp_fig)
            
            if output_message:
                print(f"Image saved at: {full_path}")
                
        except Exception as e:
            if 'temp_fig' in locals():
                plt.close(temp_fig)
            raise RuntimeError(f"Error saving image: {str(e)}")

    def save_csv(self, path: Optional[str] = None, sep: str = ',', 
                 output_message: bool = True):
        """
        Save CSV in the same directory as the original image.
        
        Args:
            path (str, optional): Output path. If None, generated automatically.
            sep (str): CSV separator.
            output_message (bool): Whether to show confirmation message.
        """
        if not self.table:
            raise ValueError("No results data available to save")
        
        try:
            if path is None:
                if not self.image_path:
                    raise ValueError("No path provided and no original image reference available")
                
                original_dir = os.path.dirname(self.image_path)
                base_name = os.path.splitext(os.path.basename(self.image_path))[0]
                path = os.path.join(original_dir, f"{base_name}_results.csv")
            
            full_path = self._ensure_dir_exists(path)
            pd.DataFrame(self.table).to_csv(full_path, sep=sep, index=False)
            
            if output_message:
                print(f"CSV saved at: {full_path}")
                
        except Exception as e:
            raise RuntimeError(f"Error saving CSV: {str(e)}")

    def save_all(self, base_name: Optional[str] = None, output_dir: Optional[str] = None, 
                 format: str = 'jpg', dpi: int = 300, sep: str = ',', 
                 output_message: bool = True):
        """
        Save both files (image and CSV) using the base name.
        
        Args:
            base_name (str, optional): Base name for files. 
                If None, uses original image name.
            output_dir (str, optional): Output directory. 
                If None, uses original image directory.
            format (str): Image format.
            dpi (int): Image resolution.
            sep (str): CSV separator.
            output_message (bool): Whether to show confirmation messages.
        """
        try:
            # Determine base name
            if base_name is None:
                if not self.image_path:
                    raise ValueError("Cannot determine base name: no original image available")
                base_name = os.path.splitext(os.path.basename(self.image_path))[0]

            # Determine output directory
            if output_dir is None:
                if not self.image_path:
                    raise ValueError("Cannot determine directory: no original image available")
                output_dir = os.path.dirname(self.image_path)
            
            # Ensure output directory exists
            output_dir = self._ensure_dir_exists(output_dir)
            
            # Build complete paths
            img_path = os.path.join(output_dir, f"{base_name}_annotated.{format.lower()}")
            csv_path = os.path.join(output_dir, f"{base_name}_results.csv")
            
            # Save files
            self.save_img(img_path, format=format, dpi=dpi, 
                         output_message=output_message)
            self.save_csv(csv_path, sep=sep, output_message=output_message)
            
        except Exception as e:
            raise RuntimeError(f"Error in save_all: {str(e)}")


class ImageAnalyzer:
    """
    Prepares image data for morphological analysis.
    
    Typical workflow:
    1. read_image()       - Load the image
    2. create_mask()      - Create segmentation mask  
    3. find_fruits()      - Detect fruits and locules
    4. analyze_image()    - Analyze using morpho.analyze_fruits()
    
    OR use analyze_folder() for batch processing
    """
    
    def __init__(self, image_path: str):
        self.image_path = image_path
        self.is_directory = os.path.isdir(image_path)

        self.img = None
        self.img_inverted = None
        self.mask = None
        self.mask_fruits = None
        self.contours = None
        self.fruit_locules_map = None  # Changed from fruit_locus_map for consistency
        self.label_text = None
        self.label_coord = None
        self.label_id = None
        self.img_name = None
        self.px_per_cm_width = None
        self.px_per_cm_length = None
        self.w_cm = None
        self.h_cm = None
        self.results = None
        

    @staticmethod
    def pdf_to_img(pdf_path: str, dpi: int = 300, output_dir: Optional[str] = None, 
                   n_threads: Optional[int] = None, output_message: bool = True) -> List[str]:
        """
        Converts a PDF file to JPEG images (one per page).

        Args:
            pdf_path: Path to the input PDF file.
            dpi: Conversion resolution (dots per inch).
            output_dir: Directory to save the images. If None, creates 'images_from_pdf' in the same folder as the PDF.
            n_threads: Number of threads for parallel processing. If None, uses 1 thread.
            output_message: Whether to print progress messages.

        Returns:
            List of paths to the generated and renamed image files.

        Raises:
            ValueError: If the input file is not a valid PDF.
            RuntimeError: If the conversion process fails.
        """
        # Input validation
        if not os.path.isfile(pdf_path):
            raise ValueError(f"File not found: {pdf_path}")
        
        if not pdf_path.lower().endswith('.pdf'):
            raise ValueError("Input file must be a PDF (.pdf extension)")

        # Set up output paths
        pdf_dir = os.path.dirname(pdf_path)
        pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        if output_dir is None:
            output_dir = os.path.join(pdf_dir, 'images_from_pdf')
        
        os.makedirs(output_dir, exist_ok=True)

        try:
            if output_message:
                print("Extracting images may take a few minutes... ⋆✧｡٩(ˊᗜˋ )و✧*｡")

            # Convert PDF to images
            if n_threads is None:
                n_threads = 1
            
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                thread_count=n_threads
            )
            
            # Save images
            saved_paths = []
            for i, image in enumerate(images):
                img_name = f"{pdf_name}_page{i+1}.jpg"
                output_path = os.path.join(output_dir, img_name)
                image.save(output_path, 'JPEG')
                saved_paths.append(output_path)
            
            if output_message:
                print(f"{len(images)} images saved in: {output_dir}")

            return saved_paths

        except Exception as e:
            error_msg = f"PDF conversion error: {str(e)}"
            if output_message:
                print(error_msg)
            raise RuntimeError(error_msg) from e


    def read_image(self, plot: bool = False, output_message: bool = True, 
                   plot_size: Tuple[int, int] = (5, 5), plot_axis: bool = False) -> None:
        """
        Load and optionally display the image.
        
        Args:
            plot: Whether to display the image
            output_message: Whether to show confirmation message
            plot_size: Figure size for plotting
            plot_axis: Whether to show axis on plot
        """
        self.img = ms.load_img(self.image_path, plot=plot, plot_size=plot_size, fig_axis=plot_axis)
        if self.img is None:
            raise ValueError(f"Failed to load image: {self.image_path}")

        if output_message: 
            img_name = os.path.basename(self.image_path)
            print(f'{img_name} successfully loaded ✧*｡')

        return None

    def create_mask(self, n_kernel: int = 5, plot: bool = False, plot_size: Tuple[int, int] = (5, 5), 
                    stamp: bool = False, plot_axis: bool = False, n_iteration: int = 1, 
                    canny_min: int = 30, canny_max: int = 100, lower_hsv: Optional[List[int]] = None,
                    upper_hsv: Optional[List[int]] = None, locules_filled: bool = False, 
                    min_locule_size: int = 300, n_blur: int = 11, clip_limit: int = 4, 
                    tile_grid_size: int = 8) -> None:
        """
        Create a mask for fruit detection and segmentation.
        
        Args:
            n_kernel: Kernel size for morphological operations
            plot: Whether to display the mask
            plot_size: Figure size for plotting
            stamp: Whether the image is a stamp (inverted colors)
            plot_axis: Whether to show axis on plot
            n_iteration: Number of iterations for morphological operations
            canny_min: Canny edge detector minimum threshold
            canny_max: Canny edge detector maximum threshold
            lower_hsv: Lower HSV range for color-based masking
            upper_hsv: Upper HSV range for color-based masking
            locules_filled: Whether to perform special processing for filled locules
            min_locule_size: Minimum size for locule detection
            n_blur: Blur kernel size for noise reduction
            clip_limit: CLAHE clip limit for contrast enhancement
            tile_grid_size: CLAHE tile grid size
        """
        if stamp:
            self.img_inverted = cv2.bitwise_not(self.img)
        else:
            self.img_inverted = self.img.copy()
        
        # Create base mask - only calculate once
        self.mask = morpho.create_mask(
            self.img_inverted,
            n_kernel=n_kernel, 
            n_iteration=n_iteration,
            plot=False,
            figsize=plot_size,
            axis=plot_axis,
            canny_max=canny_max,
            canny_min=canny_min,
            lower_hsv=lower_hsv,
            upper_hsv=upper_hsv
        )
        
        if locules_filled:
            # Use the already calculated mask instead of recalculating
            base_mask = self.mask.copy()
            
            # Fill fruit contours
            contours, _ = cv2.findContours(base_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.drawContours(base_mask, [cnt], -1, 255, -1)
                
            # Convert image to Lab for locule processing
            lab = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
            l_channel = lab[:,:,0]

            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
            l_clahe = clahe.apply(l_channel)

            _, locule_mask = cv2.threshold(l_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            locule_mask = cv2.medianBlur(locule_mask, n_blur)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n_kernel, n_kernel))
            opened = cv2.morphologyEx(locule_mask, cv2.MORPH_OPEN, kernel)
            closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

            # Detect only internal contours (locules)
            inv_closed = cv2.bitwise_not(closed)
            contours, hierarchy = cv2.findContours(inv_closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            mask_lobules_only = np.zeros_like(closed)
            
            for i, cnt in enumerate(contours):
                if hierarchy[0][i][3] != -1 and cv2.contourArea(cnt) > min_locule_size:
                    cv2.drawContours(mask_lobules_only, [cnt], -1, 255, -1)
            
            mask_lobules_only = cv2.medianBlur(mask_lobules_only, n_blur)

            # Overlap fruits mask with locule mask
            mask_fruits_rgb = cv2.cvtColor(cv2.bitwise_not(base_mask), cv2.COLOR_GRAY2BGR)
            mask_fruits_rgb[mask_lobules_only == 255] = [255, 255, 255]
            
            self.mask_fruits = base_mask.copy()
            self.mask = cv2.bitwise_not(mask_fruits_rgb[:,:,0])

        if plot:
            ms.plot_img(self.mask, metadata=False, plot_size=plot_size, fig_axis=plot_axis)

        return None

    def find_fruits(self, min_circularity: float = 0.5, output_message: bool = True, 
                    min_locule_area: int = 50, min_locule_per_fruit: int = 1, 
                    max_circularity: float = 1.0, min_aspect_ratio: float = 0.3, 
                    max_aspect_ratio: float = 3.0, contour_filters: Optional[Dict] = None,
                    language_label: List[str] = ['es', 'en'], min_area_label: int = 500,
                    min_canny_label: int = 0, max_canny_label: int = 150, 
                    blur_label: Tuple[int, int] = (11, 11)) -> None:
        """
        Detect fruits and their locules in the mask.
        
        Args:
            min_circularity: Minimum circularity for fruit detection
            output_message: Whether to show detection results
            min_locule_area: Minimum area for locule detection
            min_locule_per_fruit: Minimum locules per fruit
            max_circularity: Maximum circularity for filtering
            min_aspect_ratio: Minimum aspect ratio for filtering
            max_aspect_ratio: Maximum aspect ratio for filtering
            contour_filters: Additional contour filters
            language_label: Languages for label detection
            min_area_label: Minimum area for label detection
            min_canny_label: Minimum Canny threshold for label detection
            max_canny_label: Maximum Canny threshold for label detection
            blur_label: Blur kernel size for label detection
        """
        self.contours, self.fruit_locules_map = morpho.find_fruits(
            self.mask, 
            min_circularity=min_circularity,
            min_locule_area=min_locule_area,
            max_circularity=max_circularity,
            min_locules_per_fruit=min_locule_per_fruit,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            contour_filters=contour_filters
        )
        
        # Detect label text and coordinates
        self.label_text, self.label_coord = ms.detect_label(
            self.img, self.contours, self.fruit_locules_map, 
            language=language_label,
            min_area_label=min_area_label, 
            blur_label=blur_label, 
            min_canny_label=min_canny_label,
            max_canny_label=max_canny_label
        )
        
        # Find label contour ID if exists
        if self.label_coord is not None:
            self.label_id = next((i for i, c in enumerate(self.contours) 
                                if np.array_equal(c, self.label_coord)), None)
        else:
            self.label_id = None
        
        # Calculate number of fruits detected (excluding label if present)
        if self.label_id is not None:
            n_fruits_detected = len(self.fruit_locules_map) - 1
        else:
            n_fruits_detected = len(self.fruit_locules_map)

        if output_message:
            print(f'Total detected objects: {len(self.contours)}')
            print(f'Detected fruits after filtering: {n_fruits_detected}')

        return None

    def detect_metadata(self) -> str:
        """Detect and return image metadata including name."""
        self.img_name = ms.detect_img_name(self.image_path)
        return self.img_name

    def analyze_image(self, plot: bool = True, plot_size: Tuple[int, int] = (10, 10), 
                      font_scale: float = 1.5, font_thickness: int = 2, 
                      plot_title_fontsize: int = 12, use_ellipse: bool = False, 
                      contour_mode: str = 'raw', stamp: bool = False, plot_axis: bool = False, 
                      epsilon_hull: float = 0.001, padding: int = 15, 
                      label_bg_color: Tuple[int, int, int] = (255, 255, 255), 
                      line_spacing: int = 15, min_locule_area: int = 100, 
                      plot_title_pos: str = 'center', scale_size: str = 'letter_ansi',
                      min_distance: int = 0, max_distance: int = 100,
                      max_locule_area: Optional[int] = None, n_shifts: int = 500, 
                      width_cm: Optional[float] = None, length_cm: Optional[float] = None,
                      angle_weight: float = 0.5, radius_weight: float = 0.5,
                      min_radius_threshold: float = 0.1, rel_tol: float = 1e-6) -> AnnotatedImage:
        """
        Analyze detected fruits using morpho.analyze_fruits.
        
        Args:
            plot: Whether to display results
            plot_size: Figure size for plotting
            font_scale: Font scale for annotations
            font_thickness: Font thickness for annotations
            plot_title_fontsize: Title font size
            use_ellipse: Whether to use ellipse fitting
            contour_mode: Contour processing mode
            stamp: Whether image is a stamp
            plot_axis: Whether to show axis
            epsilon_hull: Epsilon value for contour simplification
            padding: Text padding in annotations
            label_bg_color: Background color for text labels
            line_spacing: Line spacing for text annotations
            min_locule_area: Minimum locule area
            plot_title_pos: Title position
            scale_size: Scale size for pixel conversion
            min_distance: Minimum distance for locule merging
            max_distance: Maximum distance for locule merging
            max_locule_area: Maximum locule area
            n_shifts: Number of shifts for symmetry analysis
            width_cm: Manual width in cm
            length_cm: Manual length in cm
            angle_weight: Weight for angle in symmetry calculation
            radius_weight: Weight for radius in symmetry calculation
            min_radius_threshold: Minimum radius threshold
            rel_tol: Relative tolerance for calculations
            
        Returns:
            AnnotatedImage object with results
        """
        # Use already calculated values if available, avoid recalculation
        if not hasattr(self, 'label_text') or self.label_text is None:
            self.label_text, self.label_coord = ms.detect_label(self.img, self.contours, self.fruit_locules_map)
        
        if not hasattr(self, 'img_name') or self.img_name is None:
            self.img_name = ms.detect_img_name(self.image_path)
        
        # Detect scale if not already detected
        if self.px_per_cm_width is None or self.px_per_cm_length is None:
            self.px_per_cm_width, self.px_per_cm_length, self.w_cm, self.h_cm = morpho.px_per_cm(
                self.img, 
                size=scale_size,
                width_cm=width_cm,
                length_cm=length_cm
            )

        # Perform fruit analysis
        self.results = morpho.analyze_fruits(
            img=self.img,
            contours=self.contours,
            fruit_locus_map=self.fruit_locules_map,
            px_per_cm_width=self.px_per_cm_width,
            px_per_cm_length=self.px_per_cm_length,
            img_name=self.img_name,
            label_text=self.label_text,
            use_ellipse=use_ellipse,
            contour_mode=contour_mode,
            plot=plot,
            fig_axis=plot_axis,
            stamp=stamp,
            plot_size=plot_size,
            font_scale=font_scale,
            font_thickness=font_thickness,
            title_fontsize=plot_title_fontsize,
            title_location=plot_title_pos,
            padding=padding,
            line_spacing=line_spacing,
            min_locule_area=min_locule_area,
            max_locule_area=max_locule_area,
            bg_color=label_bg_color,
            epsilon_hull=epsilon_hull,
            min_dist=min_distance,
            max_dist=max_distance,
            path=self.image_path,
            label_id=self.label_id,
            num_shifts=n_shifts,
            angle_weight=angle_weight,
            radius_weight=radius_weight,
            min_radius_threshold=min_radius_threshold,
            rel_tol=rel_tol
        )

        return self.results

    def analyze_folder(self, output_dir: Optional[str] = None, stamp: bool = False, 
                       contour_mode: str = 'raw', n_kernel: int = 7, min_circularity: float = 0.3, 
                       font_scale: float = 1.5, font_thickness: int = 2, padding: int = 15, 
                       line_spacing: int = 15, min_locule_area: int = 300, 
                       label_bg_color: Tuple[int, int, int] = (255, 255, 255),
                       epsilon_hull: float = 0.005, use_ellipse_fruit: bool = False, 
                       min_distance: int = 2, max_distance: int = 30,
                       max_locule_area: Optional[int] = None, n_shifts: int = 500, 
                       width_cm: Optional[float] = None, length_cm: Optional[float] = None, 
                       scale_size: str = 'letter_ansi', angle_weight: float = 0.5, 
                       radius_weight: float = 0.5, min_radius_threshold: float = 0.1, 
                       rel_tol: float = 1e-6, **kwargs) -> None:
        """
        Process all images in a folder with comprehensive error handling.
        
        Args:
            output_dir: Output directory for results
            stamp: Whether images are stamps
            contour_mode: Contour processing mode
            n_kernel: Kernel size for morphological operations
            min_circularity: Minimum circularity for fruit detection
            font_scale: Font scale for annotations
            font_thickness: Font thickness for annotations
            padding: Text padding
            line_spacing: Line spacing for text
            min_locule_area: Minimum locule area
            label_bg_color: Label background color
            epsilon_hull: Epsilon for contour simplification
            use_ellipse_fruit: Whether to use ellipse for fruits
            min_distance: Minimum distance for locule merging
            max_distance: Maximum distance for locule merging
            max_locule_area: Maximum locule area
            n_shifts: Number of shifts for symmetry analysis
            width_cm: Manual width in cm
            length_cm: Manual length in cm
            scale_size: Scale size for pixel conversion
            angle_weight: Weight for angle in symmetry
            radius_weight: Weight for radius in symmetry
            min_radius_threshold: Minimum radius threshold
            rel_tol: Relative tolerance
            **kwargs: Additional parameters for create_mask
        """
        if not self.is_directory:
            raise ValueError("This instance was initialized with a single image. Use analyze_image() instead.")
        
        path_input = self.image_path
        output_dir = os.path.join(path_input, "Results") if output_dir is None else output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize counters
        processed_count = 0
        skipped_no_contours = 0
        skipped_errors = 0
        all_results = []
        errors_report = []
        
        start_time = time.time()
        process = psutil.Process()

        print("Reading folder files ⋆✧｡٩(ˊᗜˋ )و✧*｡   ")

        for filename in tqdm([f for f in os.listdir(path_input) 
                             if os.path.splitext(f)[1].lower() in valid_extensions]):
            try:
                # Process each image with its own instance
                analyzer = AnalyzingImage(os.path.join(path_input, filename))
                
                try:
                    analyzer.read_image(output_message=False)
                    analyzer.create_mask(**kwargs, stamp=stamp, n_kernel=n_kernel)
                    analyzer.find_fruits(min_circularity=min_circularity, output_message=False)
                    
                    if not analyzer.contours:
                        errors_report.append({'filename': filename, 'status': 'No contours found'})
                        skipped_no_contours += 1
                        continue
                    
                    analyzer.detect_metadata()

                    # Final analysis
                    results = analyzer.analyze_image(
                        plot=kwargs.get('plot', False),
                        contour_mode=contour_mode,
                        stamp=kwargs.get('stamp', False),
                        font_scale=font_scale,
                        font_thickness=font_thickness,
                        padding=padding,
                        line_spacing=line_spacing,
                        min_locule_area=min_locule_area,
                        epsilon_hull=epsilon_hull,
                        use_ellipse=use_ellipse_fruit,
                        min_distance=min_distance,
                        max_distance=max_distance,
                        label_bg_color=label_bg_color,
                        max_locule_area=max_locule_area,
                        n_shifts=n_shifts,
                        angle_weight=angle_weight,
                        radius_weight=radius_weight,
                        min_radius_threshold=min_radius_threshold,
                        rel_tol=rel_tol,
                        width_cm=width_cm,
                        length_cm=length_cm,
                        scale_size=scale_size
                    )
                    
                    current_results = results.table if hasattr(results, 'table') else []
                    
                    # Save annotated image
                    annotated_path = os.path.join(output_dir, f"annotated_{filename}")
                    if hasattr(results, 'rgb_image'):
                        img_to_save = results.rgb_image
                        cv2.imwrite(annotated_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
                    
                    if current_results:
                        df = pd.DataFrame(current_results)
                        df['source_file'] = filename
                        all_results.append(df)
                        processed_count += 1
                        
                except Exception as e:
                    print(f"Error processing image {filename}: {str(e)}")
                    errors_report.append({'filename': filename, 'status': f'Processing error: {str(e)}'})
                    skipped_errors += 1
                    
            except Exception as e:
                print(f"Error creating analyzer for {filename}: {str(e)}")
                errors_report.append({'filename': filename, 'status': f'Initialization error: {str(e)}'})
                skipped_errors += 1

        # Save results
        if all_results:
            pd.concat(all_results).to_csv(os.path.join(output_dir, "all_results.csv"), index=False)
        
        if errors_report:
            error_csv_path = os.path.join(output_dir, "error_report.csv")
            pd.DataFrame(errors_report).to_csv(error_csv_path, index=False)
            error_message = f"║ ⚠️ Error report saved in: {error_csv_path}"
        else: 
            error_message = f"║ ⚠️ No errors found"
        
        # Calculate statistics
        end_time = time.time()
        elapsed_min = (end_time - start_time) / 60
        elapsed_seg = (end_time - start_time)
        ram_used_gb = (process.memory_info().rss / 1024**2) / 1024
        total_imgs = skipped_no_contours + processed_count + skipped_errors
        seg_per_img = elapsed_seg / total_imgs if total_imgs > 0 else 0

        # Print summary
        print()
        print(" ( ദ്ദി ˙ᗜ˙ )   Processing completed !")               
        print("══════════════════════════════════════════════════════════════════════")
        print(f" 🕒 Total time:      {elapsed_min:.2f} minutes")
        print(f" 🕒 Average time per image:      {seg_per_img:.2f} seg")
        print(f" 🖼️ Total images processed: {total_imgs}") 
        print(f" ✅ Successfully annotated: {processed_count}") 
        print(f" ⚠️ Skipped (no contours): {skipped_no_contours}") 
        print(f" ❌ Failed (errors): {skipped_errors}") 
        print(f" 💾 RAM used:        {ram_used_gb:.2f} GB")
        print(f" 📁 Output folder:   {output_dir}")
        print(error_message)
    

        return None

    def metadata_summary(self) -> None:
        """Print a summary of detected metadata."""
        print(f"Image: {self.img_name}")
        print(f"Label detected: {self.label_text}")
        print(f"Contours detected: {len(self.contours)}")
        print(f"Fruits detected: {len(self.fruit_locules_map)}")
        if hasattr(self, 'w_cm') and self.w_cm is not None:
            print(f"Image size: {self.w_cm:.2f} cm x {self.h_cm:.2f} cm")

