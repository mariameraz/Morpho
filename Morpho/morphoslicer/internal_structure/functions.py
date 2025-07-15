import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List
import numpy as np
import morphoslicer.internal_structure.core as morpho
from ..utils import common_functions as ms
from tqdm import tqdm
import psutil
from .. import valid_extensions
import time
from pdf2image import convert_from_path
from typing import List, Optional
### Save table results and annotated image

class AnnotatedImage:
    def __init__(self, cv2_image: np.ndarray, results: list = None, image_path: Optional[str] = None):
        self.rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        self.table = results if results else []
        self.image_path = image_path  

    def _ensure_dir_exists(self, path: str) -> str:
        """Garantiza que el directorio exista y devuelve la ruta absoluta"""
        if not os.path.dirname(path):
            path = os.path.join(os.getcwd(), path)
        
        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        return path
    
    @staticmethod
    def pdf_to_img(self, path_pdf, dpi = 300, path_img = None, n_threads = None, output_message = True):
        pdf_basename = os.path.basename(path_pdf)
        pdf_name, ext = os.path.splitext(pdf_basename)
        valid_ext = '.pdf'

        if ext.lower() not in valid_ext:
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
                    path_res = os.path.join(dirname, 'images_from_pdf')
                    os.makedirs(path_res, exist_ok = True)
                    path_img = path_res # <- Update the new dir path
                
                for i, image in enumerate(images):
                    img_name = f'{pdf_name}_page{i+1}.jpg'
                    output_path = os.path.join(path_img, img_name)
                    image.save(output_path, 'JPEG')
                
                if output_message: 
                    print(f'{len(images)} images saved in: {path_res}')
            except Exception as e:
                print(f'An expected error ocurred: {str(e)}')

    def save_img(self, path: Optional[str] = None, format: Optional[str] = None, dpi: int = 300,  output_message: bool = True, **kwargs):
        """Guarda la imagen en el mismo directorio que la imagen original"""
        try:
            if path is None:
                if not hasattr(self, 'image_path') or not self.image_path:
                    raise ValueError("No se proporcionó ruta y no hay imagen original de referencia")
                
                original_dir = os.path.dirname(self.image_path)
                base_name = os.path.splitext(os.path.basename(self.image_path))[0]
                ext = format.lower() if format else 'jpg'
                path = os.path.join(original_dir, f"{base_name}_annotated.{ext}")

            full_path = self._ensure_dir_exists(path)
            format = format or os.path.splitext(full_path)[1][1:].lower()
            
            # Figura temporal para guardado
            output_image = self.rgb_image.copy()  # <-- Mantener referencia
            temp_fig = plt.figure()
            plt.imshow(output_image)
            plt.axis('off')
            plt.savefig(full_path, format=format, dpi=dpi, bbox_inches='tight', pad_inches=0, **kwargs)
            plt.close(temp_fig)
            
            if output_message:
                print(f"Imagen guardada en: {full_path}")
                
        except Exception as e:
            if 'temp_fig' in locals():
                plt.close(temp_fig)
            raise RuntimeError(f"Error al guardar imagen: {str(e)}")

    def save_csv(self, path: Optional[str] = None, sep: str = ',', output_message: bool = True):
        """Guarda el CSV en el mismo directorio que la imagen original si no se especifica `path`."""
        if not self.table:
            raise ValueError("No hay datos en 'results' para guardar")
        
        try:
            if path is None:
                if not hasattr(self, 'image_path') or not self.image_path:
                    raise ValueError("No se proporcionó ruta y no hay imagen original de referencia")
                original_dir = os.path.dirname(self.image_path)
                csv_name = os.path.splitext(os.path.basename(self.image_path))[0] + "_results.csv"
                path = os.path.join(original_dir, csv_name)
            
            full_path = self._ensure_dir_exists(path)
            pd.DataFrame(self.table).to_csv(full_path, sep=sep, index=False)
            
            if output_message:
                print(f"CSV guardado en: {full_path}")
        except Exception as e:
            raise RuntimeError(f"Error al guardar CSV: {str(e)}")

    def save_all(self, base_name: Optional[str] = None, output_dir: Optional[str] = None, 
                format: str = 'jpg', dpi: int = 300, sep: str = ',', 
                output_message: bool = True):
        """
        Guarda ambos archivos usando el nombre base.
        """
        try:
            # Construir rutas completas
            if base_name is None or self.image_path is None:
                base_name = os.path.splitext(os.path.basename(self.image_path))[0]

            if output_dir:
                output_dir = os.path.abspath(output_dir)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                img_path = os.path.join(output_dir, f"{base_name}.{format.lower()}")
                csv_path = os.path.join(output_dir, f"{base_name}_results.csv")
            else:

                if not hasattr(self, 'image_path') or not self.image_path:
                    raise ValueError("No se proporcionó ruta y no hay imagen original de referencia")
                
                output_dir = os.path.dirname(self.image_path)
                img_path = os.path.join(output_dir, f"{base_name}_annotated.{format.lower()}")
                csv_path = os.path.join(output_dir, f"{base_name}_results.csv")
            
            # Guardar archivos propagando el parámetro output_message
            self.save_img(img_path, dpi=dpi, output_message=output_message)
            self.save_csv(csv_path, sep=sep, output_message=output_message)
            
        except Exception as e:
            raise RuntimeError(f"Error en save_all: {str(e)}")
        


##########

# Analyze a single image
class AnalyzingImage:
    def __init__(self, image_path):
        self.image_path = image_path
        self.is_directory = os.path.isdir(image_path)

        self.img = None
        self.img_inverted = None
        self.mask = None
        self.contours = None
        self.fruit_locus_map = None
        self.label_text = None
        self.img_name = None
        self.px_per_cm_x = None
        self.px_per_cm_y = None
        self.w_cm = None
        self.h_cm = None
        self.results = None
        

    @staticmethod
    def pdf_to_img(pdf_path: str, 
                dpi: int = 300, 
                output_dir: Optional[str] = None, 
                n_threads: Optional[int] = None,
                output_message: bool = True) -> List[str]:
        """
        Converts a PDF file to JPEG images (one per page), and renames them to a simple format (e.g., pdf_page1.jpg).

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
            print("╔══════════════════════════════════════╗")
            print("║  Extracting images may take a few minutes... ⋆✧｡٩(ˊᗜˋ )و✧*｡      ║")
            print("╚══════════════════════════════════════╝")

            # Convert PDF to images
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                thread_count=n_threads or 1,
                output_folder=output_dir,
                fmt='jpeg',
                output_file=f'{pdf_name}_page',
                paths_only=True
            )

            # Rename output files to clean format: pdf_page1.jpg, pdf_page2.jpg, ...
            renamed_images = []
            for i, old_path in enumerate(sorted(images), 1):
                new_name = f"{pdf_name}_page{i}.jpg"
                new_path = os.path.join(output_dir, new_name)
                os.rename(old_path, new_path)
                renamed_images.append(new_path)

            if output_message:
                print(f"Converted {len(renamed_images)} pages to images.")
                print(f"Saved in: {output_dir}")

            return None

        except Exception as e:
            error_msg = f"PDF conversion error: {str(e)}"
            if output_message:
                print(error_msg)
            raise RuntimeError(error_msg) from e


    def read_image(self, plot=False, output_message= True, plot_size=(5,5), plot_axis = False, plot_title = None, plot_title_pos = 'center', plot_title_fontsize = '12'):
        self.img = ms.load_image(self.image_path, plot=plot, figsize=plot_size, axis = plot_axis, title = plot_title, title_location = plot_title_pos,
                                 title_fontsize = plot_title_fontsize)
        if self.img is None:
            raise ValueError(f"No se pudo cargar la imagen: {self.image_path}")

        if output_message: 
            img_name = os.path.basename(self.image_path)
            output_message = f'{img_name} successfully loaded ♡'
            print(output_message)

        return None

    def create_mask(self, n_kernel=5, plot=False, plot_size=(5,5), stamp=False, plot_axis = False, n_iteration = 1, canny_min = 30, canny_max = 100, lower_black = None,
                    upper_black = None):
        """Crea máscara con opción de invertir colores para estampas"""
        if stamp:
            self.img_inverted = cv2.bitwise_not(self.img)
        else:
            self.img_inverted = self.img.copy()  # Usamos copia para no modificar la original
        
        self.mask = morpho.create_mask(self.img_inverted,
                                       n_kernel=n_kernel, 
                                       n_iteration = n_iteration,
                                       plot=plot, 
                                       figsize=plot_size,
                                       axis = plot_axis,
                                       canny_max = canny_max,
                                       canny_min = canny_min,
                                       lower_black = lower_black,
                                       upper_black = upper_black)
        return None

    def find_fruits(self, min_circularity=0.5, output_message = True, min_locule_area = 50, min_locule_per_fruit = 1, max_circularity = 1,
                    min_aspect_ratio = 0.3, max_aspect_ratio = 3.0, contour_filters = None):
        self.contours, self.fruit_locus_map = morpho.find_fruits(self.mask, 
                                                                 min_circularity=min_circularity,
                                                                 min_loculus_area = min_locule_area,
                                                                 max_circularity = max_circularity,
                                                                 min_loculi_per_fruit = min_locule_per_fruit,
                                                                 min_aspect_ratio = min_aspect_ratio,
                                                                 max_aspect_ratio = max_aspect_ratio,
                                                                 contour_filters = contour_filters
                                                                 )

        if output_message == True:
            print(f'Total detected objects: {len(self.contours)}')
            print(f'Detected fruits after filtering:  {len(self.fruit_locus_map)} ')

        return None

    def detect_metadata(self):
        self.label_text = ms.detect_label(self.img, self.contours, self.fruit_locus_map)
        self.img_name = ms.detect_img_name(self.image_path)
        return self.label_text, self.img_name

    def get_scale(self, scale_size='letter_ansi'):
        self.px_per_cm_x, self.px_per_cm_y, self.w_cm, self.h_cm = morpho.pixels_per_cm(self.img, size=scale_size)
        return self.px_per_cm_x, self.px_per_cm_y

    def analyze_image(self, plot=True, 
                    plot_size=(10,10), font_scale=1.5, 
                    font_thickness=2, plot_title_fontsize=12, 
                    use_ellipse_loculi=False, contour_mode='raw', 
                    stamp=False, plot_axis=False, epsilon_hull=0.005,
                    padding=15, label_bg_color=(255,255,255), line_spacing=15, 
                    min_locule_area=300, plot_title_pos='center', scale_size='letter_ansi',
                    use_ellipse_fruit=False, min_distance=2, max_distance=30): 
        
        # Get image metadata
        self.label_text = ms.detect_label(self.img, self.contours, self.fruit_locus_map)
        self.img_name = ms.detect_img_name(self.image_path)
        self.px_per_cm_x, self.px_per_cm_y, self.w_cm, self.h_cm = morpho.pixels_per_cm(self.img, 
                                                                                        size = scale_size,)
    

        self.results = morpho.analyze_fruits(
            img=self.img,  # Siempre usa la imagen original
            contours=self.contours,
            fruit_locus_map=self.fruit_locus_map,
            px_per_cm_x=self.px_per_cm_x,
            px_per_cm_y=self.px_per_cm_y,
            img_name=self.img_name,
            label_text=self.label_text,
            use_ellipse_loculi=use_ellipse_loculi,
            contour_mode=contour_mode,
            plot=plot,
            fig_axis = plot_axis,
            stamp = stamp,
            figsize=plot_size,
            font_scale=font_scale,
            font_thickness=font_thickness,
            fig_title_fontsize=plot_title_fontsize,
            fig_title_loc = plot_title_pos,
            padding = padding, 
            line_spacing = line_spacing, 
            min_locule_area = min_locule_area,
            bg_color = label_bg_color,
            epsilon_hull = epsilon_hull,
            use_ellipse_fruit = use_ellipse_fruit,
            min_dist = min_distance, 
            max_dist = max_distance,
            path = self.image_path

        )
        
        


        return self.results

    def analyze_folder(self, output_dir=None, stamp=False, contour_mode='raw', 
                      n_kernel=7, min_circularity=0.3, font_scale = 1.5, font_thickness = 2, 
                      use_ellipse_loculi = False, padding = 15, line_spacing = 15, 
                       min_locule_area = 300, label_bg_color = (255,255,255),
                        epsilon_hull = 0.005, use_ellipse_fruit = False, 
                          min_distance = 2, max_distance = 30, **kwargs):
        """
        Process all images in the directory (instance must be initialized with directory path)
        """
        if not self.is_directory:
            raise ValueError("This instance was initialized with a single image. Use analyze_image() instead.")
        
        path_input = self.image_path
        output_dir = os.path.join(path_input, "Results") if output_dir is None else output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        
        
        # Inicialización de contadores
        processed_count = 0
        skipped_no_contours = 0
        skipped_errors = 0
        all_results = []
        errors_report = []
        error_csv_path = None
        
        start_time = time.time() # Get start time information
        process = psutil.Process() # Get process resource information

        print("╔════════════════════════╗")
        print("║    Reading folder files ⋆✧｡٩(ˊᗜˋ )و✧*｡    ║")
        print("╚════════════════════════╝")

        for filename in tqdm([f for f in os.listdir(path_input) 
                                    if os.path.splitext(f)[1].lower() in valid_extensions]):
            try:
                        # Cada imagen se procesa con su propia instancia
                analyzer = AnalyzingImage(os.path.join(path_input, filename))
                analyzer.read_image(output_message=False)
                
                # Procesamiento con manejo de stamps
                analyzer.create_mask(**kwargs, stamp = stamp, n_kernel = n_kernel)
                analyzer.find_fruits(min_circularity=min_circularity, output_message = False)
                
                if not analyzer.contours:
                    errors_report.append({'filename': filename, 'status': 'No contours found'})
                    skipped_no_contours += 1
                    continue
                
                analyzer.detect_metadata()
                analyzer.get_scale()
                
                # Análisis final (siempre usa imagen original)
                results = analyzer.analyze_image(
                    plot=kwargs.get('plot', False),
                    use_ellipse_loculi=kwargs.get('ellipse_loculi', False),
                    contour_mode= contour_mode,
                    stamp = kwargs.get('stamp', False),
                    font_scale=font_scale,
                    font_thickness=font_thickness,
                    padding = padding, 
                    line_spacing = line_spacing, 
                    min_locule_area = min_locule_area,
                    epsilon_hull = epsilon_hull,
                    use_ellipse_fruit = use_ellipse_fruit,
                    min_distance = min_distance, 
                    max_distance = max_distance, 
                    label_bg_color = label_bg_color
                    

                )
                
                # Manejo de resultados
                current_results = results.table if hasattr(results, 'table') else []
                
                # Guardar imagen anotada
                annotated_path = os.path.join(output_dir, f"annotated_{filename}")
                if hasattr(results, 'rgb_image'):
                    img_to_save = results.rgb_image
                    cv2.imwrite(annotated_path, cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR))
                
                # Procesar resultados
                if current_results:
                    df = pd.DataFrame(current_results)
                    df['source_file'] = filename
                    all_results.append(df)
                    processed_count += 1
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                errors_report.append({'filename': filename, 'status': f'Error: {str(e)}'})
                skipped_errors += 1

        # Guardar resultados y mostrar estadísticas (igual que antes)
        if all_results:
            pd.concat(all_results).to_csv(os.path.join(output_dir, "all_results.csv"), index=False)
        if errors_report:
            error_csv_path = os.path.join(output_dir, "error_report.csv")
            pd.DataFrame(errors_report).to_csv(error_csv_path, index=False)
            error_message = f"║ ⚠️ Error report saved in: {error_csv_path}"
        else: 
            error_message = f"║ ⚠️ No errors found"
        
        end_time = time.time()
        elapsed_min = (end_time - start_time) / 60
        elapsed_seg = (end_time - start_time)
        ram_used_gb = (process.memory_info().rss / 1024**2) / 1024
        total_imgs = skipped_no_contours + processed_count + skipped_errors
        seg_per_img = elapsed_seg / total_imgs if total_imgs > 0 else 0

        final_res = pd.concat(all_results)

        # Mostrar estadísticas finales
        print()
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║   ( ദ്ദി ˙ᗜ˙ )   Processing completed !")               
        print("╠══════════════════════════════════════════════════════════════════════╣")
        print(f"║ 🕒 Total time:      {elapsed_min:.2f} minutes")
        print(f"║ 🕒 Average time per image:      {seg_per_img:.2f} seg")
        print(f"║ 🖼️ Total images processed: {total_imgs}") 
        print(f"║ ✅ Successfully annotated: {processed_count}") 
        print(f"║ ⚠️ Skipped (no contours): {skipped_no_contours}") 
        print(f"║ ❌ Failed (errors): {skipped_errors}") 
        print(f"║ 💾 RAM used:        {ram_used_gb:.2f} GB")
        print(f"║ 📁 Output folder:   {output_dir}")
        print(error_message)
        print("╚══════════════════════════════════════════════════════════════════════╝")

        return None   

    def metadata_summary(self):
        print(f"Image: {self.img_name}")
        print(f"Label detected: {self.label_text}")
        print(f"Contours detected: {len(self.contours)}")
        print(f"Fruits detected: {len(self.fruit_locus_map)}")
        print(f"Image size: {self.w_cm:.2f} cm x {self.h_cm:.2f} cm")