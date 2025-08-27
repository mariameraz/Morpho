
# User-Facing Classes

--- 

## 1. Classes:

---

##  <code>AnalyzingImage</code>

<h2> Description: </h2>

Main class for **preparing and analyzing images** of fruits (or similar objects).
Provides methods for **reading**, **masking**, **detecting contours/locules**, and **running morphological analysis**.
Can also batch-process entire folders of images.

<h2> Implementation Details: </h2>

**Initialization**

   * Accepts a path to either a single image or a directory of images.
   * Tracks state (original image, masks, contours, label text, calibration).

**PDF conversion**

   * `pdf_to_img()` converts multi-page PDFs into JPEG images.
   * Creates an output folder automatically if none is provided.

**Reading images**

   * `read_image()` loads the image into memory using utility functions.
   * Optionally displays it with customizable figure size.

**Mask creation**

   * `create_mask()` segments fruits from the background.
   * Supports options for inverted stamps, Canny thresholds, HSV thresholds, CLAHE contrast, and locule detection.

**Fruit detection**

   * `find_fruits()` filters contours using circularity, aspect ratio, and area thresholds.
   * Builds a mapping between fruits and their locules.
   * Detects labels within the image for metadata.

**Metadata handling**

   * `detect_metadata()` extracts image name and label text.

**Image analysis**

   * `analyze_image()` computes morphological descriptors and generates annotated results.
   * Returns an `AnnotatedImage` with both annotated image and results table.

**Batch processing**

   * `analyze_folder()` loops over all images in a directory.
   * Applies the full workflow (mask → contours → analysis).
   * Saves annotated images, results (`all_results.csv`), and error reports.

**Summaries**

   * `metadata_summary()` prints quick metadata (labels, contours, fruit count, physical dimensions).



!!! note "**Parameters**"

    **Required**:

    * `image_path` (`str`): Path to an input image file **or** a folder containing multiple images.

    **Optional**:

    - For a detailed description of method parameters, see  <font style="color:red">User-Facing Classes: 2. Methods</font>.

!!! tip "**Returns**"
    
    Instance of `AnalyzingImage`, with multiple methods to run analysis.


**Example**:

```python
# Analyze a single image
analyzer = AnalyzingImage("fruits/sample.jpg")
analyzer.read_image(plot=True)
analyzer.create_mask(stamp=False)
analyzer.find_fruits()
analyzer.detect_metadata()

results = analyzer.analyze_image(plot=True)
results.save_all()

# Analyze all images in a folder
batch = AnalyzingImage("fruits/")
batch.analyze_folder(output_dir="fruits/Results", stamp=True)
```

<br>

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>


## <code>AnnotatedImage</code>

<h2> Description: </h2>

Represents an **analyzed image** along with its **associated results** (tables, metrics).
Provides functionality to **save outputs** in different formats, including annotated images (`.jpg`, `.png`, etc.) and result tables (`.csv`).

This class is typically created as the **output** of the analysis workflow performed by **<code>ImageAnalyzer</code>**.
It acts as a **container** that holds both the **visual annotation** (image with contours/labels) and the **quantitative results** (CSV-ready table).

<h2>Implementation Details: </h2>

**Initialization**

   * Converts the input `cv2_image` (BGR) to RGB.
   * Stores results in both `results` and `table`.
   * Optionally stores the original image path.

**Directory handling**

   * `_ensure_dir_exists()` checks if a directory exists before saving files.
   * Expands `~` and relative paths to absolute paths.

**Saving images**

   * `save_img()` exports the annotated RGB image to disk.
   * Uses `matplotlib` to remove axes and margins, producing a clean figure.
   * Supports DPI and format customization.

**Saving tables**

   * `save_csv()` converts the `table` (list of results) into a Pandas DataFrame.
   * Saves the results as a `.csv` file.

**Combined saving**

   * `save_all()` writes both the annotated image and CSV results using a common base name.
   * Allows flexible control of output directory and file names.

**Relationship with <code>ImageAnalyzer</code>**:

* The class **`ImageAnalyzer`** performs the full workflow (masking, contour detection, morphological analysis).
* When analysis is completed, `ImageAnalyzer.analyze_image()` returns an **`AnnotatedImage`** instance.
* This means **`AnnotatedImage` is the final product of the analysis pipeline**, used for reporting, exporting, and visualization.

!!! note "**Parameters**"

    **Required**:

    * `cv2_image` (`np.ndarray`): Input image in **BGR** format.

    **Optional**:

    * `results` (`list`): Analysis results (default = empty list).
    * `image_path` (`str | None`): Path to the original image file.

!!! tip "**Returns**"

    Instance of `AnnotatedImage` with methods to save outputs.

**Example**:

```python
# Run analysis with ImageAnalyzer
analyzer = ImageAnalyzer("fruits/sample.jpg")
analyzer.read_image()
analyzer.create_mask()
analyzer.find_fruits()

# Perform analysis and retrieve an AnnotatedImage
annotated = analyzer.analyze_image(plot=True)

# Save outputs
annotated.results.save_img() # Save image
annotated.results.save_csv() # Save csv file
annotated.results.save_all() # Save both image and csv file

# Access the result table
annotated.results.table

```

<br>

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>

## 2. Methods:


## <code>read_image</code>

<h2>Description:</h2>

Loads the image from `image_path` into memory and optionally displays it.
Works directly with the file path provided at initialization and stores the result internally in `self.img`.

For algorithmic details, refer to <code>Utilities → ms.load\_img</code>.

!!! note "**Parameters**"

    **Optional:**

    * `plot` (`bool`, default `False`) — Whether to display the loaded image.
    * `output_message` (`bool`, default `True`) — Print a confirmation message after loading.
    * `plot_size` (`tuple[int,int]`, default `(5,5)`) — Size of the display figure.
    * `plot_axis` (`bool`, default `False`) — Whether to show axes in the display.

!!! tip "**Returns**"

    * `None` — does not return a value. Instead, stores the loaded image in:

        * `self.img` → original image in **BGR format** (as loaded with OpenCV).

**Example**:

```python
analyzer = AnalyzingImage("fruits/sample.jpg")
analyzer.read_image(plot=True, plot_size=(6,6))

# Access the loaded image directly
img = analyzer.img
print(img.shape)   # (H, W, 3) -> BGR image
```


!!! danger "**Notes**"

    * Internally calls <code>ms.load\_img</code> for image reading and optional plotting.
    * The image is preserved in memory for use in subsequent steps (`create_mask`, `find_fruits`, etc.).




<br>

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>


## <code>create_mask</code>

<h2> Description </h2>

Segments fruits from the background. Wrapper around the core masking routine that adds state storage (class attributes), stamp handling, and optional plotting.
Works directly with the image and metadata managed by <code>ImageAnalyzer</code> and is built on top of <code>core.create\_mask</code>. 

For algorithmic details, see <code>Internal/Core Utilities → core.create\_mask</code>.

!!! note "**Parameters**"

    **Optional:**

    **Morphology**:

      * `n_kernel` (`int`, default `5`) — size of the structuring element used in morphological operations.
      * `n_iteration` (`int`, default `1`) — number of times morphological operations are applied.

    **Edges & Color**:

      * `canny_min` (`int`, default `30`) — minimum threshold for Canny edge detection.
      * `canny_max` (`int`, default `100`) — maximum threshold for Canny edge detection.
      * `lower_hsv` (`list[int] | None`, default `None`) — lower bound for HSV color filtering (e.g., background removal).
      * `upper_hsv` (`list[int] | None`, default `None`) — upper bound for HSV color filtering.

    **Stamps & Locules**:

      * `stamp` (`bool`, default `False`) — if `True`, the input image is inverted before processing (for stamp-like images).
      * `locules_filled` (`bool`, default `False`) — if `True`, applies extra CLAHE + Otsu thresholding to detect internal locules (even when not hollow).
      * `min_locule_size` (`int`, default `300`) — minimum pixel area required to consider a locule valid.
      * `n_blur` (`int`, default `11`) — kernel size for median blur used to smooth the locule mask.
      * `clip_limit` (`int`, default `4`) — contrast limit for CLAHE enhancement.
      * `tile_grid_size` (`int`, default `8`) — CLAHE grid size (local histogram equalization regions).

      **Visualization**:

      * `plot` (`bool`, default `False`) — whether to display the generated mask.
      * `plot_size` (`tuple[int,int]`, default `(5,5)`) — size of the plot figure.
      * `plot_axis` (`bool`, default `False`) — if `True`, shows axis ticks/labels in the plot.

!!! tip "**Returns**:"

    * `None` — does not return a value. Instead, stores results internally:

        * `self.mask` → main binary mask (always set).
        * `self.mask_fruits` → set when `locules_filled=True`.
        * `self.img_inverted` → set when `stamp=True`.

**Example**:

```python
analyzer = AnalyzingImage("fruits/sample.jpg")
analyzer.read_image()
analyzer.create_mask(stamp=True, locules_filled=True, plot=True)

# Access the generated mask directly
mask = analyzer.mask
print(mask.shape)   # (H, W)
```


!!! danger "**Notes**:"

    * When `locules_filled=True`, the method enhances the L channel (LAB space) using CLAHE, then applies Otsu thresholding and morphology to detect internal regions.
    * This allows segmentation of **internal locules** that may not appear as hollow cavities (i.e., filled or partially filled locules).


<br>

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>

## <code>find_fruits</code>

<h2>Description:</h2>

Detects fruits and locules from the mask, applying geometric filters and label detection.
Stores contours and fruit–locule mappings as class attributes.

For algorithmic details, refer to <code>Internal/Core Utilities → core.find\_fruits</code>.

!!! note "**Parameters**""

    **Optional**

    **Shape filters**:

      * `min_circularity` (`float`, default `0.5`) — minimum circularity to accept a fruit contour.
      * `max_circularity` (`float`, default `1.0`) — maximum circularity to accept a fruit contour.
      * `min_aspect_ratio` (`float`, default `0.3`) — minimum aspect ratio (width/height) for valid contours.
      * `max_aspect_ratio` (`float`, default `3.0`) — maximum aspect ratio (width/height) for valid contours.
      * `contour_filters` (`dict | None`) — custom thresholds to override defaults.

    **Locules**:

      * `min_locule_area` (`int`, default `50`) — minimum area (px²) for a locule to be considered valid.
      * `min_locule_per_fruit` (`int`, default `1`) — minimum number of locules required to keep a fruit.

    **Labels**:

      * `language_label` (`list[str]`, default `['es','en']`) — languages to use in OCR for label detection.
      * `min_area_label` (`int`, default `500`) — minimum contour area to classify as label region.
      * `min_canny_label` (`int`, default `0`), `max_canny_label` (`int`, default `150`) — Canny thresholds for label edge detection.
      * `blur_label` (`tuple[int,int]`, default `(11,11)`) — blur kernel applied before label detection.

    **Misc**:

      * `output_message` (`bool`, default `True`) — whether to print detection summary.

!!! tip "**Returns**:"

    * `None` — does not return a value. Instead, stores results internally:

        * `self.contours` → list of detected contours.
        * `self.fruit_locules_map` → mapping fruit → locule IDs.
        * `self.label_text`, `self.label_coord`, `self.label_id` (if a label is detected).

**Example**:

```python
analyzer = AnalyzingImage("fruits/sample.jpg")
analyzer.read_image()
analyzer.create_mask()
analyzer.find_fruits()

# Access contours and fruit-locule mapping
print(len(analyzer.contours))         
print(analyzer.fruit_locules_map.keys())
```

!!! danger "**Notes**:"

    * Internally wraps <code>core.find\_fruits</code> and adds label detection (via <code>ms.detect\_label</code>) plus state storage.


<br>

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>

## <code>analyze_image</code>

<h2>Description:</h2>

Runs the full morphological analysis on detected fruits and locules.
Handles label detection (if not done), calibration of px/cm, and contour analysis.

Returns an <code>AnnotatedImage</code> with annotated RGB image and results table.

For algorithmic details, refer to <code>Internal/Core Utilities → core.px\_per\_cm</code> and <code>core.analyze\_fruits</code>.

!!! note "**Parameters**"

    **Optional:**

    **Visualization**:

      * `plot` (`bool`, default `True`) — display results with annotations.
      * `plot_size` (`tuple[int,int]`, default `(10,10)`) — figure size for display.
      * `plot_axis` (`bool`, default `False`) — show axis in plots.
      * `font_scale` (`float`, default `1.5`) — font size for text annotations.
      * `font_thickness` (`int`, default `2`) — line thickness for text.
      * `plot_title_fontsize` (`int`, default `12`) — font size for title.
      * `label_bg_color` (`tuple[int,int,int]`, default `(255,255,255)`) — background color for label boxes.
      * `line_spacing` (`int`, default `15`) — spacing between annotation lines.
      * `plot_title_pos` (`str`, default `'center'`) — title alignment.

    **Contours & Shape**:

      * `use_ellipse` (`bool`, default `False`) — fit ellipses to contours instead of raw polygon.
      * `contour_mode` (`str`, default `'raw'`) — contour approximation mode.
      * `epsilon_hull` (`float`, default `0.001`) — tolerance for convex hull simplification.

    **Locules**:

      * `min_locule_area` (`int`, default `100`) — minimum area (px²) to consider a locule.
      * `max_locule_area` (`int | None`) — maximum allowed locule area.
      * `min_distance` (`int`, default `0`) — minimum distance for merging locules.
      * `max_distance` (`int`, default `100`) — maximum distance for merging locules.

    **Symmetry**:

      * `n_shifts` (`int`, default `500`) — number of shifts for symmetry analysis.
      * `angle_weight` (`float`, default `0.5`) — weight for angle symmetry.
      * `radius_weight` (`float`, default `0.5`) — weight for radius symmetry.
      * `min_radius_threshold` (`float`, default `0.1`) — minimum radius threshold.
      * `rel_tol` (`float`, default `1e-6`) — relative tolerance in calculations.

    **Calibration**:

      * `width_cm` (`float | None`) — manual physical width in cm.
      * `length_cm` (`float | None`) — manual physical length in cm.
      * `scale_size` (`str`, default `'letter_ansi'`) — reference scale for px/cm calibration.


!!! tip "**Returns**"

    * `AnnotatedImage` — containing:

        * `rgb_image` → annotated image (RGB).
        * `table` → list of results (metrics per fruit/locule).


**Example**:

```python
analyzer = AnalyzingImage("fruits/sample.jpg")
analyzer.read_image()
analyzer.create_mask()
analyzer.find_fruits()

# Perform final analysis
annotated = analyzer.analyze_image(plot=True)

# Access image and results
print(type(annotated))            # AnnotatedImage
print(len(annotated.results.table))       # number of rows in results
```

!!! danger "**Notes**"

    * Internally calls <code>core.px\_per\_cm</code> for calibration and <code>core.analyze\_fruits</code> for analysis.
    * Also uses <code>ms.detect\_label</code> if label metadata was not detected before.
  

<br>

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>

## <code>analyze_folder</code>

<h2>Description</h2>

Batch wrapper that processes all valid images in a directory.

Applies the full pipeline (read → mask → find\_fruits → analyze\_image) to each file, saving annotated images, results, and error reports.

For algorithmic details, see <code>Internal/Core Utilities → core.analyze\_fruits</code> (used internally via <code>analyze\_image</code>).

!!! note "**Parameters**"
    
    **Optional**
    
    **General**:

      * `output_dir` (`str | None`, default `None`) — folder for saving results (defaults to `<input>/Results`).
      * `stamp` (`bool`, default `False`) — whether input images are inverted stamps.
      * `contour_mode` (`str`, default `'raw'`) — contour approximation mode.
      * `n_kernel` (`int`, default `7`) — kernel size for morphology in mask creation.
      * `min_circularity` (`float`, default `0.3`) — minimum circularity filter for fruits.

    **Visualization**:

      * `font_scale` (`float`, default `1.5`) — font size for text annotations.
      * `font_thickness` (`int`, default `2`) — thickness of annotation text.
      * `padding` (`int`, default `15`) — padding around text.
      * `line_spacing` (`int`, default `15`) — spacing between annotation lines.
      * `label_bg_color` (`tuple[int,int,int]`, default `(255,255,255)`) — background color for labels.

    **Locules**:

      * `min_locule_area` (`int`, default `300`) — minimum area (px²) for locule acceptance.
      * `max_locule_area` (`int | None`) — maximum locule area.
      * `min_distance` (`int`, default `2`) — minimum distance for merging locules.
      * `max_distance` (`int`, default `30`) — maximum distance for merging locules.

    **Shape analysis**:

      * `epsilon_hull` (`float`, default `0.005`) — tolerance for convex hull simplification.
      * `use_ellipse_fruit` (`bool`, default `False`) — whether to fit ellipses to fruits.
      * `n_shifts` (`int`, default `500`) — number of shifts for symmetry analysis.
      * `angle_weight` (`float`, default `0.5`) — weight for angle symmetry.
      * `radius_weight` (`float`, default `0.5`) — weight for radius symmetry.
      * `min_radius_threshold` (`float`, default `0.1`) — minimum radius threshold.
      * `rel_tol` (`float`, default `1e-6`) — relative tolerance in symmetry.

    **Calibration**:

      * `width_cm` (`float | None`) — manual width in cm.
      * `length_cm` (`float | None`) — manual length in cm.
      * `scale_size` (`str`, default `'letter_ansi'`) — reference scale for px/cm calibration.

    **Extra**:

      * `**kwargs` — forwarded to <code>create\_mask()</code> (e.g., HSV thresholds, Canny parameters).

!!! tip "**Returns**"

    * `None` — does not return a value. Instead, writes results to disk:

        * `annotated_<filename>` → annotated images.
        * `all_results.csv` → concatenated table of all analyzed images.
        * `error_report.csv` → errors and skipped images.

**Example**:

```python
batch = AnalyzingImage("fruits/")
batch.analyze_folder(output_dir="fruits/Results", stamp=True)

# Outputs are saved directly in the Results folder
# including annotated images, all_results.csv, and error_report.csv
```

!!! danger "**Notes**"

    * Internally loops over images in the input folder and applies <code>analyze\_image</code> to each one.
    * Uses <code>psutil</code> for runtime and memory reporting.



<br>

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>

## <code>pdf_to_img</code>

<h2>Description</h2>

Converts a multi-page PDF into a sequence of JPEG images (one per page).

Wrapper utility embedded in <code>ImageAnalyzer</code>, built on top of <code>pdf2image.convert\_from\_path</code>.

If `output_dir` is not provided, a new folder called <code>images\_from\_pdf</code> is automatically created next to the PDF file.

Each page is saved as `<pdf_name>_pageN.jpg`.

!!! note "**Parameters**"

    **Required** 

    * `pdf_path` (`str`) — path to the input PDF file. Must have `.pdf` extension.

    **Optional**

    * `dpi` (`int`, default `300`) — resolution in dots per inch for conversion.
    * `output_dir` (`str | None`, default `None`) — destination folder. If `None`, creates `<pdf_dir>/images_from_pdf`.
    * `n_threads` (`int | None`, default `None`) — number of CPU threads for conversion. If `None`, defaults to `1`.
    * `output_message` (`bool`, default `True`) — whether to print progress and summary messages.

!!! tip "**Returns**"

    * `list[str]` — list of paths to the saved JPEG images.

**Example**:

```python
# Convert a PDF to images with default settings
images = AnalyzingImage.pdf_to_img("docs/report.pdf")

# Example output:
# [
#   "docs/images_from_pdf/report_page1.jpg",
#   "docs/images_from_pdf/report_page2.jpg",
#   ...
# ]

# Save into a custom directory, at higher resolution
images = AnalyzingImage.pdf_to_img(
    pdf_path="docs/report.pdf",
    dpi=600,
    output_dir="exports/pdf_images",
    n_threads=4
)
```

!!! danger "**Notes**"

    * Input validation ensures the file exists and has `.pdf` extension.
    * If no `output_dir` is given, a new folder `images_from_pdf` is created automatically.
    * Filenames follow the pattern `<pdf_name>_page{i+1}.jpg`.
    

<br>

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>

## <code>save_img</code>

<h2>Description:</h2>

Embedded in <code>AnnotatedImage</code>. Saves the annotated RGB image (produced and stored by <code>ImageAnalyzer</code>) to disk without axes or margins.

If the output directory does not exist, it is created automatically.

!!! note "**Parameters**"

    **Optional**

    * `path` (`str | None`, default `None`) — output file path. If `None`, uses `<original_dir>/<image_name>_annotated.<ext>`.
    * `format` (`str | None`, default `None`) — image format (e.g., `'jpg'`, `'png'`). If `None`, inferred from the file extension in `path`, or defaults to `'jpg'`.
    * `dpi` (`int`, default `300`) — resolution in dots per inch for raster image export.
    * `output_message` (`bool`, default `True`) — whether to print a confirmation message after saving.
    * `**kwargs` — additional keyword arguments passed to `matplotlib.pyplot.savefig` (e.g., `transparent=True`).

!!! tip "**Returns**"

    * `None` — writes the file to disk.

**Example**:

```python
annotated = analyzer.analyze_image(plot=True)

# Save annotated image (auto-creates directory if missing)
annotated.save_img(format="png", dpi=200)

# Access default path
print(annotated.image_path)  # original image path reference
```


<br>

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>

## <code>save_csv</code>

<h2>Description:</h2>

Embedded in <code>AnnotatedImage</code>. Saves the results table (computed by <code>ImageAnalyzer</code> and stored in the object) to a CSV file.

If the output directory does not exist, it is created automatically.

!!! note "**Parameters**"

    * `path` (`str | None`, default `None`) — output CSV path. If `None`, uses `<original_dir>/<image_name>_results.csv`.
    * `sep` (`str`, default `','`) — column separator to use in the CSV file.
    * `output_message` (`bool`, default `True`) — whether to print a confirmation message after saving.

!!! tip "**Returns**"

    * `None` — writes the file to disk.

!!! warning "**Raises**"

    * Raises `ValueError` if `self.table` is empty (i.e., no results to save).

**Example**:

```python
annotated = analyzer.analyze_image(plot=False)

# Save results table (auto-creates directory if missing)
annotated.save_csv(sep=";")

# The CSV will contain the per-fruit and per-locule measurements
```


<br>

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>

## <code>save_all</code>


<h2>Description:</h2>

Embedded in <code>AnnotatedImage</code>. Convenience method that saves both the annotated image and the results table (produced by <code>ImageAnalyzer</code>) in one call.

If the output directory does not exist, it is created automatically.

!!! note "**Parameters**"

    **Optional**

    * `base_name` (`str | None`, default `None`) — base name for the files, without extension. If `None`, uses the original image name.
    * `output_dir` (`str | None`, default `None`) — directory where the files will be saved. If `None`, uses the original image directory.
    * `format` (`str`, default `'jpg'`) — image format for export (e.g., `'jpg'`, `'png'`).
    * `dpi` (`int`, default `300`) — resolution in dots per inch for raster export.
    * `sep` (`str`, default `','`) — column separator for the CSV file.
    * `output_message` (`bool`, default `True`) — whether to print confirmation messages after saving.

!!! tip "**Returns**"

    * `None` — writes both files (annotated image + CSV) to disk.

**Example**:

```python
annotated = analyzer.analyze_image(plot=True)

# Save both outputs in a custom directory (auto-created if missing)
annotated.save_all(output_dir="results/fruits", format="png")

# This creates:
# results/fruits/sample_annotated.png
# results/fruits/sample_results.csv
```
