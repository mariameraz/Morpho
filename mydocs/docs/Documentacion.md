
### 1.2 Traits Analyzed {#traits-analyzed}

Table 1. Internal structure traits obtained by Traitly.

| Column                      | Trait description (type & range)                                       | Function                                       |
| --------------------------- | ---------------------------------------------------------------------- | ---------------------------------------------- |
| `image_name`                | Input image name (string)                                              | — (metadata)                                   |
| `label`                     | User-defined label (string)                                             | — (metadata)                                   |
| `fruit_id`                  | Sequential fruit identifier (int ≥1)                                    | `analyze_fruits`                               |
| `n_locules`                 | Number of detected locules (int ≥0)                                    | `analyze_fruits`                               |
| `major_axis_cm`             | Longest distance across fruit (float, cm >0)                            | `calculate_axes`                               |
| `minor_axis_cm`             | Maximum thickness perpendicular to major axis (float, cm >0)            | `calculate_axes`                               |
| `fruit_area_cm2`            | Total fruit area (float, cm² >0)                                        | `get_fruit_morphology`                         |
| `fruit_perimeter_cm`        | Fruit perimeter length (float, cm >0)                                   | `get_fruit_morphology`                         |
| `fruit_circularity`         | Roundness index (float, 0–1; 1 = circle)                                | `get_fruit_morphology`                         |
| `fruit_aspect_ratio`        | Width ÷ length ratio (float, 0–1; 1 = circle, <1 = elongated)           | `analyze_fruits` + `rotate_box`                |
| `fruit_solidity`            | Area ÷ convex hull area (float, 0–1)                                    | `get_fruit_morphology`                         |
| `fruit_compactness`         | Shape irregularity (float, >0; higher = less compact)                   | `get_fruit_morphology`                         |
| `box_length_cm`             | Rotated bounding box larger side (length) (float, cm > 0)               | `rotate_box`                                   |
| `box_width_cm`              | Rotated bounding box shorter side (width) (float, cm > 0)               | `rotate_box`                                   |
| `compactness_index`         | Fruit area ÷ bounding box area (float, 0–1)                             | `analyze_fruits`                               |
| `inner_pericarp_area_cm2`   | Area enclosing all locules (float, cm² ≥0)                              | `inner_pericarp_area`                          |
| `outer_pericarp_area_cm2`   | Flesh area (fruit total area − inner pericarp area) (float, cm² ≥0)     | `analyze_fruits`                               |
| `avg_pericarp_thickness_cm` | Avg. thickness between inner & outer ellipses (float, cm ≥0)            | `analyze_fruits`                               |
| `mean_locule_area_cm2`      | Mean locule area (float, cm² ≥0)                                        | `precalculate_locules_data` + `analyze_fruits` |
| `std_locule_area_cm2`       | Standard deviation of locule areas (float, cm² ≥0)                      | `precalculate_locules_data` + `analyze_fruits` |
| `total_locule_area_cm2`**   | Sum of all locule areas (float, cm² ≥0)                                 | `precalculate_locules_data` + `analyze_fruits` |
| `cv_locule_area`            | Coefficient of variation in locule area (float, ≥0; unitless CV)         | `precalculate_locules_data` + `analyze_fruits` |
| `mean_locule_circularity`   | Avg. locule roundness (float, 0–1)                                      | `precalculate_locules_data`                    |
| `std_locule_circularity`    | Variation in locule circularity (float, 0–1)                            | `precalculate_locules_data`                    |
| `cv_locule_circularity`     | Coefficient of variation of locule circularity (float, ≥0)               | `precalculate_locules_data`                    |
| `angular_symmetry`          | Deviation of **locule centroids** from perfect angular spacing around the **fruit centroid** (float, rad ≥0; 0 = perfect)| `angular_symmetry`   |
| `radial_symmetry`           | Variation in radial distances of **locule centroids** from the **fruit centroid** (float, CV ≥0; 0 = perfect) | `radial_symmetry`    |
| `rotational_symmetry`       | Combined measure of angular and radial symmetry of **locule centroids** relative to the **fruit centroid** (float, 0–1; 0 = perfect)  | `rotational_symmetry`                          |
| `locules_density`           | Locules per cm² of fruit (float, ≥0)                                    | `analyze_fruits`                               |
| `inner_area_ratio`          | Inner pericarp ÷ fruit area (float, 0–1)                                | `analyze_fruits`                               |
| `locule_area_ratio`         | Largest ÷ smallest locule area (float, ≥1 if >1 locule)                 | `analyze_fruits`                               |
| `locule_area_percentage`    | % fruit area occupied by locules (float, 0–100%)                        | `analyze_fruits`                               |
| `locule_packing_efficiency`  | % of inner pericarp filled with locules (float, 0–100%)                  | `analyze_fruits`                               |



### Notes & caveats

* **Units & scaling**: All cm/cm² values respect **isotropic vs anisotropic** pixel scales (`px_per_cm_x`, `px_per_cm_y`).
* **NaNs/zeros**: Some metrics may be `NaN` (e.g., undefined symmetry with <2 locules) or `0` (e.g., guarded ratios when denominators are 0).
* **Circularity**: The theoretical range is 0–1; minor floating-point artifacts can yield values slightly >1 on noisy, low-res contours.
* **Symmetry metrics**:

  * `angular_symmetry` is in **radians** (not normalized).
  * `radial_symmetry` is a **CV** (scale-free, ≥0).
  * `rotational_symmetry` is **clipped to \[0,1]** after combining angular and radial terms.
* **Aspect ratio**: Defined from the rotated box as width/length (≤1).
* **Pericarp thickness**: Reported as a **scalar average**, not a spatial map.


### 1.3 **Multi-Platform Support**

***Traitly*** offers multiple interfaces to accommodate different user preferences and workflows:

* **Python API**: Allows users to integrate the software directly into Python scripts or interactive environments like JupyterLab for programmatic control and automation.
* **Command-Line Interface (CLI)**: Optimized for efficient batch processing and analysis of large image datasets.
* **Graphical User Interface (GUI)**: Provides a user-friendly visual interface via a Streamlit web application, allowing users to analyze single or multiple images through intuitive point-and-click interactions without programming.

This multi-interface approach ensures flexibility and accessibility across diverse use cases and skill levels.



## 2. Using the Python Code (Python API)

This manual is organized to guide users through the different layers of functionality provided by the software. The components are grouped based on their intended use and level of abstraction:

* **User-Facing Classes**  are designed for direct user interaction and orchestrate the main workflows. They provide straightforward, high-level interfaces that simplify the pipeline and give users the essential tools to perform common tasks without needing to directly interact with or fully understand the underlying implementation.
* **Advanced Utilities** cater to power users who need to customize or extend workflows beyond the basic use cases.
* **Helper Functions** include commonly used small utilities that simplify repetitive tasks and support higher-level components.
* **Internal/Core Utilities** consist of functions that support internal operations and computations, used by the user-facing classes, and are primarily intended for users who want to understand or modify the internal workings.


### 2.1 Installation and requirements

### 2.2 Modules Overview

***Traitly*** is organized into specialized modules that focus on distinct types of analyses:
- `internal_structure`: Tools and classes for analyzing the internal spatial organization and morphology of locules or other structures within images.
- `color_correction`: Functions to preprocess images by correcting color imbalances, lighting variations, and enhancing consistency across datasets.

Each module contains dedicated components grouped based on their intended use and level of abstraction:

* <span style="color:#c3adc4; font-weight:bold;">User-Facing Classes</span> are designed for direct user interaction and orchestrate the main workflows. They provide straightforward, high-level interfaces that simplify the pipeline and give users the essential tools to perform common tasks without needing to directly interact with or fully understand the underlying implementation.
* <span style="color:#DAA520; font-weight:bold;">Advanced Utilities</span> cater to power users who need to customize or extend workflows beyond the basic use cases.
* <span style="color:#BC8F8F; font-weight:bold;">Helper Functions</span> include commonly used small utilities that simplify repetitive tasks and support higher-level components.
* <span style="color:#8fadd7; font-weight:bold;">Internal/Core Utilities</span> consist of functions that support internal operations and computations, used by the user-facing classes, and are primarily intended for users who want to understand or modify the internal workings.

<br>

### 2.3 Module: Internal Structure

<br>

<div style="background-color:#D8BFD8; padding:8px; border-radius:6px; 
text-align:center;">

#### **User-Facing Classes**

</div>

<br>

<div style="background-color:rgba(234, 212, 243, 0.6); padding:8px; border-radius:6px; text-align:left;">

  <h4 style="margin:0;">High-level Classes:</h4>

</div>


<h4>Class: <code>AnalyzingImage</code></h4>

**Description**:
Main class for **preparing and analyzing images** of fruits (or similar objects).
Provides methods for **reading**, **masking**, **detecting contours/locules**, and **running morphological analysis**.
Can also batch-process entire folders of images.

**Implementation Details**:

1. **Initialization**

   * Accepts a path to either a single image or a directory of images.
   * Tracks state (original image, masks, contours, label text, calibration).

2. **PDF conversion**

   * `pdf_to_img()` converts multi-page PDFs into JPEG images.
   * Creates an output folder automatically if none is provided.

3. **Reading images**

   * `read_image()` loads the image into memory using utility functions.
   * Optionally displays it with customizable figure size.

4. **Mask creation**

   * `create_mask()` segments fruits from the background.
   * Supports options for inverted stamps, Canny thresholds, HSV thresholds, CLAHE contrast, and locule detection.

5. **Fruit detection**

   * `find_fruits()` filters contours using circularity, aspect ratio, and area thresholds.
   * Builds a mapping between fruits and their locules.
   * Detects labels within the image for metadata.

6. **Metadata handling**

   * `detect_metadata()` extracts image name and label text.

7. **Image analysis**

   * `analyze_image()` computes morphological descriptors and generates annotated results.
   * Returns an `AnnotatedImage` with both annotated image and results table.

8. **Batch processing**

   * `analyze_folder()` loops over all images in a directory.
   * Applies the full workflow (mask → contours → analysis).
   * Saves annotated images, results (`all_results.csv`), and error reports.

9. **Summaries**

   * `metadata_summary()` prints quick metadata (labels, contours, fruit count, physical dimensions).

**Arguments**:

*Required*:

* `image_path` (`str`): Path to an input image file **or** a folder containing multiple images.

*Optional*:

*(Set within methods, e.g. thresholds, kernel sizes, etc.)*

**Returns**:
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



---

<h4>Class: <code>AnnotatedImage</code></h4>

**Description**:
Represents an **analyzed image** along with its **associated results** (tables, metrics).
Provides functionality to **save outputs** in different formats, including annotated images (`.jpg`, `.png`, etc.) and result tables (`.csv`).

This class is typically created as the **output** of the analysis workflow performed by **<code>ImageAnalyzer</code>**.
It acts as a **container** that holds both the **visual annotation** (image with contours/labels) and the **quantitative results** (CSV-ready table).

**Implementation Details**:

1. **Initialization**

   * Converts the input `cv2_image` (BGR) to RGB.
   * Stores results in both `results` and `table`.
   * Optionally stores the original image path.

2. **Directory handling**

   * `_ensure_dir_exists()` checks if a directory exists before saving files.
   * Expands `~` and relative paths to absolute paths.

3. **Saving images**

   * `save_img()` exports the annotated RGB image to disk.
   * Uses `matplotlib` to remove axes and margins, producing a clean figure.
   * Supports DPI and format customization.

4. **Saving tables**

   * `save_csv()` converts the `table` (list of results) into a Pandas DataFrame.
   * Saves the results as a `.csv` file.

5. **Combined saving**

   * `save_all()` writes both the annotated image and CSV results using a common base name.
   * Allows flexible control of output directory and file names.

**Relationship with <code>ImageAnalyzer</code>**:

* The class **`ImageAnalyzer`** performs the full workflow (masking, contour detection, morphological analysis).
* When analysis is completed, `ImageAnalyzer.analyze_image()` returns an **`AnnotatedImage`** instance.
* This means **`AnnotatedImage` is the final product of the analysis pipeline**, used for reporting, exporting, and visualization.

**Arguments**:

*Required*:

* `cv2_image` (`np.ndarray`): Input image in **BGR** format.

*Optional*:

* `results` (`list`): Analysis results (default = empty list).
* `image_path` (`str | None`): Path to the original image file.

**Returns**:
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
annotated.save_img()
annotated.save_csv()
annotated.save_all()
```


<br>

<div style="background-color:rgba(234, 212, 243, 0.6); padding:8px; border-radius:6px; text-align:left;">

  <h4 style="margin:0;">Wrapper Methods (ImageAnalyzer / AnnotatedImage):</h4>

</div>



<h5>Method: <code>read_image(...)</code></h5>

**Description**:
Loads the image from `image_path` into memory and optionally displays it.
Works directly with the file path provided at initialization and stores the result internally in `self.img`.
For algorithmic details, refer to <code>Utilities → ms.load\_img</code>.

**Arguments** (*Optional*):

* `plot` (`bool`, default `False`) — Whether to display the loaded image.
* `output_message` (`bool`, default `True`) — Print a confirmation message after loading.
* `plot_size` (`tuple[int,int]`, default `(5,5)`) — Size of the display figure.
* `plot_axis` (`bool`, default `False`) — Whether to show axes in the display.

**Returns**:

* `None` — does not return a value. Instead, stores the loaded image in:

  * `self.img` → original image in **BGR format** (as loaded with OpenCV).

**Notes**:

* Internally calls <code>ms.load\_img</code> for image reading and optional plotting.
* The image is preserved in memory for use in subsequent steps (`create_mask`, `find_fruits`, etc.).

**Example**:

```python
analyzer = AnalyzingImage("fruits/sample.jpg")
analyzer.read_image(plot=True, plot_size=(6,6))

# Access the loaded image directly
img = analyzer.img
print(img.shape)   # (H, W, 3) -> BGR image
```


---


<h5>Method: <code>create_mask(...)</code></h5>

**Description**:
Segments fruits from the background. Wrapper around the core masking routine that adds state storage (class attributes), stamp handling, and optional plotting.
Works directly with the image and metadata managed by <code>ImageAnalyzer</code> and is built on top of <code>core.create\_mask</code>. For algorithmic details, see <code>Internal/Core Utilities → core.create\_mask</code>.

**Arguments** (*Optional*):

* **Morphology**:

  * `n_kernel` (`int`, default `5`) — size of the structuring element used in morphological operations.
  * `n_iteration` (`int`, default `1`) — number of times morphological operations are applied.

* **Edges & Color**:

  * `canny_min` (`int`, default `30`) — minimum threshold for Canny edge detection.
  * `canny_max` (`int`, default `100`) — maximum threshold for Canny edge detection.
  * `lower_hsv` (`list[int] | None`, default `None`) — lower bound for HSV color filtering (e.g., background removal).
  * `upper_hsv` (`list[int] | None`, default `None`) — upper bound for HSV color filtering.

* **Stamps & Locules**:

  * `stamp` (`bool`, default `False`) — if `True`, the input image is inverted before processing (for stamp-like images).
  * `locules_filled` (`bool`, default `False`) — if `True`, applies extra CLAHE + Otsu thresholding to detect internal locules (even when not hollow).
  * `min_locule_size` (`int`, default `300`) — minimum pixel area required to consider a locule valid.
  * `n_blur` (`int`, default `11`) — kernel size for median blur used to smooth the locule mask.
  * `clip_limit` (`int`, default `4`) — contrast limit for CLAHE enhancement.
  * `tile_grid_size` (`int`, default `8`) — CLAHE grid size (local histogram equalization regions).

* **Visualization**:

  * `plot` (`bool`, default `False`) — whether to display the generated mask.
  * `plot_size` (`tuple[int,int]`, default `(5,5)`) — size of the plot figure.
  * `plot_axis` (`bool`, default `False`) — if `True`, shows axis ticks/labels in the plot.

**Returns**:

* `None` — does not return a value. Instead, stores results internally:

  * `self.mask` → main binary mask (always set).
  * `self.mask_fruits` → set when `locules_filled=True`.
  * `self.img_inverted` → set when `stamp=True`.

**Notes**:

* When `locules_filled=True`, the method enhances the L channel (LAB space) using CLAHE, then applies Otsu thresholding and morphology to detect internal regions.
* This allows segmentation of **internal locules** that may not appear as hollow cavities (i.e., filled or partially filled locules).

**Example**:

```python
analyzer = AnalyzingImage("fruits/sample.jpg")
analyzer.read_image()
analyzer.create_mask(stamp=True, locules_filled=True, plot=True)

# Access the generated mask directly
mask = analyzer.mask
print(mask.shape)   # (H, W)
```

---

<h5>Method: <code>find_fruits(...)</code></h5>

**Description**:
Detects fruits and locules from the mask, applying geometric filters and label detection.
Stores contours and fruit–locule mappings as class attributes.
For algorithmic details, refer to <code>Internal/Core Utilities → core.find\_fruits</code>.

**Arguments** (*Optional*):

* **Shape filters**:

  * `min_circularity` (`float`, default `0.5`) — minimum circularity to accept a fruit contour.
  * `max_circularity` (`float`, default `1.0`) — maximum circularity to accept a fruit contour.
  * `min_aspect_ratio` (`float`, default `0.3`) — minimum aspect ratio (width/height) for valid contours.
  * `max_aspect_ratio` (`float`, default `3.0`) — maximum aspect ratio (width/height) for valid contours.
  * `contour_filters` (`dict | None`) — custom thresholds to override defaults.

* **Locules**:

  * `min_locule_area` (`int`, default `50`) — minimum area (px²) for a locule to be considered valid.
  * `min_locule_per_fruit` (`int`, default `1`) — minimum number of locules required to keep a fruit.

* **Labels**:

  * `language_label` (`list[str]`, default `['es','en']`) — languages to use in OCR for label detection.
  * `min_area_label` (`int`, default `500`) — minimum contour area to classify as label region.
  * `min_canny_label` (`int`, default `0`), `max_canny_label` (`int`, default `150`) — Canny thresholds for label edge detection.
  * `blur_label` (`tuple[int,int]`, default `(11,11)`) — blur kernel applied before label detection.

* **Misc**:

  * `output_message` (`bool`, default `True`) — whether to print detection summary.

**Returns**:

* `None` — does not return a value. Instead, stores results internally:

  * `self.contours` → list of detected contours.
  * `self.fruit_locules_map` → mapping fruit → locule IDs.
  * `self.label_text`, `self.label_coord`, `self.label_id` (if a label is detected).

**Notes**:

* Internally wraps <code>core.find\_fruits</code> and adds label detection (via <code>ms.detect\_label</code>) plus state storage.

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

---

<h5>Method: <code>analyze_image(...)</code></h5>

**Description**:
Runs the full morphological analysis on detected fruits and locules.
Handles label detection (if not done), calibration of px/cm, and contour analysis.
Returns an <code>AnnotatedImage</code> with annotated RGB image and results table.
For algorithmic details, refer to <code>Internal/Core Utilities → core.px\_per\_cm</code> and <code>core.analyze\_fruits</code>.

**Arguments** (*Optional*):

* **Visualization**:

  * `plot` (`bool`, default `True`) — display results with annotations.
  * `plot_size` (`tuple[int,int]`, default `(10,10)`) — figure size for display.
  * `plot_axis` (`bool`, default `False`) — show axis in plots.
  * `font_scale` (`float`, default `1.5`) — font size for text annotations.
  * `font_thickness` (`int`, default `2`) — line thickness for text.
  * `plot_title_fontsize` (`int`, default `12`) — font size for title.
  * `label_bg_color` (`tuple[int,int,int]`, default `(255,255,255)`) — background color for label boxes.
  * `line_spacing` (`int`, default `15`) — spacing between annotation lines.
  * `plot_title_pos` (`str`, default `'center'`) — title alignment.

* **Contours & Shape**:

  * `use_ellipse` (`bool`, default `False`) — fit ellipses to contours instead of raw polygon.
  * `contour_mode` (`str`, default `'raw'`) — contour approximation mode.
  * `epsilon_hull` (`float`, default `0.001`) — tolerance for convex hull simplification.

* **Locules**:

  * `min_locule_area` (`int`, default `100`) — minimum area (px²) to consider a locule.
  * `max_locule_area` (`int | None`) — maximum allowed locule area.
  * `min_distance` (`int`, default `0`) — minimum distance for merging locules.
  * `max_distance` (`int`, default `100`) — maximum distance for merging locules.

* **Symmetry**:

  * `n_shifts` (`int`, default `500`) — number of shifts for symmetry analysis.
  * `angle_weight` (`float`, default `0.5`) — weight for angle symmetry.
  * `radius_weight` (`float`, default `0.5`) — weight for radius symmetry.
  * `min_radius_threshold` (`float`, default `0.1`) — minimum radius threshold.
  * `rel_tol` (`float`, default `1e-6`) — relative tolerance in calculations.

* **Calibration**:

  * `width_cm` (`float | None`) — manual physical width in cm.
  * `length_cm` (`float | None`) — manual physical length in cm.
  * `scale_size` (`str`, default `'letter_ansi'`) — reference scale for px/cm calibration.

**Returns**:

* `AnnotatedImage` — containing:

  * `rgb_image` → annotated image (RGB).
  * `table` → list of results (metrics per fruit/locule).

**Notes**:

* Internally calls <code>core.px\_per\_cm</code> for calibration and <code>core.analyze\_fruits</code> for analysis.
* Also uses <code>ms.detect\_label</code> if label metadata was not detected before.

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
print(len(annotated.table))       # number of rows in results
```

---

<h5>Method: <code>analyze_folder(...)</code></h5>

**Description**:
Batch wrapper that processes all valid images in a directory.
Applies the full pipeline (read → mask → find\_fruits → analyze\_image) to each file, saving annotated images, results, and error reports.
For algorithmic details, see <code>Internal/Core Utilities → core.analyze\_fruits</code> (used internally via <code>analyze\_image</code>).

**Arguments** (*Optional*):

* **General**:

  * `output_dir` (`str | None`, default `None`) — folder for saving results (defaults to `<input>/Results`).
  * `stamp` (`bool`, default `False`) — whether input images are inverted stamps.
  * `contour_mode` (`str`, default `'raw'`) — contour approximation mode.
  * `n_kernel` (`int`, default `7`) — kernel size for morphology in mask creation.
  * `min_circularity` (`float`, default `0.3`) — minimum circularity filter for fruits.

* **Visualization**:

  * `font_scale` (`float`, default `1.5`) — font size for text annotations.
  * `font_thickness` (`int`, default `2`) — thickness of annotation text.
  * `padding` (`int`, default `15`) — padding around text.
  * `line_spacing` (`int`, default `15`) — spacing between annotation lines.
  * `label_bg_color` (`tuple[int,int,int]`, default `(255,255,255)`) — background color for labels.

* **Locules**:

  * `min_locule_area` (`int`, default `300`) — minimum area (px²) for locule acceptance.
  * `max_locule_area` (`int | None`) — maximum locule area.
  * `min_distance` (`int`, default `2`) — minimum distance for merging locules.
  * `max_distance` (`int`, default `30`) — maximum distance for merging locules.

* **Shape analysis**:

  * `epsilon_hull` (`float`, default `0.005`) — tolerance for convex hull simplification.
  * `use_ellipse_fruit` (`bool`, default `False`) — whether to fit ellipses to fruits.
  * `n_shifts` (`int`, default `500`) — number of shifts for symmetry analysis.
  * `angle_weight` (`float`, default `0.5`) — weight for angle symmetry.
  * `radius_weight` (`float`, default `0.5`) — weight for radius symmetry.
  * `min_radius_threshold` (`float`, default `0.1`) — minimum radius threshold.
  * `rel_tol` (`float`, default `1e-6`) — relative tolerance in symmetry.

* **Calibration**:

  * `width_cm` (`float | None`) — manual width in cm.
  * `length_cm` (`float | None`) — manual length in cm.
  * `scale_size` (`str`, default `'letter_ansi'`) — reference scale for px/cm calibration.

* **Extra**:

  * `**kwargs` — forwarded to <code>create\_mask()</code> (e.g., HSV thresholds, Canny parameters).

**Returns**:

* `None` — does not return a value. Instead, writes results to disk:

  * `annotated_<filename>` → annotated images.
  * `all_results.csv` → concatenated table of all analyzed images.
  * `error_report.csv` → errors and skipped images.

**Notes**:

* Internally loops over images in the input folder and applies <code>analyze\_image</code> to each one.
* Uses <code>psutil</code> for runtime and memory reporting.

**Example**:

```python
batch = AnalyzingImage("fruits/")
batch.analyze_folder(output_dir="fruits/Results", stamp=True)

# Outputs are saved directly in the Results folder
# including annotated images, all_results.csv, and error_report.csv
```

---



<h5>Method: <code>pdf_to_img(...)</code></h5>

**Description**:
Converts a multi-page PDF into a sequence of JPEG images (one per page).
Wrapper utility embedded in <code>ImageAnalyzer</code>, built on top of <code>pdf2image.convert\_from\_path</code>.
If `output_dir` is not provided, a new folder called <code>images\_from\_pdf</code> is automatically created next to the PDF file.
Each page is saved as `<pdf_name>_pageN.jpg`.

**Arguments** (*Required*):

* `pdf_path` (`str`) — path to the input PDF file. Must have `.pdf` extension.

**Arguments** (*Optional*):

* `dpi` (`int`, default `300`) — resolution in dots per inch for conversion.
* `output_dir` (`str | None`, default `None`) — destination folder. If `None`, creates `<pdf_dir>/images_from_pdf`.
* `n_threads` (`int | None`, default `None`) — number of CPU threads for conversion. If `None`, defaults to `1`.
* `output_message` (`bool`, default `True`) — whether to print progress and summary messages.

**Returns**:

* `list[str]` — list of paths to the saved JPEG images.

**Notes**:

* Input validation ensures the file exists and has `.pdf` extension.
* If no `output_dir` is given, a new folder `images_from_pdf` is created automatically.
* Filenames follow the pattern `<pdf_name>_page{i+1}.jpg`.

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


---

<div style="background-color:rgba(234, 212, 243, 0.6); padding:8px; border-radius:6px; text-align:left;">

  <h4 style="margin:0;">New Utilities (AnnotatedImage):</h4>

</div>

<h5>Method: <code>save_img(...)</code></h5>

**Description**:
Embedded in <code>AnnotatedImage</code>. Saves the annotated RGB image (produced and stored by <code>ImageAnalyzer</code>) to disk without axes or margins.
If the output directory does not exist, it is created automatically.

**Arguments** (*Optional*):

* `path` (`str | None`, default `None`) — output file path. If `None`, uses `<original_dir>/<image_name>_annotated.<ext>`.
* `format` (`str | None`, default `None`) — image format (e.g., `'jpg'`, `'png'`). If `None`, inferred from the file extension in `path`, or defaults to `'jpg'`.
* `dpi` (`int`, default `300`) — resolution in dots per inch for raster image export.
* `output_message` (`bool`, default `True`) — whether to print a confirmation message after saving.
* `**kwargs` — additional keyword arguments passed to `matplotlib.pyplot.savefig` (e.g., `transparent=True`).

**Returns**:

* `None` — writes the file to disk.

**Example**:

```python
annotated = analyzer.analyze_image(plot=True)

# Save annotated image (auto-creates directory if missing)
annotated.save_img(format="png", dpi=200)

# Access default path
print(annotated.image_path)  # original image path reference
```

---

<h5>Method: <code>save_csv(...)</code></h5>

**Description**:
Embedded in <code>AnnotatedImage</code>. Saves the results table (computed by <code>ImageAnalyzer</code> and stored in the object) to a CSV file.
If the output directory does not exist, it is created automatically.

**Arguments** (*Optional*):

* `path` (`str | None`, default `None`) — output CSV path. If `None`, uses `<original_dir>/<image_name>_results.csv`.
* `sep` (`str`, default `','`) — column separator to use in the CSV file.
* `output_message` (`bool`, default `True`) — whether to print a confirmation message after saving.

**Returns**:

* `None` — writes the file to disk.

**Errors**:

* Raises `ValueError` if `self.table` is empty (i.e., no results to save).

**Example**:

```python
annotated = analyzer.analyze_image(plot=False)

# Save results table (auto-creates directory if missing)
annotated.save_csv(sep=";")

# The CSV will contain the per-fruit and per-locule measurements
```

---

<h5>Method: <code>save_all(...)</code></h5>

**Description**:
Embedded in <code>AnnotatedImage</code>. Convenience method that saves both the annotated image and the results table (produced by <code>ImageAnalyzer</code>) in one call.
If the output directory does not exist, it is created automatically.

**Arguments** (*Optional*):

* `base_name` (`str | None`, default `None`) — base name for the files, without extension. If `None`, uses the original image name.
* `output_dir` (`str | None`, default `None`) — directory where the files will be saved. If `None`, uses the original image directory.
* `format` (`str`, default `'jpg'`) — image format for export (e.g., `'jpg'`, `'png'`).
* `dpi` (`int`, default `300`) — resolution in dots per inch for raster export.
* `sep` (`str`, default `','`) — column separator for the CSV file.
* `output_message` (`bool`, default `True`) — whether to print confirmation messages after saving.

**Returns**:

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

---


<div style="background-color:#DAA520; padding:8px; border-radius:6px; 
text-align:center;">

#### **Advanced Utilities**

</div>

---

<div style="background-color:#BC8F8F; padding:8px; border-radius:6px; 
text-align:center;">

#### Helper Functions

</div>

<br>


<h4>Function: <code>load_img</code></h4>

**Description**:
Loads an image from disk in **BGR** format (OpenCV) with basic file-type validation, and optionally displays it for quick inspection.

**Implementation Details**:

1. **Extension validation**

   * Accepts only: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`.
   * Raises a `ValueError` if the extension is not in the allowed set.
2. **Image loading**

   * Uses `cv2.imread(path)` to read the image in **BGR**.
   * If loading fails (`img is None`), raises a `ValueError` with the filename.
3. **Optional plotting**

   * If `plot=True`, calls `plot_img(img, metadata=False, fig_axis=fig_axis, plot_size=plot_size)` to display the image.

**Arguments**:

*Required*:

* `path` (`str`): Full path to the image file.

*Optional*:

* `plot` (`bool`): Display the image after loading (default `True`).
* `plot_size` (`Tuple[int,int]`): Figure size in inches `(width, height)` for display (default `(20, 10)`).
* `fig_axis` (`bool`): Show axes in the plot if `True` (default `True`).

**Returns**:

* `np.ndarray | None`: The loaded image in **BGR** format if successful; otherwise `None` (on caught exceptions).

**Raises**:

* `ValueError`: If the extension is invalid or if the image cannot be loaded by OpenCV.
* Other exceptions are caught and reported; the function then returns `None`.

**Example**:

```python
img = load_img(
    path="data/slice_03.jpg",
    plot=True,
    plot_size=(16, 9),
    fig_axis=False
)
if img is not None:
    print("Image loaded:", img.shape)
```

<br>

---

<h4>Function: <code>detect_img_name</code></h4>

**Description**:
Extracts the **base filename (without extension)** from a given image path, while validating the input type and file extension. Provides a warning if the file extension is not a recognized image format.

**Implementation Details**:

1. **Input validation**

   * Ensures `path_image` is of type `str`.
   * Raises a `TypeError` if a different type is provided.
2. **Extension check**

   * Accepts extensions: `.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`.
   * If the extension is not in this set, issues a `warnings.warn`.
3. **Filename extraction**

   * Uses `os.path.basename` and `os.path.splitext` to return the filename without extension.
4. **Error handling**

   * On unexpected exceptions, prints the error and returns `None`.

**Arguments**:

*Required*:

* `path_image` (`str`): Full path to the image file.

**Returns**:

* `str`: Base name of the file (without extension) if valid.
* `None`: If an exception occurs.

**Raises**:

* `TypeError`: If `path_image` is not a string.
* `Warning`: If the extension is not a valid image format (execution continues).

**Example**:

```python
name = detect_img_name("dataset/images/sample_slice.tif")
print(name)  
# Output: "sample_slice"
```

<br>

---

<h4>Function: <code>detect_label</code></h4>

**Description**:
Detects and reads a **text label** in the input image by scanning contour candidates that look like rectangular label boxes. Uses [EasyOCR](https://www.jaided.ai/easyocr/) to extract the text from the detected region. Returns the recognized label text and the label contour coordinates if found; otherwise returns a default message and `None`.

**Implementation Details**:

1. **OCR initialization**

   * Tries to build an `easyocr.Reader` with the provided `language` list (e.g., `['es','en']`); falls back to English `['en']` if initialization fails.
   * For additional supported languages, see the official [EasyOCR documentation](https://www.jaided.ai/easyocr/).

2. **Contour screening for label candidates**

   * For each contour:

     * Computes perimeter and polygonal approximation with `cv2.approxPolyDP(contour, 0.02*peri, True)`.
     * Computes area with `cv2.contourArea`.
     * Considers the contour a **label candidate** if `area > min_area_label` **and** the approximation has **4 vertices** (rough rectangle).

3. **Region extraction & preprocessing**

   * Crops the candidate region via `cv2.boundingRect(approx)`.
   * Converts to grayscale → Gaussian blur with kernel `blur_label` → Canny edges with thresholds `min_canny_label` / `max_canny_label` → Otsu thresholding to obtain a high-contrast mask for OCR.

4. **Text recognition**

   * Runs `reader.readtext(thresh)` and, if any results are found, joins recognized strings with spaces.
   * On first successful read, returns **immediately** the `label_text` and the label contour coordinates.

5. **Failure & errors**

   - The function continues scanning other contours even if one candidate fails text extraction due to:
     - Poor image quality (blurry, low resolution)
     - Label damage (folded, torn, or obscured labels)
     - Unfavorable lighting conditions (shadows, glare)
     - Non-text labels (logos, barcodes, or color patches)
     - Unsupported characters or fonts
   - Per-contour exceptions are caught and logged, allowing the function to continue evaluating other potential label contours
   - If text extraction fails for all candidates, returns the default "no label" message rather than failing completely

**Arguments**:
*Required*:

* `img` (`np.ndarray`): Source image in **BGR** format.
* `contours` (`List[np.ndarray]`): Contours to screen for a rectangular label region.
* `filtered_fruit_locus_map` (`dict`): Mapping of fruits→locules (not used internally; kept for interface consistency).

*Optional*:

* `language` (`List[str]`): Language codes for EasyOCR. Default `['es','en']` (Spanish and English), fallback `['en']` (English).
* `min_area_label` (`int`): Minimum contour area to consider a label candidate (default `500` px$^2$).
* `blur_label` (`Tuple[int,int]`): Gaussian blur kernel for label preprocessing (default `(11,11)`).
* `min_canny_label` (`int`): Lower Canny edge threshold (default `0`).
* `max_canny_label` (`int`): Upper Canny edge threshold (default `150`).

Aquí está la documentación actualizada que refleja esos escenarios de fallo:

**Returns**:  
`Tuple[str, Optional[numpy.ndarray]]`:  
- `label_text`: The extracted text string from the most probable label contour, OR  
  `"No label included/detected"` if:
  - No contours pass the initial size/area filters
  - **No candidate contour yields readable text** (e.g., due to poor image quality, blurry labels, or OCR failure)
  - All potential label contours raise exceptions during processing
- `label_contour`: The contour coordinates of the detected label, OR `None` if no valid label was found/read


**Example**:

```python
label_text, label_coords = detect_label(
    img=img_bgr,
    contours=all_contours,
    filtered_fruit_locus_map=fruit_locules_map,
    language=['es','en'],
    min_area_label=800,
    blur_label=(9,9),
    min_canny_label=50,
    max_canny_label=180
)
print(label_text)           # e.g., "Cultivar A - Plot 12"
print(label_coords is None) # False if a label was found
```

**Notes**:

1. **Early return**: The function returns on the **first** rectangle that produces OCR output; if multiple labels exist, only the first detected is reported.
2. **Preprocessing sensitivity**: `blur_label`, `min_canny_label`, and `max_canny_label` let you tune OCR readiness for low/high-contrast labels.
3. **Coordinate format**: `label_coordinates` is the **contour** of the detected rectangle (list of points), not the bounding box.
4. **Dependency**: Requires `easyocr` (and its OCR backends).


<br>

---

<h4> Function: <code>plot_image</code></h4>

**Description**:  
Utility function to **display images** with optional axis visibility and custom titles. Converts the input image from **BGR** (OpenCV default) to **RGB** (matplotlib default) for correct color rendering. Useful for visual inspection of processed or annotated images during fruit analysis.

**Implementation Details**:

1. **Color conversion**  
   * Uses `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` to display images correctly with matplotlib.  
   * Without this conversion, colors would appear inverted (e.g., blue ↔ red).

2. **Figure setup**  
   * Creates a matplotlib figure with dimensions specified by `plot_size`.  
   * Sets a title combining the image name (`img_name`) and the provided label (`label_text`).  
   * Applies `tight_layout()` for better spacing.

3. **Axis control**  
   * If `fig_axis=True`, axes are displayed (useful for pixel-level reference).  
   * If `fig_axis=False`, axes are hidden for a cleaner visualization.



**Arguments**:

*Required*:
- `img` (`np.ndarray`): Image to display in **BGR format** (as returned by OpenCV).

*Optional*:
- `fig_axis` (`bool`): Show axes if `True`, hide them if `False` (default `False`).  
- `plot_size` (`Tuple[int,int]`): Figure size in inches (default `(10,10)`).  
- `label_text` (`str`): Custom label or annotation text displayed in the title (default `"None"`).  
- `img_name` (`str`): Image identifier shown in the title (default `"None"`).  
- `title_fontsize` (`int`): Font size of the title (default `12`).  
- `title_location` (`str`): Title alignment in the figure; options: `'left'`, `'center'`, `'right'` (default `'center'`).  


**Returns**:  
`None` — The function directly **renders the image** using matplotlib.


**Example**:

```python
plot_image(
    img=annotated_img,
    fig_axis=False,
    plot_size=(12,8),
    label_text="Cultivar A",
    img_name="slice_03.jpg",
    title_fontsize=16,
    title_location="left"
)
```

<br>

---


<h4>Function: <code>is_contour_valid</code></h4>

**Description**:
Evaluates whether a given contour satisfies a set of geometric criteria, including **area**, **circularity**, and **aspect ratio**. Returns a boolean indicating if the contour passes all thresholds.

This is typically used as a **filtering step** to discard invalid or noisy contours before further analysis (e.g., fruits or locules that are too small, too elongated, or too irregular).

**Implementation Details**:

1. **Default thresholds**

   * If no `filters` dictionary is provided, the function applies defaults:

     * `min_area = 300`
     * `min_circularity = 0.6`
     * `max_circularity = 1.0`
     * `min_aspect_ratio = 0.4`
     * `max_aspect_ratio = 1.0`
   * If `filters` is provided, only the specified keys are overridden while others remain at default values.

2. **Area check**

   * Uses `cv2.contourArea(contour)` to compute the enclosed area.
   * If the area is below the `min_area` threshold, the contour is immediately rejected.

3. **Perimeter and circularity**

   * The perimeter is computed with `cv2.arcLength(contour, True)`.
   * Circularity is defined as:

     $$
     \text{circularity} = \frac{4 \pi \times \text{area}}{\text{perimeter}^2}
     $$

     *Values close to 1 represent circles; smaller values indicate elongated or irregular shapes.*
   * The contour is rejected if circularity is outside `[min_circularity, max_circularity]`.

4. **Aspect ratio**

   * Computes the rotated bounding box with `cv2.minAreaRect(contour)`.
   * Aspect ratio is defined as:

     $$
     \text{aspect ratio} = \frac{\min(w,h)}{\max(w,h)}
     $$

     where `w` and `h` are bounding box dimensions.
   * Ensures the contour's elongation is within `[min_aspect_ratio, max_aspect_ratio]`.

5. **Final validation**

   * A contour is considered **valid** only if it passes all three checks (area, circularity, and aspect ratio).

**Arguments**:

*Required*:

* `contour` (`np.ndarray`): Input contour to evaluate (shape: `[N, 1, 2]`).

*Optional*:

* `filters` (`dict`, default `None`): Dictionary to override filter thresholds. Missing keys are filled with defaults:

  * `min_area` (`float`): Minimum area threshold (default `300`).
  * `min_circularity` (`float`): Minimum circularity (default `0.7`).
  * `max_circularity` (`float`): Maximum circularity (default `1.0`).
  * `min_aspect_ratio` (`float`): Minimum aspect ratio (default `0.8`).
  * `max_aspect_ratio` (`float`): Maximum aspect ratio (default `1.0`).

**Returns**:

* `bool`:

  * If `True`, contour passes all filters.
  * If `False`, contour rejected by at least one filter.

**Example**:

```python
filters = {
    "min_area": 200,          # override only min_area
    "min_circularity": 0.6    # override circularity threshold
}

valid = is_contour_valid(contour, filters)
if valid:
    print("Contour accepted.")
else:
    print("Contour rejected.")
```
<br>

---

<h4> Function: <code>pdf_to_img</code></h4>

**Description**:
Converts a **PDF file** into a sequence of **JPEG images**, one per page, using the `pdf2image` library.
Saves the images to a specified folder (or to a default `images_from_pdf` directory created next to the input PDF).


**Implementation Details**:

1. **Validation**:

   * Checks that the input file has a `.pdf` extension.
   * Prints an error message if the file type is invalid.

2. **Conversion**:

   * Calls `convert_from_path()` with the specified `dpi` (default = 600) and number of threads (`n_threads`).
   * Converts each page of the PDF into a `PIL.Image` object.

3. **Output directory**:

   * If `path_img` is not provided, automatically creates a folder called `images_from_pdf` in the same directory as the input PDF.
   * Uses `os.makedirs(..., exist_ok=True)` to ensure the directory exists.

4. **Saving images**:

   * Iterates through each converted page and saves it as `PDFNAME_page{i}.jpg`.
   * Output is in JPEG format regardless of input.

5. **Messages**:

   * If `output_message=True`, prints the total number of images saved and the directory path.
   * Errors are caught and reported as exceptions.

**Arguments**:

*Required*:

* `path_pdf` (`str`): Path to the input PDF file.

*Optional*:

* `dpi` (`int`): Resolution in dots per inch for rendering (default = 600).
* `path_img` (`str | None`): Destination directory for the output images. If `None`, creates `images_from_pdf/` next to the PDF.
* `n_threads` (`int | None`): Number of CPU threads to use during conversion (default = 1).
* `output_message` (`bool`): If `True`, prints a summary message after saving images (default = True).


**Returns**:
None (images are saved to disk).


**Example**:

```python
pdf_to_img(
    path_pdf="reports/experiment_results.pdf",
    dpi=300,
    n_threads=4,
    output_message=True
)
# Output:
# "10 images saved in: reports/images_from_pdf"
```

<br>

<div style="background-color:#e6f2ff; padding:8px; border-radius:6px; 
text-align:center;">

#### Internal/Core Utilities

</div>

<br>

<div style="background-color:rgba(230, 242, 255, 0.6); padding:8px; border-radius:6px; text-align:left;">

  <h4 style="margin:0;">Image processing:</h4>

</div>

<br>




<h4>Function: <code>create_mask</code></h4>

**Description**:  
Creates a binary mask to segment objects from an HSV image using color thresholding, morphological operations, and edge detection. Designed for flexible background removal with customizable processing steps.

**Implementation Details**:  
The function combines multiple computer vision techniques:
1. **Color Thresholding**: Uses HSV bounds to isolate background/foreground with `cv2.inRange()`.
2. **Morphological Refinement**: 
   - Applies opening (noise removal) and closing (hole filling) with elliptical kernels.
   - Supports separate kernel sizes for opening/closing operations.
3. **Edge Enhancement**: Blends Canny edges with the thresholded mask for precise boundaries.

**Arguments**:

*Required*:  
- `img_hsv` (`numpy.ndarray`): Input image in HSV format (shape: `(H,W,3)`, dtype: `uint8`).  

*Optional*:  
- `lower_hsv` (`Tuple[int, int, int]`): Lower HSV bounds for background (default: `[0, 0, 0]`).  
- `upper_hsv` (`Tuple[int, int, int]`): Upper HSV bounds for background (default: `[180, 255, 30]`).  
- `n_iteration` (`int`): Iterations for morphological ops (default: `1`).  
- `n_kernel` (`int`): Default kernel size (odd) when `kernel_open`/`kernel_close` are None (default: `7`).  
- `kernel_open` (`int`): Custom kernel size for opening operation (overrides `n_kernel` if set).  
- `kernel_close` (`int`): Custom kernel size for closing operation (overrides `n_kernel` if set).  
- `canny_min` (`int`): Lower threshold for Canny edge detection (default: `30`).  
- `canny_max` (`int`): Upper threshold for Canny edge detection (default: `100`).  
- `plot` (`bool`): Whether to plot the resulting mask (default: `True`).  
- `figsize` (`Tuple[int, int]`): Figure size for plotting (default: `(20, 10)`).  
- `axis` (`bool`): Whether to show axes when plotting (default: `False`).  

**Returns**:  
`numpy.ndarray`: Binary mask (shape: `(H,W)`, dtype: `uint8`) where:  
- `255` (white): Foreground pixels  
- `0` (black): Background pixels

**Raises**:  
- `TypeError`: If input is not a numpy array.  
- `ValueError`: For invalid HSV bounds, even kernel sizes, or incorrect dtypes.  
- `RuntimeError`: If OpenCV operations fail.  

**Example**:  
```python
# Basic usage (black background removal)
mask = create_mask(img_hsv)

# Custom HSV range for blue background
blue_mask = create_mask(
    img_hsv,
    lower_hsv=np.array([90, 50, 50]),
    upper_hsv=np.array([130, 255, 255]),
    kernel_open=3,
    kernel_close=9)
```

**Notes**:  
1. **HSV Defaults**: Default bounds target black/dark backgrounds (`V ≤ 30`).  
2. **Kernel Strategy**:  
   - Smaller `kernel_open` preserves fine details.  
   - Larger `kernel_close` fills bigger holes.  
3. **Edge Detection**: Canny thresholds (`canny_min/max`) should be tuned for edge sensitivity.  
4. **Performance**: For batch processing, set `plot=False`.  


<br>

---


<h4>Function: <code>px_per_cm</code></h4>

**Description**:
Calculates pixel density (pixels/cm) from scanned objects, including **documents** (paper sheets) or **biological speciments** (fruit slices/leaves) placed directly on the scanner. Uses physicl dimension to establish precise scale. Automatically determines image orientation (portrait or landscape) and treats the **longest** physical side as **length** and the **shortest** as **width**, regardless of input orientation. **When documents uses as input, For precise measurements, always use the scanner-reported dimensions rather than theorical paper size.**

**Implementation Details**:

1. **Image Dimensions (cm)**:
  * The function uses the actual scanned dimensions (including any automatic cropping or optical distorions).
  * Provides independent X/Y scaling to handle non-square pixels.
  * The physical dimensions must satisfy `length_cm ≥ width_cm`. If using predefined sizes (e.g., `'a4_iso'`), this condition is enforced automatically.
  * If custom dimensions are used and `width_cm > length_cm`, a `ValueError` is raised.

2. **Image Dimensions (pixels)**:

  * Uses the shape of the image array to determine image orientation.
  * Automatically handles orientation:

    * `length_px` = longer side (height/Y-axis)
    * `width_px` = shorter side (width/X-axis)
  
  _For example:_
    * If the image shape is 3500 px × 2500 px (portrait), then: `length_px = 3500` and `width_px = 2500`
    * If the image shape is 2500 px × 3500 px (landscape), then, the function swaps the values internally so that: `length_px = 3500` and `width_px = 2500`
  

3. **Density Calculation**:

  * `px_per_cm_width = width_px / width_cm`
  * `px_per_cm_length = length_px / length_cm`
    _These represent the **pixel density** along the width and length respectively._

**Arguments**:

*Required*:

* `img` (numpy.ndarray): Input image (2D grayscale or 3D BGR).

*Optional*:

* `size` (str): Predefined paper size. Options:
  * `'letter_ansi'` (default): 21.6 × 27.9 cm
  * `'legal_ansi'`: 21.59 × 35.56 cm
  * `'a4_iso'`: 21.0 × 29.7 cm
  * `'a3_iso'`: 29.7 × 42.0 cm

* `width_cm` (`float`): Physical width (shorter side, cm). Requires `length_cm`.
* `length_cm` (`float`): Physical length (longer side, cm). Requires `width_cm`.

**Returns**:
`Tuple[float, float, float, float]`:

- `px_per_cm_width` (`float`): Pixels per centimeter along the **shorter side** of the image (width).  
- `px_per_cm_length` (`float`): Pixels per centimeter along the **longer side** of the image (length).  
* `used_width_cm`: Validated width (cm)
* `used_length_cm`: Validated length (cm)

**Raises**:

* `ValueError`: If `length_cm < width_cm`, if dimensions are non-positive, or if `size` is invalid.
* `TypeError`: If input image is not a NumPy array or dimension values are not numeric.
* `RuntimeError`: If calculation fails (e.g., extreme aspect ratio mismatch).

**Example**:

```python
# For a landscape A4 document (21.0 × 29.7 cm)
px_x, px_y, width, length = px_per_cm(img, size='a4_iso')

# With custom dimensions (30 × 20 cm is invalid, must be 20 × 30)
px_x, px_y, width, length = px_per_cm(img, width_cm=20, length_cm=30)
```

**Notes**:

1. **Image Size Validation**: Scanner cropping typically affects paper dimensions by \~1–3%. Check the image metadata or scanner information to extract the actual scanned image dimensions. When analyzing stamps, if scanner dimensions are unavailable, you can use the actual paper dimensions. However, keep in mind that this may introduce a minimal error (±1–3% in most cases).

2. **Image Requirements**: The image must be 2D (grayscale) or 3D (e.g. BGR);  Although a BGR image is expected in the pipeline, the function itself does not depend on the specific color format (e.g. BGR, RGB, HSV, Lab, etc.), only on the array shape.
3. **Aspect Ratio**: In most cases, an equal X/Y pixel density is expected. However, this function calculates independent X/Y scaling factors (`px_per_cm_width`, `px_per_cm_length`) to handle cases where pixel-to-cm ratios differ between axes (due to non-square pixels, optical distortions, or anisotropic processing), ensuring precise dimensional measurements.
4. **Dimension Rules**: Negative or zero dimensions are not allowed.
5. **Orientation Handling**: The `px_per_cm` function does **not** rotate the image—only interprets orientation based on dimensions.

<br>


<div style="background-color:rgba(230, 242, 255, 0.6); padding:8px; border-radius:6px; text-align:left;">
  <h4 style="margin:0;">Contour processing:</h4>
</div>

<br>

<h4>Function: <code>get_fruit_contour</code></h4>

**Description**:
Retrieves the contour of a fruit from a list of contours and optionally transforms it into a simplified or fitted representation.
The function supports **raw contours**, **convex hulls**, **polygonal approximation**, and **ellipse fitting**, depending on the selected `contour_mode`.


**Implementation Details**:

1. **Raw contour (`'raw'`)**

   * Returns the original contour exactly as detected by `cv2.findContours()`.

2. **Convex hull (`'hull'`)**

   * Computes the convex hull using `cv2.convexHull()`.
   * Ensures the contour is convex and tightly encloses the fruit, potentially removing concave indentations.

3. **Polygonal approximation (`'approx'`)**

   * Uses `cv2.arcLength()` to measure contour perimeter.
   * Applies `cv2.approxPolyDP()` with tolerance `epsilon = max(1.0, epsilon_hull × perimeter)` to simplify the contour into fewer vertices.
   * Useful for reducing noise and representing smoother polygonal shapes.

4. **Ellipse fitting (`'ellipse'`)**

   * Requires at least 5 points.
   * Fits an ellipse using `cv2.fitEllipse()`.
   * Converts the fitted ellipse into a discrete polygon (`cv2.ellipse2Poly()`), returning a contour-like representation of the ellipse.
   * Good for regularizing highly irregular contours.


**Arguments**:

*Required*:

* `contours` (`List[np.ndarray]`): List of contours, typically from `cv2.findContours()`.
* `fruit_id` (`int`): Index of the contour in the list corresponding to the fruit of interest.

*Optional*:

* `contour_mode` (`str`): Mode for contour representation. Options:

  * `'raw'`: Original contour (default).
  * `'hull'`: Convex hull.
  * `'approx'`: Polygonal approximation.
  * `'ellipse'`: Ellipse fit.
* `epsilon_hull` (`float`): Factor controlling approximation tolerance (default = `0.0001`). Higher values = more simplification.


**Returns**:
`np.ndarray`: Processed fruit contour with shape `(N, 1, 2)`.


**Example**:

```python
# Get raw contour
raw_contour = get_fruit_contour(contours, fruit_id=0, contour_mode='raw')

# Get convex hull
hull_contour = get_fruit_contour(contours, fruit_id=0, contour_mode='hull')

# Get polygonal approximation
approx_contour = get_fruit_contour(contours, fruit_id=0, contour_mode='approx', epsilon_hull=0.002)

# Get ellipse-fitted contour
ellipse_contour = get_fruit_contour(contours, fruit_id=0, contour_mode='ellipse')
```

**Notes**:

1. **Contour format**: All outputs follow OpenCV’s contour format `(N, 1, 2)`.
2. **Approximation parameter**: The `epsilon_hull` value strongly influences polygonal simplification.
3. **Ellipse fitting**: If fewer than 5 points are available, ellipse mode falls back to the raw contour.


<br>

---


<h4>Function: <code>merge_locules</code></h4>


**Description**:
Merges fragmented locule contours that are spatially close into unified contours, consolidating small, nearby fragments into larger, more meaningful locule shapes. This function helps clean and simplify locule segmentation, particularly in noisy or over-segmented masks.

**Implementation Details**:
This function uses basic geometric proximity and contour approximation to merge nearby locules:

1. **Initial Filtering**:

   * Locules with area ≤ `min_area` (default: `10` px²) are excluded as noise using `cv2.contourArea`.

2. **Pairwise Distance Evaluation**:

   * Iterates over all valid locule contours.
   * Computes **minimum signed distance** between each pair using `cv2.pointPolygonTest`, checking only every second point in the candidate contour for speed.
   * If the absolute distance between locules is within the range `(min_distance, max_distance)`, they are grouped for merging.

3. **Contour Merging**:

   * Groups of nearby locules are combined using `np.vstack` to stack their contour points.
   * The resulting merged shape is simplified using `cv2.approxPolyDP` with an `epsilon` set to `0.001 × arcLength` for smoothing.

**Arguments**:

*Required*:

* `locules_indices` (`List[int]`): Indices of locule contours in `contours`.
* `contours` (`List[np.ndarray]`): List of all detected contours.

*Optional*:

* `min_distance` (`int`): Minimum allowed distance between contours to consider them separate (default: `0`).
* `max_distance` (`int`): Maximum allowed distance to consider contours for merging (default: `50`).
* `min_area` (`int`): Minimum area (in pixels²) to keep a contour (default: `10`). Smaller ones are discarded as noise.

**Returns**:
`List[np.ndarray]`:

* `result_locules`: List of merged contours. Each contour represents either a merged group of locules or an unmerged valid locule.

**Notes**:

1. **Distance-Based Merging**: Uses geometric closeness (point-to-contour distance) rather than centroid distance to handle irregular shapes more accurately.
2. **Contour Approximation**: The final merged shapes are smoothed using `cv2.approxPolyDP`. You can optionally switch to `cv2.convexHull` if a more convex merging strategy is preferred.
3. **Performance**: Only a subset of points is tested for pairwise distances to reduce computation time on large or complex contours.

**Example**:

```python
merged_locules = merge_locules(
    locules_indices, 
    contours=all_contours, 
    min_distance=0,
    max_distance=30,
    min_area=15)
```

<br>

---


<h4>Function: <code>find_fruits</code></h4> 

**Description**:  
Detects fruit contours in a binary mask using geometric filters and a parent–child hierarchy of contours. Returns all contours and a mapping of valid fruit contours (parents) to their internal locules (children).

**Implementation Details**:  
This function leverages [OpenCV’s hierarchical contour retrieval](https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html) (`cv2.RETR_TREE`) to identify fruits and their internal structures:

1. **Hierarchical Contour Detection**:  
   - Uses `cv2.findContours()` with `cv2.RETR_TREE` to retrieve nested structures (e.g., fruit–locule hierarchy). 
2. **Contour Filtering**:  
   * Applies multiple geometric filters:
      * **Area** (`min_area`)
      * **Circularity**: Calculated as `(4π × area) / perimeter²` (range: [0,1], where 1 is a perfect circle).  
   - **Aspect Ratio**: Computed from rotated bounding box (`min(width,height)/max(width,height)`).
3. **Validation Pipeline**:  
   - A contour is classified as a *fruit* if:  
     - It is a top-level contour (no parent or `hierarchy[i][3] == -1`).  
     - Passes area, circularity, and aspect ratio filters.  
     - Contains ≥ `min_locules_per_fruit` valid child contours (locules).  
     - If no contours are found, the function returns an empty contour list ([], {}).
   - A *locule* is valid if:  
     - Nested within a fruit contour.  
     - Area ≥ `min_locule_area`.  

**Arguments**:  

*Required*:  
- `binary_mask` (`numpy.ndarray`): Binary image (unit8) with value `255` (white) for foreground objects/fruits to detect and `0` (black) for background.  

*Optional*:  
- `min_locule_area` (`int`): Minimum pixel area for locule (default: `50`).  
- `min_locules_per_fruit` (`int`): Minimum locule(s) required to classify as fruit (default: `1`).  
- `min_circularity` (`float`): Minimum circularity threshold (default: `0.4`).  
- `max_circularity` (`float`): Maximum circularity threshold (default: `1.0`).  
- `min_aspect_ratio` (`float`): Minimum width/height ratio (default: `0.3`).  
- `max_aspect_ratio` (`float`): Maximum width/height ratio (default: `3.0`). 
- `rescale_factor` (`float`): Scaling factor (0.0-1.0) for faster processing (default: `None`). When used, `min_locule_area` is automatically scaled to match the resized image. All contours are returned in original coordinates (automatically rescaled), ready for immediate analysis without post-processing.
- `contour_approximation` (`int`): Contour approximation method used by OpenCV (default: `cv2.CHAIN_APPROX_SIMPLE`). For more details about available options, see the [OpenCV documentation](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#gadf1ad6a0b82947fa1fe3c3d497f260e0).  
- `contour_filters` (`Optional[Dict]`): Allows passing a single dictionary with multiple parameters, which facilitates programmatic or dynamic use when working with variable or external configurations. Supported keys include: `'min_area'`, `'min_circularity'`, `'max_circularity'`, `'min_aspect_ratio'`, `'max_aspect_ratio'`.


**Returns**:  
`Tuple[List[numpy.ndarray], Dict[int, List[int]]]`:  
- `contours`: List of all detected contours (each as `np.ndarray` of points).  
- `fruit_locules_map`: Dictionary mapping fruit contour indices → list of loculus indices.  

**Raises**:  
- `ValueError`: For invalid inputs (negative values, non-uint8 mask).  
- `cv2.error`: If OpenCV contour detection fails.  

**Example**:  
```python  
# Basic usage  
contours, fruit_map = find_fruits(  
    binary_mask,  
    min_locule_area=100,  
    min_circularity=0.3  
)  

# With custom filters and rescaling
contours, fruit_map = find_fruits(
    binary_mask,
    min_locule_area=50,  # Original value (automatically adjusted if rescale_factor is used)
    rescale_factor=0.5,  # Halves image dimensions (adjusts min_locule_area to 12)
    contour_filters={
        'min_circularity': 0.3,  # Custom filter (not affected by rescaling)
        'max_aspect_ratio': 2.0   # Custom filter
    }
)
# All contours are returned in original coordinates (automatically rescaled)
# and ready for immediate analysis - no manual coordinate transformations needed.  
```  

**Notes**:  

1. **Minimum Locules Behavior**: When `min_locules_per_fruit = 0`, the function will classify any top-level contour that meets the morphological thresholds as a valid fruit, regardless of whether it contains locules. These fruits will appear in the results with an empty locule `list ([])`.
2. **Circularity Range**: While the **theoretical maximum** for circularity is `1.0` (perfect circle), the function **accepts values >1** due to potential floating-point artifacts in perimeter/area calculations (e.g., from low-resolution contours or noisy masks).  
3. **Performance**: For large images, consider downscaling `binary_mask` using `rescale_factor` first to reduce computation time. Note that reducing image size may affect the resolution and accuracy of small locule contours. 
4. **Tuning**: Adjust `min_circularity` and `aspect_ratio` thresholds based on fruit shape variability.  
5. **Debugging**: Visualize contours with `cv2.drawContours` using indices from `fruit_locules_map`.  

<br>

----


<div style="background-color:rgba(230, 242, 255, 0.6); padding:8px; border-radius:6px; text-align:left;">
  <h4 style="margin:0;">Morphological metrics:</h4>
</div>

<br>

<h4>Function: <code>get_fruit_morphology</code></h3>

**Description**:
Computes a set of **morphological metrics** from a fruit contour, including area, perimeter, and shape descriptors.
Handles both **isotropic** and **anisotropic** pixel scaling to convert raw pixel-based measurements into **centimeter units**.

**Implementation Details**:

1. **Area & Perimeter (pixels)**

   * Uses `cv2.contourArea()` to compute the enclosed area.
   * Uses `cv2.arcLength(..., True)` to compute the perimeter.
   * If either is ≤ 0, returns a dictionary filled with `np.nan` values.

2. **Pixel → cm conversion**

   * **Isotropic case** (`px_per_cm_x ≈ px_per_cm_y`):

     * Uses the average pixel density for both area and perimeter conversion.
   * **Anisotropic case**:

     * Converts area using `px_per_cm_x * px_per_cm_y`.
     * Converts perimeter using the average of the two densities.

3. **Shape descriptors**

   * **Circularity**:

     $$
     C = \frac{4 \pi A}{P^2}
     $$

     where $A$ is area (px²), $P$ is perimeter (px). Ranges from 0 (irregular) to 1 (perfect circle).

   * **Solidity**:
     Ratio between the contour area and its convex hull area.

     $$
     S = \frac{A}{A_{hull}}
     $$

   * **Compactness**:
     Ratio of squared perimeter to area, sensitive to contour irregularity.

     $$
     K = \frac{P^2}{A}
     $$

   * **Convex Hull Area**:
     Computed using `cv2.convexHull()` and `cv2.contourArea()`.

**Arguments**:

*Required*:

* `contour` (`np.ndarray`): Contour points of the fruit (from `cv2.findContours()`).
* `px_per_cm_x` (`float`): Pixel density along the shorter side of the image.
* `px_per_cm_y` (`float`): Pixel density along the longer side of the image.

*Optional*:

* `rel_tol` (`float`): Relative tolerance to decide isotropy between `px_per_cm_x` and `px_per_cm_y` (default = `1e-6`).

**Returns**:

`dict`: Dictionary with the following keys:

* `fruit_area_px` (`float`): Area in pixels².
* `fruit_area_cm2` (`float`): Area in cm².
* `fruit_perimeter_px` (`float`): Perimeter in pixels.
* `fruit_perimeter_cm` (`float`): Perimeter in cm.
* `fruit_circularity` (`float`): Circularity index \[0–1].
* `fruit_solidity` (`float`): Solidity index (0–1).
* `fruit_compactness` (`float`): Compactness ratio.
* `fruit_convex_hull_area_px` (`float`): Convex hull area in pixels².


**Example**:

```python
morph = get_fruit_morphology(contour, px_per_cm_x=30.2, px_per_cm_y=29.8)
print(morph["fruit_area_cm2"], morph["fruit_circularity"])
```

<br>

---


<h4>Function: <code>calculate_axes</code></h4>

**Description**:  
Computes the **major** and **minor** axes of a fruit contour in **centimeters**, handling **isotropic** and **anisotropic** pixel scales. Optionally draws both axes on an image.  
- The **major axis** is the maximum pairwise distance between convex-hull vertices (searched in pixels if isotropic, or in centimeters if anisotropic).  
- The **minor axis** is the maximum thickness **perpendicular** to the major axis, measured by projecting all contour points onto the perpendicular direction and taking `max − min` (converted to cm as appropriate).


**Implementation Details**:

1. **Preprocessing**  
   * Reshapes the contour to `(N, 2)` float32 array.

2. **Isotropy check**  
   * If `px_per_cm_width ≈ px_per_cm_length` (within `rel_tol = 1e-6`), treats pixels as **isotropic** and uses the **average** pixels-per-cm factor.  
   * Otherwise, treats pixels as **anisotropic** and builds a coordinate set in **centimeters** with axis-specific scaling:  
     \[
     x_\text{cm} = \frac{x}{\text{px\_per\_cm\_width}},\quad y_\text{cm} = \frac{y}{\text{px\_per\_cm\_length}}
     \]

3. **Major axis (endpoints & length)**  
   * Computes the convex hull indices and searches all vertex pairs to find the **maximum distance** in the chosen coordinate space.  
   * Converts the **length to cm** (divide by average px/cm if isotropic; already in cm if anisotropic).  
   * Returns the **endpoints in original pixel coordinates** `(p1_px, p2_px)`.

4. **Minor axis (thickness perpendicular to major)**  
   * Forms the unit vector **perpendicular** to the major axis.  
   * Projects all points onto this perpendicular direction (in px if isotropic, in cm if anisotropic).  
   * Minor axis length = `max_projection − min_projection`; converts to **cm** if needed.  
   * Also returns the **pixel coordinates** of the points achieving the min/max projection `(p_min_px, p_max_px)`.

5. **Optional drawing**  
   * If `draw_axes=True` and `img` is provided, draws both segments on the image using the given BGR colors and thickness.



**Arguments**:

*Required*:
- `fruit_contour` (`np.ndarray`): Nx2 or Nx1x2 contour points.
- `px_per_cm_width` (`float`): Pixels per centimeter along the **shorter side** of the image (width).  
- `px_per_cm_length` (`float`): Pixels per centimeter along the **longer side** of the image (length).  

*Optional*:
- `rel_tol` (`float`): Tolerance to decide isotropy (default `1e-6`).  
- `img` (`np.ndarray | None`): Image (BGR) where axes will be drawn if `draw_axes=True`.  
- `draw_axes` (`bool`): Draw axes on `img` (default `False`).  
- `major_axis_color` (`Tuple[int,int,int]`): BGR color for major axis (default `(0,255,0)`).  
- `minor_axis_color` (`Tuple[int,int,int]`): BGR color for minor axis (default `(255,0,0)`).  
- `axis_thickness` (`int`): Line thickness in pixels (default `2`).


**Returns**:  
`tuple` of two items:
- `(major_axis_cm, p1_px, p2_px)` → **Major axis** length in **cm**, and its endpoints in **pixels**.  
- `(minor_axis_cm, p_min_px, p_max_px)` → **Minor axis** length in **cm**, and its endpoints in **pixels**.


**Example**:
```python
(major_axis, p1_major, p2_major), (minor_axis, p1_minor, p2_minor) = calculate_axes(
    fruit_contour,
    px_per_cm_width=30.0,
    px_per_cm_length=31.2,
    img=annotated_img,
    draw_axes=True
)
print(f"Major: {major_axis:.2f} cm, Minor: {minor_axis:.2f} cm")
```


**Notes**:

1. **Projection-Based Method**: Measures the contour’s extent perpendicular to the major axis without assuming an ideal ellipse.

2. **Positive and Negative Projections**: The method accounts for points on **both sides** of the perpendicular axis using `max - min`.

3. **Axis-Specific Scaling**: Each axis is converted independently to centimeters using `px_per_cm_width` and `px_per_cm_length`, ensuring accuracy even if pixel densities differ.

4. **Major Axis Dependence**: Accuracy depends on correct identification of `point1` and `point2` as defining the true major axis.

5. **Vectorized Computation**: Projections are calculated using NumPy vectorization for efficiency, without loss of precision.

<br>

----


<h4>Function: <code>rotate_box</code></h4>

**Description**:  
Computes the rotated bounding box of a contour and its dimensions in pixels/centimeters. Optionally draws the box on the image.

**Implementation Details**:  

* Uses `cv2.minAreaRect` to compute the smallest rotated rectangle enclosing the contour.  
* Converts dimensions to centimeters:
  - <a href='#isotropic-pixels'> Isotropic pixels </a> (px_per_cm_width ≈ px_per_cm_length): uses the average of both to convert length and width.
  -  <a href='#anisotropic-pixels'> Anisotropic pixels </a>: converts each side with its corresponding density  
    (`length_cm = length_px / px_per_cm_length`, `width_cm = width_px / px_per_cm_width`).

* The **length** (Y) is defined as the longer side, and the **width** (X) as the shorter side of the bounding box, regardless of rotation.
    _For example:_
    * If the rotated bounding box has dimensions 80 px × 40 px and the angle is -45°, the **length** is 80 and the **width** is 40.
    * If the object is rotated and the box dimensions become 40 px × 80 px with an angle of -90°, the **length** is still 80 and the **width** is still 40.

**Arguments**:  
*Required*:  
- `img` (`numpy.ndarray`): BGR image for drawing the bounding box.  
- `contour` (`numpy.ndarray`): Contour points (from `cv2.findContours`).  
- `px_per_cm_width` (`float`): Pixels per centimeter along the **shorter side** of the image (width).  
- `px_per_cm_length` (`float`): Pixels per centimeter along the **longer side** of the image (length).  


*Optional*:  
- `draw_box` (`bool`): Draw bounding box if `True` (default: `True`).  
- `box_color` (`Tuple[int, int, int]`): BGR color (default: light blue `(255, 180, 0)`).  
- `box_thickness` (`int`): Line thickness in pixels (default: `3`).  

**Returns**:  
`Tuple[float, float]`:  
- `box_length_cm`: Length in cm.  
- `box_width_cm`: Width in cm.  

**Example**:  
```python
box_length_cm, box_width_cm = rotate_box(
    img, contour, px_per_cm_width=150.5, px_per_cm_length=152.3, 
    box_color=(0, 255, 0))
```

<br>

----


<h4>Function: <code>inner_pericarp_area</code></h4>

**Description**:  
Calculates and visualizes the **inner pericarp area** (enclosing locules) using either convex hull approximation or ellipse fitting. Returns the computed area in **squared centimeters** (and optionally draws on the provided image).

**Implementation Details**:  
The function uses [OpenCV](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html#ga2c759ed9f497d4a618048a2f56dc97f1) (`cv2`) for geometric computations:
1. **Convex Hull Method** (`use_ellipse=False`):
   - Computes the smallest convex polygon enclosing all points with `cv2.convexHull`.
   - Applies contour smoothing using `cv2.approxPolyDP` with `epsilon` (percentage of arc length).
   - Calculates area via `cv2.contourArea` (implements [Green's theorem](https://tutorial.math.lamar.edu/classes/calciii/GreensTheorem.aspx)).

2. **Ellipse Fitting** (`use_ellipse=True`):
   - Requires ≥5 points (minimum for `cv2.fitEllipse`).
   - Fits an ellipse to the points and calculates area as `π × (a × b)`, where `a` and `b` are semi-major/minor axes.

3. **Pixels to cm$^2$ conversion**:
  - For <a href='#isotropic-pixels'>isotropic case</a> (px_per_cm_width ≈ px_per_cm_length): Uses average pixel density.
  - For <a href='#anisotropic-pixels'>anisotropic case</a>: Accounts for different scaling factors per axis.

**Arguments**:  
*Required*:  
- `locules` (`List[int]`): Indices of locule contours in `contours`.   
- `contours` (`List[numpy.ndarray]`): Detected contours from `cv2.findContours()`.  
- `px_per_cm_width` (`float`): Pixels per centimeter along the **shorter side** of the image (width).  
- `px_per_cm_length` (`float`): Pixels per centimeter along the **longer side** of the image (length).  
 

*Optional*:  
- `img` (`numpy.ndarray`): Input BGR image (uint8) for drawing contours (used only if `draw_inner_pericarp=True`).  
- `draw_inner_pericarp` (`bool`): If `True`, draws the inner pericarp on `img` (default: `False`).  
- `use_ellipse` (`bool`): Use ellipse fitting if `True`, else convex hull (default: `False`).  
- `epsilon` (`float`): Smoothing factor (range [0, 1]; default: `0.0001`).  
- `rel_tol` (`float`): Relative tolerance for isotropy determination (default: `1e-6`).  
- `contour_thickness` (`int`): Contour line thickness in pixels (default: `2`).  
- `contour_color` (`Tuple[int, int, int]`): BGR color (default: cyan `(0, 240, 240)`).  

**Returns**:  
`float`:  
- `area_cm2`: Computed inner pericarp area in cm² (`0` if `locules` is empty or ellipse fitting is not possible).  

**Example**:  
```python
area_cm2 = inner_pericarp_area(
    locules=[0, 1],
    contours=all_contours,
    px_per_cm_x=30.0,
    px_per_cm_y=31.2,
    img=img,
    draw_inner_pericarp=True,
    use_ellipse=True,
    contour_color=(0, 255, 0)
)

```

<br>

---

<div style="background-color:rgba(230, 242, 255, 0.6); padding:8px; border-radius:6px; text-align:left;">
  <h4 style="margin:0;">Centroid and geometric data:</h4>
</div>

<br>

<h4>Function: <code>calculate_fruit_centroids</code></h4>

**Description**:  
Computes the **centroid coordinates** of each contour in a list using image moments. If a contour has zero area, the centroid is undefined, and the function assigns `None` to that entry.  

This is a common preprocessing step in fruit analysis for positioning, annotation, and spatial reference.  

**Implementation Details**:  

1. Iterates through each contour in the provided list.  
2. Computes **image moments** with `cv2.moments(contour)`.  
3. If the zeroth moment (`m00`, representing contour area) is nonzero:  
   * Calculates centroid as:  
     \[
     cx = \frac{m10}{m00}, \quad cy = \frac{m01}{m00}
     \]  
   * Stores `(cx, cy)` as integer coordinates.  
4. If `m00 == 0`, appends `None` instead of coordinates to avoid division by zero.  


**Arguments**:  

*Required*:  
- `contours` (`List[np.ndarray]`): List of contours (as returned by `cv2.findContours`). Each contour is an array of points with shape `(N, 1, 2)`.  

**Returns**:  
- `List[Tuple[int, int] | None]`:  
  * A list of centroids corresponding to each contour.  
  * Each entry is either `(cx, cy)` in pixel coordinates, or `None` if the contour area is zero.  


**Example**:  
```python
contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

centroids = calculate_fruit_centroids(contours)
for i, c in enumerate(centroids):
    if c is not None:
        print(f"Contour {i} centroid at {c}")
    else:
        print(f"Contour {i} has zero area")

```

<br>

----


<h4>Function: <code>precalculate_locules_data</code></h4>


**Description**: 
Extracts and stores geometric features of a given set of locule contours. For each specified locule, the function calculates its **centroid**, **area**, **perimeter**, **polar coordinates relative to a reference centroid**, **circularity**, and stores its **original contour points**. 

**Imprementation Details**:

1. **Contour Selection**:
   - Only con tours whose indices are listed in `locules` are processed.
   - Each locule contour is treated individually.

2. **Centroid Calculation**: 
   - Uses OpenCV spatial moments (`cv2.moments`) to calculate the centroid:
     -  $\text{cx} = \frac{m_{10}}{m_{00}}$, $\text{cy} = \frac{m_{01}}{m_{00}}$
   - If the area `m00` is zero (i.e., degenerate contour), the contour is **skipped**.

3. **Area and Perimeter**: 
   - Area is calculated via `cv2.contourArea()`.
   - Perimeter (arc length) is calculated using `cv2.arcLength()` with the `closed = True` flag.
  
4. **Polar Coordinates**:

   * Calculated relative to the given reference centroid:
     * **angle**: orientation from the positive x-axis to the vector connecting the reference centroid to the locule’s centroid, in the range \[0, 2π).
     * **radius**: Euclidean distance between the reference centroid and the locule’s centroid.

5. **Circularity**:

   * Calculated as:

     $$
     \text{circularity} = \frac{4\pi \times \text{area}}{\text{perimeter}^2}
     $$

     * **1.0** = perfect circle.
     * Lower values = more irregular or elongated shapes.
   * Useful for assessing locule geometric regularity.

6. **Data Storage**:

   * Each locule's data is stored as a dictionary.
   * All dictionaries are collected into a list and returned.
  

  
**Arguments**:

_Required_:

- `contour` (`List[np.ndarray]`): List of all images contours (e.g. obtained from `traitly.find_fruits` or `cv2.findContours`).
- `locules`: List of indices refering which contours in `contours` corespond to locules.
- `centroid` (`Tuple[int, int]`): Reference centroid coordinates (x,y). 


**Returns**:

`List[Dict]`: A list of dictionaries, each containing the geometric properties of one locule:

* `contour_id` (int): Countour identifier
* `'centroid'` (`Tuple[int,int]`): Centroid `(x, y)` coordinates.
* `'area'` (`float`): Area in pixels$^2$.
* `'perimeter'` (`float`): Perimeter in pixels.
* `'contour'` (`np.ndarray`): Original contour points.
* `'polar_coord'` (`Tuple[float,float]`): (angle${_\text{in}}_{\text{ radians}}$, radius) relative to the reference centroid. 
* `'circularity'` (`float`): Circularity value, 1.0 for a perfect circle.



**Example**:

``` python
# Precalculate geometric features for specific locule contours
for fruit_id, locules in fruit_locules_map.items(): 
    locule_features = precalculate_locules_data(
        contours=all_contours,
        locules=locules,
        centroid=fruit_centroid
    )

    # Access centroid of first locule
    centroid = locule_features[0]['centroid']
    print(centroid)
```

**Notes**:

1. **Skipped Contours**:
   - Contours with no area (`m00 = 0`) are skipped to avoid division by zero when calculating centroids.
2. **Data Integrity**:
   - Output list preserves the order of input `locules`. Each dictionary maps back to the original `contours` via the '`contour'` key.

<br>

----

<div style="background-color:rgba(230, 242, 255, 0.6); padding:8px; border-radius:6px; text-align:left;">
  <h4 style="margin:0;">Symmetry analysis:</h4>
</div>

<br>

<h4>Function: <code>angular_symmetry</code></h4>

**Description**:  
Quantifies the **angular symmetry** of locules arranged around the fruit centroid (reference point) by comparing actual locule angles with an ideal symmetric arrangement. The function searches for the **rotational alignment** that minimizes the mean angular deviation. It uses the **precomputed angles** from `precalculate_locules_data()`; only **locule centroid positions** are considered. **Shape and size of locules are ignored**.

A perfectly symmetric configuration (e.g., locules equidistantly distributed in a circular pattern) returns a value close to `0.0`. Higher values indicate greater deviations from ideal angular symmetry.  
The metric is expressed in **radians** and is not normalized.

For a visual explanation of angular symmetry, refer to <a href='#locule-symmetry'>Figure 1A</a>.

**Implementation Details**:

1. **Angle Extraction**:

   * For each locule, extracts the angle from the reference centroid (`polar_coord[0]`) in radians.
   * Normalizes angles to `[0, 2π)` to maintain circular continuity.

2. **Rotation Adjustment**:

   * Computes the **circular mean** of all angles (`circmean`) to center the pattern.
   * Subtracts this mean from each angle to remove overall rotation bias, then wraps back to `[0, 2π)`.

3. **Ideal Symmetric Angles**:

   * For $n$ locules, generates `n` equally spaced angles covering the full circle:

     $$
     \theta_k^{\text{ideal}} = \frac{2\pi}{n} \cdot k, \quad k = 0, \dots, n-1
     $$
   * Represents the angular positions of locules in a perfectly symmetric circular arrangement.

4. **Error Minimization Across Rotations**:

   * Tests `num_shifts` rotations of the ideal angles to find the best alignment with the actual locule angles.
   * Computes the **angular difference matrix**, accounting for circular wrap-around:

     $$
     \text{diff} = \min(|\theta_{\text{actual}} - \theta_{\text{ideal}}|, 2\pi - |\theta_{\text{actual}} - \theta_{\text{ideal}}|)
     $$
   * Uses the **Hungarian algorithm** (`linear_sum_assignment`) to optimally assign observed locules to ideal angles, minimizing the mean angular error.
   * Keeps the **minimum mean angular error** across all tested rotations (`best_error`).

5. **Return Value**:

   * Returns `best_error` in **radians**. Smaller values indicate better angular symmetry.
   * Returns `np.nan` if fewer than 2 locules are provided.

**Arguments**:

*Required*:

* `locules_data` (`List[Dict]`): List of locule data dictionaries. Each must contain `polar_coord` = `(angle, radius)`.

*Optional*:

* `num_shifts` (`int`): Number of rotational shifts to test when aligning ideal to actual angles (default = 500).

**Returns**:  
`float`: Mean angular deviation (radians).  

* `0.0`: Perfect symmetry (locules uniformly spaced in a circle).  
* `> 0.0`: Angular error in radians (higher = less symmetric).  
* `np.nan`: Undefined if fewer than 2 locules are provided. Uses NumPy’s `np.nan`.  

**Example**:

```python
# Four symmetrically placed locules
locules = [
    {'polar_coord': (0.0, 1.0)},        # 0°
    {'polar_coord': (np.pi/2, 1.0)},    # 90°
    {'polar_coord': (np.pi, 1.0)},      # 180°
    {'polar_coord': (3*np.pi/2, 1.0)}   # 270°
]

sym = angular_symmetry(locules)
# Output: 0.0 (perfect angular symmetry)

```

**Notes**:

1. **Rotation-Invariant**: Aligns the ideal pattern to actual locules, so rotated but symmetric arrangements yield low error.
2. **No Distance Constraint**: Radial distances from the center are ignored (`radial_symmetry()`).
3. **Collinearity**: If locules are nearly collinear, angular error increases.
4. **Effect of Number of Locules:**
The angular symmetry calculation is sensitive to the number of locules. Configurations with very few locules can yield large angular errors even when absolute angular deviations are small. Care should be taken when interpreting angular symmetry for fruits with a low number of locules (2-3).

<br>

---


<h4>Function: <code>radial_symmetry</code></h4>

**Description**:

Calculates the **radial symmetry** of locules using the **coefficient of variation (CV)** of their distances from each locule centroid to the fruit centroid (reference point). The CV measures the relative variability of these distances, where `0.0` indicates perfect radial symmetry (all locules equidistant from the centroid), and higher values indicate increasing irregularity. This function uses the **precomputed radii** from `precalculate_locules_data()`; only **locule centroid positions** are considered. **Shape and size of locules are ignored**.

For a visual explanation of angular symmetry,  refer to <a href='#locule-symmetry'>Figure 1B</a>.


**Implementation Details**:

1. **Distance Extraction**:

   * For each locule, extract the precomputed radius from the reference centroid (`polar_coord[1]`), in pixels.

2. **Symmetry Evaluation**:

   * If fewer than two locules are provided, return `np.nan` since radial symmetry cannot be defined.

   * Compute the **coefficient of variation (CV)** of the radii:

     $$
     CV = \frac{\text{standard deviation of radii}}{\text{mean radius}}
     $$

   * A CV of 0 indicates perfect radial symmetry (all locules equidistant).

   * Larger CV values indicate greater radial asymmetry.


**Arguments**:

*Required*:

* `locules_data` (`List[Dict]`): List of locule data dictionaries, **as returned by `precalculate_locules_data()`**. Each must contain:

  * `'centroid'`: `(x, y)` coordinates of the locule centroid.
  * `'polar_coord'`: `(angle, radius)` precomputed polar coordinates relative to the fruit centroid.

**Returns**:

`float`: Radial symmetry metric (CV of distances).

* `0.0`: Perfect radial symmetry.
* `> 0.0`: Increasing radial asymmetry.
* `np.nan`: Undefined if fewer than 2 locules are provided.

**Example**:

```python
locules = [
    {'polar_coord': (0.0, 1.0)},        # radius = 1
    {'polar_coord': (np.pi/2, 1.0)},    # radius = 1
    {'polar_coord': (np.pi, 1.0)},      # radius = 1 
    {'polar_coord': (3*np.pi/2, 1.0)}   # radius = 1
]

cv = radial_symmetry(locules)
# Output: 0.0 (perfect radial symmetry)
```

Notes:

1. **Radial vs Angular Symmetry**: This function evaluates only radial (distance) symmetry, not angular spacing (see angular_symmetry()).

2. **Rotation Invariance**: The radial distance CV is invariant to rotation because it depends only on the distances from locules to the centroid, which remain unchanged under rotation. 

3. **Coefficient of Variation Use**: CV normalizes the standard deviation by the mean, making it scale-independent and useful for comparing radial symmetry across fruits of different sizes (e.g., small vs. large fruits), which correspond to varying numbers of pixels in image analysis.

4. **Effect of Number of Locules**: CV does **not** directly account for the number of locules. Very few locules (e.g., 2 or 3) can produce high CV values even with small absolute differences. Interpret CV cautiously for low locule counts.

<br>

---

<h4>Function: <code>rotational_symmetry</code></h4>

**Description**:

Quantifies the **rotational symmetry** of locules arranged around a fruit centroid by evaluating **both** their angular distribution and radial distances in a combined, weighted error metric. This function uses the **precomputed angles and radii** from `precalculate_locules_data()`; only **locule centroid positions** are considered. **Shape and size of locules are ignored**.

Unlike `angular_symmetry()` or `radial_symmetry()` which assess angular or radial components independently, this function integrates both aspects to measure how closely locules approximate an ideal rotationally symmetric pattern—i.e., locules evenly spaced in angle and with uniform radial distances from the center.

A **perfectly symmetric** configuration yields a score of `0.0`, while larger values indicate increasing asymmetry in either angular spacing or radial distances.  
The final combined metric is **clipped to** `[0, 1]`.

For a visual explanation of angular symmetry, refer to <a href='#locule-symmetry'>Figure 1C</a>.


**Implementation Details**:

1. **Input validation**:
   If fewer than two locules are provided, symmetry is undefined, and the function returns `np.nan`.

2. **Data extraction & preprocessing**:

   * Extracts precomputed polar coordinates (`(angle, radius)`) from each locule's data dictionary.
   * Normalizes radii by their mean to focus on relative size variation rather than absolute distance, ensuring comparability between fruits of different sizes.
   * Filters out very small locules below a fraction of the mean radius (`min_radius_threshold`) to reduce noise.

3. **Radial asymmetry calculation**:

   * Computes the **Median Absolute Deviation (MAD)** of normalized radii as a robust measure of radial spread.
   * Scales MAD using `0.6745` to approximate standard deviation under normal assumptions, then maps via `tanh` to obtain a **bounded radial error** `radius_error_norm` in roughly `[0,1)`.

4. **Angular asymmetry**:

   * If a precomputed `angle_error` is not provided, the function calculates angular asymmetry internally using `angular_symmetry()`, which returns a **mean angular deviation in radians** (not normalized).
   * This captures deviations from evenly spaced angular positions around the fruit centroid, with rotational alignment handled internally.

5. **Weighted combination**:

   * Computes a **weighted average** of the angular (in radians) and radial (normalized) errors using `angle_weight` and `radius_weight`; weights are re-normalized by their sum.
   * The resulting combined error is then **clipped to `[0,1]`** for consistency across fruits.


**Arguments**:

*Required*:

* `locules_data` (`List[Dict]`): List of dictionaries, each containing:

  * `'centroid'`: tuple `(x, y)` coordinates of the locule centroid (not used in calculation but required by convention).
  * `'polar_coord'`: tuple `(angle, radius)` representing the locule position in polar coordinates relative to the fruit centroid.

*Optional*:

* `angle_error` (`float`): Precomputed **mean angular deviation in radians**. If `None`, it is calculated internally via `angular_symmetry()`.
* `angle_weight` (`float`): Weight for angular error in the combined metric (default `0.5`).
* `radius_weight` (`float`): Weight for radial error in the combined metric (default `0.5`).
* `min_radius_threshold` (`float`): Minimum fraction of mean radius to include a locule in radial error calculation (default `0.1`).



**Returns**:

* `float`: Combined rotational symmetry metric **in `[0,1]`** (after clipping).

  * `0.0` → **perfect rotational symmetry** (even angular spacing, uniform radial distances).
  * Values closer to `1` → increasing asymmetry.
  * `np.nan` → fewer than two valid locules.



**Example**:

```python
locules = [
    {'polar_coord': (0, 5)},
    {'polar_coord': (2*np.pi/3, 5)},
    {'polar_coord': (4*np.pi/3, 5)}
]

error = rotational_symmetry(locules)
print(error)  # Output: 0.0 (perfect rotational symmetry)


```


**Notes**:

1. **Weighted Combination**: Integrates angular and radial asymmetry into a single scalar value.

2. **Precomputed Angular Error**: If `angle_error` is provided, the function uses it directly; otherwise, it calls `angular_symmetry()`, which handles rotational alignment internally.

3. **Rotation-Invariant**: Symmetric patterns yield the same score regardless of their orientation.

4. **Radius Normalization**: Radii are normalized by their mean to make symmetry comparable across fruits of different sizes.
   
5. **Minimum Locule Threshold**: Very small locules (below `min_radius_threshold` fraction of mean radius) are ignored to avoid noise.
   
6. **Effect of Number of Locules**: With few locules, small absolute deviations can produce large errors. Interpret results carefully for fruits with 2–3 locules.


<br> 

---
<div style="background-color:rgba(230, 242, 255, 0.6); padding:8px; border-radius:6px; text-align:left;">
  <h4 style="margin:0;">High-level image analysis:</h4>
</div>


<h4>Function: <code>analyze_fruits</code></h4>

**Description**:

End-to-end analyzer that measures **fruit morphology**, **pericarp geometry**, and **locule** properties from a set of contours, then renders an **annotated image** (). 

For each fruit contour, the function computes major/minor axes, rotated bounding box, compactness/circularity/solidity, inner pericarp area (from the locules), pericarp thickness, several **locule statistics** (area and circularity distributions), and **symmetry metrics** (angular, radial, and combined rotational symmetry). All visual elements (contours, centroids, axes, bounding boxes, labels) are drawn directly on the image during processing. If `plot=True`, the annotated image is displayed via `plot_image()` (e.g., in Jupyter or an interactive window).

This function **assumes** that:

1. `contours` contains the fruit and locule contours detected elsewhere.  
2. `fruit_locus_map` maps each fruit id to the indices of its locule contours.  
3. If `stamp=True`, the input image is expected to have black stamps on a white background.  
4. The label text must be enclosed within a rectangular box in the image in order to be detected correctly.  
5. The function expects the input image in **BGR format** (as returned by OpenCV).  Internally, the image is converted to **RGB** for annotation and output consistency.  


**Implementation Details**:

1. **Preprocessing & validation**
   * Copy input image (BGR) and convert internally to RGB for annotation.
   * Optionally invert colors for stamp images (`stamp=True`).
   * Validate pixel → cm scales: both must be positive and `px_per_cm_y > px_per_cm_x`. If not, raise `ValueError`.

2. **Fruit iteration & centroid lookup**

   * Precompute all fruit centroids with `calculate_fruit_centroids()`.
   * Iterate `fruit_locus_map`; skip `label_id` if provided.

3. **Fruit contour & drawing**

   * Retrieve the analysis contour via `get_fruit_contour()` using `contour_mode` and `epsilon_hull` by default.
   * Draw the fruit contour and centroid on the annotation canvas.

4. **Axes & rotated box**

   * Compute major/minor axes (in cm) with `calculate_axes()`; draw the fruit axes on the annotation canvas.
   * Compute the **rotated rectangle** via `rotate_box()`, returning (length, width) in px and cm; draw the box.

5. **Fruit morphology**

   * Call `get_fruit_morphology()` to obtain `fruit_area_cm2`, `fruit_perimeter_cm`, `fruit_circularity`, `fruit_solidity`, and `fruit_compactness`.
   * Compute **aspect ratio** from rotated box (`box_width_cm / box_length_cm`).

6. **Inner/outer pericarp area**

   * Estimate **inner pericarp area** (cm²) using `inner_pericarp_area()` over the locules of that fruit (optionally with ellipse fitting via `use_ellipse`); draw the inner pericarp contour.
   * Calculate the **outer pericarp area** (cm²) substracting `total_pericarp_area` - `inner_pericarp_area`.
 
   * Derive **average pericarp thickness** by circle-equivalent radii difference: `sqrt(outer_area/π) − sqrt(inner_area/π)`.

7. **Pericarp thickness**

   * **Regardless of the chosen contour mode** for fruit and inner pericarp, the method **approximates ellipses (oval proxies)** around both shapes for stability and precision.
   * The ratio of inner to outer ellipse areas is used to determine how much smaller the inner ellipse is.
   * This scaling is then applied to the ellipse semi-axes, and thickness is obtained by subtracting the inner from the outer semi-axes.
   * The method reports pericarp thickness as their average (`avg_pericarp_thickness_cm`).
 
8. **Locule preprocessing & merging**

   * Build `locules_data` from `precalculate_locules_data()`, filtering by `min_locule_area` (and `max_locule_area`, if set).
   * Merge close locules with `merge_locules()` using `max_dist` / `min_dist`; draw merged contours.

9. **Locule metrics**

   * Compute per-fruit summary stats: mean/std/CV of **locule area** and **locule circularity**; count (`n_locules`).
   * Draw each locule centroid for visualization.

10. **Symmetry metrics**

   * Compute `angular_symmetry()`, `radial_symmetry()`, and `rotational_symmetry()` using `num_shifts`, `angle_weight`, `radius_weight`, and `min_radius_threshold`.

11. **Derived ratios & packing**

    * `locules_density = n_locules / fruit_area_cm2`
    * `inner_area_ratio = inner_pericarp_area_cm2 / fruit_area_cm2`
    * `locule_area_ratio = max(locule_area) / min(locule_area)` (guarded)
    * `total_locule_area_cm2`, `% locule area` of fruit, and **locule packing efficiency** relative to inner pericarp.

12. **On-image text annotation**

    * Place a semi-transparent label box near the fruit’s bounding rectangle, showing id and number of locules.

13. **Plot & return**

    * If `plot=True`, display via `plot_image()`.
    * Return an `AnnotatedImage` object containing the annotated RGB image and a list of per-fruit metric dicts.
    * If `plot=False`, the annotated image is returned silenty without display.


**Arguments**:

*Required*:

* `img` (`np.ndarray`): Input image in **BGR** format.
* `contours` (`list[np.ndarray]`): All valid contours (fruits + locules + label).
* `fruit_locus_map` (`dict[int, list[int]]`): Maps fruit id → list of locule contour indices.
* `px_per_cm_x` (`float`): Pixels per centimeter along the **short** image side (width).
* `px_per_cm_y` (`float`): Pixels per centimeter along the **long** image side (length). Must be strictly greater than `px_per_cm_x`.
* `img_name` (`str`): Image identifier.
* `label_text` (`str`): Label to associate with results (e.g., cultivar/treatment).

*Optional*:

* `label_id` (`int | None`): Contour id of the label (excluded from fruit analysis).
* `contour_mode` (`str`): `'raw'` (use original) or polygonal approximation; controlled by `epsilon_hull`.
* `epsilon_hull` (`float`): Epsilon for polygon approximation of fruit/inner pericarp shapes.
* `min_locule_area` (`int`): Minimum area (px²) for a valid locule.
* `max_locule_area` (`int | None`): Maximum area (px²) for a valid locule; `None` disables upper filter.
* `max_dist` (`int`): Max distance (px) to merge two locules.
* `min_dist` (`int`): Min distance (px) to merge two locules.

**Symmetry**:

* `num_shifts` (`int`): Angular shifts to test during alignment in `angular_symmetry()`.
* `angle_weight` (`float`): Weight of angular error in `rotational_symmetry()`.
* `radius_weight` (`float`): Weight of radial error in `rotational_symmetry()`.
* `min_radius_threshold` (`float`): Fraction of mean radius to keep a locule in symmetry calcs.

**Inner pericarp**:

* `use_ellipse` (`bool`): Fit ellipses for inner pericarp instead of raw contours.
* `rel_tol` (`float`): Relative tolerance for isotropy checks when converting pixels → cm.

**Stamps**:

* `stamp` (`bool`): If `True`, invert image colors (useful for stamp-like inputs).

**Plotting & annotation**:

* `plot` (`bool`), `plot_size` (`tuple`), `font_scale` (`int`), `font_thickness` (`int`),
  `text_color` (`Tuple[int,int,int]`), `bg_color` (`Tuple[int,int,int])`, `padding` (`int`), `line_spacing` (`int`),
  `path` (`str | None`), `fig_axis` (`bool`), `title_fontsize` (`int`), `title_location` (`str`).


**Returns**:

`AnnotatedImage` object with:

* `annotated_img` (`np.ndarray`): RGB image with contours, centroids, principal axes, rotated box, and per-fruit text labels drawn.
* `results` (`list[dict]`): One dict per fruit containing:
  * `image_name` (`str`)
  * `label` (`str`)
  * `fruit_id` (`int`)
  * `n_locules` (`int`)
  * `major_axis_cm` (`float`)
  * `minor_axis_cm` (`float`)
  * `fruit_area_cm2` (`float`)
  * `fruit_perimeter_cm` (`float`)
  * `fruit_circularity` (`float`)
  * `fruit_aspect_ratio` (`float`)
  * `fruit_solidity` (`float`)
  * `fruit_compactness` (`float`)
  * `box_length_cm` (`float`)
  * `box_width_cm` (`float`)
  * `compactness_index` (`float`)
  * `inner_pericarp_area_cm2` (`float`)
  * `outer_pericarp_area_cm2` (`float`)
  * `avg_pericarp_thickness_cm` (`float`)
  * `mean_locule_area_cm2` (`float`) 
  * `std_locule_area_cm2` (`float`)
  * `total_locule_area_cm2` (`float`)
  * `cv_locule_area` (`float`)
  * `mean_locule_circularity` (`float`)
  * `std_locule_circularity` (`float`)
  * `cv_locule_circularity` (`float`)
  * `angular_symmetry` (`float`)
  * `radial_symmetry` (`float`)
  * `rotational_symmetry` (`float`)
  * `locules_density` (`float`)
  * `inner_area_ratio` (`float`)
  * `locule_area_ratio` (`float`)
  * `locule_area_percentage` (`float`)
  * `locule_packing_efficiency` (`float`)

<br> 

  For more details about these measurements, refer to the [Traits Analyzed](#traits-analyzed) section.


**Example**:

```python
annot = analyze_fruits(
    img=img,
    contours=contours,
    fruit_locus_map={0: [5,6,7], 1: [12,13]},
    px_per_cm_x=30.0,
    px_per_cm_y=31.2,
    img_name="slice_03.jpg",
    label_text="Cultivar A",
    label_id=99,
    contour_mode='raw',
    epsilon_hull=0.001,
    min_locule_area=300,
    max_locule_area=None,
    max_dist=30,
    min_dist=2,
    num_shifts=500,
    angle_weight=0.5,
    radius_weight=0.5,
    min_radius_threshold=0.1,
    use_ellipse=False,
    rel_tol=1e-6,
    stamp=False,
    plot=True,
    plot_size=(20,10)
)

print(len(annot.table), "fruits analyzed")
# Access first fruit's metrics:
print(annot.table[0]["major_axis_cm"], annot.table[0]["rotational_symmetry"])
```


**Notes**:

1. **Pixel scale constraints**: The function enforces `px_per_cm_y > px_per_cm_x` and both > 0; otherwise it raises `ValueError`.
2. **Anisotropic scaling**: Conversions to cm² use **axis-specific** scale factors to correctly handle non-square pixels.
3. **Locule merging**: `merge_locules()` helps consolidate nearby locule contours; tune `max_dist`/`min_dist` per imaging setup.
4. **Thickness approximation**: Pericarp thickness uses circle-equivalent radii; it’s a robust scalar, not a spatial thickness map.
5. **Symmetry interpretation**: With few locules (2–3), deviations can inflate error—interpret symmetry scores with care.
6. **Rendering**: The function draws fruit and locule contours, centroids, major/minor axes, rotated box, and a compact info box (`id`, `n loc`).
7. **Performance**: Core geometry relies on NumPy/OpenCV vectorization; heavy steps (e.g., symmetry alignment with many shifts) may be tuned via `num_shifts`.



----

#### **Examples**



### 2.4 Module: Color Correction

<div style="background-color:#D8BFD8; padding:8px; border-radius:6px; 
text-align:center;">

#### **User-Facing Classes**

</div>

<br>

<div style="background-color:#DAA520; padding:8px; border-radius:6px; 
text-align:center;">

#### **Advanced Utilities**

</div>

<br>

<div style="background-color:#BC8F8F; padding:8px; border-radius:6px; 
text-align:center;">

#### Helper Functions

</div>

<br>

<div style="background-color:#e6f2ff; padding:8px; border-radius:6px; 
text-align:center;">

#### Internal/Core Utilities

</div>

---

## 3. Web Interface

### 3.1 Access and requirements

### 3.2 Navigation and main features

### 3.3 Common use cases

### 3.4 Limitations and considerations

---

## 4. Command Line Interface (CLI)

### 4.1 CLI installation

### 4.2 Available commands and syntax

### 4.3 Usage examples

### 4.4 Parameters and advanced options

---

## 5. FAQ and Support/Collaboration

* Frequently asked questions
* Support contact information

## 6. Citation
---

## 7. License

---

## 8. Tutorials 

- Python API tutorials
- Web interface guides
- CLI usage examples

---



## Syllabus

<h4 id="locule-symmetry">Locule symmetry</h3>

![alt text](image-1.png)

<h4 id="isotropic-pixels">Isotropic pixels</h3>

Non-uniform scaling just means that different scales are applied to each dimension, making it anisotropic. The opposite would be isotropic scaling, where the same scale is applied to each dimension.

<h4 id="anisotropic-pixels">Anisotropic pixels</h3>
<p>Píxeles con la misma densidad en X e Y...</p>


