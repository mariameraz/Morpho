
---

# Internal/Core Functions

--- 

## <code>px_per_cm()</code>

<h2> Description: </h2>

Calculates pixel density (pixels/cm) from scanned objects, including **documents** (paper sheets) or **biological speciments** (fruit slices/leaves) placed directly on the scanner. Uses physicl dimension to establish precise scale. Automatically determines image orientation (portrait or landscape) and treats the **longest** physical side as **length** and the **shortest** as **width**, regardless of input orientation. **When documents uses as input, For precise measurements, always use the scanner-reported dimensions rather than theorical paper size.**

<h2> Implementation Details: </h2>

**Image Dimensions (cm)**:

  * The function uses the actual scanned dimensions (including any automatic cropping or optical distorions).
  * Provides independent X/Y scaling to handle non-square pixels.
  * The physical dimensions must satisfy `length_cm ≥ width_cm`. If using predefined sizes (e.g., `'a4_iso'`), this condition is enforced automatically.
  * If custom dimensions are used and `width_cm > length_cm`, a `ValueError` is raised.

**Image Dimensions (pixels)**:

  * Uses the shape of the image array to determine image orientation.
  * Automatically handles orientation:

    * `length_px` = longer side (height/Y-axis)
    * `width_px` = shorter side (width/X-axis)
    
    _For example:_

      * If the image shape is 3500 px × 2500 px (portrait), then: `length_px = 3500` and `width_px = 2500`
      * If the image shape is 2500 px × 3500 px (landscape), then, the function swaps the values internally so that: `length_px = 3500` and `width_px = 2500`
      

**Density Calculation**:

  * `px_per_cm_width = width_px / width_cm`
  * `px_per_cm_length = length_px / length_cm`
  
    _These represent the **pixel density** along the width and length respectively._


!!! note "**Parameters**"

    **Required**:

    * `img` (numpy.ndarray): Input image (2D grayscale or 3D BGR).

    **Optional**:

    * `size` (str): Predefined paper size. Options:
        * `'letter_ansi'` (default): 21.6 × 27.9 cm
        * `'legal_ansi'`: 21.59 × 35.56 cm
        * `'a4_iso'`: 21.0 × 29.7 cm
        * `'a3_iso'`: 29.7 × 42.0 cm

    * `width_cm` (`float`): Physical width (shorter side, cm). Requires `length_cm`.
    * `length_cm` (`float`): Physical length (longer side, cm). Requires `width_cm`.

!!! tip "**Returns**"

    `Tuple[float, float, float, float]`:

    - `px_per_cm_width` (`float`): Pixels per centimeter along the **shorter side** of the image (width).  
    - `px_per_cm_length` (`float`): Pixels per centimeter along the **longer side** of the image (length).  
    * `used_width_cm`: Validated width (cm)
    * `used_length_cm`: Validated length (cm)

!!! warning "**Raises**:"

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

!!! danger "**Notes**:"

    1. **Image Size Validation**: Scanner cropping typically affects paper dimensions by \~1–3%. Check the image metadata or scanner information to extract the actual scanned image dimensions. When analyzing stamps, if scanner dimensions are unavailable, you can use the actual paper dimensions. However, keep in mind that this may introduce a minimal error (±1–3% in most cases).

    2. **Image Requirements**: The image must be 2D (grayscale) or 3D (e.g. BGR);  Although a BGR image is expected in the pipeline, the function itself does not depend on the specific color format (e.g. BGR, RGB, HSV, Lab, etc.), only on the array shape.
    3. **Aspect Ratio**: In most cases, an equal X/Y pixel density is expected. However, this function calculates independent X/Y scaling factors (`px_per_cm_width`, `px_per_cm_length`) to handle cases where pixel-to-cm ratios differ between axes (due to non-square pixels, optical distortions, or anisotropic processing), ensuring precise dimensional measurements.
    4. **Dimension Rules**: Negative or zero dimensions are not allowed.
    5. **Orientation Handling**: The `px_per_cm` function does **not** rotate the image—only interprets orientation based on dimensions.


<br>

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>


## <code>create_mask()</code></h4>

<h2> Description: </h2>
Creates a binary mask to segment objects from an HSV image using color thresholding, morphological operations, and edge detection. Designed for flexible background removal with customizable processing steps.

<h2> Implementation Details: </h2>

The function combines multiple computer vision techniques:

**Color Thresholding**: 

- Uses HSV bounds to isolate background/foreground with `cv2.inRange()`.

**Morphological Refinement**: 

- Applies opening (noise removal) and closing (hole filling) with elliptical kernels.
- Supports separate kernel sizes for opening/closing operations.

**Edge Enhancement**: 

- Blends Canny edges with the thresholded mask for precise boundaries.


!!! info "**Parameters**"
    **Required**:  

    - `img_hsv` (`numpy.ndarray`): Input image in HSV format (shape: `(H,W,3)`, dtype: `uint8`).  

    **Optional**:  

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

!!! tip "**Returns**"

    - `numpy.ndarray`: Binary mask (shape: `(H,W)`, dtype: `uint8`) where:  
        - `255` (white): Foreground pixels  
        - `0` (black): Background pixels

!!! warning "**Raises**:"
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

!!! danger "**Notes**:"

     1. **HSV Defaults**: Default bounds target black/dark backgrounds (`V ≤ 30`).  
     2. **Kernel Strategy**:  
        - Smaller `kernel_open` preserves fine details.  
        - Larger `kernel_close` fills bigger holes.  
     3. **Edge Detection**: Canny thresholds (`canny_min/max`) should be tuned for edge sensitivity.  
     4. **Performance**: For batch processing, set `plot=False`.  




<br>

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>



## <code>get_fruit_contour()</code>

<h2> Description: </h2>

Retrieves the contour of a fruit from a list of contours and optionally transforms it into a simplified or fitted representation.

The function supports **raw contours**, **convex hulls**, **polygonal approximation**, and **ellipse fitting**, depending on the selected `contour_mode`.


<h2>Implementation Details: </h2>

**Raw contour (`'raw'`)**

   * Returns the original contour exactly as detected by `cv2.findContours()`.

**Convex hull (`'hull'`)**

   * Computes the convex hull using `cv2.convexHull()`.
   * Ensures the contour is convex and tightly encloses the fruit, potentially removing concave indentations.

**Polygonal approximation (`'approx'`)**

   * Uses `cv2.arcLength()` to measure contour perimeter.
   * Applies `cv2.approxPolyDP()` with tolerance `epsilon = max(1.0, epsilon_hull × perimeter)` to simplify the contour into fewer vertices.
   * Useful for reducing noise and representing smoother polygonal shapes.

**Ellipse fitting (`'ellipse'`)**

   * Requires at least 5 points.
   * Fits an ellipse using `cv2.fitEllipse()`.
   * Converts the fitted ellipse into a discrete polygon (`cv2.ellipse2Poly()`), returning a contour-like representation of the ellipse.
   * Good for regularizing highly irregular contours.

!!! note "**Parameters:**"

    **Required**:

    * `contours` (`List[np.ndarray]`): List of contours, typically from `cv2.findContours()`.
    * `fruit_id` (`int`): Index of the contour in the list corresponding to the fruit of interest.

    **Optional**:

    * `contour_mode` (`str`): Mode for contour representation. Options:

        * `'raw'`: Original contour (default).
        * `'hull'`: Convex hull.
        * `'approx'`: Polygonal approximation.
        * `'ellipse'`: Ellipse fit.
        
    * `epsilon_hull` (`float`): Factor controlling approximation tolerance (default = `0.0001`). Higher values = more simplification.


!!! tip "**Returns**"

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

!!! danger "**Notes**"

    1. **Contour format**: All outputs follow OpenCV’s contour format `(N, 1, 2)`.
    2. **Approximation parameter**: The `epsilon_hull` value strongly influences polygonal simplification.
    3. **Ellipse fitting**: If fewer than 5 points are available, ellipse mode falls back to the raw contour.




<br>

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>



## <code>merge_locules</code></h4>


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

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>




##  <code>find_fruits</code>

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

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>


## Function: <code>get_fruit_morphology</code>

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

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>



## <code>calculate_axes</code>

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

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>


## <code>rotate_box</code>

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

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>



##<code>inner_pericarp_area</code></h4>

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

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>


## <code>calculate_fruit_centroids</code></h4>

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

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>


## <code>precalculate_locules_data</code></h4>


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

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>



## <code>angular_symmetry()</code>

<h2> Description: </h2>  
Quantifies the **angular symmetry** of locules arranged around the fruit centroid (reference point) by comparing actual locule angles with an ideal symmetric arrangement. The function searches for the **rotational alignment** that minimizes the mean angular deviation. It uses the **precomputed angles** from `precalculate_locules_data()`; only **locule centroid positions** are considered. **Shape and size of locules are ignored**.

A perfectly symmetric configuration (e.g., locules equidistantly distributed in a circular pattern) returns a value close to `0.0`. Higher values indicate greater deviations from ideal angular symmetry. The metric is expressed in **radians** and is not normalized.

For a visual explanation of angular symmetry, refer to <a href='#locule-symmetry'>Figure 1A</a>.


<h2> Implementation Details: </h2>

**Angle Extraction**:

   * For each locule, extracts the angle from the reference centroid (`polar_coord[0]`) in radians.
   * Normalizes angles to `[0, 2π)` to maintain circular continuity.

**Rotation Adjustment**:

   * Computes the **circular mean** of all angles (`circmean`) to center the pattern.
   * Subtracts this mean from each angle to remove overall rotation bias, then wraps back to `[0, 2π)`.

**Ideal Symmetric Angles**:

   * For $n$ locules, generates `n` equally spaced angles covering the full circle:

     $$
     \theta_k^{\text{ideal}} = \frac{2\pi}{n} \cdot k, \quad k = 0, \dots, n-1
     $$
   * Represents the angular positions of locules in a perfectly symmetric circular arrangement.

**Error Minimization Across Rotations**:

   * Tests `num_shifts` rotations of the ideal angles to find the best alignment with the actual locule angles.
   * Computes the **angular difference matrix**, accounting for circular wrap-around:

    $$\text{diff} = \min(|\theta_{\text{actual}} - \theta_{\text{ideal}}|, 2\pi - |\theta_{\text{actual}} - \theta_{\text{ideal}}|)$$

   * Uses the **Hungarian algorithm** (`linear_sum_assignment`) to optimally assign observed locules to ideal angles, minimizing the mean angular error.
   * Keeps the **minimum mean angular error** across all tested rotations (`best_error`).

**Return Value**:

   * Returns `best_error` in **radians**. Smaller values indicate better angular symmetry.
   * Returns `np.nan` if fewer than 2 locules are provided.



!!! note "**Parameters**"

    **Required**

    - `locules_data (List[Dict])`: List of locule data dictionaries. Each must contain `polar_coord = (angle, radius)`.

    **Optional**

    - `num_shifts (int)`: Number of rotational shifts to test when aligning ideal  
      to actual angles (default = 500).

!!! tip "**Returns**"
    - `float`: Mean angular deviation (radians), where:
        - `0.0` → Perfect symmetry  
        - `>0.0` → Angular error in radians  
        - `np.nan` → Undefined if fewer than 2 locules are provided


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

!!! danger "**Notes**"

    1. **Rotation-Invariant**: Aligns the ideal pattern to actual locules, so rotated but symmetric arrangements yield low error.  
    2. **No Distance Constraint**: Radial distances from the center are ignored (`radial_symmetry()`).  
    3. **Collinearity**: If locules are nearly collinear, angular error increases.  
    4. **Effect of Number of Locules**: The angular symmetry calculation is sensitive to the number of locules. Configurations with very few locules can yield large angular errors even when absolute angular deviations are small. Care should be taken when interpreting angular symmetry for fruits with a low number of locules (2–3).


<br>

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>

## <code>radial_symmetry()</code>

<h2> Description: </h2>

Calculates the **radial symmetry** of locules using the **coefficient of variation (CV)** of their distances from each locule centroid to the fruit centroid (reference point). The CV measures the relative variability of these distances, where `0.0` indicates perfect radial symmetry (all locules equidistant from the centroid), and higher values indicate increasing irregularity. This function uses the **precomputed radii** from `precalculate_locules_data()`; only **locule centroid positions** are considered. **Shape and size of locules are ignored**.

For a visual explanation of angular symmetry,  refer to <a href='#locule-symmetry'>Figure 1B</a>.


<h2> Implementation Details: </h2>

**Distance Extraction**:

   * For each locule, extract the precomputed radius from the reference centroid (`polar_coord[1]`), in pixels.

**Symmetry Evaluation**:

   * If fewer than two locules are provided, return `np.nan` since radial symmetry cannot be defined.

   * Compute the **coefficient of variation (CV)** of the radii:

     $$
     CV = \frac{\text{standard deviation of radii}}{\text{mean radius}}
     $$

   * A CV of 0 indicates perfect radial symmetry (all locules equidistant).

   * Larger CV values indicate greater radial asymmetry.


!!! note "**Parameters**"

    *Required*:

    * `locules_data` (`List[Dict]`): List of locule data dictionaries, **as returned by `precalculate_locules_data()`**. Each must contain:

    * `'centroid'`: `(x, y)` coordinates of the locule centroid.
    * `'polar_coord'`: `(angle, radius)` precomputed polar coordinates relative to the fruit centroid.

!!! note "**Returns**"

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

!!! note "**Notes**"

    **Radial vs Angular Symmetry**: This function evaluates only radial (distance) symmetry, not angular spacing (see `angular_symmetry()`).

    **Rotation Invariance**: The radial distance CV is invariant to rotation because it depends only on the distances from locules to the centroid, which remain unchanged under rotation. 

    **Coefficient of Variation Use**: CV normalizes the standard deviation by the mean, making it scale-independent and useful for comparing radial symmetry across fruits of different sizes (e.g., small vs. large fruits), which correspond to varying numbers of pixels in image analysis.

    **Effect of Number of Locules**: CV does **not** directly account for the number of locules. Very few locules (e.g., 2 or 3) can produce high CV values even with small absolute differences. Interpret CV cautiously for low locule counts.

<br>



<br>

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>


## <code>rotational_symmetry</code></h4>

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

<hr style="border: 1.5px solid #a0a0a0ff;">

<br>


## <code>analyze_fruits</code></h4>

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


