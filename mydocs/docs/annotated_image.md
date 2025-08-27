
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

