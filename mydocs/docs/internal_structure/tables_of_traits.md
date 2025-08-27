# Traits Description

---

Table 1. Internal structure traits obtained by Traitly.

| Column                      | Trait description (type & range)                                                                                  | Function                                       |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| `image_name`                | Input image name (string)                                                                                         | — (metadata)                                   |
| `label`                     | User-defined label (string)                                                                                       | — (metadata)                                   |
| `fruit_id`                  | Sequential fruit identifier (int ≥1)                                                                              | `core.analyze_fruits`                          |
| `n_locules`                 | Number of detected locules (int ≥0)                                                                               | `core.analyze_fruits`                          |
| `major_axis_cm`             | Longest distance across fruit (float, cm >0)                                                                      | `core.calculate_axes`                          |
| `minor_axis_cm`             | Maximum thickness perpendicular to major axis (float, cm >0)                                                      | `core.calculate_axes`                          |
| `fruit_area_cm2`            | Total fruit area (float, cm² >0)                                                                                  | `core.get_fruit_morphology`                    |
| `fruit_perimeter_cm`        | Fruit perimeter length (float, cm >0)                                                                             | `core.get_fruit_morphology`                    |
| `fruit_circularity`         | Roundness index (float, 0–1; 1 = circle)                                                                          | `core.get_fruit_morphology`                    |
| `fruit_aspect_ratio`        | Width ÷ length ratio (float, 0–1; 1 = circle, <1 = elongated)                                                     | `core.analyze_fruits` + `core.rotate_box`      |
| `fruit_solidity`            | Area ÷ convex hull area (float, 0–1)                                                                              | `core.get_fruit_morphology`                    |
| `fruit_compactness`         | Shape irregularity (float, >0; higher = less compact)                                                             | `core.get_fruit_morphology`                    |
| `box_length_cm`             | Rotated bounding box larger side (length) (float, cm > 0)                                                         | `core.rotate_box`                              |
| `box_width_cm`              | Rotated bounding box shorter side (width) (float, cm > 0)                                                         | `core.rotate_box`                              |
| `compactness_index`         | Fruit area ÷ bounding box area (float, 0–1)                                                                       | `core.analyze_fruits`                          |
| `inner_pericarp_area_cm2`   | Area enclosing all locules (float, cm² ≥0)                                                                        | `core.inner_pericarp_area`                     |
| `outer_pericarp_area_cm2`   | Flesh area (fruit total area − inner pericarp area) (float, cm² ≥0)                                               | `core.analyze_fruits`                          |
| `avg_pericarp_thickness_cm` | Avg. thickness between inner & outer ellipses (float, cm ≥0)                                                      | `core.analyze_fruits`                          |
| `mean_locule_area_cm2`      | Mean locule area (float, cm² ≥0)                                                                                  | `core.precalculate_locules_data` + `core.analyze_fruits` |
| `std_locule_area_cm2`       | Standard deviation of locule areas (float, cm² ≥0)                                                                | `core.precalculate_locules_data` + `core.analyze_fruits` |
| `total_locule_area_cm2`     | Sum of all locule areas (float, cm² ≥0)                                                                           | `core.precalculate_locules_data` + `core.analyze_fruits` |
| `cv_locule_area`            | Coefficient of variation in locule area (float, ≥0; unitless CV)                                                  | `core.precalculate_locules_data` + `core.analyze_fruits` |
| `mean_locule_circularity`   | Avg. locule roundness (float, 0–1)                                                                                | `core.precalculate_locules_data`               |
| `std_locule_circularity`    | Variation in locule circularity (float, 0–1)                                                                      | `core.precalculate_locules_data`               |
| `cv_locule_circularity`     | Coefficient of variation of locule circularity (float, ≥0)                                                        | `core.precalculate_locules_data`               |
| `angular_symmetry`          | Deviation of **locule centroids** from perfect angular spacing<br>around the **fruit centroid** (float, rad ≥0; 0 = perfect) | `core.angular_symmetry`   |
| `radial_symmetry`           | Variation in radial distances of **locule centroids**<br>from the **fruit centroid** (float, CV ≥0; 0 = perfect)   | `core.radial_symmetry`    |
| `rotational_symmetry`       | Combined measure of angular and radial symmetry<br>of **locule centroids** relative to the **fruit centroid**<br>(float, 0–1; 0 = perfect) | `core.rotational_symmetry` |
| `locules_density`           | Locules per cm² of fruit (float, ≥0)                                                                              | `core.analyze_fruits`                          |
| `inner_area_ratio`          | Inner pericarp ÷ fruit area (float, 0–1)                                                                          | `core.analyze_fruits`                          |
| `locule_area_ratio`         | Largest ÷ smallest locule area (float, ≥1 if >1 locule)                                                           | `core.analyze_fruits`                          |
| `locule_area_percentage`    | % fruit area occupied by locules (float, 0–100%)                                                                  | `core.analyze_fruits`                          |
| `locule_packing_efficiency` | % of inner pericarp filled with locules (float, 0–100%)                                                           | `core.analyze_fruits`                          |

