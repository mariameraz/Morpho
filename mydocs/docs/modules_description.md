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