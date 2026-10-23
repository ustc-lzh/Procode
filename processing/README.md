# Image processing

- `image_normalization.py`: percentile background estimation, flat-field correction and intensity normalization.
- `decoding.py`: binary codeword conversion, nearest-codeword lookup and spatial spot merging.
- `cell_segmentation.py`: intensity scaling, label expansion and assignment of transcript coordinates to cell labels.

Functions accept image arrays, spot tables or label arrays. Coordinates and image dimensions must use the same pixel frame. Registration, spot detection and nuclear segmentation are performed with the external imaging software described in Methods; these functions do not replace those packages.

The original whole-image workflows also require `share_code_2d`, `make_ashlar` and `h5ad_make_whole`, which are not included.
