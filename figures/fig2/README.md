# Fig. 2 and Supplementary Fig. 3

Summary tables are provided in [fig2_data.xlsx](fig2_data.xlsx).

| Script | Purpose |
| --- | --- |
| [analysis.py](analysis.py) | Run the codebook comparison and detection analyses for Fig. 2 and Supplementary Fig. 3. |
| [crowding.py](crowding.py) | Compare predicted imaging-bit loads and optical crowding between the two codebooks. |
| [detection_metrics.py](detection_metrics.py) | Compare per-cell transcript detection and regional signal specificity between the two codebooks. |
| [regional_validation.py](regional_validation.py) | Compare regional expression with the Allen atlas and write independent Supplementary Fig. 3 brain-region, expression-map and paired-correlation panels. |
| [data_io.py](data_io.py) | Read workbook tables, H5AD expression matrices and brain-region annotations. |
