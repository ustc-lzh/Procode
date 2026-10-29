# Fig. 4 and Supplementary Fig. 6

Summary tables are provided in [fig4_data.xlsx](fig4_data.xlsx). The workbook
also lists the external cell metadata, reference UMAP, transcript-position and
ROI files needed for the spatial and Supplementary Fig. 6 panels. Pass their
containing directory or ZIP archive with `--data-root`.

| Script | Purpose |
| --- | --- |
| [reference_projection.py](reference_projection.py) | Run Fig. 4 analyses from existing annotations and mapping results. |
| [reference_mapping.R](reference_mapping.R) | Perform Seurat reference mapping and label transfer between reference and query objects. |
| [cell_type_maps.py](cell_type_maps.py) | Plot cell-type annotations in tissue space and UMAP. |
| [spatial_expression.py](spatial_expression.py) | Plot the tissue locations of selected marker-gene transcripts. |
| [spatial_domains.py](spatial_domains.py) | Plot spatial domains D1–D5 and enlarged views of selected tissue regions. |
| [gene_expression.py](gene_expression.py) | Match transcript counts to selected cells and normalize gene expression. |
| [marker_expression.py](marker_expression.py) | Calculate cell-type marker expression and detection rates, and draw the marker dot plot. |
| [domain_composition.py](domain_composition.py) | Calculate and plot the cell-type composition of each spatial domain. |
| [supplementary_analysis.py](supplementary_analysis.py) | Plot cell-quality distributions, reference mapping and domain composition for Supplementary Fig. 6. |
| [data_io.py](data_io.py) | Read workbook tables, cell annotations, transcript counts and spatial coordinates. |
