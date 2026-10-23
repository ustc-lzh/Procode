# Fig. 1 and Supplementary Fig. 2

Summary tables are provided in [fig1_data.xlsx](fig1_data.xlsx). Spatial maps also use external cell-level H5AD files.

| Script | Purpose |
| --- | --- |
| [probe_number_response.py](probe_number_response.py) | Summarize reference datasets and compare signal gain and background across probe numbers (Fig. 1B, C and E). |
| [probe_allocation.py](probe_allocation.py) | Plot the expression-based probe-number allocation model and gene assignments (Fig. 1F). |
| [prepare_probe_allocation.py](prepare_probe_allocation.py) | Check the expression reference and probe assignments used in Fig. 1F. |
| [recompute_regional_response.py](recompute_regional_response.py) | Recalculate Fig. 1C and E metrics from regional transcript counts. |
| [export_cell_level_data.py](export_cell_level_data.py) | Copy original H5AD files and check regional transcript totals against the workbook. |
| [supplementary/brain_region_maps.py](supplementary/brain_region_maps.py) | Plot brain regions, selected expression regions and signal changes for Supplementary Fig. 2A-C. |
| [supplementary/nearest_neighbour_plot.py](supplementary/nearest_neighbour_plot.py) | Plot nearest-neighbour distance boxplots from workbook summary statistics (Supplementary Fig. 2D). |
| [supplementary/spatial_inputs.py](supplementary/spatial_inputs.py) | Read cell coordinates and brain-region labels from H5AD files, with optional NPZ caching. |
| [data_io.py](data_io.py) | Load workbook tables and provide shared analysis and plotting helpers. |
