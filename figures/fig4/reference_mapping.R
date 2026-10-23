suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})


usage <- function() {
  paste(
    "Usage:",
    "Rscript reference_mapping.R",
    "--reference reference.rds --query query.rds --output-dir output_dir",
    "[--mode pilot|full] [--seed 20250821]"
  )
}


parse_cli <- function(values) {
  result <- list(mode = "full", seed = 20250821L)
  i <- 1L
  while (i <= length(values)) {
    flag <- values[[i]]
    if (!flag %in% c("--reference", "--query", "--output-dir", "--mode", "--seed")) {
      stop(usage())
    }
    if (i == length(values)) stop(usage())
    key <- sub("^--", "", flag)
    result[[key]] <- values[[i + 1L]]
    i <- i + 2L
  }
  required <- c("reference", "query", "output-dir")
  if (!all(required %in% names(result))) stop(usage())
  if (!result$mode %in% c("pilot", "full")) stop("--mode must be pilot or full")
  result$seed <- as.integer(result$seed)
  if (is.na(result$seed)) stop("--seed must be an integer")
  result
}


get_counts <- function(object, genes, cells) {
  assay <- DefaultAssay(object)
  counts <- tryCatch(
    GetAssayData(object, assay = assay, layer = "counts"),
    error = function(...) GetAssayData(object, assay = assay, slot = "counts")
  )
  shared <- intersect(genes, rownames(counts))
  counts[shared, cells, drop = FALSE]
}


pick_annotation_column <- function(metadata) {
  candidates <- c("annotation", "celltype", "cell_type", "predicted.id")
  found <- candidates[candidates %in% colnames(metadata)]
  if (length(found) == 0L) {
    stop("The reference metadata needs annotation, celltype or predicted.id.")
  }
  found[[1L]]
}


pick_prediction_column <- function(metadata) {
  candidates <- c("predicted.annotation", "predicted.id")
  found <- candidates[candidates %in% colnames(metadata)]
  if (length(found) == 0L) {
    found <- grep("^predicted\\.", colnames(metadata), value = TRUE)
  }
  if (length(found) == 0L) stop("MapQuery did not produce a predicted annotation column.")
  found[[1L]]
}


main <- function() {
  args <- parse_cli(commandArgs(trailingOnly = TRUE))
  set.seed(args$seed)
  dir.create(args[["output-dir"]], recursive = TRUE, showWarnings = FALSE)

  reference_full <- UpdateSeuratObject(readRDS(args$reference))
  query_full <- UpdateSeuratObject(readRDS(args$query))
  if ("RNA" %in% Assays(reference_full)) DefaultAssay(reference_full) <- "RNA"
  if ("RNA" %in% Assays(query_full)) DefaultAssay(query_full) <- "RNA"

  reference_annotation <- pick_annotation_column(reference_full[[]])
  shared_genes <- intersect(rownames(reference_full), rownames(query_full))
  if (length(shared_genes) < 100L) {
    stop("Fewer than 100 shared genes are available for panel-matched mapping.")
  }

  reference_cells <- colnames(reference_full)
  query_cells <- colnames(query_full)
  if (args$mode == "pilot") {
    reference_meta <- reference_full[[]]
    reference_meta$cell_id <- rownames(reference_meta)
    groups <- split(reference_meta$cell_id, as.character(reference_meta[[reference_annotation]]))
    reference_cells <- unlist(lapply(groups, function(cells) sample(cells, min(length(cells), 500L))), use.names = FALSE)
    query_cells <- sample(query_cells, min(length(query_cells), 5000L))
  }

  reference_counts <- get_counts(reference_full, shared_genes, reference_cells)
  query_counts <- get_counts(query_full, shared_genes, query_cells)
  reference_meta <- reference_full[[]][reference_cells, , drop = FALSE]
  query_meta <- query_full[[]][query_cells, , drop = FALSE]
  reference_meta[[reference_annotation]] <- as.character(reference_meta[[reference_annotation]])

  reference <- CreateSeuratObject(
    counts = reference_counts,
    meta.data = reference_meta,
    project = "GBM_scRNA_reference",
    min.cells = 0,
    min.features = 0
  )
  query <- CreateSeuratObject(
    counts = query_counts,
    meta.data = query_meta,
    project = "BASSFISH_20250821",
    min.cells = 0,
    min.features = 0
  )
  rm(reference_full, query_full, reference_counts, query_counts)
  invisible(gc())

  panel_scale_factor <- 100
  reference <- NormalizeData(reference, normalization.method = "LogNormalize", scale.factor = panel_scale_factor, verbose = FALSE)
  query <- NormalizeData(query, normalization.method = "LogNormalize", scale.factor = panel_scale_factor, verbose = FALSE)
  VariableFeatures(reference) <- shared_genes
  VariableFeatures(query) <- shared_genes
  reference <- ScaleData(reference, features = shared_genes, verbose = FALSE)
  query <- ScaleData(query, features = shared_genes, verbose = FALSE)

  npcs <- min(30L, length(shared_genes) - 1L)
  dims_use <- seq_len(min(25L, npcs))
  reference <- RunPCA(reference, features = shared_genes, npcs = npcs, approx = TRUE, seed.use = args$seed, verbose = FALSE)
  reference <- RunUMAP(
    reference,
    reduction = "pca",
    dims = dims_use,
    n.neighbors = 30,
    metric = "cosine",
    return.model = TRUE,
    seed.use = args$seed,
    verbose = FALSE
  )
  if (is.null(reference[["umap"]]@misc$model)) stop("The reference UMAP model was not retained.")

  anchors <- FindTransferAnchors(
    reference = reference,
    query = query,
    normalization.method = "LogNormalize",
    reference.reduction = "pca",
    features = shared_genes,
    dims = dims_use,
    k.anchor = 10,
    k.filter = 200,
    k.score = 30,
    verbose = FALSE
  )
  query <- MapQuery(
    anchorset = anchors,
    reference = reference,
    query = query,
    refdata = list(annotation = reference_annotation),
    reference.reduction = "pca",
    reduction.model = "umap",
    verbose = FALSE
  )
  if (!"ref.umap" %in% Reductions(query)) stop("MapQuery did not produce ref.umap.")

  reference_umap <- Embeddings(reference, reduction = "umap")
  query_umap <- Embeddings(query, reduction = "ref.umap")
  if (any(!is.finite(reference_umap)) || any(!is.finite(query_umap))) stop("Non-finite UMAP coordinates detected.")
  prediction_column <- pick_prediction_column(query[[]])
  score_candidates <- c(
    "predicted.annotation.score",
    "prediction.score.max",
    paste0(prediction_column, ".score")
  )
  score_column <- score_candidates[score_candidates %in% colnames(query[[]])]
  score_column <- if (length(score_column)) score_column[[1L]] else NA_character_

  reference_export <- data.frame(
    cell_id = rownames(reference_umap),
    UMAP_1 = reference_umap[, 1],
    UMAP_2 = reference_umap[, 2],
    dataset = "GBM scRNA reference",
    celltype = as.character(reference[[reference_annotation, drop = TRUE]]),
    stringsAsFactors = FALSE
  )
  query_export <- data.frame(
    cell_id = rownames(query_umap),
    UMAP_1 = query_umap[, 1],
    UMAP_2 = query_umap[, 2],
    dataset = "0821 BASS-FISH",
    celltype = as.character(query[[prediction_column, drop = TRUE]]),
    prediction_score = if (!is.na(score_column)) as.numeric(query[[score_column, drop = TRUE]]) else NA_real_,
    stringsAsFactors = FALSE
  )
  output_dir <- args[["output-dir"]]
  write.csv(reference_export, file.path(output_dir, "reference_panel_umap.csv"), row.names = FALSE)
  write.csv(query_export, file.path(output_dir, "query_0821_projected_umap.csv"), row.names = FALSE)
  write.csv(
    rbind(reference_export, transform(query_export, prediction_score = NULL)),
    file.path(output_dir, "combined_reference_query_umap.csv"),
    row.names = FALSE
  )
  writeLines(shared_genes, file.path(output_dir, "shared_panel_genes.txt"))
  write.csv(
    as.data.frame(table(reference_export$celltype, useNA = "ifany")),
    file.path(output_dir, "reference_annotation_counts.csv"),
    row.names = FALSE
  )
  write.csv(
    as.data.frame(table(query_export$celltype, useNA = "ifany")),
    file.path(output_dir, "query_prediction_counts.csv"),
    row.names = FALSE
  )

  score_values <- query_export$prediction_score
  summary_table <- data.frame(
    metric = c(
      "mode", "seed", "reference_cells", "query_cells", "shared_genes",
      "n_reference_annotations", "n_query_predicted_annotations",
      "n_transfer_anchors", "prediction_column", "score_column",
      "prediction_score_mean", "prediction_score_median",
      "reference_umap_model_retained"
    ),
    value = c(
      args$mode,
      args$seed,
      nrow(reference_export),
      nrow(query_export),
      length(shared_genes),
      length(unique(reference_export$celltype)),
      length(unique(query_export$celltype)),
      nrow(anchors@anchors),
      prediction_column,
      ifelse(is.na(score_column), "", score_column),
      if (all(is.na(score_values))) NA_real_ else mean(score_values, na.rm = TRUE),
      if (all(is.na(score_values))) NA_real_ else median(score_values, na.rm = TRUE),
      TRUE
    ),
    stringsAsFactors = FALSE
  )
  write.table(summary_table, file.path(output_dir, "run_summary.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
  message("Completed panel-matched reference mapping: ", output_dir)
}


main()
