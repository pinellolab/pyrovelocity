library(dplyr)
library(dyngen)
library(anndata)
library(parallel)

generate_simulated_data <- function(
    backbone_function_name,
    num_cells,
    num_targets,
    num_hks,
    num_simulations,
    tau,
    verbose = TRUE
) {
  set.seed(12)
  
  backbone <- do.call(backbone_function_name, list())
  
  config <- initialise_model(
    backbone = backbone,
    num_cells = num_cells,
    num_tfs = nrow(backbone$module_info),
    num_targets = num_targets,
    num_hks = num_hks,
    verbose = verbose,
    download_cache_dir = tools::R_user_dir("dyngen", "data"),
    simulation_params = simulation_default(
      census_interval = 1,
      ssa_algorithm = ssa_etl(tau = tau),
      experiment_params = simulation_type_wild_type(num_simulations = num_simulations),
      compute_rna_velocity = TRUE
    )
  )
  
  model <- config %>%
    generate_tf_network() %>%
    generate_feature_network() %>%
    generate_kinetics() %>%
    generate_gold_standard() %>%
    generate_cells() %>%
    generate_experiment()
  
  ad <- as_anndata(model)
  output_file_name <- paste0(backbone_function_name, ".h5ad")
  ad$write_h5ad(output_file_name)
  
  cat("Saved output for", backbone_function_name, "to", output_file_name, "\n\n")
}

map_params <- list(
    num_cells = 1000,
    num_targets = 25,
    num_hks = 25,
    num_simulations = 100,
    tau = 0.01
)

backbone_functions <- c(
    "backbone_linear",
    "backbone_linear_simple",
    "backbone_disconnected",
    "backbone_converging",
    "backbone_branching",
    "backbone_binary_tree",
    "backbone_bifurcating",
    "backbone_bifurcating_converging",
    "backbone_bifurcating_loop",
    "backbone_bifurcating_cycle",
    "backbone_consecutive_bifurcating",
    "backbone_trifurcating",
    "backbone_cycle_simple",
    "backbone_cycle"
)

# TEST
map_params <- list(
    num_cells = 100,
    num_targets = 10,
    num_hks = 10,
    num_simulations = 10,
    tau = 0.01
)

backbone_functions <- c(
    "backbone_linear",
    "backbone_branching"
)

cat("Running simulations...\n")
num_cores <- min(3, detectCores()-1)
mclapply(backbone_functions, function(bb) {
  do.call(generate_simulated_data, c(list(backbone_function_name = bb), map_params))
}, mc.cores = num_cores)
