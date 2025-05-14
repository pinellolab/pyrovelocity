Feature: Synthetic Data Generation for Recovery Validation
  As a researcher using PyroVelocity
  I want to generate synthetic data using known parameters
  So that I can validate parameter recovery in a controlled setting

  Background:
    Given I have a SyntheticDataGenerator component
    And I have a set of known parameters

  Scenario Outline: Generate synthetic data with different sample sizes
    Given a parameter set with <num_genes> genes
    When I generate synthetic data with <num_cells> cells
    Then the synthetic data should have <num_cells> cells and <num_genes> genes
    And the synthetic data should include both unspliced and spliced counts
    And the synthetic data should follow the RNA velocity model dynamics
    And the synthetic data should be compatible with PyroVelocity models

    Examples:
      | num_cells | num_genes |
      | 50        | 10        |
      | 100       | 20        |
      | 500       | 30        |

  Scenario Outline: Generate synthetic data with different noise levels
    Given a parameter set with 20 genes
    When I generate synthetic data with 100 cells and noise level <noise_level>
    Then the synthetic data should have the expected signal-to-noise ratio
    And higher noise levels should result in more variable data
    And the underlying signal should still be present in the data

    Examples:
      | noise_level |
      | 0.1         |
      | 0.5         |
      | 1.0         |

  Scenario: Generate synthetic data with cell-specific times
    Given a parameter set with cell-specific times
    When I generate synthetic data with continuous time dynamics
    Then the synthetic data should reflect the cell-specific time points
    And cells with similar times should have similar expression profiles
    And the data should show a trajectory of expression changes over time

  Scenario: Generate synthetic data with switching dynamics
    Given a parameter set with switching times
    When I generate synthetic data with switching dynamics
    Then the synthetic data should reflect the switching behavior
    And the expression profiles should change after the switching times
    And the data should show two distinct expression states

  Scenario: Generate synthetic data with multiple replicates
    Given a parameter set with 20 genes
    When I generate 5 replicate datasets with the same parameters
    Then each replicate should have the same underlying parameters
    But each replicate should have different random noise
    And the average across replicates should approximate the expected values

  Scenario: Generate synthetic data in AnnData format
    Given a parameter set with 20 genes
    When I generate synthetic data in AnnData format
    Then the result should be a valid AnnData object
    And the AnnData object should have unspliced and spliced counts in layers
    And the AnnData object should store the ground truth parameters in uns
    And the AnnData object should be ready for use with PyroVelocity models

  Scenario: Generate synthetic data with library size variation
    Given a parameter set with 20 genes
    When I generate synthetic data with variable library sizes
    Then the synthetic data should have varying total counts per cell
    And the library size variation should follow the specified distribution
    And the normalized expression should still reflect the underlying dynamics

  Scenario: Generate synthetic data with gene-specific characteristics
    Given a parameter set with gene-specific characteristics:
      | gene_group | alpha_range | beta_range | gamma_range | count_range |
      | high_expr  | high        | medium     | low         | high        |
      | low_expr   | low         | low        | high        | low         |
      | transient  | high        | high       | high        | medium      |
    When I generate synthetic data with these gene groups
    Then the synthetic data should reflect the gene-specific characteristics
    And genes in the same group should show similar expression patterns
    And the gene groups should be distinguishable in the data

  Scenario: Reproducibility of synthetic data generation
    Given a parameter set with 20 genes
    When I generate synthetic data with seed 42
    And I generate synthetic data again with the same seed 42
    Then both synthetic datasets should be identical
    But when I generate synthetic data with a different seed 43
    Then the synthetic datasets should be different

  Scenario: Synthetic data serialization and deserialization
    Given I have generated synthetic data
    When I serialize the synthetic data to a file
    And I deserialize the synthetic data from the file
    Then the deserialized synthetic data should match the original synthetic data
