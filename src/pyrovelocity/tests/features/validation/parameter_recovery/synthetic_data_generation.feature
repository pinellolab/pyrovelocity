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
    And the normalized expression should still reflect the underlying dynamics

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
