Feature: Inference Procedure for Recovery Validation
  As a researcher using PyroVelocity
  I want to recover parameter estimates from synthetic data
  So that I can validate the inference capabilities of my models

  Background:
    Given I have an InferenceProcedure component
    And I have synthetic data with known parameters
    And I have a PyroVelocity model

  Scenario Outline: Recover parameters using SVI
    Given synthetic data with <num_cells> cells and <num_genes> genes
    And a PyroVelocity model with standard dynamics and <likelihood_type> likelihood
    When I run SVI inference for <num_epochs> epochs
    Then the inference should converge
    And the loss should decrease monotonically
    And the recovered parameters should be close to the true parameters
    And the inference results should include posterior samples

    Examples:
      | num_cells | num_genes | likelihood_type | num_epochs |
      | 100       | 10        | poisson         | 500        |
      | 200       | 20        | poisson         | 1000       |

  Scenario: Recover parameters using MCMC
    Given synthetic data with 100 cells and 10 genes
    And a PyroVelocity model with standard dynamics and poisson likelihood
    When I run MCMC inference with 1000 samples and 500 warmup steps
    Then the MCMC chains should converge
    And the effective sample size should be adequate
    And the recovered parameters should be close to the true parameters
    And the inference results should include posterior samples

  Scenario: Compare SVI and MCMC inference
    Given synthetic data with 100 cells and 20 genes
    And a PyroVelocity model with standard dynamics and poisson likelihood
    When I run both SVI and MCMC inference
    Then both methods should recover parameters close to the true values
    And the posterior distributions should be similar between methods
    And the computational performance metrics should be recorded

  Scenario: Inference with different guide types
    Given synthetic data with 100 cells and 20 genes
    And a PyroVelocity model with standard dynamics and poisson likelihood
    When I run SVI inference with guide type AutoNormal
    Then the inference should converge
    And the recovered parameters should be close to the true parameters

  Scenario: Inference with early stopping
    Given synthetic data with 100 cells and 20 genes
    And a PyroVelocity model with standard dynamics
    When I run inference with early stopping based on validation loss
    Then the inference should stop when validation loss plateaus
    And the early stopping should prevent overfitting
    And the optimal number of epochs should be recorded

  Scenario: Reproducibility of inference
    Given synthetic data with 100 cells and 20 genes
    And a PyroVelocity model with standard dynamics
    When I run inference with seed 42
    And I run inference again with the same seed 42
    Then both inference results should be identical
    But when I run inference with a different seed 43
    Then the inference results should be different

  Scenario: Inference results serialization and deserialization
    Given I have completed inference on synthetic data
    When I serialize the inference results to a file
    And I deserialize the inference results from the file
    Then the deserialized inference results should match the original inference results
