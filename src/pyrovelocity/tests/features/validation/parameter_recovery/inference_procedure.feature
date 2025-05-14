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
    And a PyroVelocity model with <model_type> dynamics and <likelihood_type> likelihood
    When I run SVI inference for <num_epochs> epochs
    Then the inference should converge
    And the loss should decrease monotonically
    And the recovered parameters should be close to the true parameters
    And the inference results should include posterior samples

    Examples:
      | num_cells | num_genes | model_type   | likelihood_type | num_epochs |
      | 100       | 10        | standard     | poisson         | 500        |
      | 200       | 20        | switching    | normal          | 1000       |
      | 500       | 30        | continuous   | poisson         | 2000       |

  Scenario Outline: Recover parameters using MCMC
    Given synthetic data with <num_cells> cells and <num_genes> genes
    And a PyroVelocity model with <model_type> dynamics and <likelihood_type> likelihood
    When I run MCMC inference with <num_samples> samples and <num_warmup> warmup steps
    Then the MCMC chains should converge
    And the effective sample size should be adequate
    And the recovered parameters should be close to the true parameters
    And the inference results should include posterior samples

    Examples:
      | num_cells | num_genes | model_type   | likelihood_type | num_samples | num_warmup |
      | 100       | 10        | standard     | poisson         | 1000        | 500        |
      | 200       | 20        | switching    | normal          | 2000        | 1000       |

  Scenario: Compare SVI and MCMC inference
    Given synthetic data with 100 cells and 20 genes
    And a PyroVelocity model with standard dynamics and poisson likelihood
    When I run both SVI and MCMC inference
    Then both methods should recover parameters close to the true values
    And the posterior distributions should be similar between methods
    And the computational performance metrics should be recorded

  Scenario Outline: Inference with different guide types
    Given synthetic data with 100 cells and 20 genes
    And a PyroVelocity model with standard dynamics and poisson likelihood
    When I run SVI inference with guide type <guide_type>
    Then the inference should converge
    And the recovered parameters should be close to the true parameters
    And the guide type should affect the inference performance

    Examples:
      | guide_type      |
      | AutoNormal      |
      | AutoDiagonalNormal |
      | AutoLowRankNormal |
      | AutoMultivariateNormal |

  Scenario: Inference with parameter constraints
    Given synthetic data with 100 cells and 20 genes
    And a PyroVelocity model with parameter constraints:
      | parameter | constraint_type | value |
      | alpha     | positive        | N/A   |
      | beta      | min             | 0.1   |
      | gamma     | max             | 2.0   |
    When I run inference with these constraints
    Then the inference should converge
    And the recovered parameters should satisfy the constraints
    And the constraints should not prevent recovery of true parameters

  Scenario: Inference with different initializations
    Given synthetic data with 100 cells and 20 genes
    And a PyroVelocity model with standard dynamics
    When I run inference with multiple random initializations
    Then the inference should converge to similar results across initializations
    And the best initialization should be selected based on final loss
    And the variability due to initialization should be quantified

  Scenario: Inference with early stopping
    Given synthetic data with 100 cells and 20 genes
    And a PyroVelocity model with standard dynamics
    When I run inference with early stopping based on validation loss
    Then the inference should stop when validation loss plateaus
    And the early stopping should prevent overfitting
    And the optimal number of epochs should be recorded

  Scenario: Inference with learning rate scheduling
    Given synthetic data with 100 cells and 20 genes
    And a PyroVelocity model with standard dynamics
    When I run inference with learning rate scheduling
    Then the inference should converge faster than with constant learning rate
    And the final results should be at least as good as with constant learning rate
    And the learning rate schedule should be recorded

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
