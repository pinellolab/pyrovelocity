Feature: Evaluation Metrics for Recovery Validation
  As a researcher using PyroVelocity
  I want to quantify how well my model recovers known parameters
  So that I can assess the empirical identifiability of my model

  Background:
    Given I have an EvaluationMetrics component
    And I have true parameter values
    And I have posterior samples from inference

  Scenario: Calculate recovery accuracy metrics
    Given posterior samples for parameters alpha, beta, and gamma
    When I calculate recovery accuracy metrics
    Then I should get Mean Squared Error (MSE) for each parameter
    And I should get correlation between true and estimated parameters
    And I should get relative error for each parameter
    And I should get Mean Absolute Error (MAE) for each parameter
    And I should get R-squared values for each parameter

  Scenario: Calculate recovery precision metrics
    Given posterior samples for parameters alpha, beta, and gamma
    When I calculate recovery precision metrics
    Then I should get credible interval widths for each parameter
    And I should get coefficient of variation for each parameter
    And I should get posterior standard deviations for each parameter
    And the precision metrics should reflect the uncertainty in parameter estimates

  Scenario: Calculate coverage metrics
    Given posterior samples for parameters alpha, beta, and gamma
    When I calculate coverage metrics
    Then I should get credible interval coverage percentages for each parameter
    And the coverage should be approximately 95% for 95% credible intervals

  Scenario: Calculate bias metrics
    Given posterior samples for parameters alpha, beta, and gamma
    When I calculate bias metrics
    Then I should get mean bias for each parameter
    And the bias should be close to zero for unbiased estimation

  Scenario Outline: Evaluate metrics across different sample sizes
    Given posterior samples from inference with <num_cells> cells
    When I calculate all evaluation metrics
    Then the metrics should reflect the effect of sample size on parameter recovery
    And larger sample sizes should generally lead to better recovery metrics

    Examples:
      | num_cells |
      | 50        |
      | 100       |

  Scenario: Compare metrics across different inference methods
    Given posterior samples from SVI inference
    And posterior samples from MCMC inference
    When I calculate evaluation metrics for both methods
    Then I should be able to compare the performance of different inference methods
    And the comparison should highlight strengths and weaknesses of each method

  Scenario: Calculate aggregate metrics across multiple parameter sets
    Given evaluation metrics for multiple parameter sets
    When I calculate aggregate metrics
    Then I should get summary statistics across all parameter sets
    And I should get metrics for the worst-case scenarios
    And I should get metrics for the best-case scenarios
    And I should get metrics for the average-case scenarios

  Scenario: Metrics serialization and deserialization
    Given I have calculated evaluation metrics
    When I serialize the metrics to a file
    And I deserialize the metrics from the file
    Then the deserialized metrics should match the original metrics
