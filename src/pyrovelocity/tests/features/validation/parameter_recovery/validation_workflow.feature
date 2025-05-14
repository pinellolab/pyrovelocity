Feature: End-to-End Validation Workflow
  As a researcher using PyroVelocity
  I want to run a complete parameter recovery validation workflow
  So that I can systematically validate my model's identifiability

  Background:
    Given I have a ValidationRunner component
    And I have a PyroVelocity model to validate

  Scenario: Run basic validation workflow
    Given a validation configuration with default settings
    When I run the validation workflow
    Then the workflow should execute all validation steps
    And the workflow should generate parameter sets
    And the workflow should generate synthetic data
    And the workflow should run inference
    And the workflow should calculate evaluation metrics
    And the workflow should create visualizations
    And the workflow should generate a validation report

  Scenario Outline: Run validation with different model types
    Given a PyroVelocity model with <model_type> dynamics and <likelihood_type> likelihood
    When I run the validation workflow
    Then the workflow should adapt to the specific model type
    And the validation report should include model-specific insights
    And the validation should assess the identifiability of model-specific parameters

    Examples:
      | model_type   | likelihood_type |
      | standard     | poisson         |
      | switching    | normal          |
      | continuous   | poisson         |
      | legacy       | normal          |

  Scenario: Run comprehensive validation across parameter space
    Given a validation configuration with stratified parameter sampling
    When I run the validation workflow
    Then the workflow should test parameter recovery across the parameter space
    And the validation should identify challenging parameter regimes
    And the validation should provide insights about parameter identifiability
    And the validation report should include recommendations for each parameter regime

  Scenario Outline: Run validation with different sample sizes
    Given a validation configuration with sample sizes <sample_sizes>
    When I run the validation workflow
    Then the workflow should test parameter recovery for each sample size
    And the validation should determine minimum sample size requirements
    And the validation report should include sample size recommendations

    Examples:
      | sample_sizes       |
      | [50, 100, 200]     |
      | [100, 500, 1000]   |
      | [200, 1000, 5000]  |

  Scenario Outline: Run validation with different noise levels
    Given a validation configuration with noise levels <noise_levels>
    When I run the validation workflow
    Then the workflow should test parameter recovery for each noise level
    And the validation should determine robustness to noise
    And the validation report should include noise level insights

    Examples:
      | noise_levels      |
      | [0.1, 0.5, 1.0]   |
      | [0.2, 0.8, 1.5]   |
      | [0.05, 0.2, 0.5]  |

  Scenario: Run validation with multiple inference methods
    Given a validation configuration with inference methods ["SVI", "MCMC"]
    When I run the validation workflow
    Then the workflow should test parameter recovery with each inference method
    And the validation should compare the performance of different methods
    And the validation report should include method comparison insights
    And the validation should recommend the best method for different scenarios

  Scenario: Run validation with early stopping criteria
    Given a validation configuration with early stopping criteria
    When I run the validation workflow
    Then the workflow should stop validation for clearly failing cases
    And the workflow should allocate more resources to challenging cases
    And the validation report should highlight cases that failed early
    And the validation should be computationally efficient

  Scenario: Run validation with parallel processing
    Given a validation configuration with parallel processing enabled
    When I run the validation workflow
    Then the workflow should distribute validation tasks across processors
    And the validation should complete faster than sequential processing
    And the validation results should be the same as sequential processing
    And the validation should handle resource allocation appropriately

  Scenario: Generate comprehensive validation report
    Given I have completed a validation workflow
    When I generate a validation report
    Then the report should include an executive summary
    And the report should include detailed metrics for all parameters
    And the report should include all visualizations
    And the report should include recommendations for model usage
    And the report should include limitations and caveats
    And the report should be well-formatted and readable

  Scenario: Integrate with existing validation framework
    Given I have the Implementation Validation Framework results
    When I run the Parameter Recovery Validation workflow
    Then the workflow should integrate with the existing framework
    And the combined validation should provide a comprehensive assessment
    And the validation report should include both implementation and parameter recovery insights
    And the validation should highlight any discrepancies between the two approaches

  Scenario: Reproducibility of validation workflow
    Given a validation configuration with seed 42
    When I run the validation workflow
    And I run the validation workflow again with the same seed 42
    Then both validation results should be identical
    But when I run the validation workflow with a different seed 43
    Then the validation results should be different

  Scenario: Validation results serialization and deserialization
    Given I have completed a validation workflow
    When I serialize the validation results to a file
    And I deserialize the validation results from the file
    Then the deserialized validation results should match the original validation results
