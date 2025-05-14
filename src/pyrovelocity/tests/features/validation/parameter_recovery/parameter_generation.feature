Feature: Parameter Generation for Recovery Validation
  As a researcher using PyroVelocity
  I want to generate known parameter sets
  So that I can use them as ground truth for parameter recovery validation

  Background:
    Given I have a ParameterGenerator component

  Scenario Outline: Generate parameters from prior distribution
    Given a model with <prior_type> prior
    When I generate <num_parameter_sets> parameter sets with <num_genes> genes
    Then I should get <num_parameter_sets> unique parameter sets
    And each parameter set should have the correct shape for <num_genes> genes
    And each parameter set should have values within the expected ranges
    And each parameter set should be immutable

    Examples:
      | prior_type   | num_parameter_sets | num_genes |
      | lognormal    | 10                 | 5         |
      | normal       | 20                 | 10        |
      | informative  | 5                  | 15        |

  Scenario: Generate parameters with stratified sampling
    Given a model with lognormal prior
    When I generate parameters with stratified sampling across parameter space
    Then I should get parameter sets from different regions of the parameter space
    And the parameter sets should cover the specified regions
    And each region should have the requested number of parameter sets

  Scenario: Generate parameters with specific regions
    Given a model with lognormal prior
    When I request parameter sets from the following regions:
      | region_name | alpha_range | beta_range | gamma_range |
      | fast        | high        | high       | high        |
      | slow        | low         | low        | low         |
      | mixed       | high        | low        | high        |
    Then I should get parameter sets from each specified region
    And the parameters in each set should match the region constraints

  Scenario: Generate parameters with custom constraints
    Given a model with lognormal prior
    When I generate parameters with the following constraints:
      | parameter | min_value | max_value |
      | alpha     | 0.5       | 2.0       |
      | beta      | 0.1       | 1.0       |
      | gamma     | 0.05      | 0.5       |
    Then all generated parameter sets should satisfy the constraints
    And the distribution of parameters should be uniform within the constraints

  Scenario: Generate parameters with switching times
    Given a model with switching dynamics
    When I generate parameter sets including switching times
    Then each parameter set should include switching time parameters
    And the switching times should be within the expected range
    And the switching times should be ordered correctly for each cell

  Scenario: Generate parameters with cell-specific times
    Given a model with continuous time dynamics
    When I generate parameter sets including cell-specific times for <num_cells> cells
    Then each parameter set should include cell-specific time parameters
    And the time parameters should have the correct shape for <num_cells> cells
    And the time parameters should be within the expected range

    Examples:
      | num_cells |
      | 50        |
      | 100       |
      | 500       |

  Scenario: Reproducibility of parameter generation
    Given a model with lognormal prior
    When I generate parameter sets with seed 42
    And I generate parameter sets again with the same seed 42
    Then both sets of parameters should be identical
    But when I generate parameters with a different seed 43
    Then the parameters should be different

  Scenario: Parameter set serialization and deserialization
    Given I have generated parameter sets
    When I serialize the parameter sets to a file
    And I deserialize the parameter sets from the file
    Then the deserialized parameter sets should match the original parameter sets
