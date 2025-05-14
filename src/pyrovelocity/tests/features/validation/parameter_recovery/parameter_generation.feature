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
      | lognormal    | 20                 | 10        |

  Scenario: Generate parameters with custom constraints
    Given a model with lognormal prior
    When I generate parameters with the following constraints:
      | parameter | min_value | max_value |
      | alpha     | 0.5       | 2.0       |
      | beta      | 0.1       | 1.0       |
      | gamma     | 0.05      | 0.5       |
    Then all generated parameter sets should satisfy the constraints

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
