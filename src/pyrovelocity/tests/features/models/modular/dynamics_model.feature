Feature: Dynamics Model
  As a computational biologist
  I want to model RNA velocity dynamics
  So that I can understand transcriptional dynamics in single-cell data

  Background:
    Given I have a dynamics model component
    And I have input data with unspliced and spliced counts

  Scenario: Standard dynamics model computes expected counts
    Given I have a StandardDynamicsModel
    When I run the forward method with parameters:
      | alpha | beta | gamma |
      | 1.0   | 0.5  | 0.2   |
    Then the model should compute expected unspliced and spliced counts
    And the expected counts should follow RNA velocity dynamics

  Scenario: Standard dynamics model computes steady state
    Given I have a StandardDynamicsModel
    When I compute the steady state with parameters:
      | alpha | beta | gamma |
      | 1.0   | 0.5  | 0.2   |
    Then the steady state unspliced should equal alpha/beta
    And the steady state spliced should equal alpha/gamma

  Scenario: Legacy dynamics model matches legacy implementation
    Given I have a LegacyDynamicsModel
    When I run the forward method with the same parameters as the legacy implementation
    Then the output should match the legacy implementation output
    And the model should create deterministic nodes with event_dim=0

  Scenario: Dynamics model handles edge cases
    Given I have a StandardDynamicsModel
    When I run the forward method with zero rates
    Then the model should handle the edge case gracefully
    And should not produce NaN or infinite values

  Scenario: Dynamics model with library size correction
    Given I have a StandardDynamicsModel with library size correction
    When I run the forward method with library size factors
    Then the expected counts should be scaled by the library size factors
    And the scaling should be applied correctly
