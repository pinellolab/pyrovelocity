Feature: Observation Model
  As a computational biologist
  I want to preprocess and transform RNA velocity data
  So that it can be used effectively in the model

  Background:
    Given I have an observation model component
    And I have input data with unspliced and spliced counts

  Scenario: Standard observation model transforms data
    Given I have a StandardObservationModel
    When I run the forward method
    Then the model should preprocess the input data
    And the transformed data should maintain the original structure
    And the context should be updated with the transformed data

  Scenario: Observation model computes size factors
    Given I have a StandardObservationModel with use_size_factor=True
    When I run the forward method
    Then the model should compute size factors for the data
    And the size factors should reflect the library size
    And the context should include the computed size factors

  Scenario: Observation model handles missing values
    Given I have a StandardObservationModel
    And I have data with missing values
    When I run the forward method
    Then the model should handle missing values gracefully
    And should not produce errors or warnings

  Scenario: Observation model with normalization
    Given I have a StandardObservationModel with normalization
    When I run the forward method
    Then the model should normalize the data
    And the normalized data should have the expected properties
    And the context should include the normalized data

  Scenario: Observation model preserves data integrity
    Given I have a StandardObservationModel
    When I run the forward method
    Then the transformed data should preserve the biological signal
    And the transformation should not introduce artifacts
