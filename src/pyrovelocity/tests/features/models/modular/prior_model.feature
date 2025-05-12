Feature: Prior Model
  As a computational biologist
  I want to define prior distributions for RNA velocity parameters
  So that I can perform Bayesian inference on single-cell data

  Background:
    Given I have a prior model component
    And I have input data with unspliced and spliced counts

  Scenario: LogNormal prior model samples parameters
    Given I have a LogNormalPriorModel
    When I run the forward method
    Then the model should sample alpha, beta, and gamma parameters
    And the parameters should follow log-normal distributions
    And the parameters should be registered with Pyro

  Scenario: Prior model uses Pyro plates
    Given I have a LogNormalPriorModel
    When I run the forward method with a plate context
    Then the model should use the plate for batch dimensions
    And the parameters should have the correct shape

  Scenario: Prior model with hyperparameters
    Given I have a LogNormalPriorModel with custom hyperparameters
    When I run the forward method
    Then the sampled parameters should reflect the custom hyperparameters
    And the prior distributions should have the specified location and scale

  Scenario: Prior model with disabled sampling
    Given I have a LogNormalPriorModel
    When I run the forward method with include_prior=False
    Then the model should not sample parameters
    But should still return the expected context structure

  Scenario: Prior model with informative priors
    Given I have a LogNormalPriorModel with informative priors
    When I run the forward method
    Then the sampled parameters should be biased towards the informative priors
    And the parameters should still have appropriate uncertainty
