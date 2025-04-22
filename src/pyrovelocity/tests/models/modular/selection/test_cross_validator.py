"""
Tests for the cross-validation functionality in PyroVelocity.

This module tests the CrossValidator class, which is responsible for
performing cross-validation on multiple models to evaluate their performance.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from pyrovelocity.models.modular.selection import CrossValidator


def test_cross_validator_init(multiple_models):
    """Test the initialization of CrossValidator."""
    # Convert list of models to dict with name as key
    models_dict = {model.name: model for model in multiple_models}
    
    # Initialize with models and default parameters
    cv = CrossValidator(models=models_dict)
    assert cv.models == models_dict
    assert cv.n_splits == 5  # Default value
    assert cv.test_size is None  # Default value
    assert cv.random_state == 42  # Default value
    
    # Initialize with custom parameters
    cv = CrossValidator(models=models_dict, n_splits=3, test_size=0.3, random_state=123)
    assert cv.n_splits == 3
    assert cv.test_size == 0.3
    assert cv.random_state == 123


def test_create_train_test_splits(multiple_models, mock_adata):
    """Test the _create_train_test_splits method."""
    # Convert list of models to dict with name as key
    models_dict = {model.name: model for model in multiple_models}
    
    cv = CrossValidator(models=models_dict, n_splits=3, random_state=42)
    
    # Create splits
    splits = cv._create_train_test_splits(mock_adata)
    
    # Verify the correct number of splits was created
    assert len(splits) == 3
    
    # Verify each split contains train and test indices
    for split in splits:
        assert "train_indices" in split
        assert "test_indices" in split
        
        # Verify indices are within range
        assert all(0 <= idx < len(mock_adata.obs) for idx in split["train_indices"])
        assert all(0 <= idx < len(mock_adata.obs) for idx in split["test_indices"])
        
        # Verify no overlap between train and test indices
        assert set(split["train_indices"]).isdisjoint(set(split["test_indices"]))
        
        # Verify all indices are covered (train + test = total)
        assert len(split["train_indices"]) + len(split["test_indices"]) == len(mock_adata.obs)


def test_evaluate_model_fold(multiple_models, mock_adata):
    """Test the _evaluate_model_fold method."""
    # Convert list of models to dict with name as key
    models_dict = {model.name: model for model in multiple_models}
    
    cv = CrossValidator(models=models_dict, n_splits=3)
    
    # Create a mock split with numpy arrays
    split = {
        "train_indices": np.array(list(range(7))),  # 70% for training
        "test_indices": np.array(list(range(7, 10)))  # 30% for testing
    }
    
    # Mock the model's methods
    model = multiple_models[0]
    mock_train_result = MagicMock()
    model.train = MagicMock(return_value=mock_train_result)
    model.evaluate = MagicMock(return_value={"metric_1": 0.9, "metric_2": 0.8})
    
    # Evaluate the model on this fold
    result = cv._evaluate_model_fold(model, mock_adata, split)
    
    # Verify the model's train method was called with the correct indices
    model.train.assert_called_once()
    train_args = model.train.call_args[1]
    assert "adata" in train_args
    assert "indices" in train_args
    assert set(train_args["indices"]) == set(split["train_indices"])
    
    # Verify the model's evaluate method was called with the correct indices
    model.evaluate.assert_called_once()
    eval_args = model.evaluate.call_args[1]
    assert "adata" in eval_args
    assert "indices" in eval_args
    assert set(eval_args["indices"]) == set(split["test_indices"])
    
    # Verify the result contains the expected metrics
    assert "metric_1" in result
    assert "metric_2" in result
    assert result["metric_1"] == 0.9
    assert result["metric_2"] == 0.8


def test_cross_validate(multiple_models, mock_adata, mock_cross_validation_result):
    """Test the cross_validate method."""
    # Convert list of models to dict with name as key
    models_dict = {model.name: model for model in multiple_models}
    
    cv = CrossValidator(models=models_dict, n_splits=2)
    
    # Mock the necessary methods
    with patch.object(cv, "_create_train_test_splits") as mock_create_splits:
        # Mock splits for 2 folds
        mock_splits = [
            {"train_indices": [0, 1, 2, 3, 4, 5, 6], "test_indices": [7, 8, 9]},
            {"train_indices": [0, 1, 2, 7, 8, 9], "test_indices": [3, 4, 5, 6]}
        ]
        mock_create_splits.return_value = mock_splits
        
        with patch.object(cv, "_evaluate_model_fold") as mock_evaluate:
            # Mock evaluation results for each model and fold
            mock_evaluate.side_effect = [
                {"metric_1": 0.9, "metric_2": 0.8},  # model_0, fold_0
                {"metric_1": 0.85, "metric_2": 0.75},  # model_0, fold_1
                {"metric_1": 0.8, "metric_2": 0.7},  # model_1, fold_0
                {"metric_1": 0.75, "metric_2": 0.65},  # model_1, fold_1
                {"metric_1": 0.7, "metric_2": 0.6},  # model_2, fold_0
                {"metric_1": 0.65, "metric_2": 0.55}   # model_2, fold_1
            ]
            
            # Perform cross-validation
            result = cv.cross_validate(mock_adata)
            
            # Verify the _create_train_test_splits method was called once
            mock_create_splits.assert_called_once_with(mock_adata)
            
            # Verify the _evaluate_model_fold method was called for each model and fold
            assert mock_evaluate.call_count == len(models_dict) * len(mock_splits)
            
            # Verify the result structure
            assert len(result) == len(models_dict)
            for model_name in models_dict:
                assert model_name in result
                assert len(result[model_name]) == len(mock_splits)
                for j in range(len(mock_splits)):
                    assert f"fold_{j}" in result[model_name]
                    assert "metric_1" in result[model_name][f"fold_{j}"]
                    assert "metric_2" in result[model_name][f"fold_{j}"]


def test_select_best_model(multiple_models, mock_adata, mock_cross_validation_result):
    """Test the select_best_model method."""
    # Convert list of models to dict with name as key
    models_dict = {model.name: model for model in multiple_models}
    
    cv = CrossValidator(models=models_dict, n_splits=2)
    
    # Mock the cross_validate method
    with patch.object(cv, "cross_validate", return_value=mock_cross_validation_result):
        # Select the best model based on metric_1
        best_model, avg_scores = cv.select_best_model(mock_adata, metric="metric_1")
        
        # Verify the correct model was selected (model_0 has the highest average metric_1)
        assert best_model.name == "model_0"
        
        # Verify the average scores are correct
        assert "model_0" in avg_scores
        assert "model_1" in avg_scores
        assert avg_scores["model_0"] == (0.9 + 0.85) / 2
        assert avg_scores["model_1"] == (0.8 + 0.75) / 2
        
        # Verify selection based on a different metric
        best_model, avg_scores = cv.select_best_model(mock_adata, metric="metric_2")
        assert best_model.name == "model_0"  # Still the best for metric_2
        assert avg_scores["model_0"] == (0.8 + 0.75) / 2
        assert avg_scores["model_1"] == (0.7 + 0.65) / 2


def test_compute_average_scores(multiple_models, mock_cross_validation_result):
    """Test the _compute_average_scores method."""
    # Convert list of models to dict with name as key
    models_dict = {model.name: model for model in multiple_models}
    
    cv = CrossValidator(models=models_dict)  # Models don't matter for this test
    
    # Compute average scores for metric_1
    avg_scores = cv._compute_average_scores(mock_cross_validation_result, "metric_1")
    
    # Verify the correct averages
    assert avg_scores["model_0"] == (0.9 + 0.85) / 2
    assert avg_scores["model_1"] == (0.8 + 0.75) / 2
    
    # Test with a metric that doesn't exist
    with pytest.raises(KeyError):
        cv._compute_average_scores(mock_cross_validation_result, "nonexistent_metric")


def test_cross_validator_str_representation(multiple_models):
    """Test the string representation of CrossValidator."""
    # Convert list of models to dict with name as key
    models_dict = {model.name: model for model in multiple_models}
    
    cv = CrossValidator(models=models_dict, n_splits=3, test_size=0.25)
    
    # Get the string representation
    cv_str = str(cv)
    
    # Verify the string contains key information
    assert "CrossValidator" in cv_str
    assert "n_splits=3" in cv_str
    assert "test_size=0.25" in cv_str
    assert str(len(models_dict)) in cv_str  # Number of models 