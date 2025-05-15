#!/usr/bin/env python
"""
Script to generate a report of all class methods for PyroVelocity classes.

This script parses the _quarto.yml file to identify all PyroVelocity classes
referenced in the documentation, imports each class, and generates a report
of all class and instance methods with their full import paths.

Usage:
    python generate_class_methods_report.py [--output OUTPUT_FILE]

Options:
    --output OUTPUT_FILE    Output file path (default: class_methods_report.md)
"""

import argparse
import importlib
import inspect
import os
import re
import sys
import yaml
from typing import Dict, List, Set, Tuple, Any, Optional


def get_class_methods(cls) -> List[str]:
    """Get all class methods for a given class."""
    return [name for name, method in inspect.getmembers(cls, inspect.isfunction)]


def get_instance_methods(cls) -> List[str]:
    """Get all instance methods for a given class."""
    return [name for name, method in inspect.getmembers(cls, predicate=inspect.ismethod)]


def parse_quarto_yml(file_path: str) -> List[str]:
    """
    Parse the _quarto.yml file to extract all PyroVelocity class references.

    Args:
        file_path: Path to the _quarto.yml file

    Returns:
        List of class references in the format 'module.ClassName'
    """
    with open(file_path, 'r') as f:
        quarto_config = yaml.safe_load(f)

    class_references = []

    # Extract class references from quartodoc.sections.contents
    if 'quartodoc' in quarto_config:
        sections = quarto_config.get('quartodoc', {}).get('sections', [])
        for section in sections:
            contents = section.get('contents', [])
            if isinstance(contents, list):
                for item in contents:
                    if isinstance(item, str):
                        class_references.append(item)
                    elif isinstance(item, dict) and 'name' in item:
                        class_references.append(item['name'])

    return class_references


def import_class(class_reference: str) -> Tuple[Optional[Any], str]:
    """
    Import a class from its reference string.

    Args:
        class_reference: Class reference in the format 'module.ClassName'

    Returns:
        Tuple of (class object, full import path)
    """
    try:
        # Handle special cases for functions
        if '.' in class_reference:
            module_path, class_name = class_reference.rsplit('.', 1)

            # Special handling for create_legacy_model1 and create_legacy_model2
            if class_name in ['create_legacy_model1', 'create_legacy_model2']:
                # These functions might be in the factory module
                try:
                    module = importlib.import_module(f"pyrovelocity.{module_path}.factory")
                    func = getattr(module, class_name)
                    return func, f"from pyrovelocity.{module_path}.factory import {class_name}"
                except (ImportError, AttributeError):
                    print(f"Could not find {class_name} in factory module")
                    return None, f"# Failed to import: {class_reference}"

            # Special handling for create_* functions
            if class_name.startswith('create_'):
                module = importlib.import_module(f"pyrovelocity.{module_path}")
                func = getattr(module, class_name)
                return func, f"from pyrovelocity.{module_path} import {class_name}"

            # Special handling for modules that are not classes
            if module_path == 'io' and class_name == 'datasets':
                return "module", f"from pyrovelocity.{module_path} import {class_name}"

            if module_path == 'analysis' and class_name in ['cytotrace', 'transcriptome_properties']:
                return "module", f"from pyrovelocity.{module_path} import {class_name}"

            if module_path == 'tasks' and class_name in ['data', 'evaluate', 'postprocess', 'preprocess', 'summarize', 'train']:
                return "module", f"from pyrovelocity.{module_path} import {class_name}"

            if module_path == 'models' and class_name == 'solve_transcription_splicing_model':
                try:
                    module = importlib.import_module(f"pyrovelocity.models.experimental")
                    func = getattr(module, class_name)
                    return func, f"from pyrovelocity.models.experimental import {class_name}"
                except (ImportError, AttributeError):
                    print(f"Could not find {class_name} in experimental module")
                    return None, f"# Failed to import: {class_reference}"

            # Regular class import
            module = importlib.import_module(f"pyrovelocity.{module_path}")
            cls = getattr(module, class_name)
            return cls, f"from pyrovelocity.{module_path} import {class_name}"
        else:
            # Direct import from pyrovelocity
            module = importlib.import_module("pyrovelocity")
            cls = getattr(module, class_reference)
            return cls, f"from pyrovelocity import {class_reference}"
    except (ImportError, AttributeError) as e:
        print(f"Error importing {class_reference}: {e}")
        return None, f"# Failed to import: {class_reference}"


def generate_report(class_references: List[str]) -> str:
    """
    Generate a markdown report of all class methods.

    Args:
        class_references: List of class references

    Returns:
        Markdown formatted report
    """
    # Process each class reference to get class objects
    class_objects = {}
    for class_ref in class_references:
        cls, import_path = import_class(class_ref)
        if cls is not None:
            class_objects[class_ref] = (cls, import_path)

    # Start the report
    report = "# PyroVelocity Class Methods Report\n\n"
    report += "This report lists all class and instance methods for each PyroVelocity class referenced in the documentation.\n\n"

    # Group classes by category
    categories = {
        "Legacy Models": [],
        "Modular Models": [],
        "Component Interfaces": [],
        "Component Implementations": [],
        "Model Comparison": [],
        "JAX Implementation": [],
        "Experimental Models": [],
        "Plotting Functions": [],
        "Analysis Functions": [],
        "IO Utilities": [],
        "Other Utilities": [],
    }

    # Categorize classes
    for class_ref in class_references:
        if class_ref.startswith("models.jax"):
            categories["JAX Implementation"].append(class_ref)
        elif class_ref in ["models.PyroVelocity"]:
            categories["Legacy Models"].append(class_ref)
        elif class_ref in ["models.PyroVelocityModel", "models.ModelState", "models.modular.create_model",
                          "models.modular.create_model_from_config", "models.modular.create_legacy_model1",
                          "models.modular.create_legacy_model2", "models.modular.create_standard_model"]:
            categories["Modular Models"].append(class_ref)
        elif class_ref in ["models.DynamicsModel", "models.PriorModel", "models.LikelihoodModel",
                          "models.ObservationModel", "models.InferenceGuide"]:
            categories["Component Interfaces"].append(class_ref)
        elif class_ref in ["models.StandardDynamicsModel", "models.LegacyDynamicsModel", "models.LogNormalPriorModel",
                          "models.PoissonLikelihoodModel", "models.LegacyLikelihoodModel",
                          "models.StandardObservationModel", "models.AutoGuideFactory", "models.LegacyAutoGuideFactory"]:
            categories["Component Implementations"].append(class_ref)
        elif class_ref in ["models.BayesianModelComparison", "models.ComparisonResult", "models.ModelEnsemble",
                          "models.ModelSelection", "models.SelectionCriterion", "models.SelectionResult",
                          "models.select_best_model"]:
            categories["Model Comparison"].append(class_ref)
        elif class_ref in ["models.deterministic_transcription_splicing_probabilistic_model",
                          "models.solve_transcription_splicing_model",
                          "models.solve_transcription_splicing_model_analytical"]:
            categories["Experimental Models"].append(class_ref)
        elif class_ref.startswith("plots."):
            categories["Plotting Functions"].append(class_ref)
        elif class_ref.startswith("analysis."):
            categories["Analysis Functions"].append(class_ref)
        elif class_ref.startswith("io."):
            categories["IO Utilities"].append(class_ref)
        else:
            categories["Other Utilities"].append(class_ref)

    # Create table of contents
    report += "## Table of Contents\n\n"

    for category, class_refs in categories.items():
        if not class_refs:
            continue

        report += f"### {category}\n\n"
        for class_ref in sorted(class_refs):
            anchor = class_ref.replace('.', '').lower()
            report += f"- [{class_ref}](#{anchor})\n"
        report += "\n"

    # Process each category
    for category, class_refs in categories.items():
        if not class_refs:
            continue

        report += f"# {category}\n\n"

        # Process each class reference in this category
        for class_ref in sorted(class_refs):
            if class_ref not in class_objects:
                report += f"## {class_ref} <a id='{class_ref.replace('.', '').lower()}'></a>\n\n"
                report += "Failed to import this class.\n\n"
                continue

            cls, import_path = class_objects[class_ref]

            # Handle modules
            if cls == "module":
                report += f"## {class_ref} <a id='{class_ref.replace('.', '').lower()}'></a>\n\n"
                report += f"```python\n{import_path}\n```\n\n"
                report += "This is a module, not a class.\n\n"

                # Try to get functions from the module
                try:
                    module_name = f"pyrovelocity.{class_ref}"
                    module = importlib.import_module(module_name)

                    # Get all public functions from the module
                    public_functions = []
                    for name, obj in inspect.getmembers(module):
                        if inspect.isfunction(obj) and not name.startswith('_'):
                            public_functions.append(name)

                    if public_functions:
                        report += "**Module Functions:**\n\n"
                        report += "```python\n"
                        for func in sorted(public_functions):
                            report += f"{func}\n"
                        report += "```\n\n"
                    else:
                        report += "No public functions found in this module.\n\n"
                except (ImportError, AttributeError) as e:
                    report += f"Could not inspect module functions: {e}\n\n"

                continue

            # Skip if not a class or function
            if not inspect.isclass(cls) and not inspect.isfunction(cls):
                continue

            report += f"## {class_ref} <a id='{class_ref.replace('.', '').lower()}'></a>\n\n"
            report += f"```python\n{import_path}\n```\n\n"

            if inspect.isfunction(cls):
                # It's a function, not a class
                report += "This is a function, not a class.\n\n"
                report += f"**Function Signature:**\n\n```python\n{inspect.signature(cls)}\n```\n\n"
                continue

            # Get class and instance methods
            class_methods = get_class_methods(cls)
            instance_methods = get_instance_methods(cls)

            # Filter out private methods
            public_class_methods = [m for m in class_methods if not m.startswith('_')]
            public_instance_methods = [m for m in instance_methods if not m.startswith('_')]

            # Add class methods to report
            if public_class_methods:
                report += "### Class Methods\n\n"
                report += "```python\n"
                for method in sorted(public_class_methods):
                    report += f"{method}\n"
                report += "```\n\n"

            # Add instance methods to report
            if public_instance_methods:
                report += "### Instance Methods\n\n"
                report += "```python\n"
                for method in sorted(public_instance_methods):
                    report += f"{method}\n"
                report += "```\n\n"

            # If no public methods found
            if not public_class_methods and not public_instance_methods:
                report += "No public methods found for this class.\n\n"

    return report


def main():
    """Main function to parse arguments and generate the report."""
    parser = argparse.ArgumentParser(description="Generate a report of PyroVelocity class methods.")
    parser.add_argument("--output", default="class_methods_report.md", help="Output file path")
    args = parser.parse_args()

    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to _quarto.yml
    quarto_yml_path = os.path.join(script_dir, "..", "nbs", "_quarto.yml")

    # Parse _quarto.yml to get class references
    class_references = parse_quarto_yml(quarto_yml_path)

    # Generate the report
    report = generate_report(class_references)

    # Write the report to file
    output_path = os.path.join(script_dir, args.output)
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Report generated at {output_path}")


if __name__ == "__main__":
    main()
