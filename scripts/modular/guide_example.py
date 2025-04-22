#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating the use of different guide implementations in Pyro.

This script shows how to use different guide types (AutoNormal, AutoDelta, and AutoDiagonalNormal)
to perform variational inference on a simple model.
"""

import torch
import pyro
import pyro.distributions as dist
import pyro.optim
import pyro.infer
import pyro.infer.autoguide
import matplotlib.pyplot as plt
import numpy as np


def simple_model(data):
    """A simple model with a normal prior and normal likelihood."""
    # Sample a parameter from a normal prior
    mu = pyro.sample("mu", dist.Normal(0.0, 1.0))
    # Sample observations from a normal likelihood
    with pyro.plate("data", len(data)):
        pyro.sample("obs", dist.Normal(mu, 0.1), obs=data)
    return mu


def main():
    """Run the guide examples."""
    # Set random seed for reproducibility
    pyro.set_rng_seed(42)
    torch.manual_seed(42)

    # Generate synthetic data
    true_mu = 0.5
    n_samples = 100
    data = torch.normal(true_mu, 0.1, size=(n_samples,))

    print(f"Generated {n_samples} data points with true mu = {true_mu}")
    print(f"Sample mean: {data.mean().item():.4f}")
    print(f"Sample std: {data.std().item():.4f}")

    # Create guides
    auto_normal = pyro.infer.autoguide.AutoNormal(simple_model)
    auto_delta = pyro.infer.autoguide.AutoDelta(simple_model)
    auto_diag = pyro.infer.autoguide.AutoDiagonalNormal(simple_model)

    # Dictionary to store results
    results = {}

    # Train each guide
    for guide_name, guide in [
        ("AutoNormal", auto_normal),
        ("AutoDelta", auto_delta),
        ("AutoDiagonalNormal", auto_diag),
    ]:
        print(f"\nTraining {guide_name}...")

        # Clear parameter store
        pyro.clear_param_store()

        # Set up optimizer
        optimizer = pyro.optim.Adam({"lr": 0.01})
        svi = pyro.infer.SVI(simple_model, guide, optimizer, loss=pyro.infer.Trace_ELBO())

        # Train for a few steps
        n_steps = 1000
        losses = []
        for step in range(n_steps):
            loss = svi.step(data)
            losses.append(loss)
            if step % 100 == 0:
                print(f"Step {step}: loss = {loss:.4f}")

        # Store losses
        results[guide_name] = {
            "losses": losses,
            "final_loss": losses[-1],
        }

        # Generate posterior samples
        predictive = pyro.infer.Predictive(guide, num_samples=1000)
        samples = predictive()
        mu_samples = samples["mu"]

        # Compute posterior statistics
        mu_mean = mu_samples.mean().item()
        mu_std = mu_samples.std().item()

        print(f"Posterior mean of mu: {mu_mean:.4f}")
        print(f"Posterior std of mu: {mu_std:.4f}")

        # Store posterior statistics
        results[guide_name]["mu_mean"] = mu_mean
        results[guide_name]["mu_std"] = mu_std
        results[guide_name]["mu_samples"] = mu_samples

    # Plot losses
    plt.figure(figsize=(10, 6))
    for guide_name, result in results.items():
        plt.plot(result["losses"], label=f"{guide_name} (final={result['final_loss']:.2f})")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss for Different Guides")
    plt.legend()
    plt.grid(True)
    plt.savefig("guide_losses.png")
    plt.close()

    # Plot posterior distributions
    plt.figure(figsize=(12, 6))
    for i, (guide_name, result) in enumerate(results.items()):
        plt.subplot(1, 3, i+1)
        plt.hist(result["mu_samples"].detach().numpy(), bins=30, alpha=0.7, density=True)
        plt.axvline(true_mu, color='r', linestyle='--', label=f"True mu={true_mu}")
        plt.axvline(result["mu_mean"], color='g', linestyle='-', label=f"Posterior mean={result['mu_mean']:.4f}")
        plt.title(f"{guide_name} Posterior")
        plt.xlabel("mu")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
    plt.tight_layout()
    plt.savefig("guide_posteriors.png")
    plt.close()

    # Compare guide performance
    print("\nGuide Performance Comparison:")
    print("-" * 80)
    print(f"{'Guide Name':<20} {'Final Loss':<15} {'Posterior Mean':<20} {'Posterior Std':<15}")
    print("-" * 80)
    for guide_name, result in results.items():
        print(f"{guide_name:<20} {result['final_loss']:<15.4f} {result['mu_mean']:<20.4f} {result['mu_std']:<15.4f}")
    print("-" * 80)

    print("\nSimulation completed. Plots saved as 'guide_losses.png' and 'guide_posteriors.png'.")


if __name__ == "__main__":
    main()
