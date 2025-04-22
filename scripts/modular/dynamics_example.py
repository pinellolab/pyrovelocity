#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script demonstrating the use of dynamics models in the PyroVelocity modular implementation.

This script shows how to use the StandardDynamicsModel and StandardDynamicsModelSimulated
classes to simulate RNA velocity dynamics.
"""

import matplotlib.pyplot as plt
import torch

from pyrovelocity.models.modular.components.dynamics import (
    NonlinearDynamicsModel,
    StandardDynamicsModel,
    StandardDynamicsModelSimulated,
)


def main():
    """Run the dynamics models example."""
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create dynamics models
    standard_model = StandardDynamicsModel()
    standard_model_simulated = StandardDynamicsModelSimulated()
    nonlinear_model = NonlinearDynamicsModel()

    # Set parameters for simulation
    n_genes = 3  # Number of genes to simulate
    t_max = 10.0  # Maximum simulation time
    n_steps = 100  # Number of simulation steps

    # Generate random parameters for the models
    alpha = torch.rand(n_genes) * 0.5 + 0.5  # Transcription rates [0.5, 1.0]
    beta = torch.rand(n_genes) * 0.3 + 0.2  # Splicing rates [0.2, 0.5]
    gamma = torch.rand(n_genes) * 0.2 + 0.1  # Degradation rates [0.1, 0.3]
    scaling = torch.ones(n_genes)  # Scaling factors (all 1.0)

    # Additional parameters for nonlinear model
    k_alpha = torch.rand(n_genes) * 5.0 + 5.0  # Alpha saturation [5.0, 10.0]
    k_beta = torch.rand(n_genes) * 2.0 + 1.0  # Beta saturation [1.0, 3.0]

    # Initial conditions (start from zeros)
    u0 = torch.zeros(n_genes)
    s0 = torch.zeros(n_genes)

    # Run simulations
    print("Running standard model simulation...")
    times_standard, u_standard, s_standard = standard_model.simulate(
        u0=u0,
        s0=s0,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        scaling=scaling,
        t_max=t_max,
        n_steps=n_steps,
    )

    print("Running standard model simulated simulation...")
    times_simulated, u_simulated, s_simulated = standard_model_simulated.simulate(
        u0=u0,
        s0=s0,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        scaling=scaling,
        t_max=t_max,
        n_steps=n_steps,
    )

    print("Running nonlinear model simulation...")
    times_nonlinear, u_nonlinear, s_nonlinear = nonlinear_model.simulate(
        u0=u0,
        s0=s0,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        scaling=scaling,
        t_max=t_max,
        n_steps=n_steps,
        k_alpha=k_alpha,
        k_beta=k_beta,
    )

    # Calculate steady states
    u_ss_standard, s_ss_standard = standard_model.steady_state(alpha, beta, gamma)
    u_ss_simulated, s_ss_simulated = standard_model_simulated.steady_state(alpha, beta, gamma)
    u_ss_nonlinear, s_ss_nonlinear = nonlinear_model.steady_state(
        alpha, beta, gamma, k_alpha=k_alpha, k_beta=k_beta
    )

    # Print steady states
    print("\nSteady States:")
    print(f"Standard Model - Unspliced: {u_ss_standard}")
    print(f"Standard Model - Spliced: {s_ss_standard}")
    print(f"Standard Model Simulated - Unspliced: {u_ss_simulated}")
    print(f"Standard Model Simulated - Spliced: {s_ss_simulated}")
    print(f"Nonlinear Model - Unspliced: {u_ss_nonlinear}")
    print(f"Nonlinear Model - Spliced: {s_ss_nonlinear}")

    # Plot results for the first gene
    gene_idx = 0
    plt.figure(figsize=(15, 10))

    # Plot unspliced counts
    plt.subplot(2, 1, 1)
    plt.plot(times_standard, u_standard[:, gene_idx], label="Standard Model")
    plt.plot(times_simulated, u_simulated[:, gene_idx], label="Standard Model Simulated", linestyle="--")
    plt.plot(times_nonlinear, u_nonlinear[:, gene_idx], label="Nonlinear Model", linestyle="-.")
    plt.axhline(y=u_ss_standard[gene_idx], color="blue", linestyle=":", label="Standard Steady State")
    plt.axhline(y=u_ss_nonlinear[gene_idx], color="green", linestyle=":", label="Nonlinear Steady State")
    plt.xlabel("Time")
    plt.ylabel("Unspliced Counts")
    plt.title(f"Unspliced mRNA Dynamics for Gene {gene_idx}")
    plt.legend()
    plt.grid(True)

    # Plot spliced counts
    plt.subplot(2, 1, 2)
    plt.plot(times_standard, s_standard[:, gene_idx], label="Standard Model")
    plt.plot(times_simulated, s_simulated[:, gene_idx], label="Standard Model Simulated", linestyle="--")
    plt.plot(times_nonlinear, s_nonlinear[:, gene_idx], label="Nonlinear Model", linestyle="-.")
    plt.axhline(y=s_ss_standard[gene_idx], color="blue", linestyle=":", label="Standard Steady State")
    plt.axhline(y=s_ss_nonlinear[gene_idx], color="green", linestyle=":", label="Nonlinear Steady State")
    plt.xlabel("Time")
    plt.ylabel("Spliced Counts")
    plt.title(f"Spliced mRNA Dynamics for Gene {gene_idx}")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("dynamics_example.png")
    plt.close()

    # Phase portrait (spliced vs unspliced)
    plt.figure(figsize=(10, 8))
    plt.plot(u_standard[:, gene_idx], s_standard[:, gene_idx], label="Standard Model")
    plt.plot(u_simulated[:, gene_idx], s_simulated[:, gene_idx], label="Standard Model Simulated", linestyle="--")
    plt.plot(u_nonlinear[:, gene_idx], s_nonlinear[:, gene_idx], label="Nonlinear Model", linestyle="-.")
    plt.scatter(
        [u_ss_standard[gene_idx], u_ss_nonlinear[gene_idx]],
        [s_ss_standard[gene_idx], s_ss_nonlinear[gene_idx]],
        c=["blue", "green"],
        marker="*",
        s=200,
        label="Steady States",
    )
    plt.xlabel("Unspliced Counts")
    plt.ylabel("Spliced Counts")
    plt.title(f"Phase Portrait for Gene {gene_idx}")
    plt.legend()
    plt.grid(True)
    plt.savefig("phase_portrait.png")
    plt.close()

    print("\nSimulation completed. Plots saved as 'dynamics_example.png' and 'phase_portrait.png'.")

    # Predict future states
    print("\nPredicting future states...")
    current_state = (u_standard[50, :], s_standard[50, :])  # State at time step 50
    time_delta = 2.0  # Predict 2 time units ahead

    u_future_standard, s_future_standard = standard_model.predict_future_states(
        current_state, time_delta, alpha, beta, gamma, scaling
    )
    u_future_simulated, s_future_simulated = standard_model_simulated.predict_future_states(
        current_state, time_delta, alpha, beta, gamma, scaling
    )
    # For NonlinearDynamicsModel, we need to create a custom method to handle the additional parameters
    # since the base class doesn't have k_alpha and k_beta parameters
    def predict_nonlinear_future(current_state, time_delta, alpha, beta, gamma, scaling, k_alpha, k_beta):
        # Extract current state
        u_current, s_current = current_state

        # Define the ODE system with saturation effects
        def dudt(u, s):
            return alpha / (1 + u / k_alpha) - beta * u / (k_beta + u)

        def dsdt(u, s):
            return beta * u / (k_beta + u) - gamma * s

        # Simple Euler integration
        dt = 0.01  # Small time step for stability

        # Handle both float and tensor time_delta
        if isinstance(time_delta, float):
            time_delta_value = time_delta
        elif isinstance(time_delta, torch.Tensor):
            time_delta_value = time_delta.item()
        else:
            time_delta_value = float(time_delta)  # Try to convert to float

        steps = int(time_delta_value / dt) + 1
        dt = time_delta_value / steps  # Adjust dt for exact time_delta

        u = u_current.clone()
        s = s_current.clone()

        for _ in range(steps):
            u_new = u + dt * dudt(u, s)
            s_new = s + dt * dsdt(u, s)
            u = u_new
            s = s_new

        u_future = u
        s_future = s

        # Apply scaling if provided
        if scaling is not None:
            u_future = u_future * scaling
            s_future = s_future * scaling

        return u_future, s_future

    # Use our custom function for the nonlinear model
    u_future_nonlinear, s_future_nonlinear = predict_nonlinear_future(
        current_state, time_delta, alpha, beta, gamma, scaling, k_alpha, k_beta
    )

    print(f"Current state - Unspliced: {current_state[0]}")
    print(f"Current state - Spliced: {current_state[1]}")
    print(f"Future state (Standard) - Unspliced: {u_future_standard}")
    print(f"Future state (Standard) - Spliced: {s_future_standard}")
    print(f"Future state (Simulated) - Unspliced: {u_future_simulated}")
    print(f"Future state (Simulated) - Spliced: {s_future_simulated}")
    print(f"Future state (Nonlinear) - Unspliced: {u_future_nonlinear}")
    print(f"Future state (Nonlinear) - Spliced: {s_future_nonlinear}")

    # Verify conservation laws
    print("\nVerifying conservation laws...")
    # For standard model: du/dt = alpha - beta * u
    u_current = current_state[0]
    expected_dudt_standard = alpha - beta * u_current
    print(f"Expected du/dt (Standard): {expected_dudt_standard[gene_idx]:.6f}")

    # For nonlinear model: du/dt = alpha / (1 + u / k_alpha) - beta * u / (k_beta + u)
    expected_dudt_nonlinear = alpha / (1 + u_current / k_alpha) - beta * u_current / (k_beta + u_current)
    print(f"Expected du/dt (Nonlinear): {expected_dudt_nonlinear[gene_idx]:.6f}")

    # Numerical approximation of derivatives
    dt = 0.01
    # Use forward method for standard model to get next state
    u_ss_standard, s_ss_standard = standard_model.forward(
        current_state[0], current_state[1], alpha, beta, gamma, scaling
    )
    u_next_standard = u_ss_standard
    u_next_nonlinear, _ = predict_nonlinear_future(
        current_state, dt, alpha, beta, gamma, scaling, k_alpha, k_beta
    )

    numerical_dudt_standard = (u_next_standard - u_current) / dt
    numerical_dudt_nonlinear = (u_next_nonlinear - u_current) / dt

    print(f"Numerical du/dt (Standard): {numerical_dudt_standard[gene_idx]:.6f}")
    print(f"Numerical du/dt (Nonlinear): {numerical_dudt_nonlinear[gene_idx]:.6f}")

    # Check if they match
    print(
        f"Standard model conservation law check: "
        f"{'PASSED' if torch.allclose(expected_dudt_standard, numerical_dudt_standard, rtol=1e-2) else 'FAILED'}"
    )
    print(
        f"Nonlinear model conservation law check: "
        f"{'PASSED' if torch.allclose(expected_dudt_nonlinear, numerical_dudt_nonlinear, rtol=1e-2) else 'FAILED'}"
    )


if __name__ == "__main__":
    main()
