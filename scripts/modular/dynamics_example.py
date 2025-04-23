"""
Example script demonstrating the different dynamics models in the PyroVelocity modular architecture.

This script shows how to:
1. Create different dynamics models
2. Simulate RNA dynamics over time
3. Compute steady states
4. Visualize the results
5. Predict future states
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

from pyrovelocity.models.modular.registry import DynamicsModelRegistry


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create dynamics models
    standard_model = DynamicsModelRegistry.create("standard")
    standard_model_simulated = DynamicsModelRegistry.create("standard_simulated")
    nonlinear_model = DynamicsModelRegistry.create("nonlinear")

    # Set up parameters
    n_genes = 3
    n_times = 100
    times = torch.linspace(0, 10, n_times)

    # Initial conditions
    u0 = torch.zeros(n_genes)
    s0 = torch.zeros(n_genes)

    # Model parameters
    alpha = torch.tensor([1.0, 1.5, 0.8])  # Transcription rates
    beta = torch.tensor([0.5, 0.5, 0.4])   # Splicing rates
    gamma = torch.tensor([0.3, 0.4, 0.3])  # Degradation rates
    scaling = None

    # Additional parameters for nonlinear model
    k_alpha = torch.tensor([0.5, 0.5, 0.5])  # Saturation constant for transcription
    k_beta = torch.tensor([0.5, 0.5, 0.5])   # Saturation constant for splicing

    # Simulate dynamics over time
    # Note: The simulate method is not part of the DynamicsModel interface
    # We'll implement our own simulation function
    
    def simulate_standard(model, u0, s0, alpha, beta, gamma, times, scaling=None):
        """Simulate standard dynamics model over time."""
        n_times = len(times)
        n_genes = len(u0)
        u = torch.zeros((n_times, n_genes))
        s = torch.zeros((n_times, n_genes))
        
        # Set initial conditions
        u[0] = u0
        s[0] = s0
        
        # Simple Euler integration
        dt = times[1] - times[0]
        
        for t in range(1, n_times):
            # Standard model: du/dt = alpha - beta * u, ds/dt = beta * u - gamma * s
            u[t] = u[t-1] + dt * (alpha - beta * u[t-1])
            s[t] = s[t-1] + dt * (beta * u[t-1] - gamma * s[t-1])
            
        return times, u, s
    
    def simulate_nonlinear(model, u0, s0, alpha, beta, gamma, times, k_alpha, k_beta, scaling=None):
        """Simulate nonlinear dynamics model over time."""
        n_times = len(times)
        n_genes = len(u0)
        u = torch.zeros((n_times, n_genes))
        s = torch.zeros((n_times, n_genes))
        
        # Set initial conditions
        u[0] = u0
        s[0] = s0
        
        # Simple Euler integration
        dt = times[1] - times[0]
        
        for t in range(1, n_times):
            # Nonlinear model: du/dt = alpha / (1 + u / k_alpha) - beta * u / (k_beta + u)
            #                  ds/dt = beta * u / (k_beta + u) - gamma * s
            dudt = alpha / (1 + u[t-1] / k_alpha) - beta * u[t-1] / (k_beta + u[t-1])
            dsdt = beta * u[t-1] / (k_beta + u[t-1]) - gamma * s[t-1]
            
            u[t] = u[t-1] + dt * dudt
            s[t] = s[t-1] + dt * dsdt
            
        return times, u, s

    # Run simulations
    print("Running standard model simulation...")
    times_standard, u_standard, s_standard = simulate_standard(
        standard_model,
        u0=u0,
        s0=s0,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        times=times,
        scaling=scaling
    )

    print("Running standard model simulated simulation...")
    times_simulated, u_simulated, s_simulated = simulate_standard(
        standard_model_simulated,
        u0=u0,
        s0=s0,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        times=times,
        scaling=scaling
    )

    print("Running nonlinear model simulation...")
    times_nonlinear, u_nonlinear, s_nonlinear = simulate_nonlinear(
        nonlinear_model,
        u0=u0,
        s0=s0,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        times=times,
        k_alpha=k_alpha,
        k_beta=k_beta,
        scaling=scaling
    )

    # Compute steady states
    u_ss_standard, s_ss_standard = standard_model.steady_state(alpha, beta, gamma)
    u_ss_simulated, s_ss_simulated = standard_model_simulated.steady_state(alpha, beta, gamma)
    
    # For nonlinear model, we need to compute steady state manually
    # At steady state: du/dt = 0, ds/dt = 0
    # Solving: alpha / (1 + u / k_alpha) = beta * u / (k_beta + u)
    #          beta * u / (k_beta + u) = gamma * s
    # This is a quadratic equation for u, we'll use the model's steady_state method
    u_ss_nonlinear, s_ss_nonlinear = nonlinear_model.steady_state(
        alpha, beta, gamma, k_alpha=k_alpha, k_beta=k_beta
    )

    print("\nSteady States:")
    print(f"Standard Model - Unspliced: {u_ss_standard}")
    print(f"Standard Model - Spliced: {s_ss_standard}")
    print(f"Standard Model Simulated - Unspliced: {u_ss_simulated}")
    print(f"Standard Model Simulated - Spliced: {s_ss_simulated}")
    print(f"Nonlinear Model - Unspliced: {u_ss_nonlinear}")
    print(f"Nonlinear Model - Spliced: {s_ss_nonlinear}")

    # Visualize the results
    gene_idx = 0  # Choose a gene to visualize

    # Create a figure with two subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    # Plot unspliced mRNA dynamics
    axs[0].plot(times_standard, u_standard[:, gene_idx], label="Standard Model")
    axs[0].plot(times_simulated, u_simulated[:, gene_idx], label="Standard Model Simulated", linestyle="--")
    axs[0].plot(times_nonlinear, u_nonlinear[:, gene_idx], label="Nonlinear Model", linestyle="-.")
    axs[0].axhline(y=u_ss_standard[gene_idx].item(), color="blue", linestyle=":", label="Standard Steady State")
    axs[0].axhline(y=u_ss_nonlinear[gene_idx].item(), color="green", linestyle=":", label="Nonlinear Steady State")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Unspliced Counts")
    axs[0].set_title(f"Unspliced mRNA Dynamics for Gene {gene_idx}")
    axs[0].legend()

    # Plot spliced mRNA dynamics
    axs[1].plot(times_standard, s_standard[:, gene_idx], label="Standard Model")
    axs[1].plot(times_simulated, s_simulated[:, gene_idx], label="Standard Model Simulated", linestyle="--")
    axs[1].plot(times_nonlinear, s_nonlinear[:, gene_idx], label="Nonlinear Model", linestyle="-.")
    axs[1].axhline(y=s_ss_standard[gene_idx].item(), color="blue", linestyle=":", label="Standard Steady State")
    axs[1].axhline(y=s_ss_nonlinear[gene_idx].item(), color="green", linestyle=":", label="Nonlinear Steady State")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Spliced Counts")
    axs[1].set_title(f"Spliced mRNA Dynamics for Gene {gene_idx}")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig("dynamics_example.png")

    # Create a phase portrait
    plt.figure(figsize=(10, 8))
    plt.plot(u_standard[:, gene_idx], s_standard[:, gene_idx], label="Standard Model")
    plt.plot(u_simulated[:, gene_idx], s_simulated[:, gene_idx], label="Standard Model Simulated", linestyle="--")
    plt.plot(u_nonlinear[:, gene_idx], s_nonlinear[:, gene_idx], label="Nonlinear Model", linestyle="-.")
    plt.scatter(u_ss_standard[gene_idx].item(), s_ss_standard[gene_idx].item(), color="blue", s=100, marker="*", label="Standard Steady State")
    plt.scatter(u_ss_nonlinear[gene_idx].item(), s_ss_nonlinear[gene_idx].item(), color="green", s=100, marker="*", label="Nonlinear Steady State")
    plt.xlabel("Unspliced Counts")
    plt.ylabel("Spliced Counts")
    plt.title(f"Phase Portrait for Gene {gene_idx}")
    plt.legend()
    plt.savefig("phase_portrait.png")
    plt.close()

    print("\nSimulation completed. Plots saved as 'dynamics_example.png' and 'phase_portrait.png'.")

    # Predict future states
    print("\nPredicting future states...")
    current_state = (u_standard[50, :], s_standard[50, :])  # State at time step 50
    time_delta = float(2.0)  # Predict 2 time units ahead - must be a float, not a tensor

    # For standard model, we can compute the future state analytically
    def predict_standard_future(current_state, time_delta, alpha, beta, gamma, scaling=None):
        """Predict future state for standard model."""
        u_current, s_current = current_state
        
        # Standard model solution:
        # u(t) = u_0 * exp(-beta * t) + alpha/beta * (1 - exp(-beta * t))
        # s(t) = s_0 * exp(-gamma * t) + beta * u_0 / (gamma - beta) * (exp(-beta * t) - exp(-gamma * t))
        #        + alpha * beta / (beta * gamma) * (1 - exp(-gamma * t) - gamma/(gamma - beta) * (exp(-beta * t) - exp(-gamma * t)))
        
        # Simplified for the case where we start from a known state
        u_future = u_current * torch.exp(-beta * time_delta) + (alpha / beta) * (1 - torch.exp(-beta * time_delta))
        
        # For s, we need to be careful about the case where beta = gamma
        # We'll use a small epsilon to avoid division by zero
        epsilon = 1e-10
        beta_gamma_diff = gamma - beta
        beta_gamma_diff = torch.where(torch.abs(beta_gamma_diff) < epsilon, epsilon * torch.sign(beta_gamma_diff), beta_gamma_diff)
        
        term1 = s_current * torch.exp(-gamma * time_delta)
        term2 = (beta * u_current / beta_gamma_diff) * (torch.exp(-beta * time_delta) - torch.exp(-gamma * time_delta))
        term3 = (alpha * beta / (beta * gamma)) * (1 - torch.exp(-gamma * time_delta))
        term4 = (alpha * beta * gamma / (beta * gamma * beta_gamma_diff)) * (torch.exp(-beta * time_delta) - torch.exp(-gamma * time_delta))
        
        s_future = term1 + term2 + term3 - term4
        
        # Apply scaling if provided
        if scaling is not None:
            u_future = u_future * scaling
            s_future = s_future * scaling
            
        return u_future, s_future

    # Use our custom function for the standard model
    u_future_standard, s_future_standard = predict_standard_future(
        current_state, time_delta, alpha, beta, gamma, scaling
    )
    
    # Use the same function for the simulated model since they have the same dynamics
    u_future_simulated, s_future_simulated = predict_standard_future(
        current_state, time_delta, alpha, beta, gamma, scaling
    )

    # For NonlinearDynamicsModel, we need to create a custom method to handle the additional parameters
    def predict_nonlinear_future(current_state, time_delta, alpha, beta, gamma, scaling, k_alpha, k_beta):
        """Predict future state for nonlinear model using numerical integration."""
        # Extract current state
        u_current, s_current = current_state

        # Define the ODE system with saturation effects
        def dudt(u, s):
            return alpha / (1 + u / k_alpha) - beta * u / (k_beta + u)

        def dsdt(u, s):
            return beta * u / (k_beta + u) - gamma * s

        # Simple Euler integration
        dt = 0.01  # Small time step for stability
        time_delta_value = float(time_delta)  # Ensure it's a float
        
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
    # Use our predict function to get next state
    u_next_standard, _ = predict_standard_future(
        current_state, dt, alpha, beta, gamma, scaling
    )
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
