# PyroVelocity Piecewise Activation Parameter Recovery Validation - Implementation Plan

**Goal**: Implement parameter recovery validation for dimensionless analytical dynamics with latent time and piecewise activation in the PyroVelocity modular framework.

**Strategy**: Bottom-up implementation with incremental testing and integration to avoid implementation tributaries and ensure each component works before building on it.

**Architecture Note**: Model components should be placed into existing modules (e.g., `dynamics.py`, `priors.py`) rather than creating new subpackages. This maintains architectural consistency with the module-per-component-type pattern.

## Phase 1: Core Mathematical Components

**Goal**: Get the analytical solutions working correctly

### 1. PiecewiseActivationDynamicsModel

- [x] ~~Create `src/pyrovelocity/models/modular/components/dynamics_models/piecewise.py`~~ **Added to `dynamics.py`**
- [x] Implement analytical solutions for 3 phases (off, on, return-to-off)
- [x] Handle γ* = 1 special case with τe^(-τ) terms
- [x] Implement steady-state initial conditions
- [x] Add helper variable ξ for compact notation
- [x] Write unit tests for analytical solutions
- [x] Test γ* = 1 boundary case specifically
- [x] Verify solutions match known analytical results
- [x] **Integration check**: Can compute u*, s* for given parameters and time

### 2. PiecewiseActivationPriorModel

- [x] ~~Create `src/pyrovelocity/models/modular/components/priors/piecewise.py`~~ **Added to `priors.py`**
- [x] Implement hierarchical priors for piecewise parameters
- [x] Add LogNormal priors for α*_off, α*_on, t*_on, δ*
- [x] Implement hierarchical Gamma/Normal time structure (T*_M, t_loc, t_scale)
- [x] Add LogNormal prior for capture efficiency λ_j
- [x] Write unit tests for prior sampling
- [x] Test parameter ranges are biologically reasonable
- [x] **Integration check**: Sampled parameters work with dynamics component

## Phase 2: Model Assembly

**Goal**: Create complete working model

### 3. Factory Function

- [x] Update `src/pyrovelocity/models/modular/factory.py`
- [x] Implement `create_piecewise_activation_model()`
- [x] Register components in appropriate registries
- [x] Ensure all required protocols are implemented
- [x] Write unit tests for model creation
- [x] Test all components are properly connected
- [x] **Integration check**: Model has all required protocols, can call basic methods ✅

### 4. Basic PyroVelocityModel Methods

- [x] Implement `model()` method with piecewise dynamics ✅ **Uses existing PyroVelocityModel.forward()**
- [x] Implement `guide()` method for variational inference ✅ **Uses existing PyroVelocityModel.guide()**
- [x] Add basic training capability ✅ **Works with standard SVI**
- [x] Ensure compatibility with existing PyroVelocity training infrastructure ✅
- [x] Write unit tests for basic model functionality ✅
- [x] Test training converges on simple synthetic data ✅
- [x] **Integration check**: Training converges, produces reasonable posteriors ✅

## Phase 3: Parameter Sampling Infrastructure

**Goal**: Enable controlled parameter generation

### 5. Pattern Constraint Logic

- [x] Define quantitative parameter ranges for each pattern ✅
- [x] Implement activation pattern constraints (α*_off < 0.2, α*_on > 1.5, etc.) ✅
- [x] Implement decay pattern constraints (α*_off > 0.5, t*_on > T*_M) ✅
- [x] Implement transient pattern constraints (δ* < 0.3) ✅
- [x] Implement sustained pattern constraints (δ* > 0.6) ✅
- [x] Add fold-change validation logic ✅
- [x] Write unit tests for constraint validation ✅ **Tested with simple validation script**
- [x] Test each pattern generates expected parameter ranges ✅
- [x] **Integration check**: Pattern-constrained parameters produce expected dynamics ✅

### 6. sample_system_parameters() Method

- [x] Implement core parameter sampling method ✅
- [x] Add pattern constraint support ✅
- [x] Add set_id support for multiple parameter sets per pattern ✅
- [x] Implement constrain_to_pattern logic ✅
- [x] Add proper tensor shapes and device handling ✅
- [x] Write unit tests for parameter sampling ✅ **Tested with integration script**
- [x] Test sampling with/without pattern constraints ✅
- [x] Test correct tensor shapes returned ✅
- [x] **Integration check**: Sampled parameters work with dynamics, produce valid data ✅

## Phase 4: Data Generation

**Goal**: Create synthetic datasets for testing

### 7. generate_predictive_samples() Method

- [ ] Implement predictive sampling interface
- [ ] Support prior predictive sampling (samples=None)
- [ ] Support posterior predictive sampling (samples provided)
- [ ] Add multiple return formats (dict, anndata, inference_data)
- [ ] Implement proper AnnData structure with metadata
- [ ] Add pattern type tracking in AnnData.uns
- [ ] Write unit tests for data generation
- [ ] Test both prior and posterior predictive sampling
- [ ] Test all return formats work correctly
- [ ] **Integration check**: Generated data has expected statistical properties

### 8. AnnData Integration

- [ ] Ensure proper storage of true parameters in AnnData.uns
- [ ] Add pattern metadata to AnnData.uns
- [ ] Store model configuration information
- [ ] Add gene and cell naming conventions
- [ ] Implement proper layers structure for count data
- [ ] Write unit tests for AnnData format
- [ ] Test parameter round-trip through AnnData
- [ ] **Integration check**: Can round-trip parameters through AnnData

## Phase 5: Validation Infrastructure

**Goal**: Complete parameter recovery validation

### 9. Training and Posterior Sampling

- [ ] Implement robust training with convergence checking
- [ ] Add ELBO monitoring and convergence criteria
- [ ] Implement posterior sample generation
- [ ] Add proper tensor device handling for training
- [ ] Implement training result structure
- [ ] Write unit tests for training pipeline
- [ ] Test training on synthetic data
- [ ] Test posterior sampling produces reasonable results
- [ ] **Integration check**: Posterior samples have correct format and properties

### 10. validate_parameter_recovery() Method

- [ ] Implement main validation method
- [ ] Add parameter recovery metrics calculation
- [ ] Implement correlation-based recovery assessment
- [ ] Add overall success rate calculation
- [ ] Implement proper result structure with metadata
- [ ] Add training result integration
- [ ] Write unit tests for validation workflow
- [ ] Test full validation on simple test case
- [ ] Test validation result structure
- [ ] **Integration check**: Returns properly structured results

## Phase 6: Visualization and Polish

**Goal**: Complete user experience

### 11. Plotting Functions

- [ ] Create `src/pyrovelocity/plots/piecewise.py`
- [ ] Implement `plot_piecewise_phase_portrait()`
- [ ] Implement `plot_piecewise_trajectories()`
- [ ] Implement `plot_activation_timing()`
- [ ] Create `src/pyrovelocity/plots/predictive_checks.py`
- [ ] Implement `plot_prior_predictive_checks()`
- [ ] Implement `plot_posterior_predictive_checks()`
- [ ] Implement `plot_parameter_recovery_diagnostics()`
- [ ] Write unit tests for plotting functions
- [ ] Test plots generate without errors
- [ ] **Integration check**: Plots are informative and properly formatted

### 12. Documentation Examples

- [ ] Update parameter recovery validation documentation
- [ ] Add working code examples to documentation
- [ ] Create tutorial notebook demonstrating functionality
- [ ] Add API reference documentation
- [ ] Update existing documentation to reference new functionality
- [ ] Write integration tests using documentation examples
- [ ] Test all documentation examples run successfully
- [ ] **Integration check**: Examples demonstrate key functionality

## Testing Strategy

### Unit Tests (per component)

- [ ] Test analytical solutions match expected values
- [ ] Test γ* = 1 boundary case handling
- [ ] Test prior sampling produces reasonable ranges
- [ ] Test pattern constraints work correctly
- [ ] Test data generation produces expected formats
- [ ] Test training converges on known data
- [ ] Test validation metrics are calculated correctly

### Integration Tests (between phases)

- [ ] Test factory creates properly connected model
- [ ] Test components work together correctly
- [ ] Test parameter flow through entire pipeline
- [ ] Test AnnData format consistency
- [ ] Test end-to-end simple parameter recovery

### Validation Tests (full workflow)

- [ ] Test parameter recovery for activation pattern
- [ ] Test parameter recovery for decay pattern
- [ ] Test parameter recovery for transient pattern
- [ ] Test parameter recovery for sustained pattern
- [ ] Test validation across multiple parameter sets
- [ ] Test validation result structure and completeness

## Key Integration Checkpoints

- [x] **After Phase 1**: Can compute analytical solutions correctly ✅ **COMPLETE**
- [x] **After Phase 2**: Have working PyroVelocity model ✅ **COMPLETE**
- [x] **After Phase 3**: Can generate pattern-constrained parameters ✅ **COMPLETE**
- [ ] **After Phase 4**: Can create synthetic validation datasets
- [ ] **After Phase 5**: Can perform complete parameter recovery validation
- [ ] **After Phase 6**: Have complete, documented functionality

## Risk Mitigation Checklist

- [ ] Test each component in isolation before integration
- [ ] Use simple test cases (2 genes, 50 cells) for fast iteration
- [ ] Verify mathematical correctness at each step with known solutions
- [ ] Check protocol compliance as components are added
- [ ] Maintain working examples that demonstrate current functionality
- [ ] Commit working code at each major checkpoint
- [ ] Write tests before implementing complex functionality

## Success Criteria

- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Parameter recovery achieves >90% accuracy on test cases
- [ ] All four gene expression patterns are supported
- [ ] API follows PyroVelocity modular patterns
- [ ] Documentation examples work correctly
- [ ] Code follows PyroVelocity style conventions
