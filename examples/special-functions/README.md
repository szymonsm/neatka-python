# Special Functions Approximation Example

Mathematical function approximation benchmark comparing KAN and feedforward networks on various special functions from the KAN paper.

## Problem Description

Networks learn to approximate mathematical special functions including:
- **Elliptic functions** (`ellipj`, `ellipkinc`, `ellipeinc`)
- **Bessel functions** (`jv`, `yv`, `kv`, `iv`) 
- **Associated Legendre functions** (`lpmv_0`, `lpmv_1`, `lpmv_2`)
- **Spherical harmonics** (`sph_harm_01`, `sph_harm_11`, `sph_harm_02`, `sph_harm_12`, `sph_harm_22`)

**Input**: 2D numerical inputs (x, y)
**Output**: Function values

## Available Functions

Complete list of 15 special functions from the original KAN paper:

- `ellipj` - Jacobian elliptic functions (sn component)
- `ellipkinc` - Incomplete elliptic integral of the first kind  
- `ellipeinc` - Incomplete elliptic integral of the second kind
- `jv` - Bessel function of the first kind
- `yv` - Bessel function of the second kind
- `kv` - Modified Bessel function of the second kind
- `iv` - Modified Bessel function of the first kind
- `lpmv_0` - Associated Legendre function (m=0)
- `lpmv_1` - Associated Legendre function (m=1)
- `lpmv_2` - Associated Legendre function (m=2)
- `sph_harm_01` - Spherical harmonics (m=0, n=1) - real part
- `sph_harm_11` - Spherical harmonics (m=1, n=1) - real part
- `sph_harm_02` - Spherical harmonics (m=0, n=2) - real part
- `sph_harm_12` - Spherical harmonics (m=1, n=2) - real part
- `sph_harm_22` - Spherical harmonics (m=2, n=2) - real part

## Files

### Core Scripts

- **`evolve.py`** - Evolution with function selection and data generation
- **`test.py`** - Test networks on mathematical functions
- **`parameter_sweep.py`** - Comprehensive hyperparameter search

### Analysis & Visualization

- **`analysis.py`** - Statistical analysis of results
- **`kan_neat_analysis.py`** - KAN-specific analysis tools  
- **`simple_analysis.py`** - Basic performance comparison
- **`plot_convergence.py`** - Convergence visualization
- **`run_analysis.py`** - Automated analysis pipeline

### Configuration

- **`config-feedforward`** - Standard neural network settings
- **`config-kan`** - KAN network configuration

### Supporting Files

- **`special_functions.py`** - Function definitions and data generation
- **`pykan_trainer.py`** - Reference PyKAN implementation
- **`results_manager.py`** - Result storage and retrieval
- **`visualize.py`** - Network and function plotting

## Quick Start

### 1. Run Evolution

**Complete command examples:**

**KAN Networks:**
```bash
# Basic KAN evolution
python evolve.py --function jv --net-type kan --generations 100

# With all parameters
python evolve.py --function ellipj --net-type kan --generations 200 --seed 42 --samples 1500 --results-file my_results.csv --config config-kan
```

**Feedforward Networks:**
```bash
# Basic feedforward evolution
python evolve.py --function yv --net-type feedforward --generations 100

# With all parameters  
python evolve.py --function sph_harm_11 --net-type feedforward --generations 150 --seed 123 --samples 1000 --skip-plots --config config-feedforward
```

### 2. Test Function Approximation

**Complete command examples:**

```bash
# Test specific genome file
python test.py --genome results/winner-jv-kan.pkl --net-type kan --function jv --view

# Test with all parameters
python test.py --genome path/to/genome.pkl --net-type kan --function ellipkinc --seed 999 --no-plots --quiet

# Test all results automatically
python test.py --all --results-dir results --net-type kan
```

### 3. Parameter Sweep

**Important: Use --sequential flag for reliable execution:**

```bash
# Sequential parameter sweep (recommended)
python parameter_sweep.py --sequential

# With specific settings
python parameter_sweep.py --sequential --max-parallel 1
```

### 4. Analysis

```bash
# Run complete analysis pipeline
python run_analysis.py --results-dir results/ --generate-plots

# Specific analysis
python analysis.py --function jv --network-type kan
python plot_convergence.py --results-dir results/
```

## Available Functions

Complete list of 15 special functions from the original KAN paper:

- `ellipj` - Jacobian elliptic functions (sn component)
- `ellipkinc` - Incomplete elliptic integral of the first kind  
- `ellipeinc` - Incomplete elliptic integral of the second kind
- `jv` - Bessel function of the first kind
- `yv` - Bessel function of the second kind
- `kv` - Modified Bessel function of the second kind
- `iv` - Modified Bessel function of the first kind
- `lpmv_0` - Associated Legendre function (m=0)
- `lpmv_1` - Associated Legendre function (m=1)
- `lpmv_2` - Associated Legendre function (m=2)
- `sph_harm_01` - Spherical harmonics (m=0, n=1) - real part
- `sph_harm_11` - Spherical harmonics (m=1, n=1) - real part
- `sph_harm_02` - Spherical harmonics (m=0, n=2) - real part
- `sph_harm_12` - Spherical harmonics (m=1, n=2) - real part
- `sph_harm_22` - Spherical harmonics (m=2, n=2) - real part

## Key Parameters

- **Population size**: 100-300
- **Training samples**: 1000 (default)
- **Test samples**: 200  
- **Success criteria**: Low MSE on test set

## Expected Results

- **KAN advantage**: Superior performance on smooth mathematical functions
- **Interpretability**: KAN splines reveal function structure and behavior
- **Complex functions**: Elliptic and spherical harmonics particularly challenging
- **Convergence**: KANs often require fewer generations for mathematical functions
