# Lunar Lander Example

This example demonstrates both feedforward and KAN network training on the OpenAI Gym LunarLander-v2 environment.

## Problem Description

The lunar lander must learn to land safely on the moon using 4 discrete actions:
- Do nothing
- Fire left engine  
- Fire main engine
- Fire right engine

**Input**: 8 state variables (position, velocity, angle, angular velocity, leg contact)
**Output**: 4 action probabilities

## Files

### Core Scripts

- **`evolve.py`** - Main evolution script supporting both network types
- **`test.py`** - Test and visualize evolved networks with movie generation  
- **`parameters_sweep.py`** - Systematic hyperparameter exploration

### Configuration

- **`config-feedforward`** - Configuration for standard neural networks
- **`config-kan`** - Configuration for KAN networks

### Supporting Files

- **`lunar_lander.py`** - Environment wrapper and simulation logic
- **`fitness.py`** - Fitness evaluation functions
- **`visualize.py`** - Network visualization tools
- **`movie.py`** - Generate simulation videos
- **`results_manager.py`** - Result logging and analysis

## Quick Start

### 1. Run Evolution

**Complete command examples:**

**KAN Networks:**
```bash
# Basic KAN evolution
python evolve.py --net-type kan --generations 100

# With all parameters
python evolve.py --net-type kan --generations 150 --seed 42 --config config-kan --results-file lunar_kan_results.csv --skip-plots
```

**Feedforward Networks:**
```bash
# Basic feedforward evolution
python evolve.py --net-type feedforward --generations 100

# With all parameters
python evolve.py --net-type feedforward --generations 200 --seed 123 --config config-feedforward --results-file lunar_ff_results.csv
```

### 2. Test Best Solution

**Complete command examples:**

```bash
# Test with movie generation and plots
python test.py --net-type kan --episodes 10 --view --seed 999

# Test specific genome file with all options
python test.py --genome results/winner-kan.pkl --net-type kan --episodes 5 --seed 42 --no-movie --no-plots --quiet

# Minimal test (uses latest results automatically)
python test.py --net-type feedforward --episodes 3
```

### 3. Parameter Sweep

**Important: Use --sequential flag for reliable execution:**

```bash
# Sequential parameter sweep (recommended)
python parameters_sweep.py --sequential

# With limited parallel processes
python parameters_sweep.py --sequential --max-parallel 1
```

## Key Parameters

- **Population size**: 150 (KAN) / 100 (feedforward)
- **Hidden nodes**: 8  
- **Generations**: 50-100
- **Evaluation runs**: 10 per genome

## Expected Results

- **Successful landing**: Fitness > 200
- **KAN networks**: Often more interpretable solution paths
- **Feedforward**: Typically faster convergence
