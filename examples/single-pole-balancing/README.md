# Single Pole Balancing Example

Classic cart-pole balancing control task comparing feedforward and KAN network performance.

## Problem Description

A cart with a pole must learn to balance the pole upright by moving left or right.

**Input**: 4 state variables (cart position, cart velocity, pole angle, pole angular velocity)  
**Output**: 1 continuous control signal (force applied to cart)

## Files

### Core Scripts

- **`evolve.py`** - Main evolution script for both network types
- **`test.py`** - Test evolved networks with visualization
- **`parameters_sweep.py`** - Hyperparameter optimization

### Configuration

- **`config-feedforward`** - Standard neural network configuration
- **`config-kan`** - KAN network configuration  

### Supporting Files

- **`cart_pole.py`** - Cart-pole physics simulation
- **`fitness.py`** - Fitness evaluation (balancing time)
- **`visualize.py`** - Network and performance visualization
- **`movie.py`** - Generate simulation videos
- **`results_manager.py`** - Logging and analysis tools

## Quick Start

### 1. Run Evolution

**Complete command examples:**

**KAN Networks:**
```bash
# Basic KAN evolution
python evolve.py --net-type kan --generations 50

# With all parameters
python evolve.py --net-type kan --generations 100 --seed 42 --config config-kan --results-file pole_kan_results.csv --skip-plots
```

**Feedforward Networks:**
```bash
# Basic feedforward evolution
python evolve.py --net-type feedforward --generations 50

# With all parameters
python evolve.py --net-type feedforward --generations 75 --seed 123 --config config-feedforward --results-file pole_ff_results.csv
```

### 2. Test Solution

**Complete command examples:**

```bash
# Test with visualization
python test.py --net-type kan --view --seed 999

# Test specific genome file with all options
python test.py --genome results/winner-kan.pkl --net-type kan --seed 42 --no-movie --no-plots --quiet

# Test with movie generation
python test.py --net-type feedforward --view
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

- **Population size**: 150
- **Simulation time**: 60 seconds maximum
- **Success criteria**: Balance for full duration
- **Evaluation runs**: 5 per genome

## Expected Results

- **Success**: Balancing for 60 seconds (fitness ≈ 1000)
- **KAN advantage**: Often finds smoother control policies
- **Convergence**: Typically within 20-50 generations
