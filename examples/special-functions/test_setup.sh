#!/bin/bash

# Quick test script for special functions experiments
# This script runs a subset of experiments to verify everything works

echo "Testing Special Functions Experiments"
echo "===================================="

# Test that special functions module works
echo "1. Testing special functions module..."
python -c "import special_functions; special_functions.test_all_functions()"

if [ $? -ne 0 ]; then
    echo "❌ Special functions test failed"
    exit 1
fi

echo "✅ Special functions test passed"

# Test a single MLP-NEAT experiment
echo "2. Testing MLP-NEAT on Bessel function..."
python evolve.py --function jv --net-type feedforward --seed 42 --generations 10 --samples 100

if [ $? -ne 0 ]; then
    echo "❌ MLP-NEAT test failed"
    exit 1
fi

echo "✅ MLP-NEAT test passed"

# Test a single KAN-NEAT experiment (if KAN is available)
echo "3. Testing KAN-NEAT on Bessel function..."
python evolve.py --function jv --net-type kan --seed 42 --generations 10 --samples 100

if [ $? -ne 0 ]; then
    echo "⚠️  KAN-NEAT test failed (KAN modules may not be available)"
else
    echo "✅ KAN-NEAT test passed"
fi

# Test PyKAN (if available)
echo "4. Testing PyKAN..."
python -c "
try:
    import pykan_trainer
    if pykan_trainer.PYKAN_AVAILABLE:
        print('PyKAN is available')
        trainer = pykan_trainer.PyKANTrainer('jv')
        print('PyKAN trainer created successfully')
    else:
        print('PyKAN is not available')
except Exception as e:
    print(f'PyKAN test failed: {e}')
"

# Test parameter sweep with minimal configuration
echo "5. Testing parameter sweep (minimal)..."
python parameter_sweep.py --functions jv --methods mlp-neat --seeds 42 --generations 5 --samples 50

if [ $? -ne 0 ]; then
    echo "❌ Parameter sweep test failed"
    exit 1
fi

echo "✅ Parameter sweep test passed"

# Test analysis
echo "6. Testing analysis..."
python analysis.py special_functions_results/

if [ $? -ne 0 ]; then
    echo "⚠️  Analysis test failed (may be due to missing results)"
else
    echo "✅ Analysis test passed"
fi

echo ""
echo "🎉 All tests completed!"
echo ""
echo "To run full experiments:"
echo "  python parameter_sweep.py --methods mlp-neat kan-neat --seeds 42 123 456 --generations 50"
echo ""
echo "To analyze results:"
echo "  python analysis.py special_functions_results/"
