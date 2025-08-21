"""
Quick analysis runner script with dependency checking.
Run this to analyze KAN-NEAT vs PyKAN results.
"""

import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []
    
    try:
        import pandas
        print("✓ pandas available")
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import numpy
        print("✓ numpy available") 
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import matplotlib
        print("✓ matplotlib available")
    except ImportError:
        missing_deps.append("matplotlib")
    
    try:
        import seaborn
        print("✓ seaborn available")
    except ImportError:
        missing_deps.append("seaborn")
    
    try:
        import graphviz
        print("✓ graphviz available")
    except ImportError:
        print("⚠ graphviz not available (network visualizations will be skipped)")
    
    try:
        import neat
        print("✓ neat-python available")
    except ImportError:
        missing_deps.append("neat-python")
    
    try:
        from neat.nn.kan import KANNetwork
        print("✓ KAN networks available")
    except ImportError:
        missing_deps.append("KAN network implementation")
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Please install missing packages before running analysis.")
        return False
    else:
        print("\n✅ All dependencies satisfied!")
        return True

def main():
    print("KAN-NEAT Results Analysis")
    print("=" * 40)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check if results exist
    results_dir = "special_functions_results"
    if not os.path.exists(results_dir):
        print(f"\n❌ Results directory '{results_dir}' not found!")
        print("Please run the parameter sweep first to generate results.")
        return
    
    kan_results_dir = "results"
    if not os.path.exists(kan_results_dir):
        print(f"\n❌ KAN results directory '{kan_results_dir}' not found!")
        print("Please ensure KAN-NEAT experiments have been run.")
        return
    
    print(f"\n✅ Found results directories")
    print(f"   - {results_dir}")
    print(f"   - {kan_results_dir}")
    
    # Run analysis
    try:
        from kan_neat_analysis import KANNEATAnalyzer
        
        print("\nStarting analysis...")
        analyzer = KANNEATAnalyzer()
        summary = analyzer.run_full_analysis()
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE!")
        print("="*60)
        print("\nFiles generated:")
        print("- comparison_summary.csv: Summary table with best results")
        print("- comparison_plots/: Comparison plots for each function")
        print("- comparison_plots/*_network.png: Network visualizations")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
