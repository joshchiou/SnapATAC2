#!/usr/bin/env python3
"""
Simple test script to verify mathematical equivalence between spectral() and spectral_slepc()
for a single large dataset configuration.

Usage:
    python test_spectral_simple_equiv.py [--n_obs N] [--n_vars N] [--n_comps N]
"""

import numpy as np
import scipy.sparse as sp
import time
import sys
import os
import argparse
import anndata as ad

# Add the python directory to the path to import snapatac2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

try:
    import snapatac2
    from snapatac2.tools._embedding import spectral, spectral_slepc
    import snapatac2._snapatac2 as internal
except ImportError as e:
    print(f"Error importing snapatac2: {e}")
    sys.exit(1)


def create_test_data(n_obs=5000, n_vars=10000, sparsity=0.95, random_state=42):
    """Create synthetic single-cell ATAC-seq-like data."""
    np.random.seed(random_state)

    print(
        f"Creating test data: {n_obs} cells × {n_vars} features (sparsity={sparsity:.2f})"
    )

    # Generate sparse count matrix
    prob_nonzero = 1 - sparsity
    n_nonzero = int(n_obs * n_vars * prob_nonzero)

    row_indices = np.random.randint(0, n_obs, n_nonzero)
    col_indices = np.random.randint(0, n_vars, n_nonzero)
    counts = np.random.negative_binomial(n=2, p=0.3, size=n_nonzero) + 1

    X = sp.csr_matrix(
        (counts.astype(np.float64), (row_indices, col_indices)), shape=(n_obs, n_vars)
    )
    X.sum_duplicates()

    # Create AnnData
    adata = ad.AnnData(X=X)
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"peak_{i}" for i in range(n_vars)]

    # Select top 50% most variable features
    # Compute variance for sparse matrix manually
    X_mean = np.array(X.mean(axis=0)).flatten()
    X_squared_mean = np.array(X.multiply(X).mean(axis=0)).flatten()
    feature_vars = X_squared_mean - X_mean**2

    n_selected = n_vars // 2
    selected_indices = np.argsort(feature_vars)[-n_selected:]
    selected_mask = np.zeros(n_vars, dtype=bool)
    selected_mask[selected_indices] = True
    # Store as numpy array to avoid pandas Series issues
    adata.var["selected"] = selected_mask.copy()

    print(f"Selected {np.sum(selected_mask)} features")
    return adata


def validate_test_parameters(n_obs, n_vars, n_comps, tolerance):
    """Validate test parameters and provide recommendations."""

    warnings = []
    recommendations = []

    # Check dataset size
    if n_obs < 500:
        warnings.append(
            f"Dataset is very small ({n_obs} cells). Small datasets may show numerical differences due to poor conditioning."
        )
        recommendations.append("Consider using n_obs >= 1000 for reliable comparison")

    if n_vars < 1000:
        warnings.append(
            f"Feature count is small ({n_vars} features). This may lead to numerical instability."
        )
        recommendations.append("Consider using n_vars >= 2000 for reliable comparison")

    # Check sparsity vs size
    expected_nnz = n_obs * n_vars * 0.05  # Assuming 95% sparsity
    if expected_nnz < 10000:
        warnings.append(
            f"Matrix will be very sparse ({expected_nnz:.0f} non-zero elements). This may cause numerical issues."
        )
        recommendations.append("Consider increasing dataset size or reducing sparsity")

    # Check tolerance
    if tolerance < 1e-10:
        warnings.append(
            f"Tolerance is very strict ({tolerance}). Different numerical methods may not achieve this precision."
        )
        recommendations.append(
            "Consider using tolerance >= 1e-8 for practical comparisons"
        )

    # Check components
    if n_comps >= min(n_obs, n_vars) * 0.8:
        warnings.append(
            f"Requesting many components ({n_comps}) relative to dataset size. This may cause convergence issues."
        )
        recommendations.append("Consider using n_comps <= min(n_obs, n_vars) // 4")

    if warnings:
        print(f"\n{'='*50}")
        print("PARAMETER VALIDATION WARNINGS")
        print(f"{'='*50}")
        for w in warnings:
            print(f"⚠️  {w}")

        print("\nRecommendations:")
        for r in recommendations:
            print(f"💡 {r}")

        print(f"\nSuggested parameters for reliable testing:")
        print(f"  --n_obs 2000 --n_vars 5000 --n_comps 30 --tolerance 1e-6")
        print(f"{'='*50}")

    return len(warnings) == 0


def run_test(n_obs=5000, n_vars=10000, n_comps=30, tolerance=1e-8):
    """Run the equivalence test."""

    print(f"\n{'='*60}")
    print(f"SPECTRAL vs SLEPC EQUIVALENCE TEST")
    print(f"{'='*60}")
    print(f"Parameters: {n_obs} cells, {n_vars} features, {n_comps} components")
    print(f"Tolerance: {tolerance}")

    # Validate test parameters
    if not validate_test_parameters(n_obs, n_vars, n_comps, tolerance):
        print(
            "Parameter validation failed. Please adjust parameters based on warnings."
        )
        return False

    # Create test data
    adata = create_test_data(n_obs, n_vars)

    # Test parameters
    test_params = {
        "n_comps": n_comps,
        "features": "selected",
        "random_state": 42,
        "distance_metric": "cosine",
        "weighted_by_sd": True,
        "inplace": False,
    }

    # Add verbose=False for cleaner output in slepc
    slepc_params = test_params.copy()
    slepc_params["verbose"] = True  # Enable verbose output to debug the issue

    # Run spectral()
    print(f"\n{'='*30}")
    print("Running spectral()...")
    print(f"{'='*30}")

    start_time = time.time()
    try:
        evals_spectral, evecs_spectral = spectral(adata, **test_params)
        time_spectral = time.time() - start_time
        print(f"✓ Completed in {time_spectral:.2f}s")
        print(
            f"  Eigenvalues: {evals_spectral.shape}, range: [{evals_spectral.min():.6f}, {evals_spectral.max():.6f}]"
        )
        print(f"  Eigenvectors: {evecs_spectral.shape}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

    # Run spectral_slepc()
    print(f"\n{'='*30}")
    print("Running spectral_slepc()...")
    print(f"{'='*30}")

    start_time = time.time()
    try:
        result_slepc = spectral_slepc(adata, **slepc_params)
        time_slepc = time.time() - start_time

        if result_slepc is None:
            print(f"✓ Completed in {time_slepc:.2f}s (non-root MPI rank)")
            print("Cannot compare results on non-root MPI rank")
            return True

        evals_slepc, evecs_slepc = result_slepc
        print(f"✓ Completed in {time_slepc:.2f}s")
        print(
            f"  Eigenvalues: {evals_slepc.shape}, range: [{evals_slepc.min():.6f}, {evals_slepc.max():.6f}]"
        )
        print(f"  Eigenvectors: {evecs_slepc.shape}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

    # Compare results
    print(f"\n{'='*30}")
    print("Comparing results...")
    print(f"{'='*30}")

    n_compare = min(
        len(evals_spectral),
        len(evals_slepc),
        evecs_spectral.shape[1],
        evecs_slepc.shape[1],
    )

    print(f"Comparing first {n_compare} components")

    # Compare eigenvalues
    eval_diff = np.abs(evals_spectral[:n_compare] - evals_slepc[:n_compare])
    eval_max_diff = np.max(eval_diff)
    eval_rel_error = eval_max_diff / (
        np.max(np.abs(evals_spectral[:n_compare])) + 1e-15
    )

    print(f"\nEigenvalues:")
    print(f"  Max absolute difference: {eval_max_diff:.2e}")
    print(f"  Relative error: {eval_rel_error:.2e}")
    print(f"  Match: {'✓' if eval_max_diff < tolerance else '✗'}")

    # Compare eigenvectors (handle sign ambiguity)
    evec_diffs = []
    for i in range(n_compare):
        v1 = evecs_spectral[:, i]
        v2 = evecs_slepc[:, i]
        diff_pos = np.linalg.norm(v1 - v2)
        diff_neg = np.linalg.norm(v1 + v2)
        evec_diffs.append(min(diff_pos, diff_neg))

    evec_max_diff = np.max(evec_diffs)
    evec_rel_error = evec_max_diff / (np.sqrt(evecs_spectral.shape[0]) + 1e-15)

    print(f"\nEigenvectors:")
    print(f"  Max L2 difference (sign-corrected): {evec_max_diff:.2e}")
    print(f"  Relative error: {evec_rel_error:.2e}")
    print(f"  Match: {'✓' if evec_max_diff < tolerance else '✗'}")

    # Overall result
    overall_match = (eval_max_diff < tolerance) and (evec_max_diff < tolerance)

    print(f"\n{'='*60}")
    if overall_match:
        print("🎉 RESULT: METHODS ARE MATHEMATICALLY EQUIVALENT!")
        print(
            f"   Both eigenvalues and eigenvectors match within tolerance {tolerance}"
        )
    else:
        print("⚠️  RESULT: METHODS DIFFER BEYOND TOLERANCE")
        print(f"   Differences exceed tolerance {tolerance}")

    # Provide additional context for failed comparisons
    if not overall_match:
        print(f"\nDiagnostic Information:")
        print(f"  Dataset size: {n_obs} cells × {n_vars} features")
        print(f"  Matrix density: ~5.0% (assumed from sparsity=0.95)")
        print(
            f"  Eigenvalue range: [{evals_spectral.min():.6f}, {evals_spectral.max():.6f}]"
        )
        print(
            f"  Eigenvalue condition number: {evals_spectral.max() / (evals_spectral.min() + 1e-15):.2e}"
        )

        print(f"\nPossible reasons for differences:")
        if n_obs < 1000 or n_vars < 2000:
            print(
                f"  • Dataset is small - numerical methods may differ on small problems"
            )
        if eval_max_diff > 1e-6:
            print(
                f"  • Large eigenvalue differences suggest algorithmic or convergence differences"
            )
        if evec_max_diff > 1e-3:
            print(
                f"  • Large eigenvector differences may indicate different eigenspaces"
            )

        print(f"\nTo improve agreement:")
        print(f"  • Increase dataset size (recommended: n_obs≥2000, n_vars≥5000)")
        print(f"  • Use more relaxed tolerance (e.g., 1e-6 instead of {tolerance})")
        print(f"  • Check that both methods converged properly")
    print(f"{'='*60}")

    # Performance comparison
    if time_spectral > 0:
        speedup = time_spectral / time_slepc
        print(f"\nPerformance:")
        print(f"  spectral(): {time_spectral:.2f}s")
        print(f"  spectral_slepc(): {time_slepc:.2f}s")
        print(
            f"  Speedup: {speedup:.2f}x {'(SLEPc faster)' if speedup > 1 else '(spectral faster)'}"
        )

    return overall_match


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Test spectral vs spectral_slepc equivalence"
    )
    parser.add_argument(
        "--n_obs", type=int, default=5000, help="Number of observations (cells)"
    )
    parser.add_argument(
        "--n_vars", type=int, default=10000, help="Number of variables (features)"
    )
    parser.add_argument("--n_comps", type=int, default=30, help="Number of components")
    parser.add_argument(
        "--tolerance", type=float, default=1e-8, help="Numerical tolerance"
    )

    args = parser.parse_args()

    # Validate parameters and provide recommendations
    validate_test_parameters(args.n_obs, args.n_vars, args.n_comps, args.tolerance)

    success = run_test(
        n_obs=args.n_obs,
        n_vars=args.n_vars,
        n_comps=args.n_comps,
        tolerance=args.tolerance,
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
