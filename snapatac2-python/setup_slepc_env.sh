#!/bin/bash
# setup_slepc_env.sh
# Script to set up the environment for PETSc/SLEPc with distributed spectral embedding

# CRITICAL: Force use of system OpenMPI with SLURM support
echo "Setting up OpenMPI with SLURM support..."
export PATH=/cm/shared/apps/openmpi4/gcc/4.1.5/bin:$PATH
export MPI_HOME=/cm/shared/apps/openmpi4/gcc/4.1.5
export MPI_RUN=/cm/shared/apps/openmpi4/gcc/4.1.5/bin/mpirun
export LD_LIBRARY_PATH=/cm/shared/apps/openmpi4/gcc/4.1.5/lib:$LD_LIBRARY_PATH

# Load required modules
echo "Loading required modules..."
module load gcc/11.2.0
# cmake is already loaded
module list

# Set library path to use newer libstdc++
export LD_LIBRARY_PATH=$(gcc --print-file-name=libstdc++.so.6 | xargs dirname):$LD_LIBRARY_PATH

# Verify OpenMPI setup
echo "Verifying OpenMPI setup..."
echo "Using mpirun from: $(which mpirun)"
echo "MPI version: $(mpirun --version 2>/dev/null | head -1)"

# Check that PETSc/SLEPc work
echo "Testing PETSc/SLEPc imports..."
python -c "from petsc4py import PETSc; from slepc4py import SLEPc; print('PETSc and SLEPc are working!')"

# Test that snapatac2 can access spectral_slepc
echo "Testing snapatac2.tl.spectral_slepc..."
python -c "import snapatac2 as snap; print('spectral_slepc available:', hasattr(snap.tl, 'spectral_slepc'))"

echo "Environment is ready for distributed spectral embedding!"
echo "Use 'source setup_slepc_env.sh' before running your analysis."
