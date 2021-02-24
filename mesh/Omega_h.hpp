// Copyright (c) 2010-2020, Lawrence Livermore National Security, LLC. Produced
// at the Lawrence Livermore National Laboratory. All Rights reserved. See files
// LICENSE and NOTICE for details. LLNL-CODE-806117.
//
// This file is part of the MFEM library. For more information and source code
// availability visit https://mfem.org.
//
// MFEM is free software; you can redistribute it and/or modify it under the
// terms of the BSD-3 license. We welcome feedback and contributions, see file
// CONTRIBUTING.md for details.

#ifndef MFEM_OMEGAH
#define MFEM_OMEGAH

#include "../config/config.hpp"

#ifdef MFEM_USE_OMEGAH

#include "../fem/fespace.hpp"
#include "../fem/gridfunc.hpp"
#include "../fem/pgridfunc.hpp"
#include "../fem/coefficient.hpp"
#include "../fem/bilininteg.hpp"

#include <iostream>
#include <limits>
#include <ostream>
#include <string>
#include "mesh.hpp"
#include "pmesh.hpp"

#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_comm.hpp>
#include <Omega_h_build.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_scalar.hpp>

#include "mesh_operators.hpp"

namespace mfem
{

/// Base class for Omega_h meshes
class OmegaMesh : public Mesh
{
public:
  /// Generate an MFEM mesh from a Omega_h mesh.
  OmegaMesh(Omega_h::Mesh* o_mesh, int refine = 0,
            bool fix_orientation = true, const int curved = 0);

  /// Destroys Mesh.
  virtual ~OmegaMesh() {}
};

/// Class for parallel Omega_h meshes
class ParOmegaMesh : public ParMesh
{
public:
  // ParOmegaMesh implementation
  // This function loads a parallel Omega_h mesh and returns the parallel MFEM mesh
  // corresponding to it.
  ParOmegaMesh(MPI_Comm comm, Omega_h::Mesh* o_mesh, int refine = 0,
            bool fix_orientation = true, const int curved = 0);

  // Transfer information about error estimator to Omega_h
  void ElementFieldMFEMtoOmegaH (Omega_h::Mesh* o_mesh, const Vector
                                mfem_field, const int dim, std::string const &name);

  // Transfer tag from omega_h element to omega_h vertex
  void ProjectErrorElementtoVertex (Omega_h::Mesh* o_mesh, std::string const &name);

  /// Update the mesh after adaptation.
  void UpdateMesh(const ParMesh *AdaptedpMesh);

  virtual ~ParOmegaMesh() {}
};

} // namespace mfem

#endif // MFEM_USE_OMEGAH

#endif
