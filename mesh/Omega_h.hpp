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


namespace mfem
{

/// Base class for PUMI meshes
class OmegaMesh : public Mesh
{
/*
protected:
  void CountBoundaryEntity (Omega_h::Mesh* o_mesh, const int BcDim, int &NumBC);

  // Readers for PUMI mesh formats, used in the Load() method.
  void ReadOmegaMesh (Omega_h::Mesh* o_mesh, Omega_h::LOs v_num_loc,
                      const int curved = 0);

  void OhLoad (Omega_h::Mesh* o_mesh, int generate_edges, int refine,
               bool fix_orientation);
*/
public:
  OmegaMesh(Omega_h::Mesh* o_mesh, int generate_edges, int refine,
            bool fix_orientation);
};


} // namespace mfem

#endif // MFEM_USE_OMEGAH

#endif
