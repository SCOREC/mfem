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

#include "Omega_h.hpp"

#ifdef MFEM_USE_OMEGAH
#ifdef MFEM_USE_MPI

#include "mesh_headers.hpp"

#include "../fem/fem.hpp"
#include "../general/sort_pairs.hpp"
#include "../general/text.hpp"
#include "../general/sets.hpp"

#include <iostream>
#include <sstream>
#include <fstream>
#include <limits>
#include <cmath>
#include <cstring>
#include <ctime>

namespace oh = Omega_h;

namespace mfem
{

OmegaMesh::OmegaMesh(oh::Mesh* o_mesh, int generate_edges, int refine,
                   bool fix_orientation)
{
  printf("ok constructor\n");
}


void OmegaMesh::CountBoundaryEntity(oh::Mesh* o_mesh, const int BcDim,
                                   int &NumBc)
{
}

void OmegaMesh::OhLoad(oh::Mesh* o_mesh, int generate_edges, int refine,
                    bool fix_orientation)
{
   int  curved = 0, read_gf = 1;

   // Add a check on o_mesh just in case
   Clear();

  /*
   // First number vertices
   apf::Field* apf_field_crd = o_mesh->getCoordinateField();
   apf::FieldShape* crd_shape = apf::getShape(apf_field_crd);
   apf::Numbering* v_num_loc = apf::createNumbering(o_mesh, "VertexNumbering",
                                                crd_shape, 1);
  */
   auto v_num_loc = oh::Write<oh::LO>(o_mesh->nverts(), 0, 1);
   auto crd = o_mesh->coords();
   // Check if it is a curved mesh
   //will not be for now
    //curved = (crd_shape->getOrder() > 1) ? 1 : 0;

   // Read mesh
   ReadOmegaMesh(o_mesh, v_num_loc, curved);
#ifdef MFEM_DEBUG
   mfem::out << "After ReadOmegaMesh" << std::endl;
#endif
   // at this point the following should be defined:
   //  1) Dim
   //  2) NumOfElements, elements
   //  3) NumOfBdrElements, boundary
   //  4) NumOfVertices, with allocated space in vertices
   //  5) curved
   //  5a) if curved == 0, vertices must be defined
   //  5b) if curved != 0 and read_gf != 0,
   //         'input' must point to a GridFunction
   //  5c) if curved != 0 and read_gf == 0,
   //         vertices and Nodes must be defined

   // FinalizeTopology() will:
   // - assume that generate_edges is true
   // - assume that refine is false
   // - does not check the orientation of regular and boundary elements
   FinalizeTopology();

   Finalize(refine, fix_orientation);
}

void OmegaMesh::ReadOmegaMesh(oh::Mesh* o_mesh, oh::LOs v_num_loc,
                              const int curved)
//***Question: the mfem mesh contents are getting allocated and set on host,
//will it be feasible to set this in device memory//
{
   // Here fill the element table from SCOREC MESH
   // The vector of element pointers is generated with attr and connectivity
   NumOfVertices = o_mesh->nverts();
  
   Dim = o_mesh->dim();
   NumOfElements = o_mesh->nelems();
   elements.SetSize(NumOfElements);

   auto verts = o_mesh->ask_down(o_mesh->dim(), 0);

}


} //end namespace mfem

#endif // MFEM_USE_MPI
#endif // MFEM_USE_OMEGAH
