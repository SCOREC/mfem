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

#include <Omega_h_for.hpp>
#include <Omega_h_element.hpp>
#include <Omega_h_mark.hpp>

namespace oh = Omega_h;

namespace mfem {

OmegaMesh::OmegaMesh(oh::Mesh* o_mesh, int generate_edges, int refine,
                   bool fix_orientation) {
  printf("ok constructor\n");
  // things needed from omegaH mesh //
  int Dim = o_mesh->oh::Mesh::dim();
  auto coords = o_mesh->oh::Mesh::coords();
  const unsigned long int nverts = coords.size()/Dim;
  const unsigned long int nelems = o_mesh->oh::Mesh::nelems();
  auto elem2vert = o_mesh->oh::Mesh::ask_down(Dim, oh::VERT);
  auto elem2vert_degree = oh::element_degree(OMEGA_H_SIMPLEX,
    Dim, oh::VERT);
  // to get the boundary and boundary elements, we will need to bring in the
  // ids of geom ents? or i think ther is an api which will give me the
  // classified elems.
  // can look at mark exposed sides and mark by exposure
  
  oh::Write<oh::LO> NumOfBdrElements = 0;

  auto exposed_sides = mark_exposed_sides (o_mesh);
  auto ns = o_mesh->nents (Dim - 1); // num. of sides
  auto s2sc = o_mesh->ask_up (Dim - 1, Dim).a2ab;
  auto sc2c = o_mesh->ask_up (Dim - 1, Dim).ab2b;
  auto f = OMEGA_H_LAMBDA (oh::LO s) {
    if ((s2sc[s + 1] - s2sc[s]) < 2) {
      NumOfBdrElements[0] = NumOfBdrElements[0] + 1;
    }
  };
  oh::parallel_for(ns, f, "count_bdrElems");
  oh::Write<oh::LO> boundary(NumOfBdrElements);// note the mfem boundary array is of
  // type Arrary<Element *>, so this boundary will need to be type cast
  // now get IDs of boundary elements
  oh::Write<oh::LO> iter_bdrElems = 0;
  auto get_bdrElemId = OMEGA_H_LAMBDA (oh::LO s) {
    if ((s2sc[s + 1] - s2sc[s]) < 2) {
      ++iter_bdrElems[0];
      boundary[iter_bdrElems[0]] = sc2c[s2sc[s]];// get the id of the side's
      // adjacent cell
    }
  };
  oh::parallel_for(ns, get_bdrElemId, "get_bdrElemId");
  // after this get the verts of each doundary element using the ask_down
  // note that following the readPumiElement, 
  auto NumOfBdrElements_h = oh::HostWrite<oh::LO>(NumOfBdrElements);
/*
  int NumOfBdrElements_int = (NumOfBdrElements_h);
  auto get_bdrElemVerts = OMEGA_H_LAMBDA (oh::LO i) {
    boundary[i] = sc2c[s2sc[i]];// get the id of the side's
  };
  oh::parallel_for(NumOfBdrElements_h, get_bdrElemVerts, "get_bdrElemVerts");
*/

  // the logic to create elem in mfem
  // now the ReadPumiElement, i.e. read elem2verts for BdrElements
  // note here an Element *el, which is ptr to mfem element will be created
  // and vertices will be assigned to it
  //note here the elements to vert connectivity will have to be created/
  elements.SetSize(NumOfElements);
  elements[0] = NewElement(3); // ptr to mfem element // arg is dim of elem type
  auto el = elements[0];
  int nv, *v;
  // Create element in MFEM
  nv = el->GetNVertices();
  v  = el->GetVertices();
/*
  // Fill the connectivity
  for (int i = 0; i < nv; ++i) {
    v[i] = apf::getNumber(vert_num, Verts[i], 0, 0);
  }
*/
  //v_num_loc is number of local vertices which apf creates. dont think its
  //needded from omegah mesh
  printf("end constructor\n");
}

void OmegaMesh::CountBoundaryEntity(oh::Mesh* o_mesh, const int BcDim,
                                   int &NumBc) {
}

void OmegaMesh::ReadOmegaMesh(oh::Mesh* o_mesh, oh::LOs v_num_loc,
                              const int curved) {
//***Question: the mfem mesh contents are getting allocated and set on host,
//will it be feasible to set this in device memory//
   // Here fill the element table from SCOREC MESH
   // The vector of element pointers is generated with attr and connectivity
   NumOfVertices = o_mesh->nverts();
  
   Dim = o_mesh->dim();
   NumOfElements = o_mesh->nelems();
   elements.SetSize(NumOfElements);

   auto verts = o_mesh->ask_down(o_mesh->dim(), 0);

}

void OmegaMesh::OhLoad(oh::Mesh* o_mesh, int generate_edges, int refine,
                    bool fix_orientation) {
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

} //end namespace mfem

#endif // MFEM_USE_MPI
#endif // MFEM_USE_OMEGAH
