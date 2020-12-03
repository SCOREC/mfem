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

#include "Omega_h_for.hpp"
#include "Omega_h_element.hpp"
#include "Omega_h_mark.hpp"
#include "Omega_h_atomics.hpp"

namespace oh = Omega_h;

namespace { // anonymous namespace

// TODO add unit tests for this function
oh::Read<oh::LO> mark_exposedCells (oh::Mesh* o_mesh) {
  const auto exposed_sides = oh::mark_exposed_sides (o_mesh);
  const int Dim = o_mesh->oh::Mesh::dim();
  const auto ns = o_mesh->nents (Dim - 1); // num. of sides
  const auto s2sc = o_mesh->ask_up (Dim - 1, Dim).a2ab;
  const auto sc2c = o_mesh->ask_up (Dim - 1, Dim).ab2b;
  const auto nc = o_mesh->nents (Dim); // num. of cells
  oh::Write<oh::LO> exposed_cells (nc, 0);
  auto get_exposedCells = OMEGA_H_LAMBDA (oh::LO s) {
    if (exposed_sides[s]) {
      auto exposed_cell = sc2c[s2sc[s]];
      exposed_cells[exposed_cell] = 1;
    }
  };
  oh::parallel_for(ns, get_exposedCells);
  return read (exposed_cells);
}

int count_exposedCells (oh::Read<oh::LO> exposed_cells) {
  auto nc = exposed_cells.size();
  oh::Write<oh::LO> NumOfBdrElements (1, 0);
  auto get_numExposedCells = OMEGA_H_LAMBDA (oh::LO c) {
    if (exposed_cells[c]) {
      oh::atomic_increment(&NumOfBdrElements[0]);
    }
  };
  oh::parallel_for(nc, get_numExposedCells);
  oh::HostRead<oh::LO> nbe(NumOfBdrElements);
  return nbe[0];
}

oh::Read<oh::LO> get_boundary (oh::Read<oh::LO> exposed_cells,
  const int num_bdryElems) {
  auto nc = exposed_cells.size();
  oh::HostWrite<oh::LO> iter_exposedCells (1, 0, 0);
  oh::HostRead<oh::LO> exposed_cells_h(exposed_cells);
  oh::HostWrite<oh::LO> boundary_h(num_bdryElems, -1, 0);
  for (oh::LO c = 0; c < nc; ++c) {
    if (exposed_cells_h[c]) {
      // this iterative transfer will have to be done through host
      boundary_h[iter_exposedCells[0]] = c;
      ++iter_exposedCells[0];
    }
  }
  return read(boundary_h.write());
}

oh::Read<oh::LO> get_bdryElemVerts (oh::Mesh* o_mesh,
  oh::Read<oh::LO> ev2v, oh::Read<oh::LO> bdryElems) {
  const int Dim = o_mesh->oh::Mesh::dim();
  auto e2v_degree = oh::element_degree (OMEGA_H_SIMPLEX, Dim, oh::VERT);
  oh::Write<oh::LO> bv2v (bdryElems.size()*e2v_degree);
  auto get_bdrElemVerts = OMEGA_H_LAMBDA (oh::LO b) {
    for (oh::LO v = 0; v < e2v_degree; ++v) {
      bv2v[b*e2v_degree + v] = ev2v[bdryElems[b]*e2v_degree + v];
      // get the id of the boundary element's adjacent verts
    }
  };
  oh::parallel_for (bdryElems.size(), get_bdrElemVerts);
  return read(bv2v);
}

} // end anonymous namespace

namespace mfem {

OmegaMesh::OmegaMesh (oh::Mesh* o_mesh, int generate_edges, int refine,
                      bool fix_orientation) {
  const int Dim = o_mesh->oh::Mesh::dim();
  const unsigned long int nverts = o_mesh->oh::Mesh::nverts();
  const unsigned long int nelems = o_mesh->oh::Mesh::nelems();
  auto ev2v = o_mesh->oh::Mesh::ask_down (Dim, oh::VERT).ab2b;
  
  auto exposed_cells = mark_exposedCells(o_mesh);

  const int NumOfBdrElements = count_exposedCells(exposed_cells);

  auto boundary = get_boundary(exposed_cells, NumOfBdrElements);
  // the mfem boundary array is of
  // type Array<Element *>, so this boundary will need to be cast

  auto bv2v = get_bdryElemVerts(o_mesh, ev2v, boundary);
  // boundary elems to verts adjacency array

  // check output
  oh::HostRead<oh::LO> boundary_h(boundary);
  oh::HostRead<oh::LO> bv2v_h(bv2v);
  auto e2v_degree = oh::element_degree (OMEGA_H_SIMPLEX, Dim, oh::VERT);
  for (int b = 0; b < NumOfBdrElements; ++b) {
    printf("\nbdry elem with ID=%d has verts with IDs", boundary_h[b]);
    for (int v = 0; v < e2v_degree; ++v) {
      printf(" %d ", bv2v_h[b*e2v_degree+v]);
    }
  }
  // end output check

  auto coords = o_mesh->oh::Mesh::coords();

  // after this get the verts of each doundary element using ask_down
  // note readPumiElement API

  // the logic to create elem in mfem
  // now the ReadPumiElement, i.e. read elem2verts for BdrElements
  // here an Element *el, which is ptr to mfem element will be created
  // and vertices will be assigned to it
  // here the elements to vert connectivity will have to be created/
  elements.SetSize(nelems);
  elements[0] = NewElement(Dim); // ptr to mfem element // arg is dim of elem type
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
  printf("\nend constructor\n");
}

/*
void OmegaMesh::ReadOmegaMesh (oh::Mesh* o_mesh, oh::LOs v_num_loc,
                              const int curved) {
   // Here fill the element table from SCOREC MESH
   // The vector of element pointers is generated with attr and connectivity
   NumOfVertices = o_mesh->nverts();
  
   Dim = o_mesh->dim();
   NumOfElements = o_mesh->nelems();
   elements.SetSize(NumOfElements);

   auto verts = o_mesh->ask_down(o_mesh->dim(), 0);

}
*/
/*
void OmegaMesh::OhLoad(oh::Mesh* o_mesh, int generate_edges, int refine,
                    bool fix_orientation) {
   int  curved = 0, read_gf = 1;

   // Add a check on o_mesh just in case
   Clear();
  
   // First number vertices
   //apf::Field* apf_field_crd = o_mesh->getCoordinateField();
   //apf::FieldShape* crd_shape = apf::getShape(apf_field_crd);
   //apf::Numbering* v_num_loc = apf::createNumbering(o_mesh, "VertexNumbering",
                                                crd_shape, 1);
  
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
*/
} //end namespace mfem

#endif // MFEM_USE_OMEGAH
