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

oh::Read<oh::LO> mark_exposedCells (oh::Mesh* o_mesh) {
  const auto exposed_sides = oh::mark_exposed_sides (o_mesh);
  const int dim = o_mesh->oh::Mesh::dim();
  const auto ns = o_mesh->nents (dim - 1); // num. of sides
  const auto s2sc = o_mesh->ask_up (dim - 1, dim).a2ab;
  const auto sc2c = o_mesh->ask_up (dim - 1, dim).ab2b;
  const auto nc = o_mesh->nents (dim); // num. of cells
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
  oh::Write<oh::LO> nBdrElements (1, 0);
  auto get_numExposedCells = OMEGA_H_LAMBDA (oh::LO c) {
    if (exposed_cells[c]) {
      oh::atomic_increment(&nBdrElements[0]);
    }
  };
  oh::parallel_for(nc, get_numExposedCells);
  oh::HostRead<oh::LO> nbe(nBdrElements);
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
  const int dim = o_mesh->oh::Mesh::dim();
  auto e2v_degree = oh::element_degree (OMEGA_H_SIMPLEX, dim, oh::VERT);
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
  const int dim = o_mesh->oh::Mesh::dim();
  const int nverts = o_mesh->oh::Mesh::nverts();
  const int nelems = o_mesh->oh::Mesh::nelems();
  auto ev2v = o_mesh->oh::Mesh::ask_down (dim, oh::VERT).ab2b;
  
  auto exposed_cells = mark_exposedCells(o_mesh);

  const int nBdrElements = count_exposedCells(exposed_cells);

  auto boundaryElems = get_boundary(exposed_cells, nBdrElements);
  // boundary elemIDs

  auto bv2v = get_bdryElemVerts(o_mesh, ev2v, boundaryElems);
  // boundary elems to verts adjacency array

  // check output
  oh::HostRead<oh::LO> boundary_h(boundaryElems);
  oh::HostRead<oh::LO> bv2v_h(bv2v);
  auto e2v_degree = oh::element_degree (OMEGA_H_SIMPLEX, dim, oh::VERT);
  for (int b = 0; b < nBdrElements; ++b) {
    printf("\nbdry elem with ID=%d has verts with IDs", boundary_h[b]);
    for (int v = 0; v < e2v_degree; ++v) {
      printf(" %d ", bv2v_h[b*e2v_degree+v]);
    }
  }
  // end output check

  auto coords = o_mesh->oh::Mesh::coords();

  // look at readPumiElement fn
  // here an Element *el, which is ptr to mfem element will be created
  // and vertices will be assigned to it

  NumOfVertices = nverts;
  NumOfElements = nelems;
  Dim = dim;
  //auto v_num_loc = oh::Write<oh::LO>(o_mesh->nverts(), 0, 1);
  // int curved = 0, read_gf = 1;
  int curved = 0;

  printf("\nset elems =%d\n", nelems);
  elements.SetSize(NumOfElements);
  // iterate over all o_mesh::elems and create mfem elements
  oh::HostRead<oh::LO> ev2v_h(ev2v);

  for (int elem = 0; elem < nelems; ++elem) {
  //auto create_mfemElems = OMEGA_H_LAMBDA (oh::LO elem) {
    elements[elem] = NewElement(dim); 
    // ptr to mfem element // arg in the pumi constructor is geom type
    auto el = elements[elem];
    // fill in info about element
    int nv, *v;
    // Create element in MFEM
    nv = el->GetNVertices(); // these are host functions
    v  = el->GetVertices(); // these are host functions

    // Fill the connectivity
    for (int i = 0; i < nv; ++i) {
      v[i] = ev2v_h[elem*e2v_degree + i];
      // note for parallel the vertex IDs should be global
    }
  }
  //}; // end parallel_for
  //oh::parallel_for(nelems, create_mfemElems);
  // end iteration
  printf("after create elems\n");

  int BcDim = Dim - 1;
  NumOfBdrElements = nBdrElements;
  boundary.SetSize(NumOfBdrElements);
  // the mfem boundary array is of
  // type Array<Element *>, so this boundary will need to be cast
  for (int bdry = 0; bdry < NumOfBdrElements; ++bdry) {
    boundary[bdry] = NewElement(dim);
    // ptr to mfem element // arg in the pumi constructor is geom type
    auto el = boundary[bdry];
    // fill in info about element
    int nv, *v;
    // Create element in MFEM
    nv = el->GetNVertices(); // these are host functions
    v  = el->GetVertices(); // these are host functions

    // Fill the connectivity
    for (int i = 0; i < nv; ++i) {
      v[i] = bv2v_h[bdry*e2v_degree + i];
      // note for parallel the vertex IDs should be global
    }
  }

  printf("after create bdry\n");

  // Fill vertices
  oh::HostRead<oh::Real> coords_h(coords);
  vertices.SetSize(NumOfVertices);
  if (!curved) {
    spaceDim = Dim;
    // iterate over all vertices
    for (unsigned int vtx = 0; vtx < NumOfVertices; ++vtx) {
      // Fill the coords
      for (int d = 0; d < spaceDim; ++d) {
        vertices[vtx](d) = coords_h[vtx*spaceDim + d];
      }
    }
  }

  FinalizeTopology();
  Finalize(refine, fix_orientation);
  // - assume that generate_edges is true
  // - assume that refine is false
  // - does not check the orientation of regular and boundary elements
  printf ("\nend constructor\n");
}

} //end namespace mfem
// TODO add unit tests for this function & regression test

#endif // MFEM_USE_OMEGAH
