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

int count_exposedEnts (oh::Read<oh::I8> is_exposed) {
  auto nents = is_exposed.size();
  oh::Write<oh::LO> nBdrEnts (1, 0);
  auto get_numExposed = OMEGA_H_LAMBDA (oh::LO ent) {
    if (is_exposed[ent]) {
      oh::atomic_increment(&nBdrEnts[0]);
    }
  };
  oh::parallel_for(nents, get_numExposed, "get_numExposed");
  oh::HostRead<oh::LO> nbe(nBdrEnts);

  // Check if boundary is detected
  if (!nbe[0]) {
    MFEM_ABORT ("boundary elements not detected");
  }

  return nbe[0];
}

oh::Read<oh::LO> get_boundary (oh::Read<oh::I8> exposed_ents,
  const int num_bdryEnts) {
  auto n_expoEnts = exposed_ents.size();
  oh::HostWrite<oh::LO> iter_exposedSides (1, 0, 0);
  oh::HostRead<oh::I8> exposed_ents_h(exposed_ents);
  oh::HostWrite<oh::LO> boundary_h(num_bdryEnts, -1, 0);
  for (oh::LO ent = 0; ent < n_expoEnts; ++ent) {
    if (exposed_ents_h[ent]) {
      boundary_h[iter_exposedSides[0]] = ent;
      ++iter_exposedSides[0];
    }
  }
  return read(boundary_h.write());
}

oh::Read<oh::LO> get_bdry2Verts (oh::Mesh* o_mesh,
  oh::Read<oh::LO> sv2v, oh::Read<oh::LO> bdryEnts) {
  const int bDim = o_mesh->oh::Mesh::dim() - 1;
  auto b2v_degree = oh::element_degree (OMEGA_H_SIMPLEX, bDim, oh::VERT);
  oh::Write<oh::LO> bv2v (bdryEnts.size()*b2v_degree);
  auto get_bdrVerts = OMEGA_H_LAMBDA (oh::LO b) {
    for (oh::LO v = 0; v < b2v_degree; ++v) {
      bv2v[b*b2v_degree + v] = sv2v[bdryEnts[b]*b2v_degree + v];
      // get the id of the boundary element's adjacent verts
    }
  };
  oh::parallel_for (bdryEnts.size(), get_bdrVerts, "get_bdrVerts");
  return read(bv2v);
}

} // end anonymous namespace

namespace mfem {

OmegaMesh::OmegaMesh (oh::Mesh* o_mesh, int generate_edges, int refine,
                      bool fix_orientation, const int curved) {
  const int dim = o_mesh->oh::Mesh::dim();
  const int nverts = o_mesh->oh::Mesh::nverts();
  const int nelems = o_mesh->oh::Mesh::nelems();
  auto ev2v = o_mesh->oh::Mesh::ask_down (dim, oh::VERT).ab2b;
  auto sv2v = o_mesh->oh::Mesh::ask_down (dim - 1, oh::VERT).ab2b;
  // s denotes side
  
  auto exposed_sides = oh::mark_exposed_sides (o_mesh);

  const int nBdrEnts = count_exposedEnts (exposed_sides);

  auto boundaryEnts = get_boundary(exposed_sides, nBdrEnts);
  // boundary elemIDs

  auto bv2v = get_bdry2Verts(o_mesh, sv2v, boundaryEnts);
  // boundary elems to verts adjacency

  oh::HostRead<oh::LO> boundary_h(boundaryEnts);
  oh::HostRead<oh::LO> bv2v_h(bv2v);
  auto e2v_degree = oh::element_degree (OMEGA_H_SIMPLEX, dim, oh::VERT);
  auto b2v_degree = oh::element_degree (OMEGA_H_SIMPLEX, dim - 1, oh::VERT);

  // check output
  for (int b = 0; b < nBdrEnts; ++b) {
    fprintf(stderr, "\nbdry ent with ID=%d has verts with IDs", boundary_h[b]);
    for (int v = 0; v < b2v_degree; ++v) {
      fprintf(stderr, " %d ", bv2v_h[b*b2v_degree+v]);
    }
    fprintf(stderr, "\n");
  }
  // end output check

  auto coords = o_mesh->oh::Mesh::coords();

  NumOfVertices = nverts;
  NumOfElements = nelems;
  Dim = dim;

  elements.SetSize(NumOfElements);
  oh::HostRead<oh::LO> ev2v_h(ev2v);

  int dim_type = -1;
  int bdr_type = -1;
  if (dim == 3) {
    dim_type = Geometry::TETRAHEDRON;
    bdr_type = Geometry::TRIANGLE;
  }
  else {
    dim_type = Geometry::TRIANGLE;
    bdr_type = Geometry::SEGMENT;
  }

  // Create elements
  for (int elem = 0; elem < nelems; ++elem) {
    elements[elem] = NewElement(dim_type); 
    auto el = elements[elem];
    int nv, *v;
    nv = el->GetNVertices();
    v  = el->GetVertices();

    for (int i = 0; i < nv; ++i) {
      v[i] = ev2v_h[elem*e2v_degree + i];
      // note for parallel the vertex IDs should be global
    }
  }

  // Create boundary
  NumOfBdrElements = nBdrEnts;
  boundary.SetSize(NumOfBdrElements);
  for (int bdry = 0; bdry < NumOfBdrElements; ++bdry) {
    boundary[bdry] = NewElement(bdr_type);
    auto el = boundary[bdry];
    int nv, *v;
    nv = el->GetNVertices();
    v  = el->GetVertices();

    for (int i = 0; i < nv; ++i) {
      v[i] = bv2v_h[bdry*b2v_degree + i];
    }
  }

  // Fill vertices
  oh::HostRead<oh::Real> coords_h(coords);
  vertices.SetSize(NumOfVertices);
  if (!curved) {
    spaceDim = Dim;
    for (unsigned int vtx = 0; vtx < NumOfVertices; ++vtx) {
      for (int d = 0; d < spaceDim; ++d) {
        vertices[vtx](d) = coords_h[vtx*spaceDim + d];
      }
    }
  }

  FinalizeMesh();
  // assume that fix_orientation is true and refine is false
}

} //end namespace mfem

#endif // MFEM_USE_OMEGAH
