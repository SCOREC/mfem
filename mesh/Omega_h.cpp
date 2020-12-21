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

int get_type (int dim) {
  int dim_type = -1;
  if (dim == 3) {
    dim_type = mfem::Geometry::TETRAHEDRON;
  }
  else if (dim == 2) {
    dim_type = mfem::Geometry::TRIANGLE;
  }
  else if (dim == 1) {
    dim_type = mfem::Geometry::SEGMENT;
  }
  else if (dim == 0) {
    dim_type = mfem::Geometry::POINT;
  }
  else {
    Omega_h_fail ("Error: Improper dimension");
  }

  return dim_type;
}

} // end anonymous namespace

namespace mfem {

OmegaMesh::OmegaMesh (oh::Mesh* o_mesh, int refine,
                      bool fix_orientation, const int curved) {
  const int nverts = o_mesh->oh::Mesh::nverts();
  const int nelems = o_mesh->oh::Mesh::nelems();
  const int dim = o_mesh->oh::Mesh::dim();
  auto ev2v = o_mesh->oh::Mesh::ask_down (dim, oh::VERT).ab2b;
  auto sv2v = o_mesh->oh::Mesh::ask_down (dim - 1, oh::VERT).ab2b;
  // s denotes side
  
  auto exposed_sides = oh::mark_exposed_sides (o_mesh);

  const int nBdrEnts = count_exposedEnts (exposed_sides);

  auto boundaryEnts = get_boundary(exposed_sides, nBdrEnts);
  // boundary elemIDs

  auto bv2v = get_bdry2Verts(o_mesh, sv2v, boundaryEnts);
  // boundary elems to verts adjacency

  NumOfVertices = nverts;
  NumOfElements = nelems;
  Dim = dim;
  elements.SetSize(NumOfElements);

  // Create elements
  const int dim_type = get_type(dim);
  const int bdr_type = get_type(dim-1);
  oh::HostRead<oh::LO> ev2v_h(ev2v);
  oh::HostRead<oh::LO> boundary_h(boundaryEnts);
  oh::HostRead<oh::LO> bv2v_h(bv2v);
  auto e2v_degree = oh::element_degree (OMEGA_H_SIMPLEX, dim, oh::VERT);
  auto b2v_degree = oh::element_degree (OMEGA_H_SIMPLEX, dim - 1, oh::VERT);
  for (int elem = 0; elem < nelems; ++elem) {
    elements[elem] = NewElement(dim_type); 
    auto el = elements[elem];
    int nv, *v;
    nv = el->GetNVertices();
    v  = el->GetVertices();

    for (int i = 0; i < nv; ++i) {
      v[i] = ev2v_h[elem*e2v_degree + i];
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
  auto coords = o_mesh->oh::Mesh::coords();
  vertices.SetSize(NumOfVertices);
  oh::HostRead<oh::Real> coords_h(coords);
  if (!curved) {
    spaceDim = Dim;
    for (unsigned int vtx = 0; vtx < NumOfVertices; ++vtx) {
      for (int d = 0; d < spaceDim; ++d) {
        vertices[vtx](d) = coords_h[vtx*spaceDim + d];
      }
    }
  }

  FinalizeMesh();
  // assume that fix_orientation is true, refine is false
}

ParOmegaMesh::ParOmegaMesh (MPI_Comm comm, oh::Mesh* o_mesh, int refine,
                            bool fix_orientation, const int curved)
{
   // Set the communicator for gtopo
   // Global numbering of vertices. This is necessary to build a local numbering
   // that has the same ordering in each process.
   // Take this process global vertex IDs and sort
   // Create local numbering that respects the global ordering
   // Construct the numbering v_num_loc and set the coordinates of the vertices.
   // Fill the elements
   // Read elements from SCOREC Mesh
   // Count number of boundaries by classification
   // Read boundary from SCOREC mesh
   // The next two methods are called by FinalizeTopology() called below:
   // The first group is the local one
   // Determine shared faces
   // Initially sfaces[i].one holds the global face id.
   // Then it is replaced by the group id of the shared face.
      // Number the faces globally and enumerate the local shared faces
      // following the global enumeration. This way we ensure that the ordering
      // of the shared faces within each group (of processors) is the same in
      // each processor in the group.
      // Replace the global face id in sfaces[i].one with group id.
   // Determine shared edges
   // Initially sedges[i].one holds the global edge id.
   // Then it is replaced by the group id of the shared edge.
      // Number the edges globally and enumerate the local shared edges
      // following the global enumeration. This way we ensure that the ordering
      // of the shared edges within each group (of processors) is the same in
      // each processor in the group.
      // Replace the global edge id in sedges[i].one with group id.
   // Determine shared vertices
   // The entries sverts[i].one hold the local vertex ids.
      // Determine svert_group
         // Get the IDs
   // Build group_stria and group_squad.
   // Also allocate shared_trias, shared_quads, and sface_lface.
   // Build group_sedge
   // Build group_svert
   // Build shared_trias and shared_quads. They are allocated above.
   // Build shared_edges and allocate sedge_ledge
   // Build svert_lvert
   // Build the group communication topology
   // Determine sedge_ledge and sface_lface
   // Set nodes for higher order mesh

  // Set the communicator for gtopo
  gtopo.SetComm(comm);

  MyComm = comm;
  MPI_Comm_size(MyComm, &NRanks);
  MPI_Comm_rank(MyComm, &MyRank);

  Dim = o_mesh->dim();
  spaceDim = Dim;

  // Global numbering of vertices. This is necessary to build a local numbering
  // that has the same ordering in each process.
  auto vert_globals = o_mesh->globals(oh::VERT);
  oh::HostRead<oh::GO> vert_globals_h(vert_globals);
  // Take this process global vertex IDs and sort
  Array<Pair<long,int>> thisVertIds(o_mesh->nverts());
  for (int i = 0; i < o_mesh->nverts(); ++i) {
    long id = vert_globals_h[i];
    thisVertIds[i] = Pair<long,int>(id, i);
  }
  thisVertIds.Sort();
  // Set thisVertIds[i].one = j where j is such that thisVertIds[j].two = i.
  // Thus, the mapping i -> thisVertIds[i].one is the inverse of the mapping
  // j -> thisVertIds[j].two.
  for (int j = 0; j < thisVertIds.Size(); ++j) {
    const int i = thisVertIds[j].two;
    thisVertIds[i].one = j;
  }

  // Set the coordinates of the vertices.
  /*v_num_loc is pumi specific variable, not needed here*/
  NumOfVertices = thisVertIds.Size();
  vertices.SetSize(NumOfVertices);
  auto coords = o_mesh->coords();
  oh::HostRead<oh::Real> coords_h(coords);
  for (unsigned int vtx = 0; vtx < NumOfVertices; ++vtx) {
    for (int d = 0; d < spaceDim; ++d) {
      vertices[vtx](d) = coords_h[vtx*spaceDim + d];
    }
  }

  // Fill the elements
  NumOfElements = o_mesh->nelems();
  elements.SetSize(NumOfElements);
  // Read elements from Omega_h Mesh
  const int dim = o_mesh->oh::Mesh::dim();
  auto ev2v = o_mesh->oh::Mesh::ask_down (dim, oh::VERT).ab2b;
  oh::HostRead<oh::LO> ev2v_h(ev2v);
  auto e2v_degree = oh::element_degree (OMEGA_H_SIMPLEX, dim, oh::VERT);
  const int dim_type = get_type(dim);
  const int bdr_type = get_type(dim-1);
  // Create elements
  for (int elem = 0; elem < o_mesh->nelems(); ++elem) {
    elements[elem] = NewElement(dim_type); 
    auto el = elements[elem];
    int nv, *v;
    nv = el->GetNVertices();
    v  = el->GetVertices();

    for (int i = 0; i < nv; ++i) {
      v[i] = ev2v_h[elem*e2v_degree + i];
    }
  }

  // create boundary info; s denotes side
  auto sv2v = o_mesh->oh::Mesh::ask_down (dim - 1, oh::VERT).ab2b;
  auto exposed_sides = oh::mark_exposed_sides (o_mesh);
  const int nBdrEnts = count_exposedEnts (exposed_sides);
  // boundary elemIDs
  auto boundaryEnts = get_boundary(exposed_sides, nBdrEnts);
  // boundary elems to verts adjacency
  auto bv2v = get_bdry2Verts(o_mesh, sv2v, boundaryEnts);
  oh::HostRead<oh::LO> boundary_h(boundaryEnts);
  oh::HostRead<oh::LO> bv2v_h(bv2v);
  auto b2v_degree = oh::element_degree (OMEGA_H_SIMPLEX, dim - 1, oh::VERT);
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

  // The next two methods are called by FinalizeTopology() called below:
  this->FinalizeTopology();

  ListOfIntegerSets  groups;
  IntegerSet         group;

  // The first group is the local one
  group.Recreate(1, &MyRank);
  groups.Insert(group);

  MFEM_ASSERT(Dim >= 3 || GetNFaces() == 0,
             "[proc " << MyRank << "]: invalid state");

  // Determine shared faces
  //Array<Pair<long, apf::MeshEntity*>> sfaces;
  // Initially sfaces[i].one holds the global face id.
  // Then it is replaced by the group id of the shared face.
  if (Dim > 2)
  {
    // Number the faces globally and enumerate the local shared faces
    // following the global enumeration. This way we ensure that the ordering
    // of the shared faces within each group (of processors) is the same in
    // each processor in the group.
    auto GlobalFaceNum = o_mesh->globals(oh::FACE);

/*
      itr = apf_mesh->begin(2);
      while ((ent = apf_mesh->iterate(itr)))
      {
         if (apf_mesh->isShared(ent))
         {
            long id = apf::getNumber(GlobalFaceNum, ent, 0, 0);
            sfaces.Append(Pair<long,apf::MeshEntity*>(id, ent));
         }
      }
      apf_mesh->end(itr);
      sfaces.Sort();
      apf::destroyGlobalNumbering(GlobalFaceNum);

      // Replace the global face id in sfaces[i].one with group id.
      for (int i = 0; i < sfaces.Size(); i++)
      {
         ent = sfaces[i].two;

         const int thisNumAdjs = 2;
         int eleRanks[thisNumAdjs];

         // Get the IDs
         apf::Parts res;
         apf_mesh->getResidence(ent, res);
         int kk = 0;
         for (std::set<int>::iterator itr = res.begin();
              itr != res.end(); ++itr)
         {
            eleRanks[kk++] = *itr;
         }

         group.Recreate(2, eleRanks);
         sfaces[i].one = groups.Insert(group) - 1;
      }
*/
   }
}

} // end namespace mfem

#endif // MFEM_USE_OMEGAH
