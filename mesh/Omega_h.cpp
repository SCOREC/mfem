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

/* based on a private fn form_sharing from ohdolfin.c */
static void get_shared_ranks(oh::Mesh* o_mesh, oh::Int ent_dim,
    std::map<std::int32_t, std::set<unsigned int>>* shared_ents) {
  auto n = o_mesh->nents(ent_dim);
  if (!o_mesh->could_be_shared(ent_dim)) {
    return;
  }
  auto dist = o_mesh->ask_dist(ent_dim).invert();
  auto d_owners2copies = dist.roots2items();
  auto d_copies2rank = dist.items2ranks();
  auto d_copies2indices = dist.items2dest_idxs();
  auto h_owners2copies = oh::HostRead<oh::LO>(d_owners2copies);
  auto h_copies2rank = oh::HostRead<oh::I32>(d_copies2rank);
  auto h_copies2indices = oh::HostRead<oh::LO>(d_copies2indices);
  std::vector<oh::I32> full_src_ranks;
  std::vector<oh::I32> full_dest_ranks;
  std::vector<oh::LO> full_dest_indices;
  auto my_rank = o_mesh->comm()->rank();
  for (oh::LO i_osh = 0; i_osh < n; ++i_osh) {
    auto begin = h_owners2copies[i_osh];
    auto end = h_owners2copies[i_osh + 1];
    if (end - begin <= 1) continue;
    for (oh::LO copy = begin; copy < end; ++copy) {
      auto dest_rank = h_copies2rank[copy];
      auto dest_index = h_copies2indices[copy];
      for (oh::LO copy2 = begin; copy2 < end; ++copy2) {
        auto src_rank = h_copies2rank[copy2];
        full_src_ranks.push_back(src_rank);
        full_dest_ranks.push_back(dest_rank);
        full_dest_indices.push_back(dest_index);
      }
    }
  }
  auto h_full_src_ranks = oh::HostWrite<oh::I32>(oh::LO(full_src_ranks.size()));
  auto h_full_dest_ranks = oh::HostWrite<oh::I32>(oh::LO(full_src_ranks.size()));
  auto h_full_dest_indices = oh::HostWrite<oh::I32>(oh::LO(full_dest_indices.size()));
  for (oh::LO i = 0; i < h_full_src_ranks.size(); ++i) {
    h_full_src_ranks[i] = full_src_ranks[size_t(i)];
    h_full_dest_ranks[i] = full_dest_ranks[size_t(i)];
    h_full_dest_indices[i] = full_dest_indices[size_t(i)];
  }
  auto d_full_src_ranks = oh::Read<oh::I32>(h_full_src_ranks.write());
  auto d_full_dest_ranks = oh::Read<oh::I32>(h_full_dest_ranks.write());
  auto d_full_dest_indices = oh::Read<oh::I32>(h_full_dest_indices.write());
  auto dist2 = oh::Dist();
  dist2.set_parent_comm(o_mesh->comm());
  dist2.set_dest_ranks(d_full_dest_ranks);
  dist2.set_dest_idxs(d_full_dest_indices, n);
  auto d_exchd_full_src_ranks = dist2.exch(d_full_src_ranks, 1);
  auto d_shared2ranks = dist2.invert().roots2items();
  auto h_exchd_full_src_ranks = oh::HostRead<oh::I32>(d_exchd_full_src_ranks);
  auto h_shared2ranks = oh::HostRead<oh::LO>(d_shared2ranks);
/*
  auto ents_are_shared_w = HostWrite<Byte>(n);
  for (LO i_osh = 0; i_osh < n; ++i_osh) {
    auto begin = h_shared2ranks[i_osh];
    auto end = h_shared2ranks[i_osh + 1];
    ents_are_shared_w[i_osh] = ((end - begin) > 0);
  }
*/
  for (oh::LO i_osh = 0; i_osh < n; ++i_osh) {
    auto begin = h_shared2ranks[i_osh];
    auto end = h_shared2ranks[i_osh + 1];
    for (auto j = begin; j < end; ++j) {
      auto rank = h_exchd_full_src_ranks[j];
      //if (rank != my_rank) {
      // include self
      (*shared_ents)[i_osh].insert(unsigned(rank));
      //}
    }
  }
}

oh::HostRead<oh::LO> mark_shared_ents (oh::Mesh* o_mesh, int dim) {
  OMEGA_H_CHECK(o_mesh->could_be_shared(dim));
  auto rank = o_mesh->comm()->rank();
  auto owners_r = o_mesh->ask_owners(dim).ranks;
  auto owners_i = o_mesh->ask_owners(dim).idxs;
  auto nents = o_mesh->nents(dim);
  oh::Write<oh::LO> ent_is_shared(nents, -1, "ent_is_shared");

  auto dist = o_mesh->ask_dist(dim).invert();
  auto d_owners2copies = dist.roots2items();

  auto check_owner = OMEGA_H_LAMBDA (oh::LO ent) {
    if (owners_r[ent] != rank) {
      // ent is a copy
      ent_is_shared[ent] = 1;
      #ifdef DEBUG_MODE
        fprintf(stderr, "ent %d of dim %d is a copy\n", ent, dim);
      #endif
    }
    else if ((d_owners2copies[ent+1] - d_owners2copies[ent]) > 1) {
      // ent is a owner
      ent_is_shared[ent] = 1;
      #ifdef DEBUG_MODE
        fprintf(stderr, "ent %d of dim %d is a owner\n", ent, dim);
      #endif
    }
    #ifdef DEBUG_MODE
    if (ent_is_shared[ent] == 1)
      fprintf(stderr, "ent %d of dim %d is shared\n", ent, dim);
    #endif
  };
  oh::parallel_for(nents, check_owner, "check_owner");
  oh::HostRead<oh::LO> ent_is_shared_h(ent_is_shared);

  return ent_is_shared_h;
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
                            bool fix_orientation, const int curved) {
  // Set the communicator for gtopo
  gtopo.SetComm(comm);

  MyComm = comm;
  MPI_Comm_size(MyComm, &NRanks);
  MPI_Comm_rank(MyComm, &MyRank);

  Dim = o_mesh->dim();
  spaceDim = Dim;

  fprintf(stderr, "ok0\n");

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
  fprintf(stderr, "ok1\n");

  ListOfIntegerSets  groups;
  IntegerSet         group;

  // The first group is the local one
  group.Recreate(1, &MyRank);
  groups.Insert(group);

  MFEM_ASSERT(Dim >= 3 || GetNFaces() == 0,
             "[proc " << MyRank << "]: invalid state");

  // Determine shared faces
  Array<Pair<long, int>> sfaces;
  // Initially sfaces[i].one holds the global face id.
  // Then it is replaced by the group id of the shared face.
  // Initially sfaces[i].two holds the local face id.
  fprintf(stderr, "ok2\n");
  if (Dim > 2) {
    // Number the faces globally and enumerate the local shared faces
    // following the global enumeration. This way we ensure that the ordering
    // of the shared faces within each group (of processors) is the same in
    // each processor in the group.
    //TODO verify this
    oh::HostRead<oh::GO> GlobalFaceNum (o_mesh->globals(2));
    auto is_shared = mark_shared_ents(o_mesh, 2);

    for (int ent = 0; ent < o_mesh->nfaces(); ++ent) {
      if (is_shared[ent] == 1) {
        long id = GlobalFaceNum[ent];
        sfaces.Append(Pair<long,int>(id, ent));
        #ifdef DEBUG_MODE
          fprintf(stderr, "ent %d of dim %d is shared\n", ent, 2);
        #endif
      }
    }
    sfaces.Sort();

    // create groups and replace the global face id in sfaces[i].one with group id
    Array<int> eleRanks;
    std::map<std::int32_t, std::set<unsigned int>> shared_faces;
    get_shared_ranks(o_mesh, oh::FACE, &shared_faces);

    for (int i = 0; i < sfaces.Size(); i++) {
      int ent = sfaces[i].two;
      int kk = 0;
      eleRanks.SetSize(shared_faces[ent].size());
      for (std::set<unsigned int>::iterator itr = shared_faces[ent].begin();
           itr != shared_faces[ent].end(); itr++) {
        eleRanks[kk++] = *itr;
      }
      group.Recreate(eleRanks.Size(), eleRanks);
      sfaces[i].one = groups.Insert(group) - 1;
    }
  } // end conditional for faces
  fprintf(stderr, "ok3\n");

  int waiting = 0;
  while (waiting);
  // Determine shared edges
  Array<Pair<long, int>> sedges;
  // Initially sedges[i].one holds the global edge id.
  // Then it is replaced by the group id of the shared edge.
  if (Dim > 1) {
    // Number the edges globally and enumerate the local shared edges
    // following the global enumeration. This way we ensure that the ordering
    // of the shared edges within each group (of processors) is the same in
    // each processor in the group.
    //TODO verify this
    oh::HostRead<oh::GO> GlobalEdgeNum (o_mesh->globals(1));
    auto is_shared = mark_shared_ents(o_mesh, 1);

    for (int ent = 0; ent < o_mesh->nedges(); ++ent) {
      if (is_shared[ent] == 1) {
        long id = GlobalEdgeNum[ent];
        sedges.Append(Pair<long,int>(id, ent));
        #ifdef DEBUG_MODE
          fprintf(stderr, "ent %d of dim %d is shared\n", ent, 1);
        #endif
      }
    }
    sedges.Sort();

    // create groups and replace the global id in sfaces[i].one with group id
    Array<int> eleRanks;
    std::map<std::int32_t, std::set<unsigned int>> shared_edges;
    get_shared_ranks(o_mesh, oh::EDGE, &shared_edges);

    for (int i = 0; i < sedges.Size(); i++) {
      int ent = sedges[i].two;
      int kk = 0;
      eleRanks.SetSize(shared_edges[ent].size());
      for (std::set<unsigned int>::iterator itr = shared_edges[ent].begin();
           itr != shared_edges[ent].end(); itr++) {
        eleRanks[kk++] = *itr;
      }
      group.Recreate (eleRanks.Size(), eleRanks);
      sedges[i].one = groups.Insert(group) - 1;
    }
  } // end conditional for edges
  fprintf(stderr, "ok4\n");

  // Determine shared vertices
  //Array<int> sverts;
  Array<Pair<int, int>> sverts;
  //Array<Pair<long, int>> sverts;
  Array<int> svert_group;
  {
    //TODO verify this
    // the pumi implm is using local vert ids to sort
    // The entries sverts[i].one hold the local vertex ids.
    oh::HostRead<oh::GO> GlobalVertNum (o_mesh->globals(0));
    auto is_shared = mark_shared_ents (o_mesh, 0);
    fprintf(stderr, "ok4.1\n");

    for (int ent = 0; ent < o_mesh->nverts(); ++ent) {
      if (is_shared[ent] == 1) {
        long id = GlobalVertNum[ent];
        sverts.Append(Pair<int,int>(ent, ent));
        //sverts.Append(Pair<long,int>(id, ent));
        #ifdef DEBUG_MODE
          fprintf(stderr, "ent %d of dim %d is shared\n", ent, 0);
        #endif
      }
    }
    sverts.Sort();
    fprintf(stderr, "ok4.2\n");

    // create groups and replace the global id in sfaces[i].one with group id
    svert_group.SetSize(sverts.Size());
    Array<int> eleRanks;
    std::map<std::int32_t, std::set<unsigned int>> shared_verts;
    get_shared_ranks(o_mesh, oh::VERT, &shared_verts);
    fprintf(stderr, "ok4.3\n");

    for (int i = 0; i < sverts.Size(); i++) {
      int ent = sverts[i].two;
      int kk = 0;
      fprintf(stderr, "ok4.3.1 size %d\n", shared_verts[ent].size());
      eleRanks.SetSize(shared_verts[ent].size());
      fprintf(stderr, "ok4.3.2 size %d\n", shared_verts[ent].size());
      for (std::set<unsigned int>::iterator itr = shared_verts[ent].begin();
           itr != shared_verts[ent].end(); itr++) {
        eleRanks[kk++] = *itr;
      }
      fprintf(stderr, "ok4.3.3 size %d\n", shared_verts[ent].size());
      group.Recreate(eleRanks.Size(), eleRanks);
      fprintf(stderr, "ok4.3.4 ent %d size %d\n", ent, shared_verts[ent].size());
      svert_group[i] = groups.Insert(group) - 1;
      fprintf(stderr, "ok4.3.5 size %d\n", shared_verts[ent].size());
    }
    fprintf(stderr, "ok4.4\n");
  }
  fprintf(stderr, "ok5\n");

  // Build group_stria and group_squad.
  // Also allocate shared_trias, shared_quads, and sface_lface.
  // TODO simplex mesh considered for now
  group_stria.MakeI(groups.Size()-1);
  group_squad.MakeI(groups.Size()-1);
  for (int i = 0; i < sfaces.Size(); i++) {
    group_stria.AddAColumnInRow(sfaces[i].one);
  }
  group_stria.MakeJ();
  group_squad.MakeJ();
  {
    int nst = 0;
    for (int i = 0; i < sfaces.Size(); i++) {
      group_stria.AddConnection(sfaces[i].one, nst++);
    }
    shared_trias.SetSize(nst);
    shared_quads.SetSize(sfaces.Size()-nst);
    sface_lface.SetSize(sfaces.Size());
  }
  group_stria.ShiftUpI();
  group_squad.ShiftUpI();
  fprintf(stderr, "ok6\n");

  // Build group_sedge
  group_sedge.MakeI(groups.Size()-1);
  for (int i = 0; i < sedges.Size(); i++) {
    group_sedge.AddAColumnInRow(sedges[i].one);
  }
  group_sedge.MakeJ();
  for (int i = 0; i < sedges.Size(); i++) {
    group_sedge.AddConnection(sedges[i].one, i);
  }
  group_sedge.ShiftUpI();
  fprintf(stderr, "ok7\n");

  // Build group_svert
  group_svert.MakeI(groups.Size()-1);
  for (int i = 0; i < svert_group.Size(); i++) {
    group_svert.AddAColumnInRow(svert_group[i]);
  }
  group_svert.MakeJ();
  for (int i = 0; i < svert_group.Size(); i++) {
    group_svert.AddConnection(svert_group[i], i);
  }
  group_svert.ShiftUpI();
  fprintf(stderr, "ok8\n");

  // Build shared_trias and shared_quads. They are allocated above.
  {
    int nst = 0;
    oh::HostRead<oh::LO> tri2vert_h (o_mesh->ask_down(2, 0).ab2b);
    for (int i = 0; i < sfaces.Size(); i++) {
      int ent = sfaces[i].two;
      int *v, nv = 0;
      v = shared_trias[nst++].v;
      nv = 3; //simplex
      for (int j = 0; j < nv; ++j) {
        v[j] = tri2vert_h[ent*nv + j];
      }
    }
  }
  fprintf(stderr, "ok9\n");

  // Build shared_edges and allocate sedge_ledge
  shared_edges.SetSize(sedges.Size());
  sedge_ledge. SetSize(sedges.Size());
  oh::HostRead<oh::LO> edge2vert_h (o_mesh->ask_down(1, 0).ab2b);
  for (int i = 0; i < sedges.Size(); i++) {
    int ent = sedges[i].two;
    int id1, id2;
    id1 = edge2vert_h[ent*2 + 0];
    id2 = edge2vert_h[ent*2 + 1];
    if (id1 > id2) { 
      auto temp = id1;
      id1 = id2;
      id2 = temp;
    }

    fprintf(stderr, "ok9.1 edge = %d\n", ent);
    shared_edges[i] = new Segment(id1, id2, 1);
  }
  fprintf(stderr, "ok10\n");

  // Build svert_lvert
  svert_lvert.SetSize(sverts.Size());
  for (int i = 0; i < sverts.Size(); i++) {
    svert_lvert[i] = sverts[i].two; // local entity id of vert
    //svert_lvert[i] = sverts[i].one;
  }
  fprintf(stderr, "ok11\n");

  // Build the group communication topology
  gtopo.Create(groups, 822);
  fprintf(stderr, "ok12\n");

  // Determine sedge_ledge and sface_lface
  FinalizeParTopo();
  fprintf(stderr, "ok13\n");

  // Set nodes for higher order mesh
  // n.a.: linear mesh

  Finalize(refine, fix_orientation);
  fprintf(stderr, "ok14\n");
}

} // end namespace mfem

#endif // MFEM_USE_OMEGAH
