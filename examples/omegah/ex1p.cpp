//                                MFEM Example 1
//                              Example for testing Omega_h to MFEM mesh
//                              constructor
//
// Compile with: make ex1p
//
// Sample runs:
//    ex1p
//
// Description:  see note

#include "mfem.hpp"
#include <fstream>
#include <iostream>

#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>
#include <Omega_h_comm.hpp>
#include <Omega_h_build.hpp>

#ifndef MFEM_USE_OMEGAH
#error This example requires that MFEM is built with MFEM_USE_OMEGAH=YES
#endif

using namespace mfem;

namespace oh = Omega_h;

// check ent counts
void check_ents (oh::Mesh *o_mesh, Mesh *mesh) {
  auto dim = mesh->Dimension ();
  auto o_dim = o_mesh->dim ();
  auto nelems = mesh->GetNE ();
  auto o_nelems = o_mesh->nelems ();
  auto nverts = mesh->GetNV ();
  auto o_nverts = o_mesh->nverts ();
  auto nedges = mesh->GetNEdges ();
  auto o_nedges = o_mesh->nedges ();

  OMEGA_H_CHECK (dim == o_dim);
  OMEGA_H_CHECK (nelems = o_nelems);
  OMEGA_H_CHECK (nverts = o_nverts);
  OMEGA_H_CHECK (nedges = o_nedges);

  return;
}

void test_2d_mesh(oh::Library *lib) {
  oh::Mesh o_mesh(lib);
  oh::binary::read ("../../../mfem/data/omega_h/box_2d_8ele_2p.osh", lib->world(),
                    &o_mesh);

  ParMesh *mesh = new ParOmegaMesh (lib->world()->get_impl(), &o_mesh);
  auto rank = lib->world()->rank();

  check_ents (&o_mesh, mesh);
  char f_mfem_mesh [128];
  sprintf(f_mfem_mesh, "box_2d_mfem_2p_%d.vtk", rank);
  std::fstream vtkFs (f_mfem_mesh, std::ios::out);
  mesh->PrintVTK(vtkFs);
  return;
}

void test_3d_mesh(oh::Library *lib) {
  oh::Mesh o_mesh(lib);
  oh::binary::read ("../../../mfem/data/omega_h/box_3d_48k_2p.osh", lib->world(),
                    &o_mesh);

  ParMesh *mesh = new ParOmegaMesh (lib->world()->get_impl(), &o_mesh);
  auto rank = lib->world()->rank();

  check_ents (&o_mesh, mesh);
  char f_mfem_mesh [128];
  sprintf(f_mfem_mesh, "box_3d_48k_2p_mfem_rank%d.vtk", rank);
  std::fstream vtkFs (f_mfem_mesh, std::ios::out);
  mesh->PrintVTK(vtkFs);
  return;
}

int main(int argc, char *argv[])
{
  auto lib = oh::Library();

  // test for 2d and 3d simplex mesh
  test_2d_mesh (&lib);
  test_3d_mesh (&lib);

  return EXIT_SUCCESS;
}
