//                                MFEM Example 1
//                              Example for testing Omega_h to MFEM mesh
//                              constructor
//
// Compile with: make ex1
//
// Sample runs:
//    ex1
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
  double o_low[3], o_high[3];
  o_low[0] = 0.0;
  o_low[1] = 0.0;
  o_low[2] = 0.0;
  o_high[0] = 1.0;
  o_high[1] = 1.0;
  o_high[2] = 0.0;

  auto o_mesh = oh::build_box (lib->world(), OMEGA_H_SIMPLEX,
                                   o_high[0], o_high[1], o_high[2], 2, 2, 0);
  oh::vtk::write_parallel ("box_2d", &o_mesh);

  // call omegaH to mfem constructor
  Mesh *mesh = new OmegaMesh (&o_mesh, 1, 0, false, 0);

  check_ents (&o_mesh, mesh);
  Vector low, high;
  mesh->GetBoundingBox (low, high);
  for (int i = 0; i < mesh->Dimension(); ++i) {
    OMEGA_H_CHECK (low(i) == o_low[i]);
    OMEGA_H_CHECK (high(i) == o_high[i]);
  }

  std::string mfem_mesh ("box_2d_mfem.vtk");
  std::fstream vtkFs (mfem_mesh.c_str(), std::ios::out);
  mesh->PrintVTK(vtkFs);
  return;
}

void test_3d_mesh(oh::Library *lib) {
  double o_low[3], o_high[3];
  o_low[0] = 0.0; o_low[1] = 0.0; o_low[2] = 0.0;
  o_high[0] = 1.0; o_high[1] = 1.0; o_high[2] = 1.0;

  auto o_mesh = oh::build_box (lib->world(), OMEGA_H_SIMPLEX,
                               o_high[0], o_high[1], o_high[2], 2, 2, 2);
  oh::vtk::write_parallel ("box_3d", &o_mesh);

  // call omegaH to mfem constructor
  Mesh *mesh = new OmegaMesh (&o_mesh, 1, 0, false, 0);

  check_ents (&o_mesh, mesh);
  Vector low, high;
  mesh->GetBoundingBox (low, high);
  for (int i = 0; i < mesh->Dimension(); ++i) {
    OMEGA_H_CHECK (low(i) == o_low[i]);
    OMEGA_H_CHECK (high(i) == o_high[i]);
  }

  std::string mfem_mesh ("box_3d_mfem.vtk");
  std::fstream vtkFs (mfem_mesh.c_str(), std::ios::out);
  mesh->PrintVTK(vtkFs);
  return;
}

int main(int argc, char *argv[])
{
/*
  Device device("cuda");
  const MemoryType d_mt = device.GetMemoryType();
*/
  auto lib = oh::Library();

  // test for 2d and 3d simplex mesh
  test_2d_mesh (&lib);
  test_3d_mesh (&lib);

  return EXIT_SUCCESS;
}
