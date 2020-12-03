//                                MFEM Example 1
//                              Omega_h Hello World
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

#include <Omega_h_for.hpp>

#ifndef MFEM_USE_OMEGAH
#error This example requires that MFEM is built with MFEM_USE_OMEGAH=YES
#endif

using namespace mfem;

namespace oh = Omega_h;

int main(int argc, char *argv[])
{
  auto lib = oh::Library();

  Device device("cuda");
  device.Print();
  const MemoryType d_mt = device.GetMemoryType();

  auto o_mesh = oh::build_box(lib.world(), OMEGA_H_SIMPLEX,
                                   1., 1., 0, 2, 2, 0);
  oh::vtk::write_parallel("box", &o_mesh);

  // example of copying data from omega to mfem device mem.
  auto coords = o_mesh.oh::Mesh::coords();
  printf("from example nelems=%d\n", o_mesh.oh::Mesh::nelems());
  int coords_size = coords.size(); // Base vector size
  double *x = new double[coords_size]; // Allocate base vector data
  Vector V(x, coords_size);
  V.ReadWrite();
  double *coords_mfem = V.ReadWrite(); // Pointer to device memory
  //std::cout<<"Contents of V on the GPU"<<std::endl;

  auto fill = OMEGA_H_LAMBDA (oh::LO i) {
    coords_mfem[i] = coords[i];
    //printf("%.1f coords=%.1f\n", coords_mfem[i], coords[i]);
  };
  oh::parallel_for(coords_size, fill);
  printf("\n");
  // end copy example

  // call omegaH constructor
  OmegaMesh(&o_mesh, 0, 0, false);

  return 0;
}
