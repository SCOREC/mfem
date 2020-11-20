//                                MFEM Example 1
//                              Omega_h Hello World
//
// Compile with: make ex1
//
// Sample runs:
//    ex1
//
// Note:         generates a 2x2 triangle mesh
//
// Description:  see note

#include "mfem.hpp"
//#include "unit_tests.hpp"
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

//using namespace std;
using namespace mfem;

int main(int argc, char *argv[])
{
  auto lib = Omega_h::Library();

  Device device("cuda");
  device.Print();
  const MemoryType d_mt = device.GetMemoryType();

  int N = 10;   // Base vector size
  int nSub = 5; // Sub vector size

  double *x = new double[N]; // Allocate base vector data

  // Shallow copy into MFEM vector + read command to copy to the GPU
  Vector V(x, N); V.ReadWrite();
  //Vector V(x, N); V.Write();
  //Vector V(x, N); V.Read();

  // Shallow copy into MFEM vector + annotate as reference to a base
  // vector
  // Similar to doing double * Vs = &x[nSub]; // Pointer to sub-vector
  // data
  Vector Vs; Vs.MakeRef(V, nSub, (N-nSub)); // MakeRef(vector, offset,size)

  std::cout<<"V - Flags"<<std::endl;
  V.GetMemory().PrintFlags();

  std::cout<<"Vs - Flags "<<std::endl;
  Vs.GetMemory().PrintFlags();

  double *d_x = V.ReadWrite(); // Pointer to device memory
  //const double *d_x = V.Write(); // Pointer to device memory
  //const double *d_x = V.Read(); // Pointer to device memory
  std::cout<<"Contents of V on the GPU"<<std::endl;

  auto o_mesh = Omega_h::build_box(lib.world(), OMEGA_H_SIMPLEX,
                                 1., 1., 0, 2, 2, 0);
  Omega_h::vtk::write_parallel("box", &o_mesh);
  auto coords = o_mesh.Omega_h::Mesh::coords();

  auto fill = OMEGA_H_LAMBDA (Omega_h::LO i) {
  //RAJA::forall<RAJA::cuda_exec<CUDA_BLOCK_SZ>>
    //(RAJA::RangeSegment(0,N), [=] RAJA_DEVICE (int i) {
       d_x[i] = coords[i] ;
       printf("%.1f coords=%.1f\n", d_x[i], coords[i]);
  //  });
  };
  Omega_h::parallel_for(N, fill);
  printf("\n");

  return 0;
}
