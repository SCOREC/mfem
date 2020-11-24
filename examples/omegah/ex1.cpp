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
#include <Omega_h_element.hpp>
#include <Omega_h_mark.hpp>

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

  // things needed from omegaH mesh //
  int Dim = o_mesh.oh::Mesh::dim();
  auto coords = o_mesh.oh::Mesh::coords();
  const unsigned long int nverts = coords.size()/Dim;
  const unsigned long int nelems = o_mesh.oh::Mesh::nelems();
  auto elem2vert = o_mesh.oh::Mesh::ask_down(Dim, oh::VERT);
  auto elem2vert_degree = oh::element_degree(OMEGA_H_SIMPLEX,
    Dim, oh::VERT);
  // to get the boundary and boundary elements, we will need to bring in the
  // ids of geom ents? or i think ther is an api which will give me the
  // classified elems.
  // can look at mark exposed sides and mark by exposure
  
  oh::Write<oh::LO> NumOfBdrElements = 0;

  auto exposed_sides = mark_exposed_sides (&o_mesh);
  auto ns = o_mesh.nents (Dim - 1); // num. of sides
  auto s2sc = o_mesh.ask_up (Dim - 1, Dim).a2ab;
  auto sc2c = o_mesh.ask_up (Dim - 1, Dim).ab2b;
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

/*
  // the logic to create elem in mfem
  // now the ReadPumiElement, i.e. read elem2verts for BdrElements
  // note here an Element *el, which is ptr to mfem element will be created
  // and vertices will be assigned to it
  //note here the elements to vert connectivity will have to be created
  el = NewElement(classDim()); // ptr to mfem element
  int nv, *v;
  // Create element in MFEM
  nv = el->GetNVertices();
  v  = el->GetVertices();
  // Fill the connectivity
  for (int i = 0; i < nv; ++i) {
    v[i] = apf::getNumber(vert_num, Verts[i], 0, 0);
  }
*/
  //v_num_loc is number of local vertices which apf creates. dont think its
  //needded from omegah mesh

  // example of copying data from omega to mfem device mem.
  int coords_size = coords.size(); // Base vector size
  double *x = new double[coords_size]; // Allocate base vector data
  Vector V(x, coords_size);
  V.ReadWrite();
  //V.Write();
  //V.Read();
  double *coords_mfem = V.ReadWrite(); // Pointer to device memory
  std::cout<<"Contents of V on the GPU"<<std::endl;

  auto fill = OMEGA_H_LAMBDA (oh::LO i) {
    coords_mfem[i] = coords[i];
    printf("%.1f coords=%.1f\n", coords_mfem[i], coords[i]);
  };
  oh::parallel_for(coords_size, fill);
  printf("\n");

  // call omegaH constructor
  OmegaMesh(&o_mesh, 0, 0, false);

  return 0;
}
