//                       MFEM-Omega_h Example 2 - Parallel Version
//
// Description:  In this example, we define a simple finite element
//               discretization of the Laplace problem -Delta u = 0
//               on a unit box model with one corner cut out and a triangular
//               hole through the body.
//
//               Specifically, we discretize using a FE space of the specified
//               order using a continuous space. We then apply
//               Dirichlet, Neumann (homogeneous),
//               boundary conditions on different portions of a
//               predefined mesh.
//
//               The boundary conditions are defined as (where u is the solution
//               field):
//
//                  Dirichlet: u = d
//                  Neumann:   n.Grad(u) = g
//
//               The user can adjust the values of 'd', 'g'. Here d is taken
//               as a function of x for face along x axis and function of y
//               for two faces along y axis. 

#include "mfem.hpp"
#include <fstream>
#include <iostream>

using namespace mfem;

int main(int argc, char *argv[])
{
  // 1. Initialize MPI.
  int num_procs, myid;
  Mpi::Init(argc, argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // 2. Read parmesh
  //std::string infile = "/lore/joshia5/develop/RF_petram_case_files/cmod-adapt/150deg/Prat0p5/sol/case_005/solmesh_0.000000";
  std::string infile = "/lore/joshia5/develop/RF_petram_case_files/cmod-adapt/150deg/Prat0p5/sol/case_005/solmesh_0.0000";
  if (myid > 9) {
    infile += std::to_string(myid);
  }
  else {
    infile += std::to_string(0);
    infile += std::to_string(myid);
  }
  std::cout << "rank " << myid << " file " <<infile <<"\n";
  std::ifstream mesh_file(infile.c_str());
  ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, mesh_file, false);
  int dim = pmesh->Dimension();
  if (!myid) std::cout << "read parmesh dim " << dim << "\n";

  // 3. Save serial mesh
  std::string mesh_path = 
    "/lore/joshia5/Meshes/RF/assemble/Prat0p5_ini314kref1mil_p2.mesh";
  std::ofstream mesh_ofs(mesh_path);
  mesh_ofs.precision(16);
  pmesh->PrintAsSerial(mesh_ofs);
  if (!myid) printf("written serial mfem mesh %s\n", mesh_path);

  return 0;
}
