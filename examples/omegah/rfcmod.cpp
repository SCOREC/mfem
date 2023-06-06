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

#include <Omega_h_file.hpp>
#include <Omega_h_library.hpp>
#include <Omega_h_mesh.hpp>

#include <Omega_h_adapt.hpp>
#include <Omega_h_for.hpp>
#include <Omega_h_metric.hpp>
#include <Omega_h_timer.hpp>
#include <Omega_h_curve_coarsen.hpp>

using namespace std;
using namespace mfem;

namespace oh = Omega_h;

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
  int num_procs, myid;
  Mpi::Init(argc, argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // 2. Constant definition

  // 3. Read Omega_h mesh
  auto lib = oh::Library(&argc,&argv);
  oh::Mesh o_mesh(&lib);
  oh::binary::read (
      //"/lore/joshia5/Models/RF/2d_antenna_extrude/lhcd_3d_52k.osh"
      //"/lore/joshia5/Meshes/RF/assemble/708k_ref1p5mil.osh"
      //"/lore/joshia5/Meshes/RF/assemble/625k_ref1p5mil.osh"
      //"/lore/joshia5/Meshes/RF/assemble/v10_2rgn_11smallFeat_709k_p2.osh"
      //"/lore/joshia5/Meshes/RF/assemble/v10_2rgn_12smallFeat_1p1mil_p2.osh"
      //"/lore/joshia5/Meshes/RF/assemble/v10_2rgn_11smallFeat_625k_p2.osh"
      //"/lore/joshia5/develop/mfem_omega/build-omegah-python-rhel7/test_adapted_1"
      //"/lore/joshia5/Meshes/RF/assemble/v11_2rgn_12smallFeat_1286k_p2.osh"
      //"/lore/joshia5/Meshes/RF/assemble/v11_2rgn_12smallFeat_437k_p2.osh"
      "/lore/joshia5/Meshes/RF/assemble/v10_2rgn_12smallFeat_110k_p2.osh"
      //"/lore/joshia5/Meshes/RF/assemble/v10_2rgn_12smallFeat_314k_p2.osh"
      //"/lore/joshia5/Meshes/RF/assemble/v10_2rgn_11smallFeat_378k_p2.osh"
      //"/lore/joshia5/Meshes/RF/assemble/v10_2rgn_10smallFeat_343k_p2.osh"
      //"/lore/joshia5/Meshes/curved/inclusion_3p_sizes.osh"
      , lib.world(), &o_mesh);

  if (o_mesh.has_tag(1, "n_bezier_pts")) {
    printf("reading curve mesh\n");
    if (!o_mesh.has_tag(0, "bezier_pts")) {
      printf("initializing bez. shape\n");
      oh::calc_quad_ctrlPts_from_interpPts(&o_mesh);
      oh::elevate_curve_order_2to3(&o_mesh);
      o_mesh.add_tag<oh::Real>(0, "bezier_pts", o_mesh.dim(), o_mesh.coords());
    }
    else {
      assert (o_mesh.has_tag(1, "bezier_pts"));
      if (o_mesh.has_tag(2, "bezier_pts")) {
        o_mesh.set_max_order(3);
        printf("read omegah mesh of order 3\n");
      }
      else {
        printf("read omegah mesh of order 2\n");
        o_mesh.set_max_order(2);
      }
    }
  }
  int max_iter = 1;

  for (int Itr = 0; Itr < max_iter; Itr++)
  {

    Mesh *mesh = new OmegaMesh (&o_mesh);
    ParMesh *pmesh = new ParMesh (MPI_COMM_WORLD, *mesh);
    //ParMesh *pmesh = new ParOmegaMesh (MPI_COMM_WORLD, &o_mesh);
    printf("generated mfem mesh\n");
    int dim = pmesh->Dimension();

    std::string mesh_path = 
      //"/lore/joshia5/Models/RF/2d_antenna_extrude/lhcd_3d_52k.mesh"
      //"/lore/joshia5/Meshes/RF/assemble/708k_ref1p5mil.mesh"
      //"/lore/joshia5/Meshes/RF/assemble/625k_ref1p5mil.mesh"
      //"/lore/joshia5/Meshes/RF/assemble/v10_2rgn_11smallFeat_709k_p2.mesh"
      //"/lore/joshia5/Meshes/RF/assemble/v10_2rgn_12smallFeat_1p1mil_p2.mesh"
      //"/lore/joshia5/Meshes/RF/assemble/v10_2rgn_11smallFeat_625k_p2.mesh"
      //"/lore/joshia5/Meshes/RF/assemble/v11_2rgn_12smallFeat_1286k_p2.mesh"
      //"/lore/joshia5/develop/mfem_omega/build-omegah-python-rhel7/test_adapted_1.mesh"
      //"/lore/joshia5/Meshes/RF/assemble/v11_2rgn_12smallFeat_437k_p2.mesh"
      //"/lore/joshia5/Meshes/RF/assemble/v10_2rgn_11smallFeat_378k_p2.mesh"
      //"/lore/joshia5/Meshes/RF/assemble/v10_2rgn_10smallFeat_343k_p2.mesh"
      "/lore/joshia5/Meshes/RF/assemble/v10_2rgn_12smallFeat_110k_p2ELEVp3.mesh"
      //"/lore/joshia5/Meshes/curved/inclusion_3p_sizes.mesh"
      ;
    ofstream mesh_ofs(mesh_path);
    mesh_ofs.precision(16);
    pmesh->Print(mesh_ofs);
    printf("written mfem mesh file\n");

  } // end adaptation loop
  /*
*/
  return 0;
}
