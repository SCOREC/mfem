//                       MFEM Example 6 - Parallel Version
//
// Compile with: make ex6p
//
// Sample runs:  mpirun -np 4 ex6p -m ../data/square-disc.mesh -o 1
//               mpirun -np 4 ex6p -m ../data/square-disc.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/square-disc-nurbs.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/star.mesh -o 3
//               mpirun -np 4 ex6p -m ../data/escher.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/fichera.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/disc-nurbs.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/ball-nurbs.mesh
//               mpirun -np 4 ex6p -m ../data/pipe-nurbs.mesh
//               mpirun -np 4 ex6p -m ../data/star-surf.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/square-disc-surf.mesh -o 2
//               mpirun -np 4 ex6p -m ../data/amr-quad.mesh
//
// Device sample runs:
//               mpirun -np 4 ex6p -pa -d cuda
//               mpirun -np 4 ex6p -pa -d occa-cuda
//               mpirun -np 4 ex6p -pa -d raja-omp
//               mpirun -np 4 ex6p -pa -d ceed-cpu
//             * mpirun -np 4 ex6p -pa -d ceed-cuda
//               mpirun -np 4 ex6p -pa -d ceed-cuda:/gpu/cuda/shared
//
// Description:  This is a version of Example 1 with a simple adaptive mesh
//               refinement loop. The problem being solved is again the Laplace
//               equation -Delta u = 1 with homogeneous Dirichlet boundary
//               conditions. The problem is solved on a sequence of meshes which
//               are locally refined in a conforming (triangles, tetrahedrons)
//               or non-conforming (quadrilaterals, hexahedra) manner according
//               to a simple ZZ error estimator.
//
//               The example demonstrates MFEM's capability to work with both
//               conforming and nonconforming refinements, in 2D and 3D, on
//               linear, curved and surface meshes. Interpolation of functions
//               from coarse to fine meshes, as well as persistent GLVis
//               visualization are also illustrated.
//
//               We recommend viewing Example 1 before viewing this example.

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

using namespace std;
using namespace mfem;

namespace oh = Omega_h;

namespace { // anonymous namespace

/* this function is copied from the file
 * ugawg_linear.cpp of omega_h source code
 */
template <oh::Int dim>
static void set_target_metric(oh::Mesh* mesh) {
  auto coords = mesh->coords();
  auto target_metrics_w = oh::Write<oh::Real>(mesh->nverts() * oh::symm_ncomps(dim));
  auto f = OMEGA_H_LAMBDA(oh::LO v) {
    auto z = coords[v * dim + (dim - 1)];
    auto h = oh::Vector<dim>();
    for (oh::Int i = 0; i < dim - 1; ++i) h[i] = 0.1;
    h[dim - 1] = 0.001 + 0.198 * std::abs(z - 0.5);
    auto m = diagonal(metric_eigenvalues_from_lengths(h));
    set_symm(target_metrics_w, v, m);
  };
  oh::parallel_for(mesh->nverts(), f);
  mesh->set_tag(oh::VERT, "target_metric", oh::Reals(target_metrics_w));
}

/* this function is copied from the file
 * ugawg_linear.cpp of omega_h source code
 */
template <oh::Int dim>
void run_case(oh::Mesh* mesh, char const* vtk_path) {
  auto world = mesh->comm();
  mesh->set_parting(OMEGA_H_GHOSTED);
  auto implied_metrics = get_implied_metrics(mesh);
  mesh->add_tag(oh::VERT, "metric", oh::symm_ncomps(dim), implied_metrics);
  mesh->add_tag<oh::Real>(oh::VERT, "target_metric", oh::symm_ncomps(dim));
  set_target_metric<dim>(mesh);
  mesh->set_parting(OMEGA_H_ELEM_BASED);
  mesh->ask_lengths();
  mesh->ask_qualities();
  oh::vtk::FullWriter writer;
  if (vtk_path) {
    writer = oh::vtk::FullWriter(vtk_path, mesh);
    writer.write();
  }
  auto opts = oh::AdaptOpts(mesh);
  opts.verbosity = oh::EXTRA_STATS;
  opts.length_histogram_max = 2.0;
  opts.max_length_allowed = opts.max_length_desired * 2.0;
  oh::Now t0 = oh::now();
  while (approach_metric(mesh, opts)) {
    adapt(mesh, opts);
    if (mesh->has_tag(oh::VERT, "target_metric")) set_target_metric<dim>(mesh);
    if (vtk_path) writer.write();
  }
  oh::Now t1 = oh::now();
  std::cout << "total time: " << (t1 - t0) << " seconds\n";
}

} // end anonymous namespace

int main(int argc, char *argv[])
{
  // 1. Initialize MPI.
  int num_procs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  // 2. Parse command-line options.

  // 3. Read Omega_h mesh
  auto lib = oh::Library();
  oh::Mesh o_mesh(&lib);
  //oh::binary::read ("/lore/joshia5/develop/data/omegah/Kova100k_8p.osh", lib.world(),
  oh::binary::read ("/users/joshia5/new_mesh/box_3d_48k_4p.osh", lib.world(),
                    &o_mesh);

  // 4. Adapt the mesh if necessary
/*
  if (o_mesh.dim() == 2) run_case<2>(&o_mesh, "/users/joshia5/oh_2dadapt.vtk");
  if (o_mesh.dim() == 3) run_case<3>(&o_mesh, "/users/joshia5/new_mesh/ohAdapt_kova.vtk");
*/
  // 5. Create parallel mfem mesh object
  ParMesh *pmesh = new ParOmegaMesh (lib.world()->get_impl(), &o_mesh);

  int order = 1;
  bool static_cond = false;
  bool visualization = 1;
  int geom_order = 1;
  double adapt_ratio = 0.05;
  auto dim = o_mesh.dim();
  // 6. Define a parallel finite element space on the parallel mesh. Here we
  //    use continuous Lagrange finite elements of the specified order. If
  //    order < 1, we instead use an isoparametric/isogeometric space.
  FiniteElementCollection *fec;
  if (order > 0)
  {
    fec = new H1_FECollection(order, dim);
  }
  else if (pmesh->GetNodes())
  {
    fec = pmesh->GetNodes()->OwnFEC();
    if (myid == 1)
    {
      cout << "Using isoparametric FEs: " << fec->Name() << endl;
    }
  }
  else
  {
    fec = new H1_FECollection(order = 1, dim);
  }
  ParFiniteElementSpace *fespace = new ParFiniteElementSpace(pmesh, fec);
  HYPRE_Int size = fespace->GlobalTrueVSize();
  if (myid == 1)
  {
    cout << "Number of finite element unknowns: " << size << endl;
  }

  // 7. Set up the parallel linear form b(.) which corresponds to the
  //    right-hand side of the FEM linear system, which in this case is
  //    (1,phi_i) where phi_i are the basis functions in fespace.
  ParLinearForm *b = new ParLinearForm(fespace);
  ConstantCoefficient one(1.0);
  b->AddDomainIntegrator(new DomainLFIntegrator(one));

  // 8. Define the solution vector x as a parallel finite element grid function
  //    corresponding to fespace. Initialize x with initial guess of zero,
  //    which satisfies the boundary conditions.
  ParGridFunction x(fespace);
  x = 0.0;

  // 9. Connect to GLVis.
  char vishost[] = "localhost";
  int  visport   = 19916;

  socketstream sout;
  if (visualization)
  {
    sout.open(vishost, visport);
    if (!sout)
    {
      if (myid == 0)
      {
        cout << "Unable to connect to GLVis server at "
             << vishost << ':' << visport << endl;
        cout << "GLVis visualization disabled.\n";
      }
      visualization = false;
    }

    sout.precision(8);
  }

  // 10. Set up the parallel bilinear form a(.,.) on the finite element space
  //     corresponding to the Laplacian operator -Delta, by adding the
  //     Diffusion domain integrator.
  ParBilinearForm *a = new ParBilinearForm(fespace);
  a->AddDomainIntegrator(new DiffusionIntegrator(one));

  // 11. Assemble the parallel bilinear form and the corresponding linear
  //     system, applying any necessary transformations such as: parallel
  //     assembly, eliminating boundary conditions, applying conforming
  //     constraints for non-conforming AMR, static condensation, etc.
  if (static_cond) { a->EnableStaticCondensation(); }

  // 12. The main AMR loop. In each iteration we solve the problem on the
  //     current mesh, visualize the solution, and adapt the mesh.
/*
THIS WILL CHANGE TO CALL OMEGAH FIELDS & ADAPT CALLS
   apf::Field* Tmag_field = 0;
   apf::Field* temp_field = 0;
   apf::Field* ipfield = 0;
   apf::Field* sizefield = 0;
*/
  int max_iter = 1;

  for (int Itr = 0; Itr < max_iter; Itr++)
  {
    HYPRE_Int global_dofs = fespace->GlobalTrueVSize();
    if (myid == 1)
    {
      cout << "\nAMR iteration " << Itr << endl;
      cout << "Number of unknowns: " << global_dofs << endl;
    }

    // Assemble.
    a->Assemble();
    b->Assemble();

    // Essential boundary condition.
    Array<int> ess_tdof_list;
    if (pmesh->bdr_attributes.Size())
    {
      Array<int> ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);
    }

    // Form linear system.
    HypreParMatrix A;
    Vector B, X;
    const int copy_interior = 1;
    a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B, copy_interior);

    // 13. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
    //     preconditioner from hypre.
    HypreBoomerAMG amg;
    amg.SetPrintLevel(0);
    CGSolver pcg(A.GetComm());
    pcg.SetPreconditioner(amg);
    pcg.SetOperator(A);
    pcg.SetRelTol(1e-6);
    pcg.SetMaxIter(200);
    pcg.SetPrintLevel(3); // print the first and the last iterations only
    pcg.Mult(B, X);

    // 14. Recover the parallel grid function corresponding to X. This is the
    //     local finite element solution on each processor.
    a->RecoverFEMSolution(X, *b, x);

    // 15. Save in parallel the displaced mesh and the inverted solution (which
    //     gives the backward displacements to the original grid). This output
    //     can be viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
    {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_name << "sol." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      x.Save(sol_ofs);
    }

    // 16. Send the above data by socket to a GLVis server.  Use the "n" and "b"
    //     keys in GLVis to visualize the displacements.
    if (visualization)
    {
      sout << "parallel " << num_procs << " " << myid << "\n";
      sout << "solution\n" << *pmesh << x << flush;
    }

    // 17. Field transfer. Scalar solution field and magnitude field for error
    //     estimation are created the PUMI mesh.
/*
      if (order > geom_order)
      {
         Tmag_field = apf::createField(pumi_mesh, "field_mag",
                                       apf::SCALAR, apf::getLagrange(order));
         temp_field = apf::createField(pumi_mesh, "T_field",
                                       apf::SCALAR, apf::getLagrange(order));
      }
      else
      {
         Tmag_field = apf::createFieldOn(pumi_mesh, "field_mag",apf::SCALAR);
         temp_field = apf::createFieldOn(pumi_mesh, "T_field", apf::SCALAR);
      }

      ParPumiMesh* pPPmesh = dynamic_cast<ParPumiMesh*>(pmesh);
*/
    ParOmegaMesh* pOmesh = dynamic_cast<ParOmegaMesh*>(pmesh);
/*
      pPPmesh->FieldMFEMtoPUMI(pumi_mesh, &x, temp_field, Tmag_field);

      ipfield= spr::getGradIPField(Tmag_field, "MFEM_gradip", 2);
      sizefield = spr::getSPRSizeField(ipfield, adapt_ratio);

      apf::destroyField(Tmag_field);
      apf::destroyField(ipfield);
*/

    // 18. Perform adapt
    if (dim == 2) run_case<2>(&o_mesh, "/users/joshia5/oh_2dadapt.vtk");
    //if (dim == 3) run_case<3>(&o_mesh, "/users/joshia5/new_mesh/ohAdapt_kova.vtk");
    if (dim == 3) run_case<3>(&o_mesh,"/users/joshia5/new_mesh/ohAdapt_cube50k.vtk");
/*
      ma::Input* erinput = ma::configure(pumi_mesh, sizefield);
      erinput->shouldFixShape = true;
      erinput->maximumIterations = 2;
      if ( geom_order > 1)
      {
         crv::adapt(erinput);
      }
      else
      {
         ma::adapt(erinput);
      }
*/

    ParMesh *Adapmesh = new ParOmegaMesh(MPI_COMM_WORLD, &o_mesh);
    //pOmesh->UpdateMesh(Adapmesh);
    delete Adapmesh;

    // 19. Update the FiniteElementSpace, GridFunction, and bilinear form.
    fespace->Update();
    x.Update();
    x = 0.0;

/*
      pPPmesh->FieldPUMItoMFEM(pumi_mesh, temp_field, &x);
*/
    a->Update();
    b->Update();

/*
      // Destroy fields.
      apf::destroyField(temp_field);
      apf::destroyField(sizefield);
*/
  }

  // 20. Free the used memory.
  delete a;
  delete b;
  delete fespace;
  if (order > 0) { delete fec; }
  delete pmesh;

/*
   pumi_mesh->destroyNative();
   apf::destroyMesh(pumi_mesh);
   PCU_Comm_Free();
*/

  return 0;
}
