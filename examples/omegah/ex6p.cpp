//                       MFEM Example 6 - Parallel Version
//
// Compile with: make ex6p
//
// Sample runs: 
//
// Device sample runs:
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

#include<math.h>

using namespace std;
using namespace mfem;

namespace oh = Omega_h;

namespace { // anonymous namespace

/* this function is copied from the file
 * ugawg_linear.cpp of omega_h source code
 */
template <oh::Int dim>
static void set_target_metric(oh::Mesh* mesh, oh::Int scale) {
  auto coords = mesh->coords();
  auto target_metrics_w = oh::Write<oh::Real>(mesh->nverts() * oh::symm_ncomps(dim));
  auto f = OMEGA_H_LAMBDA(oh::LO v) {
    //3d mesh
    auto x = coords[v * dim ];
    auto y = coords[v * dim + (dim - 2)];
    auto z = coords[v * dim + (dim - 1)];
    auto h = oh::Vector<dim>();
    for (oh::Int i = 0; i < dim - 1; ++i) h[i] = 0.1;
    //h[dim - 1] = 0.001*(0.001 + 0.198 * std::abs(z - 0.5));
    //h[0] = 1*(0.001 + 0.198*std::abs(sqrt(y*y + z*z)));
    //h[dim - 1] = 10*(0.001 + 0.198*std::abs(sqrt(x*x + y*y + z*z))); //with x fields 
    //h[dim - 1] = 25*(0.001 + 0.198*std::abs(sqrt(x*x + y*y + z*z))); // 
    //h[dim - 1] = 1*(0.001 + 0.198 * std::abs(sqrt(x*x + y*y)));//20k elems 
    //h[dim - 1] = 10*(0.001 + 0.198 * std::abs(sqrt(x*x + z*z)));//20k elems
    //h[dim - 1] = 0.05*(0.001 + 0.198 * std::abs(sqrt(x*x + z*z)));//.5mil elems
    //h[dim - 1] = 0.05*(0.001 + 0.198 * std::abs(sqrt(x*x + z*z) - 0.5));
    h[dim - 1] = (0.001 + 0.198 * std::abs(z - 0.5));//original
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
void run_case(oh::Mesh* mesh, char const* vtk_path, oh::Int scale,
              const oh::Int myid) {
  auto world = mesh->comm();
  mesh->set_parting(OMEGA_H_GHOSTED);
  auto implied_metrics = get_implied_metrics(mesh);
  mesh->add_tag(oh::VERT, "metric", oh::symm_ncomps(dim), implied_metrics);
  mesh->add_tag<oh::Real>(oh::VERT, "target_metric", oh::symm_ncomps(dim));
  set_target_metric<dim>(mesh, scale);
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
  opts.max_length_allowed = opts.max_length_desired * 4.0;
  //opts.max_length_allowed = opts.max_length_desired * 2.0;
  opts.min_quality_allowed = 0.00001;
  oh::Now t0 = oh::now();
  while (approach_metric(mesh, opts)) {
    adapt(mesh, opts);
    if (mesh->has_tag(oh::VERT, "target_metric")) set_target_metric<dim>(mesh,
                      scale);
    if (vtk_path) writer.write();
  }
  oh::Now t1 = oh::now();
  if (!myid) std::cout << "total time: " << (t1 - t0) << " seconds\n";
}

template <oh::Int dim>
void run_case_givenMetric(oh::Mesh* mesh, char const* vtk_path, oh::Int scale,
              const oh::Int myid) {
  auto world = mesh->comm();
  mesh->set_parting(OMEGA_H_GHOSTED);
  auto implied_metrics = get_implied_metrics(mesh);
  //mesh->add_tag(oh::VERT, "metric", oh::symm_ncomps(dim), implied_metrics);
  mesh->add_tag<oh::Real>(oh::VERT, "target_metric", oh::symm_ncomps(dim));
  //set_target_metric<dim>(mesh, scale);
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
  //opts.max_length_allowed = opts.max_length_desired * 2.0;
  opts.min_quality_allowed = 0.00001;
  oh::Now t0 = oh::now();
  while (approach_metric(mesh, opts)) {
    adapt(mesh, opts);
    if (mesh->has_tag(oh::VERT, "target_metric")) set_target_metric<dim>(mesh,
                      scale);
    if (vtk_path) writer.write();
  }
  oh::Now t1 = oh::now();
  if (!myid) std::cout << "total time: " << (t1 - t0) << " seconds\n";
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
  oh::binary::read
    ("../../../mfem/data/omega_h/cube_with_cutTriCube5k_4p.osh",
                    lib.world(), &o_mesh);

  // 4. The main adaptive loop. In each iteration we create mfem mesh from
  // oh::mesh, initialize fe and solve, then adapt the oh::mesh for use in next
  // iteration
  int max_iter = 1;

  for (int Itr = 0; Itr < max_iter; Itr++)
  {
  // 5. Create parallel mfem mesh object
  ParMesh *pmesh = new ParOmegaMesh (lib.world()->get_impl(), &o_mesh);

  int order = 1;
  bool static_cond = false;
  bool visualization = 1;
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
  BilinearFormIntegrator *integ = new DiffusionIntegrator(one);
  a->AddDomainIntegrator(integ);
  //a->AddDomainIntegrator(new DiffusionIntegrator(one));

  // 11. Assemble the parallel bilinear form and the corresponding linear
  //     system, applying any necessary transformations such as: parallel
  //     assembly, eliminating boundary conditions, applying conforming
  //     constraints for non-conforming AMR, static condensation, etc.
  if (static_cond) { a->EnableStaticCondensation(); }

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

    // 12. Define and apply a parallel PCG solver for AX=B with the BoomerAMG
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

    // 13. Recover the parallel grid function corresponding to X. This is the
    //     local finite element solution on each processor.
    a->RecoverFEMSolution(X, *b, x);

    // 14. Save in parallel the displaced mesh and the inverted solution (which
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

    // 15. Send the above data by socket to a GLVis server. Use the "n" and "b"
    //     keys in GLVis to visualize the displacements.
    if (visualization)
    {
      sout << "parallel " << num_procs << " " << myid << "\n";
      sout << "solution\n" << *pmesh << x << flush;
    }

   /* ### for zz estimator ### */
   // 16. Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
   //     with L2 projection in the smoothing step to better handle hanging
   //     nodes and parallel partitioning. We need to supply a space for the
   //     discontinuous flux (L2) and a space for the smoothed flux (H(div) is
   //     used here).
   int sdim = pmesh->SpaceDimension();
  printf("space dim %d\n", sdim);

   L2_FECollection flux_fec(order, dim);
   ParFiniteElementSpace flux_fes(pmesh, &flux_fec, sdim);
   //RT_FECollection smooth_flux_fec(order-1, dim);
   //ParFiniteElementSpace smooth_flux_fes(pmesh, &smooth_flux_fec);
   // Another possible option for the smoothed flux space:
   H1_FECollection smooth_flux_fec(order, dim);
   ParFiniteElementSpace smooth_flux_fes(pmesh, &smooth_flux_fec, dim);
   L2ZienkiewiczZhuEstimator estimator(*integ, x, flux_fes, smooth_flux_fes);

/*
   FiniteElementSpace flux_fespace(pmesh, fec, sdim);
   ZienkiewiczZhuEstimator estimator(*integ, u, flux_fespace);
   estimator.SetAnisotropic();
*/
   const Vector mfem_err = estimator.GetLocalErrors();
   ParOmegaMesh* pOmesh = dynamic_cast<ParOmegaMesh*>(pmesh);
   pOmesh->ElementFieldMFEMtoOmegaH (&o_mesh, mfem_err, dim, "metric");
   pOmesh->ProjectFieldElementtoVertex (&o_mesh, "metric");
   //pOmesh->ElementFieldMFEMtoOmegaH (&o_mesh, mfem_err, dim, "mfem_field");
   //pOmesh->ProjectErrorElementtoVertex (&o_mesh, "mfem_field");

   // 17. Save data in the ParaView format

  //create gridfunction from estimator
  /* the next 4 lines were suggested by morteza */
  FiniteElementCollection *errorfec = new L2_FECollection(0, dim);
  ParFiniteElementSpace errorfespace(pmesh, errorfec);
  ParGridFunction l2errors(&errorfespace);
  l2errors = estimator.GetLocalErrors();
  /* */


   ParaViewDataCollection paraview_dc("Example6P_5k", pmesh);
   paraview_dc.SetPrefixPath("CutTriCube");
   paraview_dc.SetLevelsOfDetail(1);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(false);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0);
   paraview_dc.RegisterField("temperature",&x);
   paraview_dc.RegisterField("zzErrors",&l2errors);
   paraview_dc.Save();
    // 18. Perform adapt

    char Fname[128];
    sprintf(Fname,
      "cube_with_cutTriCube5k_4p_mfemMetric.vtk");
    char iter_str[8];
    sprintf(iter_str, "_%d", Itr);
    strcat(Fname, iter_str);
    puts(Fname);

    //run_case<3>(&o_mesh, Fname, Itr, myid);
    //run_case_givenMetric<3>(&o_mesh, Fname, Itr, myid);
    oh::vtk::write_parallel(Fname, &o_mesh, false);

    // 19. Update the FiniteElementSpace, GridFunction, and bilinear form.
    fespace->Update();
    x.Update();
    x = 0.0;

    a->Update();
    b->Update();

    // 20. Free the used memory.
    delete a;
    delete b;
    delete fespace;
    if (order > 0) { delete fec; }
    delete pmesh;
  }

  return 0;
}
