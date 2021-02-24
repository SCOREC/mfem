//                       MFEM-Omega_h Example 27 - Parallel Version
//
// Compile with: make ex27p
//
// Description:  This example code demonstrates the use of MFEM to define a
//               simple finite element discretization of the Laplace problem
//               -Delta u = 0 with a variety of boundary conditions.
//
//               Specifically, we discretize using a FE space of the specified
//               order using a continuous or discontinuous space. We then apply
//               Dirichlet, Neumann (both homogeneous and inhomogeneous), Robin,
//               and Periodic boundary conditions on different portions of a
//               predefined mesh.
//
//               The boundary conditions are defined as (where u is the solution
//               field):
//
//                  Dirichlet: u = d
//                  Neumann:   n.Grad(u) = g
//                  Robin:     n.Grad(u) + a u = b
//
//               The user can adjust the values of 'd', 'g', 'a', and 'b' with
//               command line options.
//
//               This example highlights the differing implementations of
//               boundary conditions with continuous and discontinuous Galerkin
//               formulations of the Laplace problem.

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

static double a_ = 0.2;

// Normal to hole with boundary attribute 4
void n4Vec(const Vector &x, Vector &n) { n = x; n[0] -= 0.5; n /= -n.Norml2(); }

Mesh * GenerateSerialMesh(int ref);

// Compute the average value of alpha*n.Grad(sol) + beta*sol over the boundary
// attributes marked in bdr_marker. Also computes the L2 norm of
// alpha*n.Grad(sol) + beta*sol - gamma over the same boundary.
double IntegrateBC(const ParGridFunction &sol, const Array<int> &bdr_marker,
                   double alpha, double beta, double gamma,
                   double &err);

namespace { // anonymous namespace

/* this function is copied from the file
 * ugawg_linear.cpp of omega_h source code
 */
template <oh::Int dim>
static void set_target_metric(oh::Mesh* mesh, oh::Int scale) {
  auto coords = mesh->coords();
  auto target_metrics_w = oh::Write<oh::Real>(mesh->nverts() * oh::symm_ncomps(dim));
  auto f = OMEGA_H_LAMBDA(oh::LO v) {
    auto x = coords[v * dim ];
    auto y = coords[v * dim + (dim - 2)];
    auto z = coords[v * dim + (dim - 1)];
    auto h = oh::Vector<dim>();
    for (oh::Int i = 0; i < dim - 1; ++i) h[i] = 0.1;
    h[dim - 1] = 10*(0.001 + 0.198 * std::abs(sqrt(x*x + y*y + z*z)));
    h[0] = (0.001 + 0.198 * std::abs(sqrt(y*y + z*z)));
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

} // end anonymous namespace

int main(int argc, char *argv[])
{
   // 1. Initialize MPI.
   MPI_Session mpi;
  int num_procs, myid;
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   if (!mpi.Root()) { mfem::out.Disable(); mfem::err.Disable(); }

   // 2. Parse command-line options.
   int ser_ref_levels = 2;
   int par_ref_levels = 1;
   int order = 1;
   double sigma = -1.0;
   double kappa = -1.0;
   bool h1 = true;
   bool visualization = true;

   double mat_val = 5.0;
   double dbc_val = 18.0;
   double nbc_val = 12.0;
   double rbc_a_val = 1.0; // du/dn + a * u = b
   double rbc_b_val = 1.0;

   OptionsParser args(argc, argv);
   args.AddOption(&h1, "-h1", "--continuous", "-dg", "--discontinuous",
                  "Select continuous \"H1\" or discontinuous \"DG\" basis.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree) or -1 for"
                  " isoparametric space.");
   args.AddOption(&sigma, "-s", "--sigma",
                  "One of the two DG penalty parameters, typically +1/-1."
                  " See the documentation of class DGDiffusionIntegrator.");
   args.AddOption(&kappa, "-k", "--kappa",
                  "One of the two DG penalty parameters, should be positive."
                  " Negative values are replaced with (order+1)^2.");
   args.AddOption(&ser_ref_levels, "-rs", "--refine-serial",
                  "Number of times to refine the mesh uniformly in serial.");
   args.AddOption(&par_ref_levels, "-rp", "--refine-parallel",
                  "Number of times to refine the mesh uniformly in parallel.");
   args.AddOption(&mat_val, "-mat", "--material-value",
                  "Constant value for material coefficient "
                  "in the Laplace operator.");
   args.AddOption(&dbc_val, "-dbc", "--dirichlet-value",
                  "Constant value for Dirichlet Boundary Condition.");
   args.AddOption(&nbc_val, "-nbc", "--neumann-value",
                  "Constant value for Neumann Boundary Condition.");
   args.AddOption(&rbc_a_val, "-rbc-a", "--robin-a-value",
                  "Constant 'a' value for Robin Boundary Condition: "
                  "du/dn + a * u = b.");
   args.AddOption(&rbc_b_val, "-rbc-b", "--robin-b-value",
                  "Constant 'b' value for Robin Boundary Condition: "
                  "du/dn + a * u = b.");
   args.AddOption(&a_, "-a", "--radius",
                  "Radius of holes in the mesh.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");

  printf("coeffs dbc %f nbc %f mat coef %f vis %d\n", dbc_val, nbc_val,
          mat_val, visualization);


   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(mfem::out);
      return 1;
   }
   if (kappa < 0 && !h1)
   {
      kappa = (order+1)*(order+1);
   }
   args.PrintOptions(mfem::out);

   if (a_ < 0.01)
   {
      mfem::out << "Hole radius too small, resetting to 0.01.\n";
      a_ = 0.01;
   }
   if (a_ > 0.49)
   {
      mfem::out << "Hole radius too large, resetting to 0.49.\n";
      a_ = 0.49;
   }

   // 3. Construct the (serial) mesh and refine it if requested.
   //Mesh *mesh = GenerateSerialMesh(ser_ref_levels);

   // 4. Define a parallel mesh by a partitioning of the serial mesh. Refine
   //    this mesh further in parallel to increase the resolution. Once the
   //    parallel mesh is defined, the serial mesh can be deleted.
   //ParMesh pmesh(MPI_COMM_WORLD, *mesh);
   //delete mesh;
    

  // 3. Read Omega_h mesh
  auto lib = oh::Library();
  oh::Mesh o_mesh(&lib);
  oh::binary::read ("/users/joshia5/Meshes/oh-mfem/cube_with_cutTriCube5k_4p.osh",
  //oh::binary::read ("/users/joshia5/new_mesh/box_3d_48k_4p.osh",
                    lib.world(), &o_mesh);
  int max_iter = 2;

  for (int Itr = 0; Itr < max_iter; Itr++)
  {

   ParMesh *pmesh = new ParOmegaMesh (MPI_COMM_WORLD, &o_mesh);
   int dim = pmesh->Dimension();
   for (int l = 0; l < par_ref_levels; l++)
   {
      pmesh->UniformRefinement();
   }

   // 5. Define a parallel finite element space on the parallel mesh. Here we
   //    use either continuous Lagrange finite elements or discontinuous
   //    Galerkin finite elements of the specified order.
   FiniteElementCollection *fec =
      h1 ? (FiniteElementCollection*)new H1_FECollection(order, dim) :
      (FiniteElementCollection*)new DG_FECollection(order, dim);
   ParFiniteElementSpace fespace(pmesh, fec);
   HYPRE_Int size = fespace.GlobalTrueVSize();
   mfem::out << "Number of finite element unknowns: " << size << endl;

   // 6. Create "marker arrays" to define the portions of boundary associated
   //    with each type of boundary condition. These arrays have an entry
   //    corresponding to each boundary attribute. Placing a '1' in entry i
   //    marks attribute i+1 as being active, '0' is inactive.
   auto max_attr = pmesh->bdr_attributes.Size();
   mfem::out << "attribute size : " << max_attr << endl;
   Array<int> nbc_bdr(pmesh->bdr_attributes.Max());
   Array<int> rbc_bdr(pmesh->bdr_attributes.Max());
   Array<int> dbc_bdr(pmesh->bdr_attributes.Max());

   //nbc_bdr = 0; dbc_bdr[75] = 1;
   //dbc_bdr = 0; nbc_bdr[213] = 1; nbc_bdr[23] = 1;
   nbc_bdr = 0; nbc_bdr[75] = 1;
   dbc_bdr = 0; dbc_bdr[213] = 1; dbc_bdr[23] = 1;
   rbc_bdr = 0;
   //rbc_bdr = 0; rbc_bdr[1] = 1;

   Array<int> ess_tdof_list(0);
   if (h1 && pmesh->bdr_attributes.Size())
   {
      // For a continuous basis the linear system must be modifed to enforce an
      // essential (Dirichlet) boundary condition. In the DG case this is not
      // necessary as the boundary condition will only be enforced weakly.
      fespace.GetEssentialTrueDofs(dbc_bdr, ess_tdof_list);
   }

   // 7. Setup the various coefficients needed for the Laplace operator and the
   //    various boundary conditions. In general these coefficients could be
   //    functions of position but here we use only constants.
   ConstantCoefficient matCoef(mat_val);
   ConstantCoefficient dbcCoef(dbc_val);
   ConstantCoefficient nbcCoef(nbc_val);
   ConstantCoefficient rbcACoef(rbc_a_val);
   ConstantCoefficient rbcBCoef(rbc_b_val);

   // Since the n.Grad(u) terms arise by integrating -Div(m Grad(u)) by parts we
   // must introduce the coefficient 'm' into the boundary conditions.
   // Therefore, in the case of the Neumann BC, we actually enforce m n.Grad(u)
   // = m g rather than simply n.Grad(u) = g.
   ProductCoefficient m_nbcCoef(matCoef, nbcCoef);
   ProductCoefficient m_rbcACoef(matCoef, rbcACoef);
   ProductCoefficient m_rbcBCoef(matCoef, rbcBCoef);

   // 8. Define the solution vector u as a parallel finite element grid function
   //    corresponding to fespace. Initialize u with initial guess of zero.
   ParGridFunction u(&fespace);
   u = 0.0;

   // 9. Set up the parallel bilinear form a(.,.) on the finite element space
   //    corresponding to the Laplacian operator -Delta, by adding the Diffusion
   //    domain integrator.
   ParBilinearForm a(&fespace);
   BilinearFormIntegrator *integ = new DiffusionIntegrator(matCoef);
   a.AddDomainIntegrator(integ);
   if (h1)
   {
      // Add a Mass integrator on the Robin boundary
      a.AddBoundaryIntegrator(new MassIntegrator(m_rbcACoef), rbc_bdr);
   }
   else
   {
      // Add the interfacial portion of the Laplace operator
      a.AddInteriorFaceIntegrator(new DGDiffusionIntegrator(matCoef,
                                                            sigma, kappa));

      // Counteract the n.Grad(u) term on the Dirichlet portion of the boundary
      a.AddBdrFaceIntegrator(new DGDiffusionIntegrator(matCoef, sigma, kappa),
                             dbc_bdr);

      // Augment the n.Grad(u) term with a*u on the Robin portion of boundary
      a.AddBdrFaceIntegrator(new BoundaryMassIntegrator(m_rbcACoef),
                             rbc_bdr);
   }
   a.Assemble();

   // 10. Assemble the parallel linear form for the right hand side vector.
   ParLinearForm b(&fespace);

   if (h1)
   {
      // Set the Dirichlet values in the solution vector
      u.ProjectBdrCoefficient(dbcCoef, dbc_bdr);

      // Add the desired value for n.Grad(u) on the Neumann boundary
      b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_nbcCoef), nbc_bdr);

      // Add the desired value for n.Grad(u) + a*u on the Robin boundary
      b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_rbcBCoef), rbc_bdr);
   }
   else
   {
      // Add the desired value for the Dirichlet boundary
      b.AddBdrFaceIntegrator(new DGDirichletLFIntegrator(dbcCoef, matCoef,
                                                         sigma, kappa),
                             dbc_bdr);

      // Add the desired value for n.Grad(u) on the Neumann boundary
      b.AddBdrFaceIntegrator(new BoundaryLFIntegrator(m_nbcCoef),
                             nbc_bdr);

      // Add the desired value for n.Grad(u) + a*u on the Robin boundary
      b.AddBdrFaceIntegrator(new BoundaryLFIntegrator(m_rbcBCoef),
                             rbc_bdr);
   }
   b.Assemble();

   // 11. Construct the linear system.
   OperatorPtr A;
   Vector B, X;
   a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);

   // 12. Solve the linear system A X = B.
   HypreSolver *amg = new HypreBoomerAMG;
   if (h1 || sigma == -1.0)
   {
      HyprePCG pcg(MPI_COMM_WORLD);
      pcg.SetTol(1e-12);
      pcg.SetMaxIter(200);
      pcg.SetPrintLevel(2);
      pcg.SetPreconditioner(*amg);
      pcg.SetOperator(*A);
      pcg.Mult(B, X);
   }
   else
   {
      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetAbsTol(0.0);
      gmres.SetRelTol(1e-12);
      gmres.SetMaxIter(200);
      gmres.SetKDim(10);
      gmres.SetPrintLevel(1);
      gmres.SetPreconditioner(*amg);
      gmres.SetOperator(*A);
      gmres.Mult(B, X);
   }
   delete amg;

   // 13. Recover the parallel grid function corresponding to U. This is the
   //     local finite element solution on each processor.
   a.RecoverFEMSolution(X, b, u);

   // 14. Build a mass matrix to help solve for n.Grad(u) where 'n' is a surface
   //     normal.
   ParBilinearForm m(&fespace);
   m.AddDomainIntegrator(new MassIntegrator);
   m.Assemble();

   ess_tdof_list.SetSize(0);
   OperatorPtr M;
   m.FormSystemMatrix(ess_tdof_list, M);

   // 15. Compute the various boundary integrals.
   mfem::out << endl
             << "Verifying boundary conditions" << endl
             << "=============================" << endl;
   {
      // Integrate the solution on the Dirichlet boundary and compare to the
      // expected value.
      double err, avg = IntegrateBC(u, dbc_bdr, 0.0, 1.0, dbc_val, err);

      bool hom_dbc = (dbc_val == 0.0);
      err /=  hom_dbc ? 1.0 : fabs(dbc_val);
      mfem::out << "Average of solution on Gamma_dbc:\t"
                << avg << ", \t"
                << (hom_dbc ? "absolute" : "relative")
                << " error " << err << endl;
   }
   {
      // Integrate n.Grad(u) on the inhomogeneous Neumann boundary and compare
      // to the expected value.
      double err, avg = IntegrateBC(u, nbc_bdr, 1.0, 0.0, nbc_val, err);

      bool hom_nbc = (nbc_val == 0.0);
      err /=  hom_nbc ? 1.0 : fabs(nbc_val);
      mfem::out << "Average of n.Grad(u) on Gamma_nbc:\t"
                << avg << ", \t"
                << (hom_nbc ? "absolute" : "relative")
                << " error " << err << endl;
   }
   {
      // Integrate n.Grad(u) on the homogeneous Neumann boundary and compare to
      // the expected value of zero.
      Array<int> nbc0_bdr(pmesh->bdr_attributes.Max());
      nbc0_bdr = 0;
      nbc0_bdr[3] = 1;

      double err, avg = IntegrateBC(u, nbc0_bdr, 1.0, 0.0, 0.0, err);

      bool hom_nbc = true;
      mfem::out << "Average of n.Grad(u) on Gamma_nbc0:\t"
                << avg << ", \t"
                << (hom_nbc ? "absolute" : "relative")
                << " error " << err << endl;
   }
   {
      // Integrate n.Grad(u) + a * u on the Robin boundary and compare to the
      // expected value.
      double err, avg = IntegrateBC(u, rbc_bdr, 1.0, rbc_a_val, rbc_b_val, err);

      bool hom_rbc = (rbc_b_val == 0.0);
      err /=  hom_rbc ? 1.0 : fabs(rbc_b_val);
      mfem::out << "Average of n.Grad(u)+a*u on Gamma_rbc:\t"
                << avg << ", \t"
                << (hom_rbc ? "absolute" : "relative")
                << " error " << err << endl;
   }

   // 16. Save the refined mesh and the solution in parallel. This output can be
   //     viewed later using GLVis: "glvis -np <np> -m mesh -g sol".
   {
      ostringstream mesh_name, sol_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << mpi.WorldRank();
      sol_name << "sol." << setfill('0') << setw(6) << mpi.WorldRank();

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_ofs(sol_name.str().c_str());
      sol_ofs.precision(8);
      u.Save(sol_ofs);
   }

   // 17. Save data in the ParaView format
   ParaViewDataCollection paraview_dc("Example27P", pmesh);
   paraview_dc.SetPrefixPath("CutTriCube");
   paraview_dc.SetLevelsOfDetail(1);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(false);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0);
   paraview_dc.RegisterField("temperature",&u);
   paraview_dc.Save();

   // 17. Send the solution by socket to a GLVis server.
   if (visualization)
   {
      string title_str = h1 ? "H1" : "DG";
      char vishost[] = "localhost";
      int  visport   = 19916;
      socketstream sol_sock(vishost, visport);
      sol_sock << "parallel " << mpi.WorldSize()
               << " " << mpi.WorldRank() << "\n";
      sol_sock.precision(8);
      sol_sock << "solution\n" << pmesh << u
               << "window_title '" << title_str << " Solution'"
               << " keys 'mmc'" << flush;
   }

   // 18. Free the used memory.
   delete fec;

   /* ### for zz estimator ### */
   // 19. Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
   //     with L2 projection in the smoothing step to better handle hanging
   //     nodes and parallel partitioning. We need to supply a space for the
   //     discontinuous flux (L2) and a space for the smoothed flux (H(div) is
   //     used here).
   int order = 1;
   int sdim = pmesh->SpaceDimension();
  printf("space dim %d\n", sdim);

   L2_FECollection flux_fec(order, dim);
   ParFiniteElementSpace flux_fes(pmesh, &flux_fec, sdim);
   //RT_FECollection smooth_flux_fec(order-1, dim);
   //ParFiniteElementSpace smooth_flux_fes(pmesh, &smooth_flux_fec);
   // Another possible option for the smoothed flux space:
   H1_FECollection smooth_flux_fec(order, dim);
   ParFiniteElementSpace smooth_flux_fes(pmesh, &smooth_flux_fec, dim);
   L2ZienkiewiczZhuEstimator estimator(*integ, u, flux_fes, smooth_flux_fes);

/*
   FiniteElementSpace flux_fespace(pmesh, fec, sdim);
   ZienkiewiczZhuEstimator estimator(*integ, u, flux_fespace);
   estimator.SetAnisotropic();
*/
   const Vector mfem_err = estimator.GetLocalErrors();
   ParOmegaMesh* pOmesh = dynamic_cast<ParOmegaMesh*>(pmesh);
   pOmesh->ErrorEstimatorMFEMtoOmegaH (&o_mesh, mfem_err);

   // 20. Perform adapt

    char Fname[128];
    sprintf(Fname,
      "/lore/joshia5/Meshes/oh-mfem/cube_with_cutTriCube_5k_ex27p.vtk");
      //"/users/joshia5/Meshes/oh-mfem/cube_with_cutTriCube5k_4p.vtk");
    //sprintf(Fname, "/users/joshia5/new_mesh/ohAdapt1p5XIter_cube.vtk");
    char iter_str[8];
    sprintf(iter_str, "_%d", Itr);
    strcat(Fname, iter_str);
    puts(Fname);

    if ((Itr+1) < max_iter) run_case<3>(&o_mesh, Fname, Itr, myid);

  } // end adaptation loop

   return 0;
}

void quad_trans(double u, double v, double &x, double &y, bool log = false)
{
   double a = a_; // Radius of disc

   double d = 4.0 * a * (M_SQRT2 - 2.0 * a) * (1.0 - 2.0 * v);

   double v0 = (1.0 + M_SQRT2) * (M_SQRT2 * a - 2.0 * v) *
               ((4.0 - 3 * M_SQRT2) * a +
                (8.0 * (M_SQRT2 - 1.0) * a - 2.0) * v) / d;

   double r = 2.0 * ((M_SQRT2 - 1.0) * a * a * (1.0 - 4.0 *v) +
                     2.0 * (1.0 + M_SQRT2 *
                            (1.0 + 2.0 * (2.0 * a - M_SQRT2 - 1.0) * a)) * v * v
                    ) / d;

   double t = asin(v / r) * u / v;
   if (log)
   {
      mfem::out << "u, v, r, v0, t "
                << u << " " << v << " " << r << " " << v0 << " " << t
                << endl;
   }
   x = r * sin(t);
   y = r * cos(t) - v0;
}

void trans(const Vector &u, Vector &x)
{
   double tol = 1e-4;

   if (u[1] > 0.5 - tol || u[1] < -0.5 + tol)
   {
      x = u;
      return;
   }
   if (u[0] > 1.0 - tol || u[0] < -1.0 + tol || fabs(u[0]) < tol)
   {
      x = u;
      return;
   }

   if (u[0] > 0.0)
   {
      if (u[1] > fabs(u[0] - 0.5))
      {
         quad_trans(u[0] - 0.5, u[1], x[0], x[1]);
         x[0] += 0.5;
         return;
      }
      if (u[1] < -fabs(u[0] - 0.5))
      {
         quad_trans(u[0] - 0.5, -u[1], x[0], x[1]);
         x[0] += 0.5;
         x[1] *= -1.0;
         return;
      }
      if (u[0] - 0.5 > fabs(u[1]))
      {
         quad_trans(u[1], u[0] - 0.5, x[1], x[0]);
         x[0] += 0.5;
         return;
      }
      if (u[0] - 0.5 < -fabs(u[1]))
      {
         quad_trans(u[1], 0.5 - u[0], x[1], x[0]);
         x[0] *= -1.0;
         x[0] += 0.5;
         return;
      }
   }
   else
   {
      if (u[1] > fabs(u[0] + 0.5))
      {
         quad_trans(u[0] + 0.5, u[1], x[0], x[1]);
         x[0] -= 0.5;
         return;
      }
      if (u[1] < -fabs(u[0] + 0.5))
      {
         quad_trans(u[0] + 0.5, -u[1], x[0], x[1]);
         x[0] -= 0.5;
         x[1] *= -1.0;
         return;
      }
      if (u[0] + 0.5 > fabs(u[1]))
      {
         quad_trans(u[1], u[0] + 0.5, x[1], x[0]);
         x[0] -= 0.5;
         return;
      }
      if (u[0] + 0.5 < -fabs(u[1]))
      {
         quad_trans(u[1], -0.5 - u[0], x[1], x[0]);
         x[0] *= -1.0;
         x[0] -= 0.5;
         return;
      }
   }
   x = u;
}

double IntegrateBC(const ParGridFunction &x, const Array<int> &bdr,
                   double alpha, double beta, double gamma,
                   double &glb_err)
{
   double loc_vals[3];
   double &nrm = loc_vals[0];
   double &avg = loc_vals[1];
   double &err = loc_vals[2];

   nrm = 0.0;
   avg = 0.0;
   err = 0.0;

   const bool a_is_zero = alpha == 0.0;
   const bool b_is_zero = beta == 0.0;

   const ParFiniteElementSpace &fes = *x.ParFESpace();
   MFEM_ASSERT(fes.GetVDim() == 1, "");
   ParMesh &mesh = *fes.GetParMesh();
   Vector shape, loc_dofs, w_nor;
   DenseMatrix dshape;
   Array<int> dof_ids;
   for (int i = 0; i < mesh.GetNBE(); i++)
   {
      if (bdr[mesh.GetBdrAttribute(i)-1] == 0) { continue; }

      FaceElementTransformations *FTr = mesh.GetBdrFaceTransformations(i);
      if (FTr == nullptr) { continue; }

      const FiniteElement &fe = *fes.GetFE(FTr->Elem1No);
      MFEM_ASSERT(fe.GetMapType() == FiniteElement::VALUE, "");
      const int int_order = 2*fe.GetOrder() + 3;
      const IntegrationRule &ir = IntRules.Get(FTr->FaceGeom, int_order);

      fes.GetElementDofs(FTr->Elem1No, dof_ids);
      x.GetSubVector(dof_ids, loc_dofs);
      if (!a_is_zero)
      {
         const int sdim = FTr->Face->GetSpaceDim();
         w_nor.SetSize(sdim);
         dshape.SetSize(fe.GetDof(), sdim);
      }
      if (!b_is_zero)
      {
         shape.SetSize(fe.GetDof());
      }
      for (int j = 0; j < ir.GetNPoints(); j++)
      {
         const IntegrationPoint &ip = ir.IntPoint(j);
         IntegrationPoint eip;
         FTr->Loc1.Transform(ip, eip);
         FTr->Face->SetIntPoint(&ip);
         double face_weight = FTr->Face->Weight();
         double val = 0.0;
         if (!a_is_zero)
         {
            FTr->Elem1->SetIntPoint(&eip);
            fe.CalcPhysDShape(*FTr->Elem1, dshape);
            CalcOrtho(FTr->Face->Jacobian(), w_nor);
            val += alpha * dshape.InnerProduct(w_nor, loc_dofs) / face_weight;
         }
         if (!b_is_zero)
         {
            fe.CalcShape(eip, shape);
            val += beta * (shape * loc_dofs);
         }

         // Measure the length of the boundary
         nrm += ip.weight * face_weight;

         // Integrate alpha * n.Grad(x) + beta * x
         avg += val * ip.weight * face_weight;

         // Integrate |alpha * n.Grad(x) + beta * x - gamma|^2
         val -= gamma;
         err += (val*val) * ip.weight * face_weight;
      }
   }

   double glb_vals[3];
   MPI_Allreduce(loc_vals, glb_vals, 3, MPI_DOUBLE, MPI_SUM, fes.GetComm());

   double glb_nrm = glb_vals[0];
   double glb_avg = glb_vals[1];
   glb_err = glb_vals[2];

   // Normalize by the length of the boundary
   if (std::abs(glb_nrm) > 0.0)
   {
      glb_err /= glb_nrm;
      glb_avg /= glb_nrm;
   }

   // Compute l2 norm of the error in the boundary condition (negative
   // quadrature weights may produce negative 'err')
   glb_err = (glb_err >= 0.0) ? sqrt(glb_err) : -sqrt(-glb_err);

   // Return the average value of alpha * n.Grad(x) + beta * x
   return glb_avg;
}
