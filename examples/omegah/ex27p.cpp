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
//
//               The user can adjust the values of 'd', 'g', with
//               command line options.

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

/* parts of this function is derived from the file
 * ugawg_linear.cpp of omega_h source code
 */
template <oh::Int dim>
static void set_target_metric(oh::Mesh* mesh, oh::Int scale, ParOmegaMesh
  *pOmesh) {
  auto coords = mesh->coords();
  auto target_metrics_w = oh::Write<oh::Real>
    (mesh->nverts() * oh::symm_ncomps(dim));
  pOmesh->ProjectErrorElementtoVertex (mesh, "zz_error");
  auto zz_error = mesh->get_array<oh::Real> (0, "zz_error");
  auto f = OMEGA_H_LAMBDA(oh::LO v) {
    auto x = coords[v*dim];
    auto y = coords[v*dim + 1];
    auto z = coords[v*dim + 2];
    auto h = oh::Vector<dim>();
    auto vtxError = zz_error[v];
    for (oh::Int i = 0; i < dim; ++i)
      h[i] = 0.025/(std::abs((vtxError)));
      //h[i] = 0.001 + 650* std::abs((vtxError)); // error small-->fine mesh
    auto m = diagonal(metric_eigenvalues_from_lengths(h));
    set_symm(target_metrics_w, v, m);
  };
  oh::parallel_for(mesh->nverts(), f);
  mesh->set_tag(oh::VERT, "target_metric", oh::Reals(target_metrics_w));
}

/* parts of this function is derived from the file
 * ugawg_linear.cpp of omega_h source code
 */
template <oh::Int dim>
void run_case(oh::Mesh* mesh, char const* vtk_path, oh::Int scale,
              const oh::Int myid, ParOmegaMesh *pOmesh) {
  auto world = mesh->comm();
  mesh->set_parting(OMEGA_H_GHOSTED);
  auto implied_metrics = get_implied_metrics(mesh);
  mesh->add_tag(oh::VERT, "metric", oh::symm_ncomps(dim), implied_metrics);
  mesh->add_tag<oh::Real>(oh::VERT, "target_metric", oh::symm_ncomps(dim));
  set_target_metric<dim>(mesh, scale, pOmesh);
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
  opts.min_quality_allowed = 0.00001;
  opts.xfer_opts.type_map["zz_error"] = OMEGA_H_POINTWISE;
  oh::Now t0 = oh::now();
  while (approach_metric(mesh, opts)) {
    adapt(mesh, opts);
    if (mesh->has_tag(oh::VERT, "target_metric")) set_target_metric<dim>(mesh,
                      scale, pOmesh);
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
  int order = 1;
  double sigma = -1.0;
  bool h1 = true;

  double mat_val = 5.0;
  double dbc_val1 = 18.0;
  double dbc_val2 = 50.0;
  double dbc_val3 = 90.0;
  double nbc_val = 12.0;

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

    // 5. Define a parallel finite element space on the parallel mesh. Here we
    //    use either continuous Lagrange finite elements or discontinuous
    //    Galerkin finite elements of the specified order.
    FiniteElementCollection *fec = new H1_FECollection(order, dim);
    ParFiniteElementSpace fespace(pmesh, fec, 1);

    // 6. Create "marker arrays" to define the portions of boundary associated
    //    with each type of boundary condition. These arrays have an entry
    //    corresponding to each boundary attribute. Placing a '1' in entry i
    //    marks attribute i+1 as being active, '0' is inactive.
    auto max_attr = pmesh->bdr_attributes.Size();
    Array<int> nbc_bdr(pmesh->bdr_attributes.Max());
    Array<int> dbc_bdr1(pmesh->bdr_attributes.Max());
    Array<int> dbc_bdr2(pmesh->bdr_attributes.Max());
    Array<int> dbc_bdr3(pmesh->bdr_attributes.Max());
    Array<int> ess_bdr(pmesh->bdr_attributes.Max());

    ess_bdr = 0; 
    ess_bdr[213] = 1;
    ess_bdr[75] = 1;
    ess_bdr[617] = 1;
    ess_bdr[81] = 1;
    ess_bdr[393] = 1;
    ess_bdr[389] = 1;
    ess_bdr[397] = 1;
    dbc_bdr1 = 0; 
    dbc_bdr2 = 0; 
    dbc_bdr3 = 0; 
    dbc_bdr1[213] = 1;
    dbc_bdr2[75] = 1;
    dbc_bdr3[617] = 1;
    dbc_bdr3[81] = 1;
    dbc_bdr3[393] = 1;
    dbc_bdr3[389] = 1;
    dbc_bdr3[397] = 1;
    nbc_bdr = 0;
    //nbc_bdr[23] = 1;

    Array<int> ess_tdof_list(0);
    // For a continuous basis the linear system must be modifed to enforce an
    // essential (Dirichlet) boundary condition. In the DG case this is not
    // necessary as the boundary condition will only be enforced weakly.
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 7. Setup the various coefficients needed for the Laplace operator and the
   //    various boundary conditions. In general these coefficients could be
   //    functions of position but here we use only constants.
   ConstantCoefficient matCoef(mat_val);
   ConstantCoefficient dbcCoef1(dbc_val1);
   ConstantCoefficient dbcCoef2(dbc_val2);
   ConstantCoefficient dbcCoef3(dbc_val3);
   ConstantCoefficient nbcCoef(nbc_val);

   // Since the n.Grad(u) terms arise by integrating -Div(m Grad(u)) by parts we
   // must introduce the coefficient 'm' into the boundary conditions.
   // Therefore, in the case of the Neumann BC, we actually enforce m n.Grad(u)
   // = m g rather than simply n.Grad(u) = g.
   ProductCoefficient m_nbcCoef(matCoef, nbcCoef);

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
   a.Assemble();

   // 10. Assemble the parallel linear form for the right hand side vector.
   ParLinearForm b(&fespace);

   // Set the Dirichlet values in the solution vector
   u.ProjectBdrCoefficient(dbcCoef1, dbc_bdr1);
   u.ProjectBdrCoefficient(dbcCoef2, dbc_bdr2);
   u.ProjectBdrCoefficient(dbcCoef3, dbc_bdr3);

   // Add the desired value for n.Grad(u) on the Neumann boundary
   b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_nbcCoef), nbc_bdr);
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

   // 19. Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
   //     with L2 projection in the smoothing step to better handle hanging
   //     nodes and parallel partitioning. We need to supply a space for the
   //     discontinuous flux (L2) and a space for the smoothed flux (H(div) is
   //     used here).
   int sdim = pmesh->SpaceDimension();

   L2_FECollection flux_fec(order, dim);
   ParFiniteElementSpace flux_fes(pmesh, &flux_fec, sdim);
   H1_FECollection smooth_flux_fec(order, dim);
   ParFiniteElementSpace smooth_flux_fes(pmesh, &smooth_flux_fec, dim);
   L2ZienkiewiczZhuEstimator estimator(*integ, u, flux_fes, smooth_flux_fes);

   const Vector mfem_err = estimator.GetLocalErrors();
   ParOmegaMesh* pOmesh = dynamic_cast<ParOmegaMesh*>(pmesh);
   pOmesh->ElementFieldMFEMtoOmegaH (&o_mesh, mfem_err, dim, "zz_error");
   pOmesh->ProjectErrorElementtoVertex (&o_mesh, "zz_error");

   // 17. Save data in the ParaView format
  //create gridfunction from estimator
  /* the next 4 lines were suggested by morteza */
  FiniteElementCollection *errorfec = new L2_FECollection(0, dim);
  ParFiniteElementSpace errorfespace(pmesh, errorfec);
  ParGridFunction l2errors(&errorfespace);
  l2errors = estimator.GetLocalErrors();
  /* */

   // 17. Save data in the ParaView format
   ParaViewDataCollection paraview_dc("Example27P", pmesh);
   paraview_dc.SetPrefixPath("CutTriCube");
   paraview_dc.SetLevelsOfDetail(1);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(false);
   paraview_dc.SetCycle(0);
   paraview_dc.SetTime(0.0);
   paraview_dc.RegisterField("temperature",&u);
   paraview_dc.RegisterField("zzErrors",&l2errors);
   paraview_dc.Save();

   // 18. Free the used memory.
   delete fec;

   // 20. Perform adapt

    char Fname[128];
    sprintf(Fname,
      "/lore/joshia5/Meshes/oh-mfem/cube_with_cutTriCube_5k_ex27p.vtk");
    char iter_str[8];
    sprintf(iter_str, "_%d", Itr);
    strcat(Fname, iter_str);
    puts(Fname);

    if ((Itr+1) < max_iter) run_case<3>(&o_mesh, Fname, Itr, myid, pOmesh);

  } // end adaptation loop

   return 0;
}
