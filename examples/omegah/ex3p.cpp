//                       MFEM-Omega_h Example 3 - Parallel Version
//
// Description:  In this example, we define a simple finite element
//               discretization of the Laplace problem -Delta u = 0
//               on a unit box model with one quarter cut out.
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

using namespace std;
using namespace mfem;

namespace oh = Omega_h;

/* the next 8 lines were copied from morteza's example*/
double f_239(const Vector& x)
{
  return 10. - (x(0) - 0.5)/ 0.05;
}
double f_243(const Vector& x)
{
  return 10. - (x(1) - 0.5)/ 0.05;
}
/**/

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
  pOmesh->ProjectFieldElementtoVertex (mesh, "zz_error");
  auto zz_error = mesh->get_array<oh::Real> (0, "zz_error");
  auto f = OMEGA_H_LAMBDA(oh::LO v) {
    auto x = coords[v*dim];
    auto y = coords[v*dim + 1];
    auto z = coords[v*dim + 2];
    auto h = oh::Vector<dim>();
    auto vtxError = zz_error[v];
    for (oh::Int i = 0; i < dim; ++i)
      h[i] = 0.00075/std::pow(std::abs(vtxError), 0.6);//1k, 0.8mil
      //h[i] = 0.001/std::pow(std::abs(vtxError), 0.6);//1k, 0.33mil
      //h[i] = 0.001/std::sqrt(std::abs(vtxError));//1k, 1.73mil
      //h[i] = 0.0015/std::sqrt(std::abs(vtxError));//1k, 488k
      //h[i] = 0.0001/(std::abs(vtxError));//1.5mil
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
  int num_procs, myid;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  int order = 1;

  // Read Omega_h mesh
  auto lib = oh::Library();
  oh::Mesh o_mesh(&lib);
  oh::binary::read ("../../../mfem/data/omega_h/unitbox_cutQuart_1k_4p.osh",
                    lib.world(), &o_mesh);

  //number of adaptation iterations
  int max_iter = 2;
  for (int Itr = 0; Itr < max_iter; Itr++)  {

    ParMesh *mfem_mesh = new ParOmegaMesh (MPI_COMM_WORLD, &o_mesh);

    /* from here till line 268, i.e., BC settings and mfem solution setup
     * copied from morteza's example or suggested by him, also model for
     * this example created by morteza */
    int dim  = mfem_mesh->Dimension();
    int sdim = mfem_mesh->SpaceDimension();

    FiniteElementCollection *fec = new H1_FECollection(order, dim);
    ParFiniteElementSpace *fes = new ParFiniteElementSpace(mfem_mesh, fec, 1);

    // Define the solution vector and initialize to 0. This will hold the fem solution
    ParGridFunction u_fem(fes);
    u_fem = 0.;

    // Setting the Dirichlet boundary conditions
    // face 239 gets the f_239 (linear function in x going from 10 at the corner to 0)
    Array<int> bdr_mask_dirichlet_239(mfem_mesh->bdr_attributes.Max());
    bdr_mask_dirichlet_239 = 0;
    for (int i = 0; i < bdr_mask_dirichlet_239.Size(); i++) {
      if (i+1 == 239)
        bdr_mask_dirichlet_239[i] = 1;
    }
    FunctionCoefficient f239_coeff(f_239);
    u_fem.ProjectBdrCoefficient(f239_coeff, bdr_mask_dirichlet_239);
    // face 243 gets the f_243 (linear function in y going from 10 at the corner to 0)
    Array<int> bdr_mask_dirichlet_243(mfem_mesh->bdr_attributes.Max());
    bdr_mask_dirichlet_243 = 0;
    for (int i = 0; i < bdr_mask_dirichlet_243.Size(); i++) {
      if (i+1 == 243)
        bdr_mask_dirichlet_243[i] = 1;
    }
    FunctionCoefficient f243_coeff(f_243);
    u_fem.ProjectBdrCoefficient(f243_coeff, bdr_mask_dirichlet_243);

    // Setting the Neumann boundary conditions
    // zero flux on faces with tag 9 and 36
    Array<int> bdr_mask_neumann(mfem_mesh->bdr_attributes.Max());
    bdr_mask_neumann = 0;
    for (int i = 0; i < bdr_mask_neumann.Size(); i++) {
      if (i+1 == 9) // bottom edge with boundary attribute 13
        bdr_mask_neumann[i] = 1;
      else if (i+1 == 36) // top edge with boundary attribute 17
        bdr_mask_neumann[i] = 1;
      else
        bdr_mask_neumann[i] = 0;
    }

    // Add the right-hand side corresponding to the force term and the zero-flux conditions
    // in mfem terminology this is known as the LinearForm
    ParLinearForm b(fes);
    // force will be zero for order 1 and will be 1 for order 2 (this is because for this we
    // know the exact solutions)
    ConstantCoefficient force(0);
    b.AddDomainIntegrator(new DomainLFIntegrator(force));
    // add the zero flux term to the right hand side
    ConstantCoefficient kappa(1.);
    ConstantCoefficient flux(0.);
    ProductCoefficient kappa_flux(kappa, flux);
    b.AddBoundaryIntegrator(new BoundaryLFIntegrator(kappa_flux), bdr_mask_neumann);
    b.Assemble();

    // Add the right-hand-side corresponding to the diffusive term
    // in mfem terminology this is known as the Bilinear form
    ParBilinearForm a(fes);
    BilinearFormIntegrator *integ = new DiffusionIntegrator(kappa);
    a.AddDomainIntegrator(integ);
    a.Assemble();

    // Get the essential dofs
    // all faces that are not Neumann are essential. These are 2, 27, 30, 33, and 239
    Array<int> ess_tdof_list;
    Array<int> ess_bdr(mfem_mesh->bdr_attributes.Max());
    ess_bdr = 0;
    for (int i = 0; i < ess_bdr.Size(); i++) {
      if (i+1 == 2)
        ess_bdr[i] = 1;
      else if (i+1 == 27)
        ess_bdr[i] = 1;
      else if (i+1 == 30)
        ess_bdr[i] = 1;
      else if (i+1 == 33)
        ess_bdr[i] = 1;
      else if (i+1 == 239)
        ess_bdr[i] = 1;
      else if (i+1 == 243)
        ess_bdr[i] = 1;
      else
        ess_bdr[i] = 0;
    }
    fes->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    // Form the linear system and solve
    OperatorPtr A;
    Vector B, X;
    a.FormLinearSystem(ess_tdof_list, u_fem, b, A, X, B);

    // solve step
    #ifdef MFEM_USE_SUPERLU // if MFEM has super lu use a direct solve
    SuperLUSolver *superlu = new SuperLUSolver(MPI_COMM_WORLD);
    Operator *SLU_A = new SuperLURowLocMatrix(*A.As<HypreParMatrix>());
    superlu->SetPrintStatistics(true);
    superlu->SetSymmetricPattern(false);
    superlu->SetColumnPermutation(superlu::METIS_AT_PLUS_A);
    superlu->SetRowPermutation(superlu::LargeDiag_MC64);
    superlu->SetIterativeRefine(superlu::SLU_DOUBLE);
    superlu->SetOperator(*SLU_A);
    superlu->Mult(B, X);
    #else // if not use an iterative solve
    HypreBoomerAMG amg(*A.As<HypreParMatrix>());
    amg.SetSystemsOptions(dim);
    GMRESSolver gmres(MPI_COMM_WORLD);
    gmres.SetKDim(100);
    gmres.SetRelTol(1e-12);
    gmres.SetMaxIter(500);
    gmres.SetPrintLevel(1);
    gmres.SetOperator(*A.As<HypreParMatrix>());
    gmres.SetPreconditioner(amg);
    gmres.Mult(B, X);
    /* GSSmoother M((SparseMatrix&)(*A)); */
    /* GMRES(*A, M, B, X, 0, 200, 50, 1e-24, 0.0); */
    #endif

    // recover the solution
    a.RecoverFEMSolution(X, b, u_fem);

    /*End of code from morteza's example*/
  
    // adapt
    char Fname[128];
    sprintf(Fname,
      "unitbox_cutQuart_1k_4p_smooth_bef.vtk");
    char iter_str[8];
    sprintf(iter_str, "_%d", Itr);
    strcat(Fname, iter_str);
    puts(Fname);

    // Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
    L2_FECollection flux_fec(order, dim);
    ParFiniteElementSpace flux_fes(mfem_mesh, &flux_fec, sdim);
    H1_FECollection smooth_flux_fec(order, dim);
    ParFiniteElementSpace smooth_flux_fes(mfem_mesh, &smooth_flux_fec, dim);
    L2ZienkiewiczZhuEstimator estimator(*integ, u_fem, flux_fes, smooth_flux_fes);
    FiniteElementCollection *errorfec = new L2_FECollection(0, dim);
    ParFiniteElementSpace errorfespace(mfem_mesh, errorfec);
    ParGridFunction l2errors(&errorfespace);
    l2errors = estimator.GetLocalErrors();
    const Vector mfem_err = estimator.GetLocalErrors();
    ParOmegaMesh* pOmesh = dynamic_cast<ParOmegaMesh*>(mfem_mesh);
    pOmesh->ElementFieldMFEMtoOmegaH (&o_mesh, mfem_err, dim, "zz_error");
    pOmesh->SmoothElementField (&o_mesh, "zz_error");
    pOmesh->ProjectFieldElementtoVertex (&o_mesh, "zz_error");

    // Save data in the ParaView format
    ParaViewDataCollection paraview_dc("Example3P_1ksmooth_bef", mfem_mesh);
    paraview_dc.SetPrefixPath("CutQuart");
    paraview_dc.SetLevelsOfDetail(1);
    paraview_dc.SetDataFormat(VTKFormat::BINARY);
    paraview_dc.SetHighOrderOutput(false);
    paraview_dc.SetCycle(0);
    paraview_dc.SetTime(0.0);
    paraview_dc.RegisterField("temperature",&u_fem);
    paraview_dc.RegisterField("zzErrors",&l2errors);
    paraview_dc.Save();

    //if ((Itr+1) < max_iter) run_case<3>(&o_mesh, Fname, Itr, myid, pOmesh);

    delete fes;
    delete fec;
    /* MPI_Finalize(); */
  } // end iterative adaptation loop

  return 0;
}
