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
      h[i] = 0.001/std::pow(std::abs(vtxError), 0.6);// 1k to .33 mil 
      //h[i] = 0.00075/std::pow(std::abs(vtxError), 0.6);// 1k to 1.6mil
      //h[i] = 0.000175/(std::abs((vtxError)));// 1k to 1.3 mil, 4p
      //h[i] = 0.0005/(std::abs((vtxError)));//1k to 51k, 4p
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

  // 2. Constant definition
  int order = 1;
  double mat_val = 1.0;
  double nbc_val = 0.0;

  // 3. Read Omega_h mesh
  auto lib = oh::Library();
  oh::Mesh o_mesh(&lib);
  oh::binary::read ("../../../mfem/data/omega_h/unitbox_cutTriCube_1k_4p.osh",
                    lib.world(), &o_mesh);
  int max_iter = 2;

  for (int Itr = 0; Itr < max_iter; Itr++)
  {

    ParMesh *pmesh = new ParOmegaMesh (MPI_COMM_WORLD, &o_mesh);
    int dim = pmesh->Dimension();

    FiniteElementCollection *fec = new H1_FECollection(order, dim);
    ParFiniteElementSpace fespace(pmesh, fec, 1);

    auto max_attr = pmesh->bdr_attributes.Size();
    Array<int> nbc_bdr(pmesh->bdr_attributes.Max());
    Array<int> ess_bdr(pmesh->bdr_attributes.Max());
    Array<int> dbc_linx(pmesh->bdr_attributes.Max());
    Array<int> dbc_liny(pmesh->bdr_attributes.Max());

    nbc_bdr = 0;
    ess_bdr = 0;
    dbc_linx = 0;
    dbc_liny = 0;

    nbc_bdr[668 -1] = 1;
    nbc_bdr[672 -1] = 1;
    nbc_bdr[676 -1] = 1;
    nbc_bdr[36 -1] = 1;
    nbc_bdr[30 -1] = 1;
    nbc_bdr[33 -1] = 1;
    nbc_bdr[27 -1] = 1;
    nbc_bdr[9 -1] = 1;
    nbc_bdr[2 -1] = 1;

    ess_bdr[239 -1] = 1;
    ess_bdr[243 -1] = 1;
    ess_bdr[476 -1] = 1;

    dbc_linx[239 -1] = 1;

    dbc_liny[243 -1] = 1;
    dbc_liny[476 -1] = 1;

    Array<int> ess_tdof_list(0);
    fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    ConstantCoefficient matCoef(mat_val);
    ConstantCoefficient nbcCoef(nbc_val);
    ProductCoefficient m_nbcCoef(matCoef, nbcCoef);

    ParGridFunction u(&fespace);
    u = 0.0;

    ParBilinearForm a(&fespace);
    BilinearFormIntegrator *integ = new DiffusionIntegrator(matCoef);
    a.AddDomainIntegrator(integ);
    a.Assemble();

    ParLinearForm b(&fespace);

    // Set the Dirichlet values in the solution vector
    FunctionCoefficient flinX_coeff(f_239);
    u.ProjectBdrCoefficient(flinX_coeff, dbc_linx);
    FunctionCoefficient flinY_coeff(f_243);
    u.ProjectBdrCoefficient(flinY_coeff, dbc_liny);

    // Add the desired value for n.Grad(u) on the Neumann boundary
    b.AddBoundaryIntegrator(new BoundaryLFIntegrator(m_nbcCoef), nbc_bdr);
    b.Assemble();

    // 11. Construct the linear system.
    OperatorPtr A;
    Vector B, X;
    a.FormLinearSystem(ess_tdof_list, u, b, A, X, B);

    // 12. Solve the linear system A X = B.
    HypreSolver *amg = new HypreBoomerAMG;
    HyprePCG pcg(MPI_COMM_WORLD);
    pcg.SetTol(1e-12);
    pcg.SetMaxIter(200);
    pcg.SetPrintLevel(2);
    pcg.SetPreconditioner(*amg);
    pcg.SetOperator(*A);
    pcg.Mult(B, X);
    delete amg;

    a.RecoverFEMSolution(X, b, u);

    // 19. Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
    int sdim = pmesh->SpaceDimension();

    L2_FECollection flux_fec(order, dim);
    ParFiniteElementSpace flux_fes(pmesh, &flux_fec, sdim);
    H1_FECollection smooth_flux_fec(order, dim);
    ParFiniteElementSpace smooth_flux_fes(pmesh, &smooth_flux_fec, dim);
    L2ZienkiewiczZhuEstimator estimator(*integ, u, flux_fes, smooth_flux_fes);
    //create gridfunction from estimator
    /* the next 4 lines were suggested by morteza */
    FiniteElementCollection *errorfec = new L2_FECollection(0, dim);
    ParFiniteElementSpace errorfespace(pmesh, errorfec);
    ParGridFunction l2errors(&errorfespace);
    l2errors = estimator.GetLocalErrors();
    /* */
    const Vector mfem_err = estimator.GetLocalErrors();
    ParOmegaMesh* pOmesh = dynamic_cast<ParOmegaMesh*>(pmesh);
    pOmesh->ElementFieldMFEMtoOmegaH (&o_mesh, mfem_err, dim, "zz_error");
    pOmesh->SmoothElementField (&o_mesh, "zz_error");
    pOmesh->ProjectFieldElementtoVertex (&o_mesh, "zz_error");

    // Save data in the ParaView format
    ParaViewDataCollection paraview_dc("Example2P_1k_bef", pmesh);
    paraview_dc.SetPrefixPath("CutTriCube");
    paraview_dc.SetLevelsOfDetail(1);
    paraview_dc.SetDataFormat(VTKFormat::BINARY);
    paraview_dc.SetHighOrderOutput(false);
    paraview_dc.SetCycle(0);
    paraview_dc.SetTime(0.0);
    paraview_dc.RegisterField("temperature",&u);
    paraview_dc.RegisterField("zzErrors",&l2errors);
    paraview_dc.Save();

    // Free the used memory.
    delete fec;

    // Perform adapt

    char Fname[128];
    sprintf(Fname,
      "/unitbox_cutTriCube_1k_4p.vtk");
    char iter_str[8];
    sprintf(iter_str, "_%d", Itr);
    strcat(Fname, iter_str);
    puts(Fname);
    //if ((Itr+1) < max_iter) run_case<3>(&o_mesh, Fname, Itr, myid, pOmesh);

  } // end adaptation loop

  return 0;
}
