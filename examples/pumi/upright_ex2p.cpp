//                       MFEM Example 2 - Parallel Version
//
//
// Description:  This example code solves a simple linear elasticity problem
//               describing on a complex domain (known as upright)
//
//               Specifically, we approximate the weak form of -div(sigma(u))=0
//               where sigma(u)=lambda*div(u)*I+mu*(grad*u+u*grad) is the stress
//               tensor corresponding to displacement field u, and lambda and mu
//               are the material Lame constants. The boundary conditions are
//               u=0 on the fixed part of the boundary with attribute 1, and
//               sigma(u).n=f on the remainder with f being a constant pull down
//               vector on boundary elements with attribute 2, and zero
//               otherwise.
//Sample PUMI RUN
//mpirun -np 2 ./pumi_upright_ex2p -p ../../data/pumi/geom/upright_no_ring_geomsim.smd   -bf ../../data/pumi/geom/upright.def -m ../../data/pumi/parallel/upright/upright_no_ring_geomsim-10k-2p/
//mpirun -np 1 ./pumi_upright_ex2p -p ../../data/pumi/geom/upright_no_ring_geomsim.smd   -bf ../../data/pumi/geom/upright.def -m ../../data/pumi/serial/upright_no_ring_geomsim-10k.smb
//

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <sstream>

#include "../../general/text.hpp" //text parsing

#ifdef MFEM_USE_SIMMETRIX
#include <MeshSim.h>
#include <SimModel.h>
#include <SimUtil.h>
#include <gmi_sim.h>
#endif
#include <apfMDS.h>
#include <gmi_null.h>
#include <PCU.h>
#include <lionPrint.h>
#include <spr.h>
#include <apfConvert.h>
#include <gmi_mesh.h>
#include <crv.h>

using namespace std;
using namespace mfem;

void writeVtk(apf::Mesh* m, int itr) {
   std::stringstream ss;
   ss << "upright_" << itr;
   std::string vtkName = ss.str();
   apf::writeVtkFiles(vtkName.c_str(), m);
}

void updateVolumeAttributes(ParMesh* pmesh)
{
   // models spans the interval [0,0.21] in the y directions
   // we want entities with
   // a) center(1) <= 0.06 to have attribute 1
   // b) center(1) > 0.06 and center(1) <= 0.12 to have attribute 2
   // c) center(1) > 0.12 to have attribute 3
   //
   // center(1) denotes the y-component
   double ppt[3];
   Vector cent(ppt, 3);
   for (int el = 0; el < pmesh->GetNE(); el++)
   {
      (pmesh->GetElementTransformation(el))->Transform(Geometries.GetCenter(
                                                         pmesh->GetElementBaseGeometry(el)),cent);
      if (cent(1) <= 0.06)
      {
         pmesh->SetAttribute(el , 1);
      }
      else if (cent(1) > 0.06 && cent(1) <= 0.12)
      {
         pmesh->SetAttribute(el , 2);
      }
      else
      {
         pmesh->SetAttribute(el , 3);
      }

   }
   pmesh->SetAttributes();
}


void updateBoundaryAttributes(
    apf::Mesh2* pumi_mesh,
    ParMesh* pmesh,
    const Array<int>& Dirichlet,
    const Array<int>& load_bdr)
{
   int dim = pumi_mesh->getDimension();
   apf::MeshIterator* itr = pumi_mesh->begin(dim-1);
   apf::MeshEntity* ent ;
   int bdr_cnt = 0;
   while ((ent = pumi_mesh->iterate(itr)))
   {
      apf::ModelEntity *me = pumi_mesh->toModel(ent);
      if (pumi_mesh->getModelType(me) == (dim-1))
      {
         //Evrywhere 3 as initial
         (pmesh->GetBdrElement(bdr_cnt))->SetAttribute(3);
         int tag = pumi_mesh->getModelTag(me);
         if (Dirichlet.Find(tag) != -1)
         {
            //Dirichlet attr -> 1
            (pmesh->GetBdrElement(bdr_cnt))->SetAttribute(1);
         }
         else if (load_bdr.Find(tag) != -1)
         {
            //load attr -> 2
            (pmesh->GetBdrElement(bdr_cnt))->SetAttribute(2);
         }
         bdr_cnt++;
      }
   }
   pumi_mesh->end(itr);
}


// This function returns the trace of the stress tensor:
// sigma_11 + sigma_22 + sigma_33
void getTraceStress(
    ParGridFunction& u, // input grid
    PWConstCoefficient& lambda,
    PWConstCoefficient& mu,
    ParGridFunction& sigma) // output grid
{
  sigma = 0.0;
  ParMesh* pmesh = u.ParFESpace()->GetParMesh();
  const FiniteElementCollection* fec = u.ParFESpace()->FEColl();
  ParFiniteElementSpace* fes = new ParFiniteElementSpace(pmesh, fec, 1); // this are scalar finite elements hence the 1

  ParGridFunction lambda_grid(fes);
  ParGridFunction mu_grid(fes);

  lambda_grid.ProjectCoefficient(lambda);
  mu_grid.ProjectCoefficient(mu);

  mu_grid *= 2.;
  lambda_grid *= 3.;

  // after the next line mu_grid will actually hold 2*mu + 3*lambda
  mu_grid += lambda_grid;


  // trance of stress for an isotropic material is simply
  // (2mu + 3lambda) * div(u)
  ParGridFunction u11(fes);
  ParGridFunction u22(fes);
  ParGridFunction u33(fes);

  u.GetDerivative(1,0,u11);
  u.GetDerivative(2,1,u22);
  u.GetDerivative(3,2,u33);


  ParGridFunction divu(fes);
  divu = 0.;

  divu += u11;
  divu += u22;
  divu += u33;

  sigma = divu;

  sigma *= mu_grid;
}

int main(int argc, char *argv[])
{
   // 1. Parse command-line options.
   //initilize mpi
   int num_procs, myId;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myId);

   // 1. Parse command-line options.
   const char *mesh_file = "../../data/pumi/parallel/upright/parallel/2p_10k/";
   const char *boundary_file = "../../data/pumi/geom/upright.def";
#ifdef MFEM_USE_SIMMETRIX
   const char *model_file = "../../data/pumi/geom/upright_no_ring_nat.x_t";
#else
   fprintf(stderr, "Rebuild with MFEM_USE_SIMMETRIX=on\n");
   return 0;
#endif

   bool static_cond = false;
   bool visualization = 1;
   int geom_order = 1;
   int order = 1;
   bool amg_elast = 0;
   double adapt_ratio = 0.15;
   int verbose = 0;
   bool shouldCoarsen = false;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&static_cond, "-sc", "--static-condensation", "-no-sc",
                  "--no-static-condensation", "Enable static condensation.");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.AddOption(&amg_elast, "-elast", "--amg-for-elasticity", "-sys",
                  "--amg-for-systems",
                  "Use the special AMG elasticity solver (GM/LN approaches), "
                  "or standard AMG for systems (unknown approach).");
   args.AddOption(&model_file, "-p", "--parasolid",
                  "Parasolid model to use.");
   args.AddOption(&geom_order, "-go", "--geometry_order",
                  "Geometric order of the model");
   args.AddOption(&boundary_file, "-bf", "--txt",
                  "txt file containing boundary tags");
   args.AddOption(&adapt_ratio, "-ar", "--adapt_ratio",
                  "adaptation factor used in MeshAdapt");
   args.AddOption(&verbose, "-v", "--verbose",
                  "increase the output from PUMI; 0:silent, >0:not silent");
   args.AddOption(&shouldCoarsen, "-c", "--enable_coarsening", "-nc", "-disable_coarsening",
                  "Enable or disable coarsening in mesh adaptation.");

   args.Parse();
   if (!args.Good())
   {
      if (myId == 0)
      {
         args.PrintUsage(cout);
      }
      MPI_Finalize();
      return 1;
   }
   if (myId == 0)
   {
      args.PrintOptions(cout);
   }

   // 2. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral or hexahedral elements with the same code.
   // 3. Read the SCOREC Mesh
   PCU_Comm_Init();
#ifdef MFEM_USE_SIMMETRIX
   MS_init();
   SimModel_start();
   Sim_readLicenseFile(0);
   gmi_sim_start();
   gmi_register_sim();
#endif
   gmi_register_mesh();

   lion_set_verbosity(verbose);
   apf::Mesh2* pumi_mesh;
   pumi_mesh = apf::loadMdsMesh(model_file, mesh_file);

   // 4. Increase the geometry order if necessary.
   if (geom_order > 1)
   {
      crv::BezierCurver bc(pumi_mesh, geom_order, 0);
      bc.run();
   }
   pumi_mesh->verify();


   //Read boundary
   string bdr_tags;
   named_ifgzstream input_bdr(boundary_file);
   input_bdr >> ws;
   getline(input_bdr, bdr_tags);
   filter_dos(bdr_tags);
   if (myId == 0) cout << " the boundary tag is : " << bdr_tags << endl;
   Array<int> Dirichlet;
   int numOfent;
   if (bdr_tags == "Dirichlet")
   {
      input_bdr >> numOfent;
      if (myId == 0) cout << " num of Dirirchlet bdr conditions : " << numOfent << endl;
      Dirichlet.SetSize(numOfent);
      for (int kk = 0; kk < numOfent; kk++)
      {
         input_bdr >> Dirichlet[kk];
      }
   }
   Dirichlet.Print();

   Array<int> load_bdr;
   skip_comment_lines(input_bdr, '#');
   input_bdr >> bdr_tags;
   filter_dos(bdr_tags);
   if (myId == 0) cout << " the boundary tag is : " << bdr_tags << endl;
   if (bdr_tags == "Load")
   {
      input_bdr >> numOfent;
      load_bdr.SetSize(numOfent);
      if (myId == 0) cout << " num of load bdr conditions : " << numOfent << endl;
      for (int kk = 0; kk < numOfent; kk++)
      {
         input_bdr >> load_bdr[kk];
      }
   }
   load_bdr.Print();

   // 3. Read the mesh from the given mesh file on all processors.
   ParMesh *pmesh = new ParPumiMesh(MPI_COMM_WORLD, pumi_mesh);
   int dim = pumi_mesh->getDimension();

   updateBoundaryAttributes(pumi_mesh, pmesh, Dirichlet, load_bdr);
   updateVolumeAttributes(pmesh);

   cout << " elem attr max " << pmesh->attributes.Max() << " bdr attr max " <<
        pmesh->bdr_attributes.Max() <<endl;
   if (pmesh->attributes.Max() < 2 || pmesh->bdr_attributes.Max() < 2)
   {
      cerr << "\nInput mesh should have at least two materials and "
           << "two boundary attributes! (See schematic in ex2.cpp)\n"
           << endl;
      return 3;
   }


   // 7. Define a parallel finite element space on the parallel mesh. Here we
   //    use vector finite elements, i.e. dim copies of a scalar finite element
   //    space. We use the ordering by vector dimension (the last argument of
   //    the FiniteElementSpace constructor) which is expected in the systems
   //    version of BoomerAMG preconditioner. For NURBS meshes, we use the
   //    (degree elevated) NURBS space associated with the mesh nodes.
   FiniteElementCollection *fec;
   ParFiniteElementSpace *fespace;  // vector version
   ParFiniteElementSpace *fespaces; // scalar version
   /* const bool use_nodal_fespace = pmesh->NURBSext && !amg_elast; */
   /* if (use_nodal_fespace) */
   /* { */
   /*    fec = NULL; */
   /*    fespace = (ParFiniteElementSpace *)pmesh->GetNodes()->FESpace(); */
   /* } */
   /* else */
   /* { */
   fec = new H1_FECollection(order, dim);
   fespace  = new ParFiniteElementSpace(pmesh, fec, dim);
   fespaces = new ParFiniteElementSpace(pmesh, fec, 1);

   /* } */
   HYPRE_Int size = fespace->GlobalTrueVSize();
   if (myId == 0)
   {
      cout << "Number of finite element unknowns: " << size << endl
           << "Assembling: " << flush;
   }

   /* ParFiniteElementSpace* fespace_scalar = new ParFiniteElementSpace(pmesh, fec, 1); */

   // 8. Determine the list of true (i.e. parallel conforming) essential
   //    boundary dofs. In this example, the boundary conditions are defined by
   //    marking only boundary attribute 1 from the mesh as essential and
   //    converting it to a list of true dofs.
   //Array<int> ess_tdof_list, ess_bdr(pmesh->bdr_attributes.Max());
   //ess_bdr = 0;
   //ess_bdr[0] = 1;
   //fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

   // 9. Set up the parallel linear form b(.) which corresponds to the
   //    right-hand side of the FEM linear system. In this case, b_i equals the
   //    boundary integral of f*phi_i where f represents a "pull down" force on
   //    the Neumann part of the boundary and phi_i are the basis functions in
   //    the finite element fespace. The force is defined by the object f, which
   //    is a vector of Coefficient objects. The fact that f is non-zero on
   //    boundary attribute 2 is indicated by the use of piece-wise constants
   //    coefficient for its last component.
   VectorArrayCoefficient f(dim);
   f.Set(0, new ConstantCoefficient(0.0));
   f.Set(1, new ConstantCoefficient(0.0));
   f.Set(2, new ConstantCoefficient(0.0));


   //ParLinearForm *b = new ParLinearForm(fespace);
   //b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));
   if (myId == 0)
   {
      cout << "r.h.s. ... " << flush;
   }
   //b->Assemble();

   // 10. Define the solution vector x as a parallel finite element grid
   //     function corresponding to fespace. Initialize x with initial guess of
   //     zero, which satisfies the boundary conditions.
   ParGridFunction x(fespace);
   ParGridFunction sigma(fespaces);
   x = 0.0;
   sigma = 0.0;

   // 11. Set up the parallel bilinear form a(.,.) on the finite element space
   //     corresponding to the linear elasticity integrator with piece-wise
   //     constants coefficient lambda and mu.
   Vector lambda(pmesh->attributes.Max());
   lambda = 1.0;
   /* lambda(1) = lambda(0)*10.; */
   PWConstCoefficient lambda_func(lambda);
   Vector mu(pmesh->attributes.Max());
   mu = 1.0;
   /* mu(1) = mu(0)*10.; */
   PWConstCoefficient mu_func(mu);

   ParBilinearForm *a = new ParBilinearForm(fespace);
   a->AddDomainIntegrator(new ElasticityIntegrator(lambda_func, mu_func));


    f.Set(0, new ConstantCoefficient(0.0));
    f.Set(1, new ConstantCoefficient(0.0));
    f.Set(2, new ConstantCoefficient(0.0));
    {
	  Vector pull_force(pmesh->bdr_attributes.Max());
	  pull_force = 0.0;
	  /* pull_force(1) =  1.e-1; */
	  pull_force(2) =  1.e-1;
	  f.Set(1, new PWConstCoefficient(pull_force));
    }
    ParLinearForm *b = new ParLinearForm(fespace);
    b->AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(f));


    // 12. Assemble the parallel bilinear form and the corresponding linear
    //     system, applying any necessary transformations such as: parallel
    //     assembly, eliminating boundary conditions, applying conforming
    //     constraints for non-conforming AMR, static condensation, etc.
    if (myId == 0) { cout << "matrix ... " << flush; }
    if (static_cond) { a->EnableStaticCondensation(); }

   apf::Field* disp_field = 0;
   apf::Field* disp_field_mag = 0;
   apf::Field* trace_stress = 0;
   apf::Field* trace_stress_mag = 0;
   apf::Field* ipfield = 0;
   apf::Field* sizefield = 0;

   int max_iter = 3;

   for (int Itr = 0; Itr < max_iter; Itr++)
   {

      a->Assemble();
      b->Assemble();

      Array<int> ess_tdof_list, ess_bdr(pmesh->bdr_attributes.Max());
      ess_bdr = 0;
      ess_bdr[0] = 1;
      fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);


      HypreParMatrix A;
      Vector B, X;
      a->FormLinearSystem(ess_tdof_list, x, *b, A, X, B);
      if (myId == 0)
        {
           cout << "done." << endl;
           cout << "Size of linear system: " << A.GetGlobalNumRows() << endl;
        }

      // 13. Define and apply a parallel PCG solver for A X = B with the BoomerAMG
      //     preconditioner from hypre.
      HypreBoomerAMG *amg = new HypreBoomerAMG(A);
      if (amg_elast && !a->StaticCondensationIsEnabled())
      {
           amg->SetElasticityOptions(fespace);
      }
      else
      {
           amg->SetSystemsOptions(dim);
      }
      HyprePCG *pcg = new HyprePCG(A);
      pcg->SetTol(1e-7);
      pcg->SetMaxIter(1000);
      pcg->SetPrintLevel(1);
      pcg->SetPreconditioner(*amg);
      pcg->Mult(B, X);

      // 14. Recover the parallel grid function corresponding to X. This is the
      //     local finite element solution on each processor.
      a->RecoverFEMSolution(X, *b, x);


      // 17. Send the above data by socket to a GLVis server.  Use the "n" and "b"
      //     keys in GLVis to visualize the displacements.
      if (visualization)
       {
           char vishost[] = "localhost";
           int  visport   = 19916;
           socketstream sol_sock(vishost, visport);
           sol_sock << "parallel " << num_procs << " " << myId << "\n";
           sol_sock.precision(8);
           sol_sock << "solution\n" << *pmesh << x << flush;
       }

       // 18. Field transfer. Scalar solution field and magnitude field for
       //     error estimation are created the pumi mesh.
       if (order > geom_order)
        {
              disp_field_mag = apf::createField(pumi_mesh, "|u|",
                                            apf::SCALAR, apf::getLagrange(order));
              disp_field = apf::createField(pumi_mesh, "u",
                                            apf::VECTOR, apf::getLagrange(order));
              trace_stress = apf::createField(pumi_mesh, "sigma",
                                            apf::SCALAR, apf::getLagrange(order));
              trace_stress_mag = apf::createField(pumi_mesh, "|sigma|",
                                            apf::SCALAR, apf::getLagrange(order));
        }
        else
        {
             disp_field_mag = apf::createFieldOn(pumi_mesh, "|u|",apf::SCALAR);
             disp_field = apf::createFieldOn(pumi_mesh, "u", apf::VECTOR);
             trace_stress = apf::createFieldOn(pumi_mesh, "sigma", apf::SCALAR);
             trace_stress_mag = apf::createFieldOn(pumi_mesh, "|sigma|", apf::SCALAR);
        }

        ParPumiMesh* pPPmesh = dynamic_cast<ParPumiMesh*>(pmesh);
        pPPmesh->VectorFieldMFEMtoPUMI(pumi_mesh, &x, disp_field, disp_field_mag);

	getTraceStress(x, lambda_func, mu_func, sigma);
        pPPmesh->FieldMFEMtoPUMI(pumi_mesh, &sigma, trace_stress, trace_stress_mag);


        ipfield= spr::getGradIPField(trace_stress, "gradip", 2);
        sizefield = spr::getSPRSizeField(ipfield, adapt_ratio);

        pumi_mesh->removeField(ipfield);
        apf::destroyField(ipfield);

        // 19. Perform MesAdapt
        auto erInputAdv = ma::makeAdvanced(ma::configure(pumi_mesh, sizefield));
        erInputAdv->shouldFixShape = true;
        erInputAdv->shouldCoarsen = shouldCoarsen;
        erInputAdv->maximumIterations = 3;
        /* erinput->shouldRunMidParma = true; */
        if ( geom_order > 1)
        {
            crv::adapt(erInputAdv);
        }
         else
        {
            ma::adapt(erInputAdv);
        }
        pumi_mesh->verify();

        //write vtk file
        writeVtk(pumi_mesh,Itr);

        ParMesh* Adapmesh = new ParPumiMesh(MPI_COMM_WORLD, pumi_mesh);
        pPPmesh->UpdateMesh(Adapmesh);
        delete Adapmesh;

	updateBoundaryAttributes(pumi_mesh, pmesh, Dirichlet, load_bdr);
	updateVolumeAttributes(pmesh);

        fespace->Update();
        fespaces->Update();
        x.Update();
        sigma.Update();
        x = 0.0;
        sigma = 0.0;

        a->Update();
        b->Update();

        //Destroy fields
        pumi_mesh->removeField(disp_field);
        pumi_mesh->removeField(disp_field_mag);
        pumi_mesh->removeField(trace_stress);
        pumi_mesh->removeField(trace_stress_mag);
        pumi_mesh->removeField(sizefield);

        apf::destroyField(disp_field);
        apf::destroyField(disp_field_mag);
        apf::destroyField(trace_stress);
        apf::destroyField(trace_stress_mag);
        apf::destroyField(sizefield);

        delete pcg;
        delete amg;
   }

   writeVtk(pumi_mesh,max_iter);

   // 18. Free the used memory.
   delete a;
   if (fec)
   {
      delete fespace;
      delete fespaces;
      delete fec;
   }
   delete pmesh;

   pumi_mesh->destroyNative();
   apf::destroyMesh(pumi_mesh);
   PCU_Comm_Free();

#ifdef MFEM_USE_SIMMETRIX
   gmi_sim_stop();
   Sim_unregisterAllKeys();
#endif

   MPI_Finalize();

   return 0;
}
