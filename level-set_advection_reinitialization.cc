// Level-set advection and reinitialization.

// This file is based on the advection_solver_variable.cc file 
// of the advection_miniapp repository,
// see https://github.com/kronbichler/advection_miniapp.

// Author: Andreas Ritthaler, Technical University of Munich, 2023

#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/solution_transfer.h>
#include <deal.II/distributed/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q_generic.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h> 
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/point_value_history.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>

#include <deal.II/base/parameter_handler.h>

#include <deal.II/fe/fe_system.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#include "smoothness_indicator.h"
#include "preconditioner.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>



// create surface triangulation of the bubble for curvature calculation
namespace dealii::GridTools
{
  template <int dim, typename VectorType>
  void
  create_triangulation_with_marching_cube_algorithm(Triangulation<dim - 1, dim> &tria,
                                                    const Mapping<dim> &         mapping,
                                                    const DoFHandler<dim> &background_dof_handler,
                                                    const VectorType &     ls_vector,
                                                    const double           iso_level,
                                                    const unsigned int     n_subdivisions = 1,
                                                    const double           tolerance      = 1e-10)
  {
    std::vector<Point<dim>>        vertices;
    std::vector<CellData<dim - 1>> cells;
    SubCellData                    subcelldata;

    const GridTools::MarchingCubeAlgorithm<dim, VectorType> mc(mapping,
                                                               background_dof_handler.get_fe(),
                                                               n_subdivisions,
                                                               tolerance);

    const bool vector_is_ghosted = ls_vector.has_ghost_elements();

    if (vector_is_ghosted == false)
      ls_vector.update_ghost_values();

    mc.process(background_dof_handler, ls_vector, iso_level, vertices, cells);

    if (vector_is_ghosted == false)
      ls_vector.zero_out_ghost_values();

    std::vector<unsigned int> considered_vertices;

    // note: the following operation does not work for simplex meshes yet
    //GridTools::delete_duplicated_vertices(vertices, cells, subcelldata,
    //considered_vertices);

    if (vertices.size() > 0)
      tria.create_triangulation(vertices, cells, subcelldata);
  }
} // namespace dealii::GridTools



namespace LevelSet
{
  using namespace dealii;

  class LevelSetProblemParameters
  {
  public:

    static void declare_parameters (ParameterHandler &prm);

    void get_parameters (ParameterHandler &prm);

    ////////////////////////////////////////
    ////////// General parameters //////////
    ////////////////////////////////////////

    // Space dimension
    unsigned int dimension = 2;
    // 1D-polynomial degree of the elements
    unsigned int fe_degree = 4;
    // The initial mesh (consisting of a single square) is refined by
    // doubling the number of elements for every increase in number. Thus, the
    // number of elements is given by 2^(dim * n_global_refinements).
    // If an adaptive mesh refinement is used, n_global_refinements corresponds
    // to the coarsest element size.
    unsigned int n_global_refinements = 6;
    // Number of maximum additional mesh refinement levels
    unsigned int n_refinement_levels = 2;
    // Final simulation time
    double FINAL_TIME = 4.0;
    // Frequency of output in s
    double output_tick = 0.2;
    // Enable or disable writing of result files for visualization with ParaView or VisIt
    bool print_vtu = true;
    // Factor for the calculation of the grid refinement and coarsening interval
    // dt_refinement = factor_refinement_interval * dt_advection
    unsigned int factor_refinement_interval = 8;
    // Use adaptive mesh refinement?
    bool use_adaptive_mesh_refinement = true;
    // Should the element size in a narrow band around the zero level be constant?
    // (Might be relevent for applying forces. The computational effort will increase.)
    bool use_const_element_size_at_interface = false;
    // Fraction of the reinitialization distance epsilon, which should have constant element
    // size in the vicinity of the zero level.
    // (only relevant if use_const_element_size_at_interface = true)
    double factor_distance_const_element_size_at_interface = 0.15;
    // Round the edge between the reinitialized region and the outer flat region?
    bool do_edge_rounding = true;
    // Test case
    // 0: 2D/3D bubble vortex
    // 1: 2D slotted disk
    // 2: 2D Poiseuille flow (adjust grid to 1x3)
    // 3: 2D static reinitialization of a circular interface with highly distorted gradient (adjust grid to 2x2)
    // 4: 2D static reinitialization of a square interface
    // 5: 2D static reinitialization of multiple interfaces (adjust grid to 2x2)
    // 6: 2D/3D static reinitialization of two intersecting circles/spheres (adjust grid to 2x2(x2))
    unsigned int test_case = 0;
    // Do a static (pure reinitializatin) or dynamic simulation?
    bool dynamic_sim = true;

    //////////////////////////////////////////
    ////////// Advection parameters //////////
    //////////////////////////////////////////

    // Time step calculation for advection, it is updated adaptively
    // dt_advection = min(courant_number_advection * h / (transport_speed_norm * fe_degree^2)
    double courant_number_advection = 0.5;

    /////////////////////////////////////////////////
    ////////// Reinitialization parameters //////////
    /////////////////////////////////////////////////

    // Perform reinitialization every RI_inverval advection time steps
    // (only relevant if use_gradient_based_RI_indicator = false)
    unsigned int RI_interval = 1;
    // Factor for the calculation of the reinitialization distance
    // RI_distance = factor_RI_distance * h_max/fe_degree
    // (Interpretation as number of linear elements based on the coursest cell size)
    double factor_RI_distance = 6.0;
    // Safety factor for reinitialization time step calculation
    // Used for implicit and explicit time discretization for the reinitialization
    double courant_number_RI = 0.5;
    // Interior penalty parameter for symmetric interior penalty method for viscous term.
    // The smoothness of the zero level-set contour and the simulation result quality 
    // can be considerably improved, if IP>1 is used. However, IP=1 is sufficient for a stable simulation.
    // If the smoothness of the gradient field is of high importance, use IP>>1, e. g. 100 and IMEX
    // time discretization.
    double IP_diffusion = 100.0;
    // Calculation of the cell-wise required artificial viscosity value: nu = factor_diffusivity * h / N;
    double factor_diffusivity = 0.25;
    // Parameters for the calculation of the local required artificial viscosity value
    double art_diff_k = 2.0;      // ~ threshold
    double art_diff_kappa = 2.8;  // ~ bandwidth
    // Parameters for the grid coarsening and refinement
    double factor_refinement_flag = 0.65;
    double factor_coarsen_flag = 0.2;
    // Number of RI pseudo time steps per reinitialization procedure
    unsigned int RI_steps = 1;
    // RI quality criteria, has to be adjusted according to the simulation case
    // (only relevant if use_gradient_based_RI_indicator = true)
    // general hint: low RI quality requirements: 0.25
    //               high RI quality requirements: 0.15
    double RI_quality_criteria = 0.20;
    // Use gradient based reinitialization indicator?
    bool use_gradient_based_RI_indicator = true;
    // Use IMEX time discretization?
    // (Especally recommended for IP>3 to save computational effort.)
    bool use_IMEX = true;
    // Use constant gradient for each reinitialization process to save computational effort?
    // (Useful expecially in explicit reinitialication time discretization.)
    bool use_const_gradient_in_RI = false;
  };



  void LevelSetProblemParameters::declare_parameters (ParameterHandler &prm)
  {
    prm.enter_subsection ("GENERAL");
    {
      prm.declare_entry ("DIMENSION", "2", Patterns::Integer());
      prm.declare_entry ("FE DEGREE", "4", Patterns::Integer());
      prm.declare_entry ("N GLOBAL REFINEMENTS", "6", Patterns::Integer());
      prm.declare_entry ("N REFINEMENT LEVELS", "2", Patterns::Integer());
      prm.declare_entry ("FINAL TIME", "4.0", Patterns::Double());
      prm.declare_entry ("OUTPUT TICK", "0.2", Patterns::Double());
      prm.declare_entry ("PRINT VTU", "true", Patterns::Bool());
      prm.declare_entry ("FACTOR REFINEMENT INTERVAL", "8", Patterns::Integer());
      prm.declare_entry ("USE ADAPTIVE MESH REFINEMENT", "true", Patterns::Bool());
      prm.declare_entry ("USE CONST ELEMENT SIZE AT INTERFACE", "false", Patterns::Bool());
      prm.declare_entry ("FACTOR DISTANCE CONST ELEMENT SIZE AT INTERFACE", "0.15", Patterns::Double());
      prm.declare_entry ("DO EDGE ROUNDING", "true", Patterns::Bool());
      prm.declare_entry ("TEST CASE", "0", Patterns::Integer());
      prm.declare_entry ("DYNAMIC SIM", "true", Patterns::Bool());
    }
    prm.leave_subsection ();

    prm.enter_subsection ("ADVECTION");
    {
      prm.declare_entry ("COURANT NUMBER ADVECTION", "0.5", Patterns::Double());
    }
    prm.leave_subsection ();

    prm.enter_subsection ("REINITIALIZATION");
    {
      prm.declare_entry ("RI INTERVAL", "1", Patterns::Integer());
      prm.declare_entry ("FACTOR RI DISTANCE", "6.0", Patterns::Double());
      prm.declare_entry ("COURANT NUMBER RI", "0.5", Patterns::Double());
      prm.declare_entry ("INTERIOR PENALTY DIFFUSION", "100.0", Patterns::Double());
      prm.declare_entry ("FACTOR ARTIFICIAL VISCOSITY", "0.35", Patterns::Double());
      prm.declare_entry ("ART DIFF K", "1.8", Patterns::Double());
      prm.declare_entry ("ART DIFF KAPPA", "3.0", Patterns::Double());
      prm.declare_entry ("FACTOR REFINEMENT FLAG", "0.65", Patterns::Double());
      prm.declare_entry ("FACTOR COARSEN FLAG", "0.2", Patterns::Double());
      prm.declare_entry ("RI STEPS", "1", Patterns::Integer());
      prm.declare_entry ("RI QUALITY CRITERIA", "0.20", Patterns::Double());
      prm.declare_entry ("USE GRADIENT BASED RI INDICATOR", "true", Patterns::Bool());
      prm.declare_entry ("USE IMEX", "false", Patterns::Bool());
      prm.declare_entry ("USE CONST GRADIENT IN RI", "true", Patterns::Bool());
    }
    prm.leave_subsection ();
  }



  void LevelSetProblemParameters::get_parameters (ParameterHandler &prm)
  {
    prm.enter_subsection ("GENERAL");
    {
      dimension = prm.get_integer ("DIMENSION");
      fe_degree = prm.get_integer ("FE DEGREE");
      n_global_refinements = prm.get_integer ("N GLOBAL REFINEMENTS");
      n_refinement_levels = prm.get_integer ("N REFINEMENT LEVELS");
      FINAL_TIME = prm.get_double ("FINAL TIME");
      output_tick = prm.get_double ("OUTPUT TICK");
      print_vtu = prm.get_bool ("PRINT VTU");
      factor_refinement_interval = prm.get_integer ("FACTOR REFINEMENT INTERVAL");
      use_adaptive_mesh_refinement = prm.get_bool ("USE ADAPTIVE MESH REFINEMENT");
      use_const_element_size_at_interface = prm.get_bool("USE CONST ELEMENT SIZE AT INTERFACE");
      factor_distance_const_element_size_at_interface = prm.get_double("FACTOR DISTANCE CONST ELEMENT SIZE AT INTERFACE");
      do_edge_rounding = prm.get_bool("DO EDGE ROUNDING");
      test_case = prm.get_integer ("TEST CASE");
      dynamic_sim = prm.get_bool ("DYNAMIC SIM");
    }
    prm.leave_subsection ();

    prm.enter_subsection ("ADVECTION");
    {
      courant_number_advection = prm.get_double ("COURANT NUMBER ADVECTION");
    }
    prm.leave_subsection ();

    prm.enter_subsection ("REINITIALIZATION");
    {
      RI_interval = prm.get_integer ("RI INTERVAL");
      factor_RI_distance = prm.get_double ("FACTOR RI DISTANCE");
      courant_number_RI = prm.get_double ("COURANT NUMBER RI");
      IP_diffusion = prm.get_double ("INTERIOR PENALTY DIFFUSION");
      factor_diffusivity = prm.get_double("FACTOR ARTIFICIAL VISCOSITY");
      art_diff_k = prm.get_double("ART DIFF K");
      art_diff_kappa = prm.get_double("ART DIFF KAPPA");
      factor_refinement_flag = prm.get_double("FACTOR REFINEMENT FLAG");
      factor_coarsen_flag = prm.get_double("FACTOR COARSEN FLAG");
      RI_steps = prm.get_integer ("RI STEPS");
      RI_quality_criteria = prm.get_double ("RI QUALITY CRITERIA");
      use_gradient_based_RI_indicator = prm.get_bool ("USE GRADIENT BASED RI INDICATOR");
      use_IMEX = prm.get_bool ("USE IMEX");
      use_const_gradient_in_RI = prm.get_bool ("USE CONST GRADIENT IN RI");
    }
    prm.leave_subsection ();
  }



  // Runge-Kutta time integrator schemes
  enum LowStorageRungeKuttaScheme
  {
    stage_3_order_3, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_5_order_4, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_7_order_4, /* Tselios, Simos, 2007 */
    stage_9_order_5, /* Kennedy, Carpenter, Lewis, 2000 */
  };
  // Currently, for both the advection and the reinitialization, 
  // the same time integrator scheme is used.
  constexpr LowStorageRungeKuttaScheme lsrk_scheme = stage_5_order_4;



  // Analytical solution of the considered level-set function at initial conditions.
  // The DoF values are determined via projection.
  template <int dim>
  class LevelSetFunction : public Function<dim>
  {
  public:
    LevelSetFunction(const double time, const LevelSetProblemParameters &param): Function<dim>(1, time), param(param)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      return value<double>(p);
    }

    template <typename Number>
    Number
    value(const Point<dim, Number> &p) const
    {
      if (dim == 2)
      {
        switch (param.test_case)
              {
                case 0:
                  {
                    // bubble vortex test case, (domain: [0,1]^2)
                    return std::sqrt((p[0] - 0.5) * (p[0] - 0.5) + (p[1] - 0.75) * (p[1] - 0.75))-0.15;

                    break;
                  }
                
                case 1:
                  {
                    // slotted disk test case, (domain: [0,1]^2)
                    return std::max(std::min(std::min(p[0]-0.475, -(p[0]-0.525)),-(p[1]-0.8)),
                    std::sqrt((p[0] - 0.5) * (p[0] - 0.5) + (p[1] - 0.75) * (p[1] - 0.75))-0.15);

                    break;
                  }
                
                case 2:
                  {
                    // Poiseuille flow test case (channel flow), (domain: [0,3]x[0,1])
                    return std::sqrt((p[0] - 0.5) * (p[0] - 0.5) + (p[1] - 0.5) * (p[1] - 0.5))-0.35;

                    break;
                  }

                case 3:
                  {
                    // static reinitialization of a circular interface with highly distorted gradient, (domain: [2,2]^2)
                    return ((p[0] - 1.) * (p[0] - 1.) + (p[1] - 1.) * (p[1] - 1.) + 0.1) * (std::sqrt(p[0] * p[0] + p[1] * p[1])-1.);

                    break;
                  }

                case 4:
                  {
                    // static reinitialization of a square interface (domain: [0,1]^2)
                    return 0.8 * (std::max(std::abs(p[0]-0.5), std::abs(p[1]-0.5))-0.25);

                    break;
                  }

                case 5:
                  {
                    // static reinitialization of multiple interfaces (domain: [-2,2]^2)
                    const auto d_1 = std::sqrt((p[0]+0.8)*(p[0]+0.8)+(p[1]-0.8)*(p[1]-0.8))-0.15;
                    const auto d_2 = std::min(std::sqrt((p[0]-0.8)*(p[0]-0.8)+(p[1]-0.8)*(p[1]-0.8))-0.15, d_1);
                    const auto d_3 = std::min(std::sqrt((p[0]-0.8)*(p[0]-0.8)+(p[1]+0.8)*(p[1]+0.8))-0.15, d_2);
                    const auto d_4 = std::min(std::sqrt((p[0]+0.8)*(p[0]+0.8)+(p[1]+0.8)*(p[1]+0.8))-0.15, d_3);

                    const auto d_5 = std::min(std::sqrt((p[0]+0.)*(p[0]+0.)+(p[1]-0.8)*(p[1]-0.8))-0.15, d_4);
                    const auto d_6 = std::min(std::sqrt((p[0]+0.)*(p[0]+0.)+(p[1]+0.8)*(p[1]+0.8))-0.15, d_5);
                    const auto d_7 = std::min(std::sqrt((p[0]+0.8)*(p[0]+0.8)+(p[1]+0.)*(p[1]+0.))-0.15, d_6);
                    const auto d_8 = std::min(std::sqrt((p[0]-0.8)*(p[0]-0.8)+(p[1]+0.)*(p[1]+0.))-0.15, d_7);

                    const auto d_9 = std::min(std::sqrt((p[0]+0.4)*(p[0]+0.4)+(p[1]-0.4)*(p[1]-0.4))-0.15, d_8);
                    const auto d_10 = std::min(std::sqrt((p[0]-0.4)*(p[0]-0.4)+(p[1]-0.4)*(p[1]-0.4))-0.15, d_9);
                    const auto d_11 = std::min(std::sqrt((p[0]-0.4)*(p[0]-0.4)+(p[1]+0.4)*(p[1]+0.4))-0.15, d_10);
                    const auto d_12 = std::min(std::sqrt((p[0]+0.4)*(p[0]+0.4)+(p[1]+0.4)*(p[1]+0.4))-0.15, d_11);

                    return d_12 * ((p[0]-1.)*(p[0]-1.)+(p[1]-1.)*(p[1]-1.)+0.1);

                    break;
                  }

                case 6:
                  {
                    // static reinitialization of two intersecting circles (domain: [2,2]^2)
                    return std::min(0.8*(std::sqrt((p[0] - 0.5) * (p[0] - 0.5) + (p[1] - 0.65) * (p[1] - 0.65))-0.2),
                                    0.8*(std::sqrt((p[0] - 0.5) * (p[0] - 0.5) + (p[1] - 0.35) * (p[1] - 0.35))-0.2));

                    break;
                  }

                default:
                  AssertThrow(false, ExcNotImplemented());
              }
      }
      else if (dim ==3)
      {
        switch (param.test_case)
              {
                case 0:
                  {
                    // bubble vortex test case (domain: [0,1]^3)
                    return std::sqrt((p[0] - 0.35) * (p[0] - 0.35) + (p[1] - 0.35) * (p[1] - 0.35) + (p[2] - 0.35) * (p[2] - 0.35))-0.15;

                    break;
                  }

                case 6:
                  {
                    // static reinitialization of two intersecting spheres (domain: [2,2]^2)
                    return std::min((std::sqrt((p[0]-0.2) * (p[0]-0.2) + (p[1]-0.2) * (p[1]-0.2) + (p[2]-0.) * (p[2]-0.))-0.4),
                                 (std::sqrt((p[0]+0.2) * (p[0]+0.2) + (p[1]+0.2) * (p[1]+0.2) + (p[2]+0.) * (p[2]+0.))-0.65)) *
                                 ((p[0] - 0.3) * (p[0] - 0.3) + (p[1] - 0.3) * (p[1] - 0.3) + (p[2] - 0.3) * (p[2] - 0.3) + 0.5);

                    break;
                  }
                
                default:
                  AssertThrow(false, ExcNotImplemented());
              }
      }
    }

  private:
    const LevelSetProblemParameters &param;
  };



  // Compute radius
  // (for calculating norm in vicinity of zero-level of a circular interface)
  template <int dim>
  class ComputeRadius : public Function<dim>
  {
  public:
    ComputeRadius(const double time,
                  const double x_center, 
                  const double y_center, 
                  const double z_center): 
    Function<dim>(1, time), 
    x_center(x_center),
    y_center(y_center),
    z_center(z_center)
    {}

    virtual double
    value(const Point<dim> &p, const unsigned int /*component*/ = 0) const override
    {
      return value<double>(p);
    }

    template <typename Number>
    Number
    value(const Point<dim, Number> &p) const
    {
      if (dim == 2)
      {
        return std::sqrt((p[0]-x_center)*(p[0]-x_center)+(p[1]-y_center)*(p[1]-y_center));
      }
      else if (dim == 3)
      {
        return std::sqrt((p[0]-x_center)*(p[0]-x_center)+(p[1]-y_center)*(p[1]-y_center)+(p[2]-z_center)*(p[2]-z_center));
      }
    }

  private:
    const double x_center;
    const double y_center;
    const double z_center;
  };



  // Analytical function for the transport speed
  template <int dim>
    class TransportSpeed : public Function<dim>
    {
    public:
      TransportSpeed(const double time, const LevelSetProblemParameters &param)
        : Function<dim>(dim, time), param(param)
      {}
  
      virtual double value(const Point<dim> & p,
                           const unsigned int component = 0) const override;
    
    private:
      const LevelSetProblemParameters &param;
    };

    template <int dim>
    double TransportSpeed<dim>::value(const Point<dim> & p,
                                     const unsigned int component) const
    {
      const double t = this->get_time();
  
            if (dim == 2)
            {
              switch (param.test_case)
              {
                case 0:
                  {
                    // bubble vortex test case
                    const double factor = std::cos(numbers::PI * t / param.FINAL_TIME);

                    if (component == 0)
                    {
                      return factor * (std::sin(2 * numbers::PI * p[1]) *
                                          std::sin(numbers::PI * p[0]) *
                                          std::sin(numbers::PI * p[0]));
                    }
                    else if (component == 1)
                    {
                      return -factor * (std::sin(2 * numbers::PI * p[0]) *
                                          std::sin(numbers::PI * p[1]) *
                                          std::sin(numbers::PI * p[1]));
                    }

                    break;
                  }

                case 1:
                  {
                    // slotted disk test case
                    if (component == 0)
                    {
                      return numbers::PI * (0.5 - p[1])/numbers::PI;
                    }
                    else if (component == 1)
                    {
                      return numbers::PI * (p[0] - 0.5)/numbers::PI;
                    }

                    break;
                  }

                case 2:
                  {
                    // Poiseuille flow test case (channel flow) with modified periodic velocity field
                    if (component == 0)
                    {
                      const double factor = std::sin(numbers::PI * t / param.FINAL_TIME);
                      return 1.*(1.-(1.-factor)*(p[1]-0.5)*(p[1]-0.5)/(0.5*0.5));
                    }
                    else if (component == 1)
                    {
                      return 0.;
                    }

                    break;
                  }

                default:
                  AssertThrow(false, ExcNotImplemented());
              }
            }
            else if (dim == 3)
            {
              switch (param.test_case)
              {
                case 0:
                  {
                    // bubble vortex test case
                    const double factor = std::cos(numbers::PI * t / param.FINAL_TIME);

                    if (component == 0)
                    {
                      return 2. * factor * (std::sin(2 * numbers::PI * p[1]) *
                              std::sin(2 * numbers::PI * p[2]) *
                              std::sin(numbers::PI * p[0]) *
                              std::sin(numbers::PI * p[0]));
                    }
                    else if (component == 1)
                    {
                      return - factor * (std::sin(2 * numbers::PI * p[0]) *
                              std::sin(2 * numbers::PI * p[2]) *
                              std::sin(numbers::PI * p[1]) *
                              std::sin(numbers::PI * p[1]));
                    }
                    else if (component == 2)
                    {
                      return - factor * (std::sin(2 * numbers::PI * p[0]) *
                              std::sin(2 * numbers::PI * p[1]) *
                              std::sin(numbers::PI * p[2]) *
                              std::sin(numbers::PI * p[2]));
                    }

                    break;
                  }

                default:
                  AssertThrow(false, ExcNotImplemented());
              }
            }

             return 0;
    }



  template <int dim, int fe_degree>
  class LevelSetOperation
  {
  public:
    using Number = double;

    LevelSetOperation(const LevelSetProblemParameters &param): param(param)
    {}

    void
    reinit(const DoFHandler<dim> &dof_handler,
           const DoFHandler<dim> &dof_handler_legendre,
           const DoFHandler<dim> &dof_handler_vel);

    void
    initialize_dof_vector(LinearAlgebra::distributed::Vector<Number> &vec,
                          const unsigned int                          dof_handler_index)
    {
      data.initialize_dof_vector(vec, dof_handler_index);
    }

    template <bool is_right, uint component>
    void
    apply_RI_grad(const LinearAlgebra::distributed::Vector<Number> &src,
                  LinearAlgebra::distributed::Vector<Number>       &dst);

    void
    perform_stage(const Number                                     current_time,
                  const Number                                     factor_solution,
                  const Number                                     factor_ai,
                  const LinearAlgebra::distributed::Vector<Number> &current_ri,
                  LinearAlgebra::distributed::Vector<Number>       &vec_ki,
                  LinearAlgebra::distributed::Vector<Number>       &solution,
                  LinearAlgebra::distributed::Vector<Number>       &next_ri,
                  TimerOutput                                      &computing_timer) const;

    void
    perform_stage_RI(const Number                                  current_time,
                     const Number                                  factor_solution,
                     const Number                                  factor_ai,
                     LinearAlgebra::distributed::Vector<Number>    &vec_ki,
                     LinearAlgebra::distributed::Vector<Number>    &solution,
                     LinearAlgebra::distributed::Vector<Number>    &next_ri,
                     LinearAlgebra::distributed::Vector<Number>    &num_Hamiltonian,
                     TimerOutput                                   &computing_timer) const;

    void
    project_initial(LinearAlgebra::distributed::Vector<Number> &dst) const;

    double 
    compute_L2_norm_in_interface_region(const LinearAlgebra::distributed::Vector<Number> &solution) const;

    Tensor<1, 2>
    compute_mass_and_energy(const LinearAlgebra::distributed::Vector<Number> &vec) const;

    void
    Godunov_Hamiltonian(const LinearAlgebra::distributed::Vector<Number>  &solution,
                        LinearAlgebra::distributed::Vector<Number>        &num_Hamiltonian,
                        LinearAlgebra::distributed::Vector<Number>        &Signum_smoothed,
                        TimerOutput                                       &computing_timer);

    void
    apply_Hamiltonian(LinearAlgebra::distributed::Vector<Number>          &num_Hamiltonian,
                      const LinearAlgebra::distributed::Vector<Number>    &Signum_smoothed,
                      const LinearAlgebra::distributed::Vector<Number>    &solution) const;

    void
    Godunov_gradient(const LinearAlgebra::distributed::Vector<Number>     &solution,
                     LinearAlgebra::distributed::Vector<Number>           &God_grad,
                     TimerOutput                                          &computing_timer);

    void
    Smoothed_signum(const LinearAlgebra::distributed::Vector<Number>      &solution,
                    LinearAlgebra::distributed::Vector<Number>            &Signum_smoothed,
                    LinearAlgebra::distributed::Vector<Number>            &God_grad,
                    TimerOutput                                           &computing_timer,
                    const uint                                            max_vertex_distance) const;

    void
    flatten_level_set(LinearAlgebra::distributed::Vector<Number> &solution) const;

    void
    compute_local_viscosity(const LinearAlgebra::distributed::Vector<Number> &solution) const;	

    void
    compute_RI_indicator(const LinearAlgebra::distributed::Vector<Number> & sol);

    const Number &
    get_RI_indicator() const;

    const Number &
    get_viscosity_value() const;

    double
    compute_area(const LinearAlgebra::distributed::Vector<Number>    &solution,
                 const unsigned int                                   ls_dof_idx  = 0,
                 const unsigned int                                   ls_quad_idx = 0) const;

    double
    compute_circularity(const LinearAlgebra::distributed::Vector<Number>  &solution,
                        const DoFHandler<dim>                             &dof_handler) const;

    void
    set_artificial_viscosity_refinement_flags(const LinearAlgebra::distributed::Vector<Number> &sol) const;

    void
    reinit_grad_vectors(const LinearAlgebra::distributed::Vector<Number> &solution);

    void
    set_velocity_vector(const LinearAlgebra::distributed::Vector<Number> &velocity);

    void
    compute_viscosity_value(const double vertex_distance);

    void
    compute_RI_distance(const double vertex_distance);

    void
    compute_penalty_parameter();

    void
    vmult(LinearAlgebra::distributed::Vector<Number>        &dst,
          const LinearAlgebra::distributed::Vector<Number>  &src) const;

    void
    apply_viscosity_implicit(LinearAlgebra::distributed::Vector<Number> &solution,
                             LinearAlgebra::distributed::Vector<Number> &rk_register,
                             const double                                dt_RI,
                             TimerOutput                                &computing_timer) const;

    const unsigned int &
    get_CG_iterations() const
    {
      return CG_iterations;
    }

    void
    reset_CG_iterations() const
    {
      CG_iterations = 0;
    }

    double
    get_cond_number() const
    {
     return cond_number;
    }

    void
    reset_cond_number() const
    {
      cond_number = 0.;
    }

    double
    compute_time_step_advection() const;

  private:
    MatrixFree<dim, Number>                         data;
    
    const LevelSetProblemParameters                &param;

    mutable Number                                  time;
    mutable Number                                  dt_RI = 1.;
    mutable Number                                  RI_ind = 0.;
    mutable AlignedVector<VectorizedArray<Number>>  artificial_viscosity;
    mutable AlignedVector<VectorizedArray<Number>>  array_penalty_parameter;
    mutable Number                                  viscosity = 1.;
    mutable Number                                  area = 1.;
    mutable unsigned int                            CG_iterations = 0;
    mutable Number                                  cond_number = 0.;
    mutable Number                                  RI_distance = 0.;

    // auxiliary vectors for Godunov's scheme
    LinearAlgebra::distributed::Vector<Number>      grad_x_l;
    LinearAlgebra::distributed::Vector<Number>      grad_x_r;
    LinearAlgebra::distributed::Vector<Number>      grad_y_l;
    LinearAlgebra::distributed::Vector<Number>      grad_y_r;
    LinearAlgebra::distributed::Vector<Number>      grad_z_l;
    LinearAlgebra::distributed::Vector<Number>      grad_z_r;

    LinearAlgebra::distributed::Vector<Number>      velocity_operator;

    SmoothnessIndicator<dim, fe_degree>             indicator;

    void
    local_apply_inverse_mass_matrix(
      const MatrixFree<dim, Number>                     &data,
      LinearAlgebra::distributed::Vector<Number>        &dst,
      const LinearAlgebra::distributed::Vector<Number>  &src,
      const std::pair<unsigned int, unsigned int>       &cell_range) const;

    void
    local_apply_domain(
      const MatrixFree<dim, Number>                     &data,
      LinearAlgebra::distributed::Vector<Number>        &dst,
      const LinearAlgebra::distributed::Vector<Number>  &src,
      const std::pair<unsigned int, unsigned int>       &cell_range) const;

    template <uint component>
    void
    local_apply_domain_RI_grad(
      const MatrixFree<dim, Number>                     &data,
      LinearAlgebra::distributed::Vector<Number>        &dst,
      const LinearAlgebra::distributed::Vector<Number>  &src,
      const std::pair<unsigned int, unsigned int>       &cell_range) const;

    void
    local_apply_domain_num_Hamiltonian(
      const MatrixFree<dim, Number>                     &data,
      LinearAlgebra::distributed::Vector<Number>        &dst,
      const LinearAlgebra::distributed::Vector<Number>  &src,
      const std::pair<unsigned int, unsigned int>       &cell_range) const;

    void
    local_apply_domain_RI_diffusion(
      const MatrixFree<dim, Number>                     &data,
      LinearAlgebra::distributed::Vector<Number>        &dst,
      const LinearAlgebra::distributed::Vector<Number>  &src,
      const std::pair<unsigned int, unsigned int>       &cell_range) const;

    void
    local_apply_domain_RI_diffusion_implicit(
      const MatrixFree<dim, Number>                     &data,
      LinearAlgebra::distributed::Vector<Number>        &dst,
      const LinearAlgebra::distributed::Vector<Number>  &src,
      const std::pair<unsigned int, unsigned int>       &cell_range) const;

    void
    local_apply_inner_face(
      const MatrixFree<dim, Number>                     &data,
      LinearAlgebra::distributed::Vector<Number>        &dst,
      const LinearAlgebra::distributed::Vector<Number>  &src,
      const std::pair<unsigned int, unsigned int>       &cell_range) const;

    template <bool is_right, uint component>
    void
    local_apply_inner_face_RI_grad(
      const MatrixFree<dim, Number>                     &data,
      LinearAlgebra::distributed::Vector<Number>        &dst,
      const LinearAlgebra::distributed::Vector<Number>  &src,
      const std::pair<unsigned int, unsigned int>       &cell_range) const;

    void
    local_apply_inner_face_RI_diffusion(
      const MatrixFree<dim, Number>                     &data,
      LinearAlgebra::distributed::Vector<Number>        &dst,
      const LinearAlgebra::distributed::Vector<Number>  &src,
      const std::pair<unsigned int, unsigned int>       &cell_range) const;

    void
    local_apply_inner_face_RI_diffusion_implicit(
      const MatrixFree<dim, Number>                     &data,
      LinearAlgebra::distributed::Vector<Number>        &dst,
      const LinearAlgebra::distributed::Vector<Number>  &src,
      const std::pair<unsigned int, unsigned int>       &cell_range) const;

    void
    local_apply_boundary_face(
      const MatrixFree<dim, Number>                     &data,
      LinearAlgebra::distributed::Vector<Number>        &dst,
      const LinearAlgebra::distributed::Vector<Number>  &src,
      const std::pair<unsigned int, unsigned int>       &cell_range) const;

    template <bool is_right, uint component>
    void
    local_apply_boundary_face_RI_grad(
      const MatrixFree<dim, Number>                     &data,
      LinearAlgebra::distributed::Vector<Number>        &dst,
      const LinearAlgebra::distributed::Vector<Number>  &src,
      const std::pair<unsigned int, unsigned int>       &cell_range) const;

    void
    local_apply_boundary_face_RI_diffusion(
      const MatrixFree<dim, Number>                     &data,
      LinearAlgebra::distributed::Vector<Number>        &dst,
      const LinearAlgebra::distributed::Vector<Number>  &src,
      const std::pair<unsigned int, unsigned int>       &cell_range) const;

    void
    local_apply_boundary_face_RI_diffusion_implicit(
      const MatrixFree<dim, Number>                     &data,
      LinearAlgebra::distributed::Vector<Number>        &dst,
      const LinearAlgebra::distributed::Vector<Number>  &src,
      const std::pair<unsigned int, unsigned int>       &cell_range) const;

    void
    compute_artificial_viscosity() const;

    void
    create_rhs(
      LinearAlgebra::distributed::Vector<Number>        &dst,
      const LinearAlgebra::distributed::Vector<Number>  &src) const;
  };



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, Number>                       &data,
    LinearAlgebra::distributed::Vector<Number>          &dst,
    const LinearAlgebra::distributed::Vector<Number>    &src,
    const std::pair<unsigned int, unsigned int>         &cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval(data, 0, 1);

    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, 1, Number> inverse(eval);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval.read_dof_values(src);

        inverse.apply(eval.begin_dof_values(), eval.begin_dof_values());

        eval.set_dof_values(dst);
      }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::local_apply_domain(
    const MatrixFree<dim, Number>                       &data,
    LinearAlgebra::distributed::Vector<Number>          &dst,
    const LinearAlgebra::distributed::Vector<Number>    &src,
    const std::pair<unsigned int, unsigned int>         &cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>    eval(data);
    FEEvaluation<dim, fe_degree, fe_degree + 1, dim, Number>  eval_vel(data, 2);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);
        eval_vel.reinit(cell);

        eval.gather_evaluate(src, EvaluationFlags::values);
        eval_vel.gather_evaluate(velocity_operator, EvaluationFlags::values);

        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            const auto speed = eval_vel.get_value(q);
            const auto u     = eval.get_value(q);
            const auto flux  = speed * u;
            eval.submit_gradient(flux, q);
          }

        eval.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }



  template <int dim, int fe_degree>
  template <uint component>
  void
  LevelSetOperation<dim, fe_degree>::local_apply_domain_RI_grad(
    const MatrixFree<dim, Number>                       &data,
    LinearAlgebra::distributed::Vector<Number>          &dst,
    const LinearAlgebra::distributed::Vector<Number>    &src,
    const std::pair<unsigned int, unsigned int>         &cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval(data);

    // unit_vector * gradient = gradient[component], required for submit
    Tensor<1, dim, Number> unit_vector;
    unit_vector[component] = 1.0;

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);

        eval.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            const auto u    = eval.get_value(q);
            const auto flux = unit_vector * u;
            eval.submit_gradient(-flux, q);
          }

        eval.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::local_apply_domain_num_Hamiltonian(
    const MatrixFree<dim, Number>                       &data,
    LinearAlgebra::distributed::Vector<Number>          &dst,
    const LinearAlgebra::distributed::Vector<Number>    &src,
    const std::pair<unsigned int, unsigned int>         &cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);

        eval.gather_evaluate(src, EvaluationFlags::values);

        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            const auto u = eval.get_value(q);
            // minus sign because the term is shifted to the right-hand side of the RI equation
            eval.submit_value(-u, q);
          }

        eval.integrate_scatter(EvaluationFlags::values, dst);
      }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::local_apply_domain_RI_diffusion(
    const MatrixFree<dim, Number>                       &data,
    LinearAlgebra::distributed::Vector<Number>          &dst,
    const LinearAlgebra::distributed::Vector<Number>    &src,
    const std::pair<unsigned int, unsigned int>         &cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);

        eval.gather_evaluate(src, EvaluationFlags::gradients);
        
        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            const auto flux  = viscosity * eval.get_gradient(q);
            eval.submit_gradient(-flux, q);
          }

        eval.integrate_scatter(EvaluationFlags::gradients, dst);
      }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::local_apply_domain_RI_diffusion_implicit(
    const MatrixFree<dim, Number>                       &data,
    LinearAlgebra::distributed::Vector<Number>          &dst,
    const LinearAlgebra::distributed::Vector<Number>    &src,
    const std::pair<unsigned int, unsigned int>         &cell_range) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval(data);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        eval.reinit(cell);

        eval.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        
        for (unsigned int q = 0; q < eval.n_q_points; ++q)
          {
            // mass matrix
            eval.submit_value(eval.get_value(q), q);
            // consider time step size dt_RI for implicit scheme
            const auto flux  = viscosity * eval.get_gradient(q) * dt_RI;
            eval.submit_gradient(flux, q);
          }

        eval.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }
  


  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::local_apply_inner_face(
    const MatrixFree<dim, Number>                       &data,
    LinearAlgebra::distributed::Vector<Number>          &dst,
    const LinearAlgebra::distributed::Vector<Number>    &src,
    const std::pair<unsigned int, unsigned int>         &face_range) const
  {
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>    eval_minus(data, true);
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number>    eval_plus(data, false);
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, dim, Number>  eval_vel(data, true, 2);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_plus.reinit(face);
        eval_vel.reinit(face);

        eval_minus.gather_evaluate(src, EvaluationFlags::values);
        eval_plus.gather_evaluate(src, EvaluationFlags::values);
        eval_vel.gather_evaluate(velocity_operator, EvaluationFlags::values);

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            const auto speed   = eval_vel.get_value(q);
            const auto u_minus = eval_minus.get_value(q);
            const auto u_plus  = eval_plus.get_value(q);
            const auto normal_vector_minus = eval_minus.get_normal_vector(q);

            const auto normal_times_speed         = speed * normal_vector_minus;
            const auto flux_times_normal_of_minus = 0.5 * ((u_minus + u_plus) * normal_times_speed +
                                                    std::abs(normal_times_speed) * (u_minus - u_plus));

            eval_minus.submit_value(-flux_times_normal_of_minus, q);
            eval_plus.submit_value(flux_times_normal_of_minus, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values, dst);
        eval_plus.integrate_scatter(EvaluationFlags::values, dst);
      }
  }



  template <int dim, int fe_degree>
  template <bool is_right, uint component>
  void
  LevelSetOperation<dim, fe_degree>::local_apply_inner_face_RI_grad(
    const MatrixFree<dim, Number>                       &data,
    LinearAlgebra::distributed::Vector<Number>          &dst,
    const LinearAlgebra::distributed::Vector<Number>    &src,
    const std::pair<unsigned int, unsigned int>         &face_range) const
  {
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_minus(data, true);
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_plus(data, false);

    if (is_right)
    {
      for (unsigned int face = face_range.first; face < face_range.second; face++)
        {
          eval_minus.reinit(face);
          eval_plus.reinit(face);

          eval_minus.gather_evaluate(src, EvaluationFlags::values);
          eval_plus.gather_evaluate(src, EvaluationFlags::values);

          for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
            {
              const auto u_minus = eval_minus.get_value(q);
              const auto u_plus  = eval_plus.get_value(q);
              const auto normal_vector_minus = eval_minus.get_normal_vector(q);

              const auto flux = compare_and_apply_mask<SIMDComparison::greater_than_or_equal>(
                                    normal_vector_minus[component],
                                    0.,
                                    normal_vector_minus[component] * u_plus,
                                    normal_vector_minus[component] * u_minus);

              eval_minus.submit_value(flux, q);
              eval_plus.submit_value(-flux, q);
            }

          eval_minus.integrate_scatter(EvaluationFlags::values, dst);
          eval_plus.integrate_scatter(EvaluationFlags::values, dst);
        }
    }
    else
    {
      for (unsigned int face = face_range.first; face < face_range.second; face++)
        {
          eval_minus.reinit(face);
          eval_plus.reinit(face);

          eval_minus.gather_evaluate(src, EvaluationFlags::values);
          eval_plus.gather_evaluate(src, EvaluationFlags::values);

          for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
            {
              const auto u_minus = eval_minus.get_value(q);
              const auto u_plus  = eval_plus.get_value(q);
              const auto normal_vector_minus = eval_minus.get_normal_vector(q);

              const auto flux = compare_and_apply_mask<SIMDComparison::greater_than_or_equal>(
                                    normal_vector_minus[component],
                                    0.,
                                    normal_vector_minus[component] * u_minus,
                                    normal_vector_minus[component] * u_plus);

              eval_minus.submit_value(flux, q);
              eval_plus.submit_value(-flux, q);
            }

          eval_minus.integrate_scatter(EvaluationFlags::values, dst);
          eval_plus.integrate_scatter(EvaluationFlags::values, dst);
        }
    }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::local_apply_inner_face_RI_diffusion(
    const MatrixFree<dim, Number>                       &data,
    LinearAlgebra::distributed::Vector<Number>          &dst,
    const LinearAlgebra::distributed::Vector<Number>    &src,
    const std::pair<unsigned int, unsigned int>         &face_range) const
  {
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_minus(data, true);
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_plus(data, false);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_plus.reinit(face);

        eval_minus.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        eval_plus.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        const auto sigmaF = std::max(eval_minus.read_cell_data(array_penalty_parameter),
                            eval_plus.read_cell_data(array_penalty_parameter));

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            // 1st face integral
            const auto u_minus = eval_minus.get_value(q);
            const auto u_plus  = eval_plus.get_value(q);

            const auto flux_1   = 0.5 * (u_minus - u_plus) * viscosity;

            eval_minus.submit_normal_derivative(flux_1, q);
            eval_plus.submit_normal_derivative(flux_1, q);

            // 2nd and 3rd (=penalty) face integral 
            const auto u_minus_normal_grad = eval_minus.get_normal_derivative(q);
            const auto u_plus_normal_grad  = eval_plus.get_normal_derivative(q);

            const auto flux_2 = 0.5 * (u_minus_normal_grad + u_plus_normal_grad) * viscosity
                                -(u_minus - u_plus) * viscosity * sigmaF;

            eval_minus.submit_value(flux_2, q);
            eval_plus.submit_value(-flux_2, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        eval_plus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::local_apply_inner_face_RI_diffusion_implicit(
    const MatrixFree<dim, Number>                     &data,
    LinearAlgebra::distributed::Vector<Number>        &dst,
    const LinearAlgebra::distributed::Vector<Number>  &src,
    const std::pair<unsigned int, unsigned int>       &face_range) const
  {
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_minus(data, true);
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_plus(data, false);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_plus.reinit(face);

        eval_minus.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
        eval_plus.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        const auto sigmaF = std::max(eval_minus.read_cell_data(array_penalty_parameter),
                            eval_plus.read_cell_data(array_penalty_parameter));

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            // 1st face integral
            // consider time step size dt_RI for implicit scheme
            const auto u_minus = eval_minus.get_value(q);
            const auto u_plus  = eval_plus.get_value(q);
            const auto flux_1  = 0.5 * (u_minus - u_plus) * viscosity * dt_RI;

            eval_minus.submit_normal_derivative(-flux_1, q);
            eval_plus.submit_normal_derivative(-flux_1, q);

            // 2nd and 3rd (=penalty) face integral 
            // consider time step size dt_RI for implicit scheme
            const auto u_minus_normal_grad = eval_minus.get_normal_derivative(q);
            const auto u_plus_normal_grad  = eval_plus.get_normal_derivative(q);

            const auto flux_2 = 0.5 * (u_minus_normal_grad + u_plus_normal_grad) * viscosity * dt_RI
                                -(u_minus - u_plus) * viscosity * sigmaF * dt_RI;

            eval_minus.submit_value(-flux_2, q);
            eval_plus.submit_value(flux_2, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        eval_plus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::local_apply_boundary_face(
    const MatrixFree<dim, Number>                       &data,
    LinearAlgebra::distributed::Vector<Number>          &dst,
    const LinearAlgebra::distributed::Vector<Number>    &src,
    const std::pair<unsigned int, unsigned int>         &face_range) const
  {
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_minus(data, true);
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, dim, Number> eval_vel(data, true, 2);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_minus.gather_evaluate(src, EvaluationFlags::values);
        eval_vel.reinit(face);
        eval_vel.gather_evaluate(velocity_operator, EvaluationFlags::values);

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            const auto speed = eval_vel.get_value(q);

            // Dirichlet boundary
            const auto u_minus       = eval_minus.get_value(q);
            const auto normal_vector = eval_minus.get_normal_vector(q);

            // Fix solution value outside of the reinitialization region
            const auto u_plus =  RI_distance * 1.2;

            // Compute the flux
            const auto normal_times_speed = normal_vector * speed;
            const auto flux_times_normal = 0.5 * ((u_minus + u_plus) * normal_times_speed +
                                           std::abs(normal_times_speed) * (u_minus - u_plus));

            eval_minus.submit_value(-flux_times_normal, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values, dst);
      }
  }



  template <int dim, int fe_degree>
  template <bool is_right, uint component>
  void
  LevelSetOperation<dim, fe_degree>::local_apply_boundary_face_RI_grad(
    const MatrixFree<dim, Number>                     &data,
    LinearAlgebra::distributed::Vector<Number>        &dst,
    const LinearAlgebra::distributed::Vector<Number>  &src,
    const std::pair<unsigned int, unsigned int>       &face_range) const
  {
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_minus(data, true);

      if (is_right)
      {
        for (unsigned int face = face_range.first; face < face_range.second; face++)
          {
            eval_minus.reinit(face);
            eval_minus.gather_evaluate(src, EvaluationFlags::values);

            for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
              {
                // Dirichlet boundary
                const auto u_minus             = eval_minus.get_value(q);
                const auto normal_vector_minus = eval_minus.get_normal_vector(q);

                // Compute the outer solution value
                const auto u_plus  =  RI_distance * 1.2;

                // Compute the flux
                const auto flux = compare_and_apply_mask<SIMDComparison::greater_than_or_equal>(
                                      normal_vector_minus[component],
                                      0.,
                                      normal_vector_minus[component] * u_plus,
                                      normal_vector_minus[component] * u_minus);

                eval_minus.submit_value(flux, q);
              }

            eval_minus.integrate_scatter(EvaluationFlags::values, dst);
          }
      }
      else
      {
        for (unsigned int face = face_range.first; face < face_range.second; face++)
          {
            eval_minus.reinit(face);
            eval_minus.gather_evaluate(src, EvaluationFlags::values);

            for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
              {
                // Dirichlet boundary
                const auto u_minus       = eval_minus.get_value(q);
                const auto normal_vector_minus = eval_minus.get_normal_vector(q);

                // Compute the outer solution value
                const auto u_plus  = RI_distance * 1.2;

                // Compute the flux
                const auto flux = compare_and_apply_mask<SIMDComparison::greater_than_or_equal>(
                                      normal_vector_minus[component],
                                      0.,
                                      normal_vector_minus[component] * u_minus,
                                      normal_vector_minus[component] * u_plus);

                eval_minus.submit_value(flux, q);
              }

            eval_minus.integrate_scatter(EvaluationFlags::values, dst);
          }
      }
  }
  



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::local_apply_boundary_face_RI_diffusion(
    const MatrixFree<dim, Number>                       &data,
    LinearAlgebra::distributed::Vector<Number>          &dst,
    const LinearAlgebra::distributed::Vector<Number>    &src,
    const std::pair<unsigned int, unsigned int>         &face_range) const
  {
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_minus(data, true);

      for (unsigned int face = face_range.first; face < face_range.second; face++)
        {
          eval_minus.reinit(face);
          eval_minus.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

          const auto sigmaF = eval_minus.read_cell_data(array_penalty_parameter);

          for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
            {
              // 1st face integral
              const auto u_minus = eval_minus.get_value(q);

              // Compute the outer solution value
              const auto u_plus  = RI_distance * 1.2;

              const auto flux_1  = 0.5 * (u_minus - u_plus) * viscosity;

              eval_minus.submit_normal_derivative(flux_1, q);

              // 2nd and 3rd (=penalty) face integral
              const auto u_minus_normal_grad = eval_minus.get_normal_derivative(q);

              // Assume same gradients.
              const auto u_plus_normal_grad  = - u_minus_normal_grad;

              const auto flux_2 = 0.5 * (u_minus_normal_grad + u_plus_normal_grad) * viscosity
                                  - (u_minus - u_plus) * viscosity * sigmaF;

              eval_minus.submit_value(flux_2, q);
            }

          eval_minus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::local_apply_boundary_face_RI_diffusion_implicit(
    const MatrixFree<dim, Number>                       &data,
    LinearAlgebra::distributed::Vector<Number>          &dst,
    const LinearAlgebra::distributed::Vector<Number>    &src,
    const std::pair<unsigned int, unsigned int>         &face_range) const
  {
    FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_minus(data, true);

    for (unsigned int face = face_range.first; face < face_range.second; face++)
      {
        eval_minus.reinit(face);
        eval_minus.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

        const auto sigmaF = eval_minus.read_cell_data(array_penalty_parameter);

        for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
          {
            // 1st face integral
            const auto u_minus = eval_minus.get_value(q);

            // Boundary condition for solution value. 
            // (Note: The CG solver has problems if a fix solution value at the boundary is used.)
            // Consider time step size dt_RI for implicit scheme.
            const auto u_plus = u_minus;
            const auto flux_1 = 0.5 * (u_minus - u_plus) * viscosity * dt_RI;

            eval_minus.submit_normal_derivative(-flux_1, q);

            // 2nd and 3rd (=penalty) face integral 
            // Consider time step size dt_RI for implicit scheme.
            const auto u_minus_normal_grad = eval_minus.get_normal_derivative(q);

            // Assume same gradients.
            const auto u_plus_normal_grad = - u_minus_normal_grad;

            const auto flux_2 = 0.5 * (u_minus_normal_grad + u_plus_normal_grad) * viscosity * dt_RI
                                - (u_minus - u_plus) * viscosity * sigmaF * dt_RI;

            eval_minus.submit_value(-flux_2, q);
          }

        eval_minus.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::compute_artificial_viscosity() const
  {
    // Resize
    const unsigned int n_macro_cells = this->data.n_cell_batches();
    artificial_viscosity.resize(n_macro_cells);

    const Number s_0   = param.art_diff_k * (-4. * std::log10(fe_degree));
    const auto & smoothness_indicator = indicator.get_smoothness_indicator();

    Number epsilon_0, sm_ind, distance;

    for (uint macro_cells = 0; macro_cells < n_macro_cells; ++macro_cells)
      {
        // Depending on the cell number, there might be empty lanes
        const unsigned int n_lanes_filled = this->data.n_active_entries_per_cell_batch(macro_cells);

        // Element-wise calculation of the required artificial viscosity according to Persson and Peraire (2006)
        for (uint lane = 0; lane < n_lanes_filled; ++lane)
          {
            auto cell = this->data.get_cell_iterator(macro_cells, lane);

            sm_ind = smoothness_indicator[macro_cells][lane];

            if (sm_ind < (s_0 - param.art_diff_kappa))
              artificial_viscosity[macro_cells][lane] = 0.;
            else if (sm_ind <= (s_0 + param.art_diff_kappa))
              {
                distance  = cell->minimum_vertex_distance();
                epsilon_0 = param.factor_diffusivity * distance / fe_degree;

                artificial_viscosity[macro_cells][lane] =
                  (.5 * epsilon_0) *
                  (1. + std::sin(numbers::PI * (sm_ind - s_0) / (2. * param.art_diff_kappa)));
              }
            else
              {
                distance  = cell->minimum_vertex_distance();
                epsilon_0 = param.factor_diffusivity * distance / fe_degree;
                artificial_viscosity[macro_cells][lane] = epsilon_0;
              }
          }
      }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::create_rhs(LinearAlgebra::distributed::Vector<Number>        &dst,
                                                const LinearAlgebra::distributed::Vector<Number>  &src) const
    {
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval(this->data);

      this->data.template cell_loop<LinearAlgebra::distributed::Vector<Number>,
                                    LinearAlgebra::distributed::Vector<Number>>(
        [&](const auto &, auto &dst, const auto &src, auto &range) {
          for (auto cell = range.first; cell < range.second; ++cell)
            {
              eval.reinit(cell);                
              eval.gather_evaluate(src, EvaluationFlags::values);

              for (unsigned int q = 0; q < eval.static_n_q_points; ++q)
                eval.submit_value(eval.get_value(q), q);

              eval.integrate_scatter(EvaluationFlags::values, dst);
            }
        },
        dst,
        src,
        true);
    }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::reinit(const DoFHandler<dim>   &dof_handler,
                                            const DoFHandler<dim>   &dof_handler_legendre,
                                            const DoFHandler<dim>   &dof_handler_vel)
  {
    std::vector<const DoFHandler<dim> *> dof_handlers(
        {&dof_handler, &dof_handler_legendre, &dof_handler_vel});
    MappingQGeneric<dim> mapping(fe_degree);
    Quadrature<1> quadrature      = QGauss<1>(fe_degree + 1);
    Quadrature<1> quadrature_mass = QGauss<1>(fe_degree + 1);
    // QGauss<1>(fe_degree + 1) gives inaccurate results for the norm computation.
    // Use overintegration or GaussLobatto quadrature for norm computation.
    Quadrature<1> quadrature_norm = QGauss<1>(fe_degree + 2);
    typename MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.overlap_communication_computation = false;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points |
       update_values);
    additional_data.mapping_update_flags_inner_faces =
      (update_JxW_values | update_normal_vectors | update_quadrature_points |
       update_values);
    additional_data.mapping_update_flags_boundary_faces =
      (update_JxW_values | update_normal_vectors | update_quadrature_points |
       update_values);

    AffineConstraints<Number> dummy;
    std::vector<const AffineConstraints<Number> *> constraints({&dummy, &dummy, &dummy});
    dummy.close();
    data.reinit(mapping,
                dof_handlers,
                constraints,
                std::vector<Quadrature<1>>{{quadrature, quadrature_mass, quadrature_norm}},
                additional_data);
  }



  template <int dim, int fe_degree>
  template <bool is_right, uint component>
  void
  LevelSetOperation<dim, fe_degree>::apply_RI_grad(const LinearAlgebra::distributed::Vector<Number>   &src,
                                                   LinearAlgebra::distributed::Vector<Number>         &dst)
  {
    data.loop(&LevelSetOperation<dim, fe_degree>::local_apply_domain_RI_grad<component>,
              &LevelSetOperation<dim, fe_degree>::local_apply_inner_face_RI_grad<is_right, component>,
              &LevelSetOperation<dim, fe_degree>::local_apply_boundary_face_RI_grad<is_right, component>,
              this,
              dst,
              src,
              true,
              MatrixFree<dim, Number>::DataAccessOnFaces::values,
              MatrixFree<dim, Number>::DataAccessOnFaces::values);

    data.cell_loop(
      &LevelSetOperation<dim, fe_degree>::local_apply_inverse_mass_matrix,
      this,
      dst,
      dst);
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::perform_stage(
    const Number                                        current_time,
    const Number                                        factor_solution,
    const Number                                        factor_ai,
    const LinearAlgebra::distributed::Vector<Number>   &current_ri,
    LinearAlgebra::distributed::Vector<Number>         &vec_ki,
    LinearAlgebra::distributed::Vector<Number>         &solution,
    LinearAlgebra::distributed::Vector<Number>         &next_ri,
    TimerOutput                                        &computing_timer) const
  {
    time = current_time;
    {
    TimerOutput::Scope t(computing_timer, "ADVECTION: apply integrals");

    data.loop(&LevelSetOperation<dim, fe_degree>::local_apply_domain,
              &LevelSetOperation<dim, fe_degree>::local_apply_inner_face,
              &LevelSetOperation<dim, fe_degree>::local_apply_boundary_face,
              this,
              vec_ki,
              current_ri,
              true,
              MatrixFree<dim, Number>::DataAccessOnFaces::values,
              MatrixFree<dim, Number>::DataAccessOnFaces::values);
    }

    {
    TimerOutput::Scope t(computing_timer, "ADVECTION: apply inverse mass matrix");
    
    data.cell_loop(
      &LevelSetOperation<dim, fe_degree>::local_apply_inverse_mass_matrix,
      this,
      next_ri,
      vec_ki,
      std::function<void(const unsigned int, const unsigned int)>(),
      [&](const unsigned int start_range, const unsigned int end_range) {
        const Number ai = factor_ai;
        const Number bi = factor_solution;
        if (ai == Number())
          {
            DEAL_II_OPENMP_SIMD_PRAGMA
            for (unsigned int i = start_range; i < end_range; ++i)
              {
                const Number k_i          = next_ri.local_element(i);
                const Number sol_i        = solution.local_element(i);
                solution.local_element(i) = sol_i + bi * k_i;
              }
          }
        else
          {
            DEAL_II_OPENMP_SIMD_PRAGMA
            for (unsigned int i = start_range; i < end_range; ++i)
              {
                const Number k_i          = next_ri.local_element(i);
                const Number sol_i        = solution.local_element(i);
                solution.local_element(i) = sol_i + bi * k_i;
                next_ri.local_element(i)  = sol_i + ai * k_i;
              }
          }
      });
    }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::perform_stage_RI(
    const Number                                      current_time,
    const Number                                      factor_solution,
    const Number                                      factor_ai,
    LinearAlgebra::distributed::Vector<Number>       &vec_ki,
    LinearAlgebra::distributed::Vector<Number>       &solution,
    LinearAlgebra::distributed::Vector<Number>       &next_ri,
    LinearAlgebra::distributed::Vector<Number>       &num_Hamiltonian,
    TimerOutput                                      &computing_timer) const
  {
    time = current_time;
    {
    TimerOutput::Scope t(computing_timer, "RI: apply num. Hamiltonian integral");
    
    data.cell_loop(&LevelSetOperation<dim, fe_degree>::local_apply_domain_num_Hamiltonian,
              this,
              vec_ki,
              num_Hamiltonian,
              true);
    }

    if(param.use_IMEX == false)
    {
      {
      TimerOutput::Scope t(computing_timer, "RI: apply diffusion integrals");
      
      data.loop(&LevelSetOperation<dim, fe_degree>::local_apply_domain_RI_diffusion,
                &LevelSetOperation<dim, fe_degree>::local_apply_inner_face_RI_diffusion,
                &LevelSetOperation<dim, fe_degree>::local_apply_boundary_face_RI_diffusion,
                this,
                num_Hamiltonian,
                solution,
                true,
                MatrixFree<dim, Number>::DataAccessOnFaces::values,
                MatrixFree<dim, Number>::DataAccessOnFaces::values);
      }

      // add the numerical Hamiltonian rhs and the diffusion rhs
      vec_ki.add(1.0, num_Hamiltonian);
    }

    {
    TimerOutput::Scope t(computing_timer, "RI: apply inverse mass matrix");
    
    data.cell_loop(
      &LevelSetOperation<dim, fe_degree>::local_apply_inverse_mass_matrix,
      this,
      next_ri,
      vec_ki,
      std::function<void(const unsigned int, const unsigned int)>(),
      [&](const unsigned int start_range, const unsigned int end_range) {
        const Number ai = factor_ai;
        const Number bi = factor_solution;
        if (ai == Number())
          {
            DEAL_II_OPENMP_SIMD_PRAGMA
            for (unsigned int i = start_range; i < end_range; ++i)
              {
                const Number k_i          = next_ri.local_element(i);
                const Number sol_i        = solution.local_element(i);
                solution.local_element(i) = sol_i + bi * k_i;
              }
          }
        else
          {
            DEAL_II_OPENMP_SIMD_PRAGMA
            for (unsigned int i = start_range; i < end_range; ++i)
              {
                const Number k_i          = next_ri.local_element(i);
                const Number sol_i        = solution.local_element(i);
                solution.local_element(i) = sol_i + bi * k_i;
                next_ri.local_element(i)  = sol_i + ai * k_i;
              }
          }
      });
    }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::project_initial(
    LinearAlgebra::distributed::Vector<Number> &dst) const
  {
    LevelSetFunction<dim> solution(0., param);
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(data);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, 1, Number>
      inverse(phi);
#if DEAL_II_VERSION_GTE(9, 3, 0)
    dst.zero_out_ghost_values();
#else
    dst.zero_out_ghosts();
#endif
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_dof_value(solution.value(phi.quadrature_point(q)), q);
        inverse.transform_from_q_points_to_basis(1,
                                                 phi.begin_dof_values(),
                                                 phi.begin_dof_values());
        phi.set_dof_values(dst);
      }
  }



  // This function calculates the L2-norm of the distance (phi-phi_analytical)
  // in the interface region of a circular interface
  template <int dim, int fe_degree>
  double
  LevelSetOperation<dim, fe_degree>::compute_L2_norm_in_interface_region(
    const LinearAlgebra::distributed::Vector<Number> &solution) const
  {
    // Define the coordinates of the center of the circular interface for
    // the current test case
    const Number x_center = 0.5;
    const Number y_center = 0.5;
    const Number z_center = 0.;
    // Exact radius of interface for the current test case
    const Number r_interface = 0.15;
    // Interval from the inteface, in which the norm should be calculated
    const Number epsilon = 0.02;

    Number  norm_interface_region = 0.;

    // Set up function for radius computation
    ComputeRadius<dim> compute_radius(0., x_center, y_center, z_center);
    // Set up function for exact solution
    LevelSetFunction<dim>  exact_solution(0., param);
    
    FEEvaluation<dim, -1> phi(data,0,2);

    for (unsigned int macro_cells = 0; macro_cells < data.n_cell_batches(); ++macro_cells)
      {
        phi.reinit(macro_cells);
        phi.gather_evaluate(solution, EvaluationFlags::values);

        // Depending on the cell number, there might be empty lanes
        const unsigned int n_lanes_filled = this->data.n_active_entries_per_cell_batch(macro_cells);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          // calculate radius at the current quadrature point
          auto r = compute_radius.value(phi.quadrature_point(q));

          // calculate deviation from solution
          auto sol_difference = std::abs(0.5*exact_solution.value(phi.quadrature_point(q)) - phi.get_value(q));
          // alternatively, use:
          // auto sol_difference = std::abs((r-r_interface) - phi.get_value(q));
          
          // loop over lanes of VectorizedArray
          // check, if current position is in the defined interval around the interface level
          for (unsigned int lane = 0; lane < n_lanes_filled; ++lane)
          {
            if (r[lane]<=(r_interface+epsilon) && r[lane]>=(r_interface-epsilon))
            {
              norm_interface_region += sol_difference[lane] * sol_difference[lane] * phi.JxW(q)[lane];
            }
          }
        }
      }

    norm_interface_region = Utilities::MPI::sum(norm_interface_region, MPI_COMM_WORLD);

    return std::sqrt(norm_interface_region);
  }



  template <int dim, int fe_degree>
  Tensor<1, 2>
  LevelSetOperation<dim, fe_degree>::compute_mass_and_energy(
    const LinearAlgebra::distributed::Vector<Number> &vec) const
  {
    Tensor<1, 2>    mass_energy = {};
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(data);
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(vec, EvaluationFlags::values | EvaluationFlags::gradients);
        VectorizedArray<Number> mass   = {};
        VectorizedArray<Number> energy = {};
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            mass += phi.get_value(q) * phi.JxW(q);
            energy += phi.get_value(q) * phi.get_value(q) * phi.JxW(q);
          }
        for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell);
             ++v)
          {
            mass_energy[0] += mass[v];
            mass_energy[1] += energy[v];
          }
      }
    return Utilities::MPI::sum(mass_energy, vec.get_mpi_communicator());
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::Godunov_Hamiltonian(
          const LinearAlgebra::distributed::Vector<Number>  &solution,
          LinearAlgebra::distributed::Vector<Number>        &num_Hamiltonian,
          LinearAlgebra::distributed::Vector<Number>        &Signum_smoothed,
          TimerOutput                                       &computing_timer)
  {
    {
    TimerOutput::Scope t(computing_timer, "RI: compute upwind and downwind gradients");

    // The upwind and downwind gradients are already calculated for the smoothed signum function.
    // The user has the choice if he want's to update the gradient vectors in every Runge-Kutta stage.

    // Hint: Especially for explicit reinitialization time stepping use_const_gradient_in_RI = is useful
    // to save computational effort. As rule of thumb, up to 10 explicit reinitialization time steps, the 
    // accuracy is not deteriorated due to this simplification.
    if (param.use_const_gradient_in_RI == false)
    {
      // x-direction
      apply_RI_grad<false, 0>(solution, grad_x_l);
      apply_RI_grad<true, 0>(solution, grad_x_r);
      // y-direction
      apply_RI_grad<false, 1>(solution, grad_y_l);
      apply_RI_grad<true, 1>(solution, grad_y_r);

      if (dim == 3)
      {
        // z-direction
        apply_RI_grad<false, 2>(solution, grad_z_l);
        apply_RI_grad<true, 2>(solution, grad_z_r);
      }
    }
    }

    {
    TimerOutput::Scope t(computing_timer, "RI: compute Godunov num. Hamiltonian");
    
    // calculate the numerical Hamiltonian with Godunov's method
    apply_Hamiltonian(num_Hamiltonian, Signum_smoothed, solution);
    }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::apply_Hamiltonian(
          LinearAlgebra::distributed::Vector<Number>        &num_Hamiltonian,
          const LinearAlgebra::distributed::Vector<Number>  &Signum_smoothed,
          const LinearAlgebra::distributed::Vector<Number>  &solution) const
  {
    const dealii::VectorizedArray<Number> zero_vector = 0;

    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_grad_x_l(data);
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_grad_x_r(data);
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_grad_y_l(data);
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_grad_y_r(data);

    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_sign_mod(data);
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_sol(data);

    if (dim == 2)
    {
      for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
        {
          phi_grad_x_l.reinit(cell);
          phi_grad_x_l.read_dof_values(grad_x_l);
          phi_grad_x_r.reinit(cell);
          phi_grad_x_r.read_dof_values(grad_x_r);
          phi_grad_y_l.reinit(cell);
          phi_grad_y_l.read_dof_values(grad_y_l);
          phi_grad_y_r.reinit(cell);
          phi_grad_y_r.read_dof_values(grad_y_r);
          phi_sign_mod.reinit(cell);
          phi_sign_mod.read_dof_values(Signum_smoothed);
          phi_sol.reinit(cell);
          phi_sol.read_dof_values(solution);

          for (unsigned int q = 0; q < phi_grad_x_l.dofs_per_cell; ++q)
          {
            dealii::VectorizedArray<Number> gradient_goal = 1.;

            if (param.do_edge_rounding)
            {
            // definition of the target gradient
            gradient_goal = 1.* 4. / ((std::exp(6. * (1. - std::abs(phi_sol.get_dof_value(q))/RI_distance)) 
                                      + std::exp(-6. *(1.-std::abs(phi_sol.get_dof_value(q)) / RI_distance))) 
                                      * (std::exp(6. *(1. - std::abs(phi_sol.get_dof_value(q)) / RI_distance)) 
                                      + std::exp(-6.*(1. - std::abs(phi_sol.get_dof_value(q)) / RI_distance)) ));
              
            gradient_goal = compare_and_apply_mask<SIMDComparison::less_than>(std::abs(phi_sol.get_dof_value(q)),
                                                                              RI_distance,
                                                                              1.,
                                                                              gradient_goal);
            }

            auto u = compare_and_apply_mask<SIMDComparison::greater_than>(phi_sign_mod.get_dof_value(q),
                                                                          0.,
            
            (std::sqrt(
              
            std::max((std::min(phi_grad_x_l.get_dof_value(q), zero_vector))*(std::min(phi_grad_x_l.get_dof_value(q), zero_vector)),
                    (std::max(phi_grad_x_r.get_dof_value(q), zero_vector))*(std::max(phi_grad_x_r.get_dof_value(q), zero_vector)))

            + std::max((std::min(phi_grad_y_l.get_dof_value(q), zero_vector))*(std::min(phi_grad_y_l.get_dof_value(q), zero_vector)),
                    (std::max(phi_grad_y_r.get_dof_value(q), zero_vector))*(std::max(phi_grad_y_r.get_dof_value(q), zero_vector)))   
            )    
            - gradient_goal) * phi_sign_mod.get_dof_value(q)
            ,
            (std::sqrt(
              
            std::max((std::max(phi_grad_x_l.get_dof_value(q), zero_vector))*(std::max(phi_grad_x_l.get_dof_value(q), zero_vector)),
                    (std::min(phi_grad_x_r.get_dof_value(q), zero_vector))*(std::min(phi_grad_x_r.get_dof_value(q), zero_vector)))

            + std::max((std::max(phi_grad_y_l.get_dof_value(q), zero_vector))*(std::max(phi_grad_y_l.get_dof_value(q), zero_vector)),
                    (std::min(phi_grad_y_r.get_dof_value(q), zero_vector))*(std::min(phi_grad_y_r.get_dof_value(q), zero_vector)))
            )
            - gradient_goal) * phi_sign_mod.get_dof_value(q)
            );

            phi_grad_x_l.submit_dof_value(u, q);
          }
          phi_grad_x_l.set_dof_values(num_Hamiltonian);
        }
    }

    if (dim == 3)
    {
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_grad_z_l(data);
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_grad_z_r(data);

      for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
        {
          phi_grad_x_l.reinit(cell);
          phi_grad_x_l.read_dof_values(grad_x_l);
          phi_grad_x_r.reinit(cell);
          phi_grad_x_r.read_dof_values(grad_x_r);
          phi_grad_y_l.reinit(cell);
          phi_grad_y_l.read_dof_values(grad_y_l);
          phi_grad_y_r.reinit(cell);
          phi_grad_y_r.read_dof_values(grad_y_r);
          phi_grad_z_l.reinit(cell);
          phi_grad_z_l.read_dof_values(grad_z_l);
          phi_grad_z_r.reinit(cell);
          phi_grad_z_r.read_dof_values(grad_z_r);
          phi_sign_mod.reinit(cell);
          phi_sign_mod.read_dof_values(Signum_smoothed);
          phi_sol.reinit(cell);
          phi_sol.read_dof_values(solution);

          for (unsigned int q = 0; q < phi_grad_x_l.dofs_per_cell; ++q)
          {
            dealii::VectorizedArray<Number> gradient_goal = 1.;

            if (param.do_edge_rounding)
            {
            // definition of the target gradient
            gradient_goal = 1.* 4. / ((std::exp(6. * (1. - std::abs(phi_sol.get_dof_value(q))/RI_distance)) 
                                      + std::exp(-6. *(1.-std::abs(phi_sol.get_dof_value(q)) / RI_distance))) 
                                      * (std::exp( 6. *(1. - std::abs(phi_sol.get_dof_value(q)) / RI_distance)) 
                                      + std::exp(-6.*(1. - std::abs(phi_sol.get_dof_value(q)) / RI_distance)) ));
              
            gradient_goal = compare_and_apply_mask<SIMDComparison::less_than>(std::abs(phi_sol.get_dof_value(q)),
                                                                              RI_distance,
                                                                              1.,
                                                                              gradient_goal);
            }
            
            auto u = compare_and_apply_mask<SIMDComparison::greater_than>(phi_sign_mod.get_dof_value(q),0.,
            
            (std::sqrt(
              
            std::max((std::min(phi_grad_x_l.get_dof_value(q), zero_vector))*(std::min(phi_grad_x_l.get_dof_value(q), zero_vector)),
                    (std::max(phi_grad_x_r.get_dof_value(q), zero_vector))*(std::max(phi_grad_x_r.get_dof_value(q), zero_vector)))

            + std::max((std::min(phi_grad_y_l.get_dof_value(q), zero_vector))*(std::min(phi_grad_y_l.get_dof_value(q), zero_vector)),
                    (std::max(phi_grad_y_r.get_dof_value(q), zero_vector))*(std::max(phi_grad_y_r.get_dof_value(q), zero_vector)))

            + std::max((std::min(phi_grad_z_l.get_dof_value(q), zero_vector))*(std::min(phi_grad_z_l.get_dof_value(q), zero_vector)),
                    (std::max(phi_grad_z_r.get_dof_value(q), zero_vector))*(std::max(phi_grad_z_r.get_dof_value(q), zero_vector)))   
            )
            - gradient_goal) * phi_sign_mod.get_dof_value(q)
            ,
            (std::sqrt(
              
            std::max((std::max(phi_grad_x_l.get_dof_value(q), zero_vector))*(std::max(phi_grad_x_l.get_dof_value(q), zero_vector)),
                    (std::min(phi_grad_x_r.get_dof_value(q), zero_vector))*(std::min(phi_grad_x_r.get_dof_value(q), zero_vector)))

            + std::max((std::max(phi_grad_y_l.get_dof_value(q), zero_vector))*(std::max(phi_grad_y_l.get_dof_value(q), zero_vector)),
                    (std::min(phi_grad_y_r.get_dof_value(q), zero_vector))*(std::min(phi_grad_y_r.get_dof_value(q), zero_vector)))

            + std::max((std::max(phi_grad_z_l.get_dof_value(q), zero_vector))*(std::max(phi_grad_z_l.get_dof_value(q), zero_vector)),
                    (std::min(phi_grad_z_r.get_dof_value(q), zero_vector))*(std::min(phi_grad_z_r.get_dof_value(q), zero_vector)))  
            )
            - gradient_goal) * phi_sign_mod.get_dof_value(q)
            );

            phi_grad_x_l.submit_dof_value(u, q);
          }

          phi_grad_x_l.set_dof_values(num_Hamiltonian);
        }
    }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::Godunov_gradient(
          const LinearAlgebra::distributed::Vector<Number>  &solution,
          LinearAlgebra::distributed::Vector<Number>        &God_grad,
          TimerOutput                                       &computing_timer)
  {
    {
    TimerOutput::Scope t(computing_timer, "RI: compute upwind and downwind gradients");

    //compute local upwind and downwind gradients
    // x-direction
    apply_RI_grad<false, 0>(solution, grad_x_l);
    apply_RI_grad<true, 0>(solution, grad_x_r);
    // y-direction
    apply_RI_grad<false, 1>(solution, grad_y_l);
    apply_RI_grad<true, 1>(solution, grad_y_r);

    if (dim == 3)
    {
      // z-direction
      apply_RI_grad<false, 2>(solution, grad_z_l);
      apply_RI_grad<true, 2>(solution, grad_z_r);
    }
    }

    {
    TimerOutput::Scope t(computing_timer, "RI: compute Godunov gradient");
    
    const dealii::VectorizedArray<Number> zero_vector = 0;
    
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_grad_x_l(data);
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_grad_x_r(data);
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_grad_y_l(data);
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_grad_y_r(data);

    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_sol(data);

    if (dim == 2)
    {
      for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
        {
          phi_grad_x_l.reinit(cell);
          phi_grad_x_l.read_dof_values(grad_x_l);
          phi_grad_x_r.reinit(cell);
          phi_grad_x_r.read_dof_values(grad_x_r);
          phi_grad_y_l.reinit(cell);
          phi_grad_y_l.read_dof_values(grad_y_l);
          phi_grad_y_r.reinit(cell);
          phi_grad_y_r.read_dof_values(grad_y_r);
          phi_sol.reinit(cell);
          phi_sol.read_dof_values(solution);

          for (unsigned int q = 0; q < phi_grad_x_l.dofs_per_cell; ++q)
          {
          auto u = compare_and_apply_mask<SIMDComparison::greater_than>(phi_sol.get_dof_value(q),
                                                                        0.,
          
          std::sqrt(
            
          std::max((std::min(phi_grad_x_l.get_dof_value(q), zero_vector))*(std::min(phi_grad_x_l.get_dof_value(q), zero_vector)),
                  (std::max(phi_grad_x_r.get_dof_value(q), zero_vector))*(std::max(phi_grad_x_r.get_dof_value(q), zero_vector)))

          + std::max((std::min(phi_grad_y_l.get_dof_value(q), zero_vector))*(std::min(phi_grad_y_l.get_dof_value(q), zero_vector)),
                  (std::max(phi_grad_y_r.get_dof_value(q), zero_vector))*(std::max(phi_grad_y_r.get_dof_value(q), zero_vector)))     
          )
          ,
          std::sqrt(
            
          std::max((std::max(phi_grad_x_l.get_dof_value(q), zero_vector))*(std::max(phi_grad_x_l.get_dof_value(q), zero_vector)),
                  (std::min(phi_grad_x_r.get_dof_value(q), zero_vector))*(std::min(phi_grad_x_r.get_dof_value(q), zero_vector)))

          + std::max((std::max(phi_grad_y_l.get_dof_value(q), zero_vector))*(std::max(phi_grad_y_l.get_dof_value(q), zero_vector)),
                  (std::min(phi_grad_y_r.get_dof_value(q), zero_vector))*(std::min(phi_grad_y_r.get_dof_value(q), zero_vector)))     
          )
          );
          phi_grad_x_l.submit_dof_value(u, q);
          }

          phi_grad_x_l.set_dof_values(God_grad);
        }
    }

    if (dim == 3)
    {
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_grad_z_l(data);
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi_grad_z_r(data);

      for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
        {
          phi_grad_x_l.reinit(cell);
          phi_grad_x_l.read_dof_values(grad_x_l);
          phi_grad_x_r.reinit(cell);
          phi_grad_x_r.read_dof_values(grad_x_r);
          phi_grad_y_l.reinit(cell);
          phi_grad_y_l.read_dof_values(grad_y_l);
          phi_grad_y_r.reinit(cell);
          phi_grad_y_r.read_dof_values(grad_y_r);
          phi_grad_z_l.reinit(cell);
          phi_grad_z_l.read_dof_values(grad_z_l);
          phi_grad_z_r.reinit(cell);
          phi_grad_z_r.read_dof_values(grad_z_r);
          phi_sol.reinit(cell);
          phi_sol.read_dof_values(solution);

          for (unsigned int q = 0; q < phi_grad_x_l.dofs_per_cell; ++q)
          {
          auto u = compare_and_apply_mask<SIMDComparison::greater_than>(phi_sol.get_dof_value(q),
                                                                        0.,
          
          std::sqrt(
            
          std::max((std::min(phi_grad_x_l.get_dof_value(q), zero_vector))*(std::min(phi_grad_x_l.get_dof_value(q), zero_vector)),
                  (std::max(phi_grad_x_r.get_dof_value(q), zero_vector))*(std::max(phi_grad_x_r.get_dof_value(q), zero_vector)))

          + std::max((std::min(phi_grad_y_l.get_dof_value(q), zero_vector))*(std::min(phi_grad_y_l.get_dof_value(q), zero_vector)),
                  (std::max(phi_grad_y_r.get_dof_value(q), zero_vector))*(std::max(phi_grad_y_r.get_dof_value(q), zero_vector)))

          + std::max((std::min(phi_grad_z_l.get_dof_value(q), zero_vector))*(std::min(phi_grad_z_l.get_dof_value(q), zero_vector)),
                  (std::max(phi_grad_z_r.get_dof_value(q), zero_vector))*(std::max(phi_grad_z_r.get_dof_value(q), zero_vector)))      
          )
          ,
          std::sqrt(
            
          std::max((std::max(phi_grad_x_l.get_dof_value(q), zero_vector))*(std::max(phi_grad_x_l.get_dof_value(q), zero_vector)),
                  (std::min(phi_grad_x_r.get_dof_value(q), zero_vector))*(std::min(phi_grad_x_r.get_dof_value(q), zero_vector)))

          + std::max((std::max(phi_grad_y_l.get_dof_value(q), zero_vector))*(std::max(phi_grad_y_l.get_dof_value(q), zero_vector)),
                  (std::min(phi_grad_y_r.get_dof_value(q), zero_vector))*(std::min(phi_grad_y_r.get_dof_value(q), zero_vector)))

          + std::max((std::max(phi_grad_z_l.get_dof_value(q), zero_vector))*(std::max(phi_grad_z_l.get_dof_value(q), zero_vector)),
                  (std::min(phi_grad_z_r.get_dof_value(q), zero_vector))*(std::min(phi_grad_z_r.get_dof_value(q), zero_vector))) 
          )
          );
          phi_grad_x_l.submit_dof_value(u, q);
          }
          
          phi_grad_x_l.set_dof_values(God_grad);
        }
    }
    }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::Smoothed_signum(
          const LinearAlgebra::distributed::Vector<Number>  &solution,
          LinearAlgebra::distributed::Vector<Number>        &Signum_smoothed,
          LinearAlgebra::distributed::Vector<Number>        &God_grad,
          TimerOutput                                       &computing_timer,
          const uint                                         max_vertex_distance) const
  {
    {
    TimerOutput::Scope t(computing_timer, "RI: compute smoothed Signum function");
  
    // Use maximum element size for the calculation of the argument of tanh
    const dealii::VectorizedArray<Number> eta_vector = 2. * max_vertex_distance / fe_degree;

    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> source(data);
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> God_grad_p(data);
    
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        source.reinit(cell);
        source.read_dof_values(solution);
        God_grad_p.reinit(cell);
        God_grad_p.read_dof_values(God_grad);

        for (unsigned int q = 0; q < source.dofs_per_cell; ++q)
        {
          // calculate argument of tanh: (pi*phi)/(2*grad(phi)*max_cell_size/fe_degree)
          // +0.001 to avoid division through 0 in case of a singularity in the level set function
          const auto arg = numbers::PI * source.get_dof_value(q) / (eta_vector * God_grad_p.get_dof_value(q) + 0.001);
          //tanh(x)=(e^x-e^(-x))/(e^x+e^(-x)), tanh(x) is not supported for VectorizedArray
          const auto u = (std::exp(arg)-std::exp(-arg)) / (std::exp(arg)+std::exp(-arg));

          source.submit_dof_value(u, q);
        }

        source.set_dof_values(Signum_smoothed);
      }
    }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::flatten_level_set( LinearAlgebra::distributed::Vector<Number> &solution) const
  {
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> source(data);
    
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        source.reinit(cell);
        source.read_dof_values(solution);

        for (unsigned int q = 0; q < source.dofs_per_cell; ++q)
        {
        // Flatten the solution field to the value +/- (epsilon * 1.2)
        auto u = compare_and_apply_mask<SIMDComparison::greater_than>(source.get_dof_value(q), 
                                                                            RI_distance * 1.2,
                                                                            RI_distance * 1.2,
                                                                            source.get_dof_value(q));

        u = compare_and_apply_mask<SIMDComparison::less_than>(source.get_dof_value(q), 
                                                                            -RI_distance * 1.2,
                                                                            -RI_distance * 1.2,
                                                                            u);

        source.submit_dof_value(u, q);
        }

        source.set_dof_values(solution);
      }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::compute_local_viscosity(const LinearAlgebra::distributed::Vector<Number> &solution) const
  {
    indicator.template compute_smoothness_indicator(this->data, solution);
    compute_artificial_viscosity();
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::compute_RI_indicator(const LinearAlgebra::distributed::Vector<Number> & sol)
  {
    Number u = 0.;
    Number v = 0.;

    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(data);
    
    for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(sol, EvaluationFlags::values | EvaluationFlags::gradients);

        // Depending on the cell number, there might be empty lanes
        const unsigned int n_lanes_filled = this->data.n_active_entries_per_cell_batch(cell);

        // loop over quadrature points
        for (unsigned int q = 0; q < phi.n_q_points; ++q)
        {
          const auto value = phi.get_value(q);
          const auto grad = phi.get_gradient(q);
          const auto norm_grad = grad.norm();

          // loop over lanes of VectorizedArray
          for (unsigned int lane = 0; lane < n_lanes_filled; ++lane)
          {
            // The factor 0.8 is required for the bubble vortex test case,
            // as the reinitialized areas in the vicinity of two adjacent
            // zero level sections intersect and develop a kink.
            // For non-disturbed signed distance areas +-epsilon,
            // the factor is not necessary.
            if (std::abs(value[lane]) <= 1.0 * RI_distance)
            {
            u += std::abs(norm_grad[lane] - 1.);
            v += 1.;
            }
          }
        }
      }
      u = Utilities::MPI::sum(u, MPI_COMM_WORLD);
      v = Utilities::MPI::sum(v, MPI_COMM_WORLD);

    RI_ind = u/(v+0.0001);
  }



  template <int dim, int fe_degree>
  const double &
  LevelSetOperation<dim, fe_degree>::get_RI_indicator() const
  {
    return RI_ind;
  }



  template <int dim, int fe_degree>
  const double &
  LevelSetOperation<dim, fe_degree>::get_viscosity_value() const
  {
    return RI_ind;
  }



  template <int dim, int fe_degree>
  double
  LevelSetOperation<dim, fe_degree>::compute_area(const LinearAlgebra::distributed::Vector<double> &solution,
                                                  const unsigned int                                ls_dof_idx,
                                                  const unsigned int                                ls_quad_idx) const
  {
    Number area_droplet = 0;

    FEEvaluation<dim, fe_degree, fe_degree+1, 1, Number> phi(this->data, ls_dof_idx, ls_quad_idx);

    for (unsigned int cell = 0; cell <this->data.n_cell_batches(); ++cell)
      {
        phi.reinit(cell);
        phi.gather_evaluate(solution, EvaluationFlags::values);

        // Depending on the cell number, there might be empty lanes
        const unsigned int n_lanes_filled = this->data.n_active_entries_per_cell_batch(cell);

        for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            // it is assumed that the level set is negative inside the droplet
            auto mask = compare_and_apply_mask<SIMDComparison::less_than>(phi.get_value(q), 0., 1., 0.);
            mask = mask * phi.JxW(q);

            // loop over lanes of VectorizedArray
            for (unsigned int lane = 0; lane < n_lanes_filled; ++lane)
            {
              area_droplet += mask[lane];
            }
            
          }
      }

    area_droplet = Utilities::MPI::sum(area_droplet, MPI_COMM_WORLD);

    this->area = area_droplet;
    return area_droplet;
  }



  /**
  * Compute circularity according to
  *
  *       _____
  *   2 \/ π A  
  *   ----------
  *    /
  *   | 1 dS
  *  /
  *  Γ
  *
  *  with the surface (2D) or volume(3D) of the droplet A. The denominator
  *  represents the surface area of the droplet.
  */
  template <int dim, int fe_degree>
  double
  LevelSetOperation<dim, fe_degree>::compute_circularity(const LinearAlgebra::distributed::Vector<double>   &solution,
                                                         const DoFHandler<dim>                              &dof_handler) const
  {
    // 1) compute the surface of the droplet
    Triangulation<std::max(1,dim - 1), dim> tria_droplet_surface;

    GridTools::create_triangulation_with_marching_cube_algorithm(
      tria_droplet_surface,
      *this->data.get_mapping_info().mapping,
      dof_handler,
      solution,
      0. /*iso level*/,
      1 /*n subdivisions of surface mesh*/);

    double area_droplet_boundary = 0;

    // check if partitioned domains contain surface elements in case of parallel execution
    if(tria_droplet_surface.n_cells() > 0)
      area_droplet_boundary = GridTools::volume<std::max(1,dim - 1), dim> (tria_droplet_surface); 

    area_droplet_boundary = Utilities::MPI::sum(area_droplet_boundary, MPI_COMM_WORLD);

    AssertThrow(area_droplet_boundary >= 1e-16, ExcMessage("Area of droplet is zero."));

    // 2) compute circularity
    return 2. * std::sqrt(numbers::PI * this->area) / (area_droplet_boundary + 0.000001);
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::set_artificial_viscosity_refinement_flags(
                        const LinearAlgebra::distributed::Vector<Number> & sol) const
  {
    // Resize
    const unsigned int n_macro_cells = this->data.n_cell_batches();

    // Set refinement and coarsen flags according to the artificial_viscosity value in each cell
    for (uint macro_cells = 0; macro_cells < n_macro_cells; ++macro_cells)
      {
        // Depending on the cell number, there might be empty lanes
        const unsigned int n_lanes_filled = this->data.n_active_entries_per_cell_batch(macro_cells);

        for (uint lane = 0; lane < n_lanes_filled; ++lane)
          {
            auto cell = this->data.get_cell_iterator(macro_cells, lane);

            if (cell->is_locally_owned() == false)
            {
              continue;
            }

            if (artificial_viscosity[macro_cells][lane] > param.factor_refinement_flag * viscosity)
            {
              cell->set_refine_flag();
            } 
            else if (artificial_viscosity[macro_cells][lane] < param.factor_coarsen_flag * viscosity)
            {
              cell->set_coarsen_flag();
            }
            else
            {
              cell->clear_coarsen_flag();
              cell->clear_refine_flag();
            }
          }
      }

    // set refinement flags for the elements in the vicinity of the zero level
    if (param.use_const_element_size_at_interface)
     {
      FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(data);
      
        for (unsigned int macro_cells = 0; macro_cells < data.n_cell_batches(); ++macro_cells)
        {
          phi.reinit(macro_cells);
          phi.gather_evaluate(sol, EvaluationFlags::values);

          // Depending on the cell number, there might be empty lanes
          const unsigned int n_lanes_filled = this->data.n_active_entries_per_cell_batch(macro_cells);

          // loop over quadrature points
          for (unsigned int q = 0; q < phi.n_q_points; ++q)
          {
            const auto value = phi.get_value(q);

            // loop over lanes of VectorizedArray
            for (unsigned int lane = 0; lane < n_lanes_filled; ++lane)
            {
              if (std::abs(value[lane]) <= param.factor_distance_const_element_size_at_interface * RI_distance)
              {
                auto cell = this->data.get_cell_iterator(macro_cells, lane);
                cell->clear_coarsen_flag();
                cell->set_refine_flag();
              }
            }
          }
        }
     }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::reinit_grad_vectors(const LinearAlgebra::distributed::Vector<double> &solution)
  {
    grad_x_l.reinit(solution);
    grad_x_r.reinit(solution);
    grad_y_l.reinit(solution);
    grad_y_r.reinit(solution);

    if (dim == 3)
    {
      grad_z_l.reinit(solution);
      grad_z_r.reinit(solution);
    }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::set_velocity_vector(const LinearAlgebra::distributed::Vector<double> &velocity)
  {
    initialize_dof_vector(velocity_operator, 2);
    velocity_operator.copy_locally_owned_data_from(velocity);
    velocity_operator.update_ghost_values();
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::compute_viscosity_value(const double vertex_distance)
  {
    // The value for the artificial viscosity is determined by the smallest enabled element size.
    if (param.use_adaptive_mesh_refinement)
      viscosity = param.factor_diffusivity * vertex_distance / std::pow(2., param.n_refinement_levels) / fe_degree;
    else
      viscosity = param.factor_diffusivity * vertex_distance / fe_degree;
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::compute_RI_distance(const double vertex_distance)
  {
    // Compute the reinitialization distance
    RI_distance = vertex_distance / fe_degree * param.factor_RI_distance;
  }



  template<int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::compute_penalty_parameter()
  {
    // Resize
    const unsigned int n_macro_cells = this->data.n_cell_batches() + this->data.n_ghost_cell_batches();
    array_penalty_parameter.resize(n_macro_cells);

    for (uint macro_cells = 0; macro_cells < n_macro_cells; ++macro_cells)
      {
        // Depending on the cell number, there might be empty lanes
        const unsigned int n_lanes_filled = this->data.n_active_entries_per_cell_batch(macro_cells);

        for (uint lane = 0; lane < n_lanes_filled; ++lane)
          {
            auto cell = this->data.get_cell_iterator(macro_cells, lane);

            array_penalty_parameter[macro_cells][lane] = 1. / cell->minimum_vertex_distance() * 
                                                         (param.fe_degree + 1) * (param.fe_degree + 1) * 
                                                         param.IP_diffusion;
          }
      }
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::vmult(LinearAlgebra::distributed::Vector<Number>       &dst,
                                           const LinearAlgebra::distributed::Vector<Number> &src) const
  {
    this->data.loop(
      &LevelSetOperation::local_apply_domain_RI_diffusion_implicit,
      &LevelSetOperation::local_apply_inner_face_RI_diffusion_implicit,
      &LevelSetOperation::local_apply_boundary_face_RI_diffusion_implicit,
      this,
      dst,
      src,
      true,
      MatrixFree<dim, Number>::DataAccessOnFaces::gradients,
      MatrixFree<dim, Number>::DataAccessOnFaces::gradients);
  }



  template <int dim, int fe_degree>
  void
  LevelSetOperation<dim, fe_degree>::apply_viscosity_implicit(LinearAlgebra::distributed::Vector<Number> &solution,
                                                              LinearAlgebra::distributed::Vector<Number> &rk_register,
                                                              const double                                delta_t,
                                                              TimerOutput                                &computing_timer) const
  {
    TimerOutput::Scope t(computing_timer, "RI: IMEX - solve linear system");
    
    dt_RI = delta_t;

    Preconditioner<dim, fe_degree> p(this->data);

    create_rhs(rk_register, solution);

    // 1.e-3 for residual reduction is a good trade-off between compuational effort and result quality
    ReductionControl reduction_control(1000, 1.e-20, 1.e-3);
    SolverCG<LinearAlgebra::distributed::Vector<Number>> solver(reduction_control);

    // observe condition number
    solver.connect_condition_number_slot([&](double condition_number) {
      this->cond_number += condition_number;
    });

    solver.solve(*this, solution, rk_register, p);

    this->CG_iterations += reduction_control.last_step();
  }



  template <int dim, int fe_degree>
  double
  LevelSetOperation<dim, fe_degree>::
    compute_time_step_advection() const
  {
    Number max_transport = 0.;

    FEEvaluation<dim, fe_degree, fe_degree + 1, dim, Number> phi_vel(data, 2);

      for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
       {
          phi_vel.reinit(cell);
          phi_vel.gather_evaluate(velocity_operator, EvaluationFlags::values);

          dealii::VectorizedArray<Number> local_max = 0.;

          for (unsigned int q = 0; q < phi_vel.n_q_points; ++q)
            {
              const auto velocity = phi_vel.get_value(q);

              const auto inverse_jacobian = phi_vel.inverse_jacobian(q);
              const auto convective_speed = inverse_jacobian * velocity;

              dealii::VectorizedArray<Number> convective_limit = 0.;

              for (unsigned int d = 0; d < dim; ++d)
                convective_limit = std::max(convective_limit, std::abs(convective_speed[d]));

              local_max = std::max(local_max, convective_limit);
            }

          // There might be empty cells in the cell-batch
          for (unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
            for (unsigned int d = 0; d < dim; ++d)
            max_transport = std::max(max_transport, local_max[v]);
        }

      max_transport = Utilities::MPI::max(max_transport, velocity_operator.get_mpi_communicator());

      // The advection time step size is not allowed to exceed a certain value, 
      // as for currently very low velocities, the time step size
      // would be very large and lead to inaccurate results.
      // (e. g. for the bubble vortex test case at half simulation time.)
      // Adjust max_time_step_size_advection for the current simulation case.

      const Number max_time_step_size_advection = 0.001;

      return std::min(max_time_step_size_advection, param.courant_number_advection / (std::pow(fe_degree, 2) * max_transport));
  }



  class LowStorageRungeKuttaIntegrator
  {
  public:
    LowStorageRungeKuttaIntegrator(const LowStorageRungeKuttaScheme scheme, const LevelSetProblemParameters &param): param(param)
    {
      switch (scheme)
        {
          case stage_3_order_3:
            {
              bi = {{0.245170287303492, 0.184896052186740, 0.569933660509768}};
              ai = {{0.755726351946097, 0.386954477304099}};

              break;
            }

          case stage_5_order_4:
            {
              bi = {{1153189308089. / 22510343858157.,
                     1772645290293. / 4653164025191.,
                     -1672844663538. / 4480602732383.,
                     2114624349019. / 3568978502595.,
                     5198255086312. / 14908931495163.}};
              ai = {{970286171893. / 4311952581923.,
                     6584761158862. / 12103376702013.,
                     2251764453980. / 15575788980749.,
                     26877169314380. / 34165994151039.}};

              break;
            }

          case stage_7_order_4:
            {
              bi = {{0.0941840925477795334,
                     0.149683694803496998,
                     0.285204742060440058,
                     -0.122201846148053668,
                     0.0605151571191401122,
                     0.345986987898399296,
                     0.186627171718797670}};
              ai = {{0.241566650129646868 + bi[0],
                     0.0423866513027719953 + bi[1],
                     0.215602732678803776 + bi[2],
                     0.232328007537583987 + bi[3],
                     0.256223412574146438 + bi[4],
                     0.0978694102142697230 + bi[5]}};

              break;
            }

          case stage_9_order_5:
            {
              bi = {{2274579626619. / 23610510767302.,
                     693987741272. / 12394497460941.,
                     -347131529483. / 15096185902911.,
                     1144057200723. / 32081666971178.,
                     1562491064753. / 11797114684756.,
                     13113619727965. / 44346030145118.,
                     393957816125. / 7825732611452.,
                     720647959663. / 6565743875477.,
                     3559252274877. / 14424734981077.}};
              ai = {{1107026461565. / 5417078080134.,
                     38141181049399. / 41724347789894.,
                     493273079041. / 11940823631197.,
                     1851571280403. / 6147804934346.,
                     11782306865191. / 62590030070788.,
                     9452544825720. / 13648368537481.,
                     4435885630781. / 26285702406235.,
                     2357909744247. / 11371140753790.}};

              break;
            }

          default:
            AssertThrow(false, ExcNotImplemented());
        }
    }

    unsigned int
    n_stages() const
    {
      return bi.size();
    }

    template <typename VectorType, typename Operator>
    void
    perform_time_step(const Operator         &pde_operator,
                      const double            current_time,
                      const double            time_step,
                      VectorType             &solution,
                      VectorType             &vec_ri,
                      VectorType             &vec_ki,
                      TimerOutput            &computing_timer) const
    {
      AssertDimension(ai.size() + 1, bi.size());

      pde_operator.perform_stage(current_time,
                                 bi[0] * time_step,
                                 ai[0] * time_step,
                                 solution,
                                 vec_ri,
                                 solution,
                                 vec_ri,
                                 computing_timer);
      double sum_previous_bi = 0;
      for (unsigned int stage = 1; stage < bi.size(); ++stage)
        {
          const double c_i = sum_previous_bi + ai[stage - 1];
          pde_operator.perform_stage(current_time + c_i * time_step,
                                     bi[stage] * time_step,
                                     (stage == bi.size() - 1 ?
                                        0 :
                                        ai[stage] * time_step),
                                     vec_ri,
                                     vec_ki,
                                     solution,
                                     vec_ri,
                                     computing_timer);
          sum_previous_bi += bi[stage - 1];
        }
    }

    template <typename VectorType, typename Operator>
    void
    perform_time_step_RI(Operator     &pde_operator,
                         const double  current_time,
                         const double  time_step,
                         VectorType   &solution,
                         VectorType   &vec_ri,
                         VectorType   &vec_ki,
                         VectorType   &num_Hamiltonian,
                         VectorType   &Signum_smoothed,
                         TimerOutput  &computing_timer) const
    {
      AssertDimension(ai.size() + 1, bi.size());

      // Only relevant for IMEX time discretization,
      // perform implicit time discretization for artificial viscosity term
      if (param.use_IMEX)
        pde_operator.apply_viscosity_implicit(solution, vec_ki, time_step, computing_timer);

      // calculate numerical Hamiltonian with Godunov method
      pde_operator.Godunov_Hamiltonian(solution, num_Hamiltonian, Signum_smoothed, computing_timer);
      
      pde_operator.perform_stage_RI(current_time,
                                 bi[0] * time_step,
                                 ai[0] * time_step,
                                 vec_ri,
                                 solution,
                                 vec_ri,
                                 num_Hamiltonian,
                                 computing_timer);
      double sum_previous_bi = 0;
      
      for (unsigned int stage = 1; stage < bi.size(); ++stage)
        {
          // calculate numerical Hamiltonian with Godunov method
          pde_operator.Godunov_Hamiltonian(solution, num_Hamiltonian, Signum_smoothed, computing_timer);

          const double c_i = sum_previous_bi + ai[stage - 1];
          pde_operator.perform_stage_RI(current_time + c_i * time_step,
                                     bi[stage] * time_step,
                                     (stage == bi.size() - 1 ?
                                        0 :
                                        ai[stage] * time_step),
                                     vec_ki,
                                     solution,
                                     vec_ri,
                                     num_Hamiltonian,
                                     computing_timer);
          sum_previous_bi += bi[stage - 1];
        }
    }

  private:
    std::vector<double> bi;
    std::vector<double> ai;
    const LevelSetProblemParameters &param;
  };



  template <int dim, int fe_degree>
  class AdvectionRIProblem
  {
  public:
    typedef typename LevelSetOperation<dim, fe_degree>::Number Number;
    AdvectionRIProblem(const LevelSetProblemParameters &param);
    void
    run();

  private:
    void
    make_grid();
    void
    setup_dofs();
    void
    output_results(const unsigned int timestep_number,
                   const Tensor<1, 2> mass_and_energy,
                   const Number,
                   const Number,
                   const Number,
                   const Number,
                   const Number,
                   const Number,
                   const Number,
                   const Number);

    template <typename VectorType, typename Operator>
    void
    refine_grid(Operator            &pde_operator,
                const unsigned int   number,
                VectorType          &rk_reg_1,
                VectorType          &rk_reg_2,
                VectorType          &num_Hamiltonian,
                VectorType          &Signum_smoothed,
                VectorType          &God_grad,
                VectorType          &velocity);

    LinearAlgebra::distributed::Vector<Number> solution;
    LinearAlgebra::distributed::Vector<Number> velocity;

    std::shared_ptr<Triangulation<dim>> triangulation;
    MappingQGeneric<dim>                mapping;
    FESystem<dim>                       fe;
    FESystem<dim>                       fe_legendre;
    FESystem<dim>                       fe_vel;
    DoFHandler<dim>                     dof_handler;
    DoFHandler<dim>                     dof_handler_legendre;
    DoFHandler<dim>                     dof_handler_vel;

    IndexSet locally_relevant_dofs;

    Number time, time_step_advection, time_step_RI, time_step_output;

    ConditionalOStream pcout;

    const LevelSetProblemParameters &param;

    TimerOutput computing_timer;
  };



  template <int dim, int fe_degree>
  AdvectionRIProblem<dim, fe_degree>::AdvectionRIProblem(const LevelSetProblemParameters &param)
    : mapping(fe_degree)
    , fe(FE_DGQ<dim>(fe_degree), 1)
    , fe_legendre(FE_DGQLegendre<dim>(fe_degree), 1)
    , fe_vel(FE_DGQ<dim>(fe_degree), dim)
    , time(0)
    , pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    , param(param)
    , computing_timer(MPI_COMM_WORLD,
                      pcout,
                      TimerOutput::never,
                      TimerOutput::wall_times)
  {
#ifdef DEAL_II_WITH_P4EST
    if (dim > 1)
      triangulation =
        std::make_shared<parallel::distributed::Triangulation<dim>>(
          MPI_COMM_WORLD);
    else
#endif
      triangulation = std::make_shared<Triangulation<dim>>();
  }


  template <int dim, int fe_degree>
  void
  AdvectionRIProblem<dim, fe_degree>::make_grid()
  {
    // adjust geometry according to current test case
    triangulation->clear();
    Point<dim> p1;
    Point<dim> p2;
    for (unsigned int d = 0; d < dim; ++d)
    {
      p2[d] = 1;
      //p1[d] = -2;
    }
    //p2[0] = 3;
    std::vector<unsigned int> subdivisions(dim, 1);
    //subdivisions[0] = 3;

    GridGenerator::subdivided_hyper_rectangle(*triangulation,
                                              subdivisions,
                                              p1,
                                              p2);

    triangulation->refine_global(param.n_global_refinements);
  }



  template <int dim, int fe_degree>
  void
  AdvectionRIProblem<dim, fe_degree>::setup_dofs()
  {
#if DEAL_II_VERSION_GTE(9, 3, 0)
    dof_handler.reinit(*triangulation);
    dof_handler_legendre.reinit(*triangulation);
    dof_handler_vel.reinit(*triangulation);

    dof_handler.distribute_dofs(fe);
    dof_handler_legendre.distribute_dofs(fe_legendre);
    dof_handler_vel.distribute_dofs(fe_vel);
#else
    dof_handler.initialize(*triangulation, fe);
    dof_handler_legendre.initialize(*triangulation, fe_legendre);
    dof_handler_vel.initialize(*triangulation, fe_vel);
#endif
  }



  // Compute the gradient of the level-set field for output
  template <int dim>
  class GradientPostprocessor : public DataPostprocessorVector<dim>
  {
  public:
    GradientPostprocessor ()
      :
      DataPostprocessorVector<dim> ("grad_u",
                                    update_gradients)
    {}
  


    virtual
    void
    evaluate_scalar_field
    (const DataPostprocessorInputs::Scalar<dim> &param,
     std::vector<Vector<double>>                &computed_quantities) const override
    {
      AssertDimension (param.solution_gradients.size(),
                      computed_quantities.size());
  
      for (unsigned int p=0; p<param.solution_gradients.size(); ++p)
        {
          AssertDimension (computed_quantities[p].size(), dim);
          for (unsigned int d=0; d<dim; ++d)
            computed_quantities[p][d] = param.solution_gradients[p][d];
        }
    }
  };



  template <int dim, int fe_degree>
  void
  AdvectionRIProblem<dim, fe_degree>::output_results(const unsigned int   output_number,
                                                     const Tensor<1, 2>   mass_energy,
                                                     const Number         RI_indicator,
                                                     const Number         n_RI_timestep_ave,
                                                     const Number         area,
                                                     const Number         circularity,
                                                     const Number         time_step_advection,
                                                     const Number         time_step_RI,
                                                     const Number         ave_CG_iterations,
                                                     const Number         ave_cond_number)
  {
    pcout << "  -------------------------------------------" << std::endl 
          << "   Time: " << std::setw(25) << std::setprecision(3) << time << std::endl
          << "  -------------------------------------------" << std::endl; 
    if (param.dynamic_sim)
    {
      pcout << "   time step adv: " << std::setw(16) << std::setprecision(3) << time_step_advection << std::endl;
    }
    pcout << "   time step RI: " << std::setw(17) << std::setprecision(3) << time_step_RI << std::endl
          << "   mass: " << std::setprecision(8) << std::setw(25) << mass_energy[0] << std::endl
          << "   energy: " << std::setprecision(8) << std::setw(23) << mass_energy[1] << std::endl 
          << "   area: " << std::setprecision(6) << std::setw(25) << area << std::endl
          << "   circularity: " << std::setprecision(6) << std::setw(18) << circularity << std::endl
          << "   RI quality: " << std::setw(19) << std::setprecision(6) << RI_indicator << std::endl
          << "   NoE: " << std::setw(26) << triangulation->n_global_active_cells() << std::endl
          << "   NoDoF: " << std::setw(24) << dof_handler.n_dofs() << std::endl;
    if (param.dynamic_sim)
    {
      pcout << "   ave RI/adv.: " << std::setw(18) << std::setprecision(5)<< n_RI_timestep_ave << std::endl;
    }
          if (param.use_IMEX)
            pcout << "   ave CG iterations: " << std::setw(12) << std::setprecision(5)<< ave_CG_iterations << std::endl
            << "   ave condition number: " << std::setw(9) << std::setprecision(5)<< ave_cond_number << std::endl;
    pcout << "  -------------------------------------------" << std::endl;

    if (!param.print_vtu)
      return;

    // calculate gradient of level-set field
    GradientPostprocessor<dim> gradient_postprocessor;

    // Write output to a vtu file
    DataOut<dim>          data_out;
    if (param.dynamic_sim)
    {
      DataOut<dim>          data_out_vel;
      data_out_vel.attach_dof_handler(dof_handler_vel);
      data_out_vel.add_data_vector(velocity, "velocity");
      data_out_vel.build_patches(mapping,
                                 fe_degree,
                                 DataOut<dim>::curved_inner_cells);
      const std::string filename_velocity =
        "velocity_" + Utilities::int_to_string(output_number, 3) + ".vtu";
      data_out_vel.write_vtu_in_parallel(filename_velocity, MPI_COMM_WORLD);
    }
     
    DataOutBase::VtkFlags flags;
    flags.write_higher_order_cells = true;
    data_out.set_flags(flags);

    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
    data_out.add_data_vector(solution, gradient_postprocessor);
    data_out.build_patches(mapping,
                           fe_degree,
                           DataOut<dim>::curved_inner_cells);

    const std::string filename =
      "solution_" + Utilities::int_to_string(output_number, 3) + ".vtu";
    data_out.write_vtu_in_parallel(filename, MPI_COMM_WORLD);
  }



  template <int dim, int fe_degree>
  template <typename VectorType, typename Operator>
  void
  AdvectionRIProblem<dim, fe_degree>::refine_grid(Operator            &pde_operator,
                                                  const unsigned int,
                                                  VectorType          &rk_reg1,
                                                  VectorType          &rk_reg2,
                                                  VectorType&         num_Hamiltonian,
                                                  VectorType&         Signum_smoothed,
                                                  VectorType&         God_grad,
                                                  VectorType&         velocity)
  {
    pde_operator.set_artificial_viscosity_refinement_flags(solution);

    // delete flags, if max/min level is reached
    if (triangulation->n_levels() > (param.n_global_refinements+param.n_refinement_levels))
      for (typename Triangulation<dim>::active_cell_iterator cell =
             triangulation->begin_active(param.n_global_refinements+param.n_refinement_levels);
           cell != triangulation->end();
           ++cell)
        if (cell->is_locally_owned())
          cell->clear_refine_flag();
    for (typename Triangulation<dim>::active_cell_iterator cell =
           triangulation->begin_active(param.n_global_refinements);
         cell != triangulation->end_active(param.n_global_refinements);
         ++cell)
      if (cell->is_locally_owned())
        cell->clear_coarsen_flag();

    triangulation->prepare_coarsening_and_refinement();

    parallel::distributed::Triangulation<dim> *tria =
      dynamic_cast<parallel::distributed::Triangulation<dim> *>(
        triangulation.get());

    if (tria)
      {
        parallel::distributed::SolutionTransfer<dim, VectorType>
          solution_transfer(dof_handler);

        solution_transfer.prepare_for_coarsening_and_refinement(solution);

        triangulation->execute_coarsening_and_refinement();

        // reinit data
        dof_handler.distribute_dofs(fe);
        dof_handler_legendre.distribute_dofs(fe_legendre);
        dof_handler_vel.distribute_dofs(fe_vel);

        pde_operator.reinit(dof_handler,
                              dof_handler_legendre,
                              dof_handler_vel);

        // and interpolate the solution
        VectorType interpolated_solution;
        // create VectorType in the right size (=new mesh) here
        pde_operator.initialize_dof_vector(interpolated_solution, 0);

        solution_transfer.interpolate(interpolated_solution);

        pde_operator.initialize_dof_vector(solution, 0);
        solution = interpolated_solution;
      }
    else
      {
        SolutionTransfer<dim, VectorType> solution_transfer(dof_handler);

        solution_transfer.prepare_for_coarsening_and_refinement(solution);

        triangulation->execute_coarsening_and_refinement();

        // reinit data
        dof_handler.distribute_dofs(fe);
        dof_handler_legendre.distribute_dofs(fe_legendre);
        dof_handler_vel.distribute_dofs(fe_vel);

        pde_operator.reinit(dof_handler,
                            dof_handler_legendre,
                            dof_handler_vel);

        // and interpolate the solution
        VectorType interpolated_solution;
        // create VectorType in the right size (=new mesh) here
        pde_operator.initialize_dof_vector(interpolated_solution, 0);

        solution_transfer.interpolate(solution, interpolated_solution);

        pde_operator.initialize_dof_vector(solution, 0);
        solution = interpolated_solution;
      }

    // reinit vectors
    rk_reg1.reinit(solution);
    rk_reg2.reinit(solution);
    num_Hamiltonian.reinit(solution);
    Signum_smoothed.reinit(solution);
    God_grad.reinit(solution);
    pde_operator.initialize_dof_vector(velocity, 2);
  }



  template <int dim, int fe_degree>
  void
  AdvectionRIProblem<dim, fe_degree>::run()
  {
    make_grid();

    // Initialize the advection operator
    LevelSetOperation<dim, fe_degree> level_set_operator(param);

    setup_dofs();

    level_set_operator.reinit(dof_handler, dof_handler_legendre, dof_handler_vel);
    level_set_operator.initialize_dof_vector(solution, 0);
    level_set_operator.initialize_dof_vector(velocity, 2);
    level_set_operator.project_initial(solution);

    // Initialize auxiliary vectors
    LinearAlgebra::distributed::Vector<Number> num_Hamiltonian(solution);
    LinearAlgebra::distributed::Vector<Number> Signum_smoothed(solution);
    LinearAlgebra::distributed::Vector<Number> God_grad(solution);

    // Initialize timer
    Timer        timer;

    // Initialize auxiliary variables
    Number       RI_indicator = 0.;
    Number       circularity = 0.;
    Number       area = 1.;
    Number       ave_CG_iterations = 0.;
    Number       ave_cond_number = 0.;
    unsigned int timestep_RI_counter = 0;
    unsigned int RI_counter = 0;
    Number       n_RI_timestep_ave = 0.;
    Number       glob_min_vertex_distance = 0.;
    Number       glob_max_vertex_distance = 0.;
    Number       min_vertex_distance = 0.;
    Number       wtime = 0.;
    Number       output_time = 0.;
    unsigned int timestep_number = 0;
    unsigned int timestep_number_RI = 0;
    unsigned int n_output = 0;

    // Prepare for output of initial conditions
    level_set_operator.compute_local_viscosity(solution);
    area = level_set_operator.compute_area(solution, 0, 0);
    circularity = level_set_operator.compute_circularity(solution, dof_handler);

    if (param.dynamic_sim)
    {
      TransportSpeed<dim> transport_speed(time, param);
      VectorTools::interpolate(dof_handler_vel,
                              transport_speed,
                              velocity);
      level_set_operator.set_velocity_vector(velocity);
    }

    pcout << "          __         __               __    __  ___     "  << std::endl;
    pcout << "   |     |__  \\  /  |__    |     _   |__   |__   |     "  << std::endl;
    pcout << "   |__   |__   \\/   |__    |__        __|  |__   |     "  << std::endl;
    pcout << std::endl;
    pcout << "                    is running ...                      "  << std::endl;
    pcout << std::endl;
    pcout << std::endl;

    // Output of initial data
    output_results(n_output++,
                   level_set_operator.compute_mass_and_energy(solution),
                   RI_indicator,
                   n_RI_timestep_ave,
                   area,
                   circularity,
                   time_step_advection,
                   time_step_RI,
                   ave_CG_iterations,
                   ave_cond_number);

    // Initialize time integrator and Runke-Kutta registers
    LinearAlgebra::distributed::Vector<Number> rk_register_1(solution), rk_register_2(solution);
    const LowStorageRungeKuttaIntegrator time_integrator(lsrk_scheme, param);

    // update minimum cell size for reinitialization time step calculation and refinement interval calculation
    min_vertex_distance = 0.;
    glob_min_vertex_distance = 0.;
    min_vertex_distance = std::numeric_limits<Number>::max();
      for (const auto &cell : triangulation->active_cell_iterators())
    min_vertex_distance = std::min(min_vertex_distance, cell->minimum_vertex_distance());
    glob_min_vertex_distance = Utilities::MPI::min(min_vertex_distance, MPI_COMM_WORLD);

    // assume not refined initial mesh (glob_max_vertex_distance = glob_min_vertex_distance)
    glob_max_vertex_distance = glob_min_vertex_distance;

    // Compute RI distance, 
    level_set_operator.compute_RI_distance(glob_max_vertex_distance);

    // Do first level-set flattening process
    level_set_operator.flatten_level_set(solution);

    // Preparations
    level_set_operator.reinit_grad_vectors(solution);
    level_set_operator.compute_penalty_parameter();
    level_set_operator.compute_viscosity_value(glob_max_vertex_distance);

    // calculate reinitialization time step size (char. velocity ~ 1.0) for first RI time step
    if (param.use_IMEX)
      time_step_RI = param.courant_number_RI * glob_min_vertex_distance / (fe_degree * fe_degree);
    else
      time_step_RI = param.courant_number_RI / param.IP_diffusion / (std::pow(fe_degree, 2) / glob_min_vertex_distance + 
                     level_set_operator.get_viscosity_value()  *
                     std::pow(fe_degree, 4) / (glob_min_vertex_distance * glob_min_vertex_distance));

    ////////////////////////////////////////////////////////////////////////
    ///////////////////////////// Time loop ////////////////////////////////
    ////////////////////////////////////////////////////////////////////////
    while (time < param.FINAL_TIME - 1e-12)
      {
        timer.restart();

        // update transport speed interpolation from analytical function
        if (param.dynamic_sim)
        {
          {
          TimerOutput::Scope t(computing_timer, "Interpolate velocity field from analytical function");

          TransportSpeed<dim> transport_speed(time, param);
          VectorTools::interpolate(dof_handler_vel,
                                   transport_speed,
                                   velocity);

          level_set_operator.set_velocity_vector(velocity);
          }

          // update advection time step size
          {
          TimerOutput::Scope t(computing_timer, "ADVECTION: Compute advection time step size");

          time_step_advection = level_set_operator.compute_time_step_advection();

          // ensure that FINAL_TIME is reached exactly with last time step
          if (time + time_step_advection > param.FINAL_TIME)
            time_step_advection = param.FINAL_TIME - time;
          }

          time_integrator.perform_time_step(level_set_operator,
                                            time,
                                            time_step_advection,
                                            solution,
                                            rk_register_1,
                                            rk_register_2,
                                            computing_timer);
        }

        if(param.use_gradient_based_RI_indicator)
        {
          // Assert, static simulation makes no sense with gradient based RI-indicator
          AssertThrow(param.dynamic_sim == true, ExcMessage("RI-indicator is not meaningful for pure reinitialization."));

          {
          TimerOutput::Scope t(computing_timer, "RI: compute gradient-based RI indicator");

          level_set_operator.compute_RI_indicator(solution);
          RI_indicator = level_set_operator.get_RI_indicator();
          }

          if (RI_indicator > param.RI_quality_criteria)
          {
            level_set_operator.Godunov_gradient(solution, God_grad, computing_timer);
            level_set_operator.Smoothed_signum(solution, Signum_smoothed, God_grad, computing_timer, glob_max_vertex_distance);

                for (unsigned int i = 0; i < param.RI_steps; i++)
                {
                  time_integrator.perform_time_step_RI(level_set_operator,
                                                    time,
                                                    time_step_RI,
                                                    solution,
                                                    rk_register_1,
                                                    rk_register_2,
                                                    num_Hamiltonian,
                                                    Signum_smoothed,
                                                    computing_timer);

                  {
                  TimerOutput::Scope t(computing_timer, "Flatten level set field");

                  level_set_operator.flatten_level_set(solution);
                  }
                }

                RI_counter += param.RI_steps;
                timestep_number_RI += param.RI_steps;
          }
        }
        else
        {
          if (timestep_number%param.RI_interval==0)
          {
            if (param.dynamic_sim == false)
            {
              // in case of a static simulation, compute smoothed signum only once at the beginning
              if (timestep_number == 0)
              {
                level_set_operator.Godunov_gradient(solution, God_grad, computing_timer);
                level_set_operator.Smoothed_signum(solution, Signum_smoothed, God_grad, computing_timer, glob_max_vertex_distance);
              }
            }
            else
            {
              level_set_operator.Godunov_gradient(solution, God_grad, computing_timer);
              level_set_operator.Smoothed_signum(solution, Signum_smoothed, God_grad, computing_timer, glob_max_vertex_distance);
            }


                for (unsigned int i = 0; i < param.RI_steps; i++)
                {
                  time_integrator.perform_time_step_RI(level_set_operator,
                                                    time,
                                                    time_step_RI,
                                                    solution,
                                                    rk_register_1,
                                                    rk_register_2,
                                                    num_Hamiltonian,
                                                    Signum_smoothed,
                                                    computing_timer);
                  
                  {
                  TimerOutput::Scope t(computing_timer, "Flatten level set field");

                  level_set_operator.flatten_level_set(solution);
                  }
                }

                RI_counter += param.RI_steps;
                timestep_number_RI += param.RI_steps;
          }
        }

        if (param.dynamic_sim)
        {
          time += time_step_advection;
        }
        else
        {
          time += time_step_RI;
        }
        timestep_number++;
        timestep_RI_counter++;


        if (param.use_adaptive_mesh_refinement)
          {
            if ((timestep_number % param.factor_refinement_interval) == 0)
              {
                {
                TimerOutput::Scope t(computing_timer, "Compute local viscosity field for mesh refinement");
                
                // compute local viscosity value for mesh refinement criteria
                level_set_operator.compute_local_viscosity(solution); 
                }
                
                {
                TimerOutput::Scope t(computing_timer, "Do adaptive mesh refinement");

                refine_grid(level_set_operator, timestep_number, rk_register_1, rk_register_2, num_Hamiltonian, Signum_smoothed, God_grad, velocity);
                level_set_operator.reinit_grad_vectors(solution);
                level_set_operator.compute_penalty_parameter();

                // update minimum cell size for reinitialization time step calculation and refinement interval calculation
                min_vertex_distance = 0.;
                glob_min_vertex_distance = 0.;
                min_vertex_distance = std::numeric_limits<Number>::max();
                  for (const auto &cell : triangulation->active_cell_iterators())
                min_vertex_distance = std::min(min_vertex_distance, cell->minimum_vertex_distance());
                glob_min_vertex_distance = Utilities::MPI::min(min_vertex_distance, MPI_COMM_WORLD);
                }

                // in the case of a static simulation with adaptive grid refinement, update signum function after mesh refinement
                level_set_operator.Godunov_gradient(solution, God_grad, computing_timer);
                level_set_operator.Smoothed_signum(solution, Signum_smoothed, God_grad, computing_timer, glob_max_vertex_distance);

                {
                TimerOutput::Scope t(computing_timer, "RI: compute reinitialization time step size");
                
                // update reinitialization time step size (char. velocity ~ 1.0)
                if (param.use_IMEX)
                  time_step_RI = param.courant_number_RI * glob_min_vertex_distance / (fe_degree * fe_degree);
                else
                  time_step_RI = param.courant_number_RI / param.IP_diffusion / (std::pow(fe_degree, 2) / glob_min_vertex_distance + 
                                 level_set_operator.get_viscosity_value() *
                                 std::pow(fe_degree, 4) / (glob_min_vertex_distance * glob_min_vertex_distance));
                }
              }
          }

        wtime += timer.wall_time();

        timer.restart();

        if (param.dynamic_sim)
          time_step_output = time_step_advection;
        else
          time_step_output = time_step_RI;

        if (static_cast<int>(time / (param.output_tick)) !=
              static_cast<int>((time - time_step_output) / param.output_tick) ||
            time >= param.FINAL_TIME - 1e-12)
          {
            TimerOutput::Scope t(computing_timer, "Output results");

            level_set_operator.compute_local_viscosity(solution);
            level_set_operator.compute_RI_indicator(solution);
            RI_indicator = level_set_operator.get_RI_indicator();
            area = level_set_operator.compute_area(solution, 0, 0);
            circularity = level_set_operator.compute_circularity(solution, dof_handler);
            n_RI_timestep_ave = (Number)RI_counter / (Number)timestep_RI_counter;
            RI_indicator = level_set_operator.get_RI_indicator();
            ave_CG_iterations = (Number) level_set_operator.get_CG_iterations()/RI_counter;
            ave_cond_number = level_set_operator.get_cond_number()/RI_counter;

            output_results(n_output++, 
                           level_set_operator.compute_mass_and_energy(solution), 
                           RI_indicator, 
                           n_RI_timestep_ave, 
                           area, 
                           circularity, 
                           time_step_advection, 
                           time_step_RI,
                           ave_CG_iterations,
                           ave_cond_number);

            level_set_operator.reset_CG_iterations();
            level_set_operator.reset_cond_number();
            timestep_RI_counter = 0;
            RI_counter = 0;
            }
                
        output_time += timer.wall_time();
      }


    pcout << std::endl
          << "   Level-set calculation has finished successfully." << std::endl;

    pcout << std::endl;
    if (param.dynamic_sim)
    {
    pcout << "   Performed " << timestep_number << " advection time steps." << std::endl;
    }
    pcout << "   Performed " << timestep_number_RI << " reinitialization time steps." << std::endl;

    if (param.use_gradient_based_RI_indicator && param.dynamic_sim)
    {
      const Number ave_RI_steps_per_ADV_step = (Number) timestep_number_RI / timestep_number;
      pcout << "   Average RI time steps per advection time step: " << std::setprecision(5) 
      << ave_RI_steps_per_ADV_step << std::endl;
    }

    if (param.dynamic_sim)
    {
    pcout << "   Average wall clock time per advection time step: "
          << wtime / timestep_number << std::endl;
    }
    else
    {
    pcout << "   Average wall clock time per reinitialization time step: "
          << wtime / timestep_number_RI << std::endl;
    }

    pcout << "   Spent " << output_time << "s on output and " << wtime
          << "s on computations." << std::endl;

    // comment scope, if norm calculation is not required
    {
    // compute error norm in the interval around a circular interface
    const Number norm_interface = level_set_operator.compute_L2_norm_in_interface_region(solution);
    pcout << std::endl
          << "   L2-Norm in zero level interface: "
          << norm_interface << std::endl;
    }

    pcout << std::endl;

    computing_timer.print_summary();
    computing_timer.reset();
 
    pcout << std::endl;
  }
} // namespace LevelSet



int
main(int argc, char *argv[])
{
  using namespace LevelSet;
  using namespace dealii;

  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  try
    {
      ParameterHandler prm;

      LevelSetProblemParameters par;

      if (argc > 1)
        {
          par.declare_parameters(prm);
          prm.parse_input(argv[1]);
          par.get_parameters (prm);
        }

      deallog.depth_console(0);

      if (par.dimension == 2)
        {
          if (par.fe_degree == 1 )
          {
          AdvectionRIProblem<2,1> advect_problem(par);
          advect_problem.run();
          }
          else if (par.fe_degree == 2)
          {
          AdvectionRIProblem<2,2> advect_problem(par);
          advect_problem.run();
          }
          else if (par.fe_degree == 3)
          {
          AdvectionRIProblem<2,3> advect_problem(par);
          advect_problem.run();
          }
          else if (par.fe_degree == 4)
          {
          AdvectionRIProblem<2,4> advect_problem(par);
          advect_problem.run();
          }
          else if (par.fe_degree == 5)
          {
          AdvectionRIProblem<2,5> advect_problem(par);
          advect_problem.run();
          }
          else if (par.fe_degree == 6)
          {
          AdvectionRIProblem<2,6> advect_problem(par);
          advect_problem.run();
          }
          else if (par.fe_degree == 7)
          {
          AdvectionRIProblem<2,7> advect_problem(par);
          advect_problem.run();
          }
          else
          {
            AssertThrow(false, ExcNotImplemented());
          }
        }
      else if (par.dimension == 3)
        {
          if (par.fe_degree ==1 )
          {
          AdvectionRIProblem<3,1> advect_problem(par);
          advect_problem.run();
          }
          else if (par.fe_degree == 2)
          {
          AdvectionRIProblem<3,2> advect_problem(par);
          advect_problem.run();
          }
          else if (par.fe_degree == 3)
          {
          AdvectionRIProblem<3,3> advect_problem(par);
          advect_problem.run();
          }
          else if (par.fe_degree == 4)
          {
          AdvectionRIProblem<3,4> advect_problem(par);
          advect_problem.run();
          }
          else if (par.fe_degree == 5)
          {
          AdvectionRIProblem<3,5> advect_problem(par);
          advect_problem.run();
          }
          else if (par.fe_degree == 6)
          {
          AdvectionRIProblem<3,6> advect_problem(par);
          advect_problem.run();
          }
          else if (par.fe_degree == 7)
          {
          AdvectionRIProblem<3,7> advect_problem(par);
          advect_problem.run();
          }
          else
          {
            AssertThrow(false, ExcNotImplemented());
          }
        }
      else
        AssertThrow(false, ExcNotImplemented());
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
