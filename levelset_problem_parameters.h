#ifndef LEVELSET_PROBLEM_PARAMETERS_H
#define LEVELSET_PROBLEM_PARAMETERS_H
#include <deal.II/base/parameter_handler.h>
    using namespace dealii;
    class LevelSetProblemParameters
    {
    public:
        static void declare_parameters(ParameterHandler &prm);
        void get_parameters(ParameterHandler &prm);
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
        //For generalized Theta time integration scheme,
        double Theta_advection=0.5;
        double time_step_size=0.5;

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
        double art_diff_k = 2.0;     // ~ threshold
        double art_diff_kappa = 2.8; // ~ bandwidth
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
        double Theta_IMEX=1.0;

        // Use constant gradient for each reinitialization process to save computational effort?
        // (Useful expecially in explicit reinitialication time discretization.)
        bool use_const_gradient_in_RI = false;
    };
    
    void LevelSetProblemParameters::LevelSetProblemParameters::declare_parameters(ParameterHandler &prm)
    {
        prm.enter_subsection("GENERAL");
        {
            prm.declare_entry("DIMENSION", "2", Patterns::Integer());
            prm.declare_entry("FE DEGREE", "4", Patterns::Integer());
            prm.declare_entry("N GLOBAL REFINEMENTS", "6", Patterns::Integer());
            prm.declare_entry("N REFINEMENT LEVELS", "2", Patterns::Integer());
            prm.declare_entry("FINAL TIME", "4.0", Patterns::Double());
            prm.declare_entry("OUTPUT TICK", "0.2", Patterns::Double());
            prm.declare_entry("PRINT VTU", "true", Patterns::Bool());
            prm.declare_entry("FACTOR REFINEMENT INTERVAL", "8", Patterns::Integer());
            prm.declare_entry("USE ADAPTIVE MESH REFINEMENT", "true", Patterns::Bool());
            prm.declare_entry("USE CONST ELEMENT SIZE AT INTERFACE", "false", Patterns::Bool());
            prm.declare_entry("FACTOR DISTANCE CONST ELEMENT SIZE AT INTERFACE", "0.15", Patterns::Double());
            prm.declare_entry("DO EDGE ROUNDING", "true", Patterns::Bool());
            prm.declare_entry("TEST CASE", "0", Patterns::Integer());
            prm.declare_entry("DYNAMIC SIM", "true", Patterns::Bool());
        }
        prm.leave_subsection();
        prm.enter_subsection("ADVECTION");
        {
            prm.declare_entry("COURANT NUMBER ADVECTION", "0.5", Patterns::Double());
            prm.declare_entry("THETA ADVECTION", "0.5", Patterns::Double());
            prm.declare_entry("TIME STEP SIZE", "0.004", Patterns::Double());

        }
        prm.leave_subsection();
        prm.enter_subsection("REINITIALIZATION");
        {
            prm.declare_entry("RI INTERVAL", "1", Patterns::Integer());
            prm.declare_entry("FACTOR RI DISTANCE", "6.0", Patterns::Double());
            prm.declare_entry("COURANT NUMBER RI", "0.5", Patterns::Double());
            prm.declare_entry("INTERIOR PENALTY DIFFUSION", "100.0", Patterns::Double());
            prm.declare_entry("FACTOR ARTIFICIAL VISCOSITY", "0.35", Patterns::Double());
            prm.declare_entry("ART DIFF K", "1.8", Patterns::Double());
            prm.declare_entry("ART DIFF KAPPA", "3.0", Patterns::Double());
            prm.declare_entry("FACTOR REFINEMENT FLAG", "0.65", Patterns::Double());
            prm.declare_entry("FACTOR COARSEN FLAG", "0.2", Patterns::Double());
            prm.declare_entry("RI STEPS", "1", Patterns::Integer());
            prm.declare_entry("RI QUALITY CRITERIA", "0.20", Patterns::Double());
            prm.declare_entry("USE GRADIENT BASED RI INDICATOR", "true", Patterns::Bool());
            prm.declare_entry("USE IMEX", "false", Patterns::Bool());
            prm.declare_entry("Theta_IMEX", "1.0", Patterns::Double());
            prm.declare_entry("THETA IMEX", "0.5", Patterns::Double());
            prm.declare_entry("USE CONST GRADIENT IN RI", "true", Patterns::Bool());
        }
        prm.leave_subsection();
    }
    void LevelSetProblemParameters::LevelSetProblemParameters::get_parameters(ParameterHandler &prm)
    {
        prm.enter_subsection("GENERAL");
        {
            dimension = prm.get_integer("DIMENSION");
            fe_degree = prm.get_integer("FE DEGREE");
            n_global_refinements = prm.get_integer("N GLOBAL REFINEMENTS");
            n_refinement_levels = prm.get_integer("N REFINEMENT LEVELS");
            FINAL_TIME = prm.get_double("FINAL TIME");
            output_tick = prm.get_double("OUTPUT TICK");
            print_vtu = prm.get_bool("PRINT VTU");
            factor_refinement_interval = prm.get_integer("FACTOR REFINEMENT INTERVAL");
            use_adaptive_mesh_refinement = prm.get_bool("USE ADAPTIVE MESH REFINEMENT");
            use_const_element_size_at_interface = prm.get_bool("USE CONST ELEMENT SIZE AT INTERFACE");
            factor_distance_const_element_size_at_interface = prm.get_double("FACTOR DISTANCE CONST ELEMENT SIZE AT INTERFACE");
            do_edge_rounding = prm.get_bool("DO EDGE ROUNDING");
            test_case = prm.get_integer("TEST CASE");
            dynamic_sim = prm.get_bool("DYNAMIC SIM");
        }
        prm.leave_subsection();
        prm.enter_subsection("ADVECTION");
        {
            courant_number_advection = prm.get_double("COURANT NUMBER ADVECTION");
            Theta_advection = prm.get_double("THETA ADVECTION");
            time_step_size = prm.get_double("TIME STEP SIZE");

        }
        prm.leave_subsection();
        prm.enter_subsection("REINITIALIZATION");
        {
            RI_interval = prm.get_integer("RI INTERVAL");
            factor_RI_distance = prm.get_double("FACTOR RI DISTANCE");
            courant_number_RI = prm.get_double("COURANT NUMBER RI");
            IP_diffusion = prm.get_double("INTERIOR PENALTY DIFFUSION");
            factor_diffusivity = prm.get_double("FACTOR ARTIFICIAL VISCOSITY");
            art_diff_k = prm.get_double("ART DIFF K");
            art_diff_kappa = prm.get_double("ART DIFF KAPPA");
            factor_refinement_flag = prm.get_double("FACTOR REFINEMENT FLAG");
            factor_coarsen_flag = prm.get_double("FACTOR COARSEN FLAG");
            RI_steps = prm.get_integer("RI STEPS");
            RI_quality_criteria = prm.get_double("RI QUALITY CRITERIA");
            use_gradient_based_RI_indicator = prm.get_bool("USE GRADIENT BASED RI INDICATOR");
            use_IMEX = prm.get_bool("USE IMEX");
            Theta_IMEX = prm.get_double("THETA IMEX");

            use_const_gradient_in_RI = prm.get_bool("USE CONST GRADIENT IN RI");
        }
        prm.leave_subsection();
    }
#endif // LEVELSET_PROBLEM_PARAMETERS_H