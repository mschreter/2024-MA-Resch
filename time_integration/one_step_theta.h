#ifndef ONE_STEP_THETA_H
#define ONE_STEP_THETA_H
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/matrix_free/tools.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/config.h>
#include <ostream>

#include "../levelset_problem_parameters.h"

using namespace dealii;


template <typename Operator, int dim, int fe_degree>
class OneStepTheta
{
public:
    using Number = double;
    OneStepTheta(Operator &pde_operator,
                 const LevelSetProblemParameters &param,
                 LinearAlgebra::distributed::Vector<Number> const &);

    OneStepTheta(Operator &pde_operator,
                 const LevelSetProblemParameters &param);

    void perform_time_step([[maybe_unused]] const double current_time,
                                                               const double time_step,
                                                                LinearAlgebra::distributed::Vector<Number> &next_solution,
                                                               [[maybe_unused]] const LinearAlgebra::distributed::Vector<Number> &vec_ri,
                                                               [[maybe_unused]] const LinearAlgebra::distributed::Vector<Number> &vec_ki,
                                                               [[maybe_unused]] TimerOutput &computing_timer) const;
    void
    vmult(LinearAlgebra::distributed::Vector<Number> &dst,
          const LinearAlgebra::distributed::Vector<Number> &src) const;


    void print_iteration_number() const
    {
        pcout << "min iterations " << min_iterations << std::endl;
        pcout << "max iterations " << max_iterations << std::endl;
        pcout << "average iterations " << (double)total_iterations / ((double)counter_) << std::endl;
    }

private:
    void create_right_hand_side(bool const zero_out) const;

    void local_apply_inverse_mass_matrix(const MatrixFree<dim, Number> &data_,
                                         LinearAlgebra::distributed::Vector<Number> &dst,
                                         const LinearAlgebra::distributed::Vector<Number> &src,
                                         const std::pair<unsigned int, unsigned int> &cell_range) const;

    Operator &pde_operator_;

    double const Theta_;

    mutable double dt_;
    mutable double current_time_;
    mutable bool update_preconditioner_;
    mutable LinearAlgebra::distributed::Vector<Number> old_solution_;
    mutable LinearAlgebra::distributed::Vector<Number> right_hand_side_;
    const LevelSetProblemParameters &param_;

    ConditionalOStream pcout;
    mutable int min_iterations = std::numeric_limits<double>::infinity();
    mutable int total_iterations = 0;
    mutable int max_iterations = 0;
    mutable int counter_ = 0;


};

template <typename Operator, int dim, int fe_degree>
OneStepTheta<Operator, dim, fe_degree>::OneStepTheta(Operator &pde_operator,
                                                     const LevelSetProblemParameters &param,
                                                     LinearAlgebra::distributed::Vector<Number> const &) : pde_operator_(pde_operator),
                                                                                                           param_(param),
                                                                                                           pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
                                                                                                           Theta_(param.Theta_advection) {}

template <typename Operator, int dim, int fe_degree>
OneStepTheta<Operator, dim, fe_degree>::OneStepTheta(Operator &pde_operator,
                                                     const LevelSetProblemParameters &param) : pde_operator_(pde_operator),
                                                                                                 param_(param),
                                                                                                 pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
                                                                                                 Theta_(param.Theta_IMEX) {}

/* Left hand operator of the solver*/
template <typename Operator, int dim, int fe_degree>
void OneStepTheta<Operator, dim, fe_degree>::vmult(LinearAlgebra::distributed::Vector<Number> &dst,
                                                   const LinearAlgebra::distributed::Vector<Number> &src) const
{
    pde_operator_.apply_operator(current_time_ + dt_, dst, src);

    pde_operator_.operator_data_.data_.cell_loop(
        &OneStepTheta<Operator, dim, fe_degree>::local_apply_inverse_mass_matrix,
        this,
        dst,
        dst,
        std::function<void(const unsigned int, const unsigned int)>(),
        [&](const unsigned int start_range, const unsigned int end_range)
        {
            DEAL_II_OPENMP_SIMD_PRAGMA
            for (unsigned int i = start_range; i < end_range; ++i)
            {
                dst.local_element(i) *= -Theta_ * dt_;
                dst.local_element(i) += src.local_element(i);
            }
        });
}

/* Creates the right hand side of the solver*/
template <typename Operator, int dim, int fe_degree>
void OneStepTheta<Operator, dim, fe_degree>::create_right_hand_side(bool const zero_out) const
{
    right_hand_side_.reinit(old_solution_);
    if (zero_out)
    {
        right_hand_side_ = 0;
    }
     pde_operator_.apply_operator(current_time_, right_hand_side_, old_solution_);

    pde_operator_.operator_data_.data_.cell_loop(
        &OneStepTheta<Operator, dim, fe_degree>::local_apply_inverse_mass_matrix,
        this,
        right_hand_side_,
        right_hand_side_,
        std::function<void(const unsigned int, const unsigned int)>(),
        [&](const unsigned int start_range, const unsigned int end_range)
        {
            DEAL_II_OPENMP_SIMD_PRAGMA
            for (unsigned int i = start_range; i < end_range; ++i)
            {
                right_hand_side_.local_element(i) *= (1.0 - Theta_) * dt_;
                right_hand_side_.local_element(i) += old_solution_.local_element(i);
            }
        });
}

/* Perform one time step of the one step theta method*/
template <typename Operator, int dim, int fe_degree>
void OneStepTheta<Operator, dim, fe_degree>::perform_time_step([[maybe_unused]] const double current_time,
                                                               const double time_step,
                                                                LinearAlgebra::distributed::Vector<Number> &next_solution,
                                                               [[maybe_unused]] const LinearAlgebra::distributed::Vector<Number> &vec_ri,
                                                               [[maybe_unused]] const LinearAlgebra::distributed::Vector<Number> &vec_ki,
                                                               [[maybe_unused]] TimerOutput &computing_timer) const
{
    old_solution_ = next_solution;
    dt_ = time_step;
    current_time_ = current_time;



    create_right_hand_side(true);

    pde_operator_.set_velocity_operator(current_time_+dt_);

    //The velocity field needs only to be updated once and not in ervy call to vmult
    pde_operator_.update_velocity_ = false;

    SolverControl solver_control(1000, old_solution_.l2_norm() * 1e-9);

    // Solver must be able to handle nonsymmetry
    SolverGMRES<LinearAlgebra::distributed::Vector<Number>> solver(solver_control);
    solver.solve(*this, next_solution, right_hand_side_, PreconditionIdentity());
    int iterations = solver_control.last_step();
    min_iterations = std::min(min_iterations, iterations);
    max_iterations = std::max(max_iterations, iterations);
    total_iterations += iterations;
    counter_++;

    pde_operator_.update_velocity_ = true;
}

template <typename Operator, int dim, int fe_degree>
void OneStepTheta<Operator, dim, fe_degree>::local_apply_inverse_mass_matrix(const MatrixFree<dim, Number> &data_,
                                                                             LinearAlgebra::distributed::Vector<Number> &dst,
                                                                             const LinearAlgebra::distributed::Vector<Number> &src,
                                                                             const std::pair<unsigned int, unsigned int> &cell_range) const
{
    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval(data_, 0, 1);

    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, 1, Number> inverse(eval);

    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
        eval.reinit(cell);
        eval.read_dof_values(src);

        inverse.apply(eval.begin_dof_values(), eval.begin_dof_values());

        eval.set_dof_values(dst);
    }
}

#endif // ONE_STEP_THETA