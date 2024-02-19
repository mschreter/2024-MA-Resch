#ifndef ONE_STEP_THETA_H
#define ONE_STEP_THETA_H

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>

using namespace dealii;

template <typename Operator>
class OneStepTheta
{
public:
    using Number = double;
    OneStepTheta(Operator &pde_operator, LinearAlgebra::distributed::Vector<Number> &solution_n);

    template <typename VectorType>
    void UpdateBuffers(VectorType &solution) const;

    template <typename VectorType>
    void perform_time_step(double const time_step_size,
                           VectorType &next_solutions) const;

    void
    vmult(LinearAlgebra::distributed::Vector<Number> &dst,
          const LinearAlgebra::distributed::Vector<Number> &src) const;

    template <typename VectorType>
    void righthandside(VectorType &right_hand_side,
                       VectorType &current_solution) const;

private:
    const double theta_ = 0.5;
    Operator const &pde_operator_;
    mutable double dt;
    mutable LinearAlgebra::distributed::Vector<Number> solution_n_;
};

///// @brief Constructor of the one step theta scheme
///// @tparam Operator
///// @param pde_operator
///// @param solution_n
template <typename Operator>
OneStepTheta<Operator>::OneStepTheta(Operator &pde_operator, LinearAlgebra::distributed::Vector<Number> &solution_n) : pde_operator_(pde_operator),
                                                                                                                       solution_n_(solution_n) {}

///// @brief Left hand operator of the solver
///// @tparam Operator
///// @param dst
///// @param src
///// @note Operator is phi(u+1)-Theta*(dt*M^-1*R(phi(u+1)))
template <typename Operator>
void OneStepTheta<Operator>::vmult(LinearAlgebra::distributed::Vector<Number> &dst,
                                   const LinearAlgebra::distributed::Vector<Number> &src) const
{
    pde_operator_.apply_advection_operator(dst, src);
    dst *= -1.0;
    dst *= theta_ * dt;
    dst.add(1.0, src);
}

///// @brief Creates the right hand side of the solver
///// @tparam Operator
///// @param right_hand_side
///// @param current_solution
///// @note Operator is (1-Theta)*(dt*M^-1*R(phi(u)))+u(n)
template <typename Operator>
template <typename VectorType>
void OneStepTheta<Operator>::righthandside(VectorType &right_hand_side,
                                           VectorType &current_solution) const
{
    pde_operator_.apply_advection_operator(right_hand_side, current_solution);
    right_hand_side *= (1.0 - theta_) * dt;
    right_hand_side.add(1.0, current_solution);
}

///// @brief Perform one time step of the one step theta method
///// @tparam Operator
///// @param time_step_size
///// @param next_solution
template <typename Operator>
template <typename VectorType>
void OneStepTheta<Operator>::perform_time_step(double const time_step_size,
                                               VectorType &next_solution) const
{
    dt = time_step_size;

    LinearAlgebra::distributed::Vector<Number> right_hand_side(next_solution);
    righthandside(right_hand_side, next_solution);
    SolverControl solver_control(1000, solution_n_.l2_norm() * 1e-8);

    // Solver must be able to handle nonsymmetry
    SolverGMRES<VectorType> solver(solver_control);
    solver.solve(*this, next_solution, right_hand_side, PreconditionIdentity());
}

///// @brief Updates the buffers of the integration scheme
///// @tparam Operator
///// @param time_step_size
///// @param next_solution
template <typename Operator>
template <typename VectorType>
void OneStepTheta<Operator>::UpdateBuffers(VectorType &solution) const
{
    // Reinit is necessary, because mesh could have changed
    solution_n_.reinit(solution);
    solution_n_ = solution;
}

#endif // ONE_STEP_THETA