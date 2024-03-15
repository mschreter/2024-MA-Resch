#ifndef REINIZILIZATION_OPERATOR_H
#define REINIZILIZATION_OPERATOR_H

#include "operator_data.h"
#include "operator_base.h"
#include "../time_integration/one_step_theta.h"
#include "ri_diffusion_operator.h"
#include "ri_grad_operator.h"
#include "advection_operator.h"

#include "../levelset_problem_parameters.h"

template <int dim, int fe_degree>
class ReinitilizationOperator : public OperatorBase<dim, fe_degree>
{
public:
  using Number = double;

  ReinitilizationOperator(OperatorData<dim, fe_degree> &operator_data,
                          LevelSetProblemParameters const &param,
                          RIDiffusionOperator<dim, fe_degree> &RI_diffusion_operator,
                          RIGradOperator<dim, fe_degree> &RI_grad_operator)
      : OperatorBase<dim, fe_degree>(operator_data),
        param_(param),
        RI_diffusion_operator(RI_diffusion_operator),
        RI_grad_operator(RI_grad_operator),
        implicit_time_integrator(RI_diffusion_operator, param)
  {
  }

  void apply_operator(double const time, LinearAlgebra::distributed::Vector<Number> &dst, LinearAlgebra::distributed::Vector<Number> const &src);

  void prepare_reinitilization(double const time, double const time_step, LinearAlgebra::distributed::Vector<Number> &solution, const double max_vertex_distance,TimerOutput &computing_timer);

  void reinit_vectors()
  {
    initialize_dof_vector(num_Hamiltonian, 0);
    initialize_dof_vector(Signum_smoothed, 0);
    initialize_dof_vector(God_grad, 0);
    initialize_dof_vector(grad_x_l, 0);
    initialize_dof_vector(grad_x_r, 0);
    initialize_dof_vector(grad_y_l, 0);
    initialize_dof_vector(grad_y_r, 0);

    if (dim == 3)
    {
      initialize_dof_vector(grad_z_l, 0);
      initialize_dof_vector(grad_z_r, 0);
    }
  }

  void
  compute_RI_indicator(const LinearAlgebra::distributed::Vector<Number> &sol);

  const Number &
  get_RI_indicator() const;

  void
  compute_viscosity_value(const double vertex_distance);

  void
  compute_RI_distance(const double vertex_distance);

  void
  compute_penalty_parameter();

  void compare(const LinearAlgebra::distributed::Vector<Number> &seins, const LinearAlgebra::distributed::Vector<Number> &meins)
  {
    for (int i = 0; i < seins.locally_owned_size(); i++)
    {
      std::cout << "seins " << std::setprecision(16) << seins.local_element(i) << std::endl;
      std::cout << "meins " << std::setprecision(16) << meins.local_element(i) << std::endl;
    }
  }

  void
  Godunov_Hamiltonian(const LinearAlgebra::distributed::Vector<Number> &solution);

  // auxiliary vectors for Godunov's scheme
  mutable LinearAlgebra::distributed::Vector<Number> grad_x_l;
  mutable LinearAlgebra::distributed::Vector<Number> grad_x_r;
  mutable LinearAlgebra::distributed::Vector<Number> grad_y_l;
  mutable LinearAlgebra::distributed::Vector<Number> grad_y_r;
  mutable LinearAlgebra::distributed::Vector<Number> grad_z_l;
  mutable LinearAlgebra::distributed::Vector<Number> grad_z_r;

  //
  mutable LinearAlgebra::distributed::Vector<Number> num_Hamiltonian;
  mutable LinearAlgebra::distributed::Vector<Number> Signum_smoothed;
  mutable LinearAlgebra::distributed::Vector<Number> God_grad;

private:
  LevelSetProblemParameters const &param_;

  mutable Number RI_ind = 0.;
  mutable Number RI_distance = 0.;

  // operators
  RIDiffusionOperator<dim, fe_degree> &RI_diffusion_operator;
  RIGradOperator<dim, fe_degree> &RI_grad_operator;

  OneStepTheta<RIDiffusionOperator<dim, fe_degree>, dim, fe_degree> implicit_time_integrator;

  void
  local_apply_inverse_mass_matrix(
      const MatrixFree<dim, Number> &data,
      LinearAlgebra::distributed::Vector<Number> &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

  void
  apply_Hamiltonian(LinearAlgebra::distributed::Vector<Number> &num_Hamiltonian,
                    const LinearAlgebra::distributed::Vector<Number> &Signum_smoothed,
                    const LinearAlgebra::distributed::Vector<Number> &solution) const;

  void
  local_apply_domain_num_Hamiltonian(
      const MatrixFree<dim, Number> &data,
      LinearAlgebra::distributed::Vector<Number> &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

  void
  initialize_dof_vector(LinearAlgebra::distributed::Vector<Number> &vec,
                        const unsigned int dof_handler_index)
  {
    this->operator_data_.data_.initialize_dof_vector(vec, dof_handler_index);
  }

  void
  Godunov_gradient(const LinearAlgebra::distributed::Vector<Number> &solution);

  void
  Smoothed_signum(const LinearAlgebra::distributed::Vector<Number> &solution,

                  const uint max_vertex_distance) const;
};

template <int dim, int fe_degree>
void ReinitilizationOperator<dim, fe_degree>::apply_operator(double const time, LinearAlgebra::distributed::Vector<Number> &dst, LinearAlgebra::distributed::Vector<Number> const &src)
{
  Godunov_Hamiltonian(src);

  this->operator_data_.data_.cell_loop(&ReinitilizationOperator<dim, fe_degree>::local_apply_domain_num_Hamiltonian,
                                       this,
                                       dst,
                                       num_Hamiltonian,
                                       true);
  
      if(param_.use_IMEX == false)
    {
      RI_diffusion_operator.apply_operator(time, num_Hamiltonian, src);
      dst.add(1.0, num_Hamiltonian);
    }
}

template <int dim, int fe_degree>
void ReinitilizationOperator<dim, fe_degree>::prepare_reinitilization(double const time, double const time_step, LinearAlgebra::distributed::Vector<Number> &solution, const double max_vertex_distance, TimerOutput &computing_timer)
{
  Godunov_gradient(solution);
  Smoothed_signum(solution, max_vertex_distance);

  LinearAlgebra::distributed::Vector<Number> b;
  if (param_.use_IMEX)
  {
    implicit_time_integrator.perform_time_step(time,
                                               time_step,
                                               solution,
                                               b,
                                               b,
                                               computing_timer);
  }
}

template <int dim, int fe_degree>
void ReinitilizationOperator<dim, fe_degree>::compute_RI_indicator(const LinearAlgebra::distributed::Vector<Number> &sol)
{
  auto &data = this->operator_data_.data_;
  Number u = 0.;
  Number v = 0.;

  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> phi(data);

  for (unsigned int cell = 0; cell < data.n_cell_batches(); ++cell)
  {
    phi.reinit(cell);
    phi.gather_evaluate(sol, EvaluationFlags::values | EvaluationFlags::gradients);

    // Depending on the cell number, there might be empty lanes
    const unsigned int n_lanes_filled = data.n_active_entries_per_cell_batch(cell);

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

  RI_ind = u / (v + 0.0001);
}

template <int dim, int fe_degree>
const double &
ReinitilizationOperator<dim, fe_degree>::get_RI_indicator() const
{
  return RI_ind;
}

template <int dim, int fe_degree>
void ReinitilizationOperator<dim, fe_degree>::compute_viscosity_value(const double vertex_distance)
{
  RI_diffusion_operator.compute_viscosity_value(vertex_distance);
}

template <int dim, int fe_degree>
void ReinitilizationOperator<dim, fe_degree>::compute_RI_distance(const double vertex_distance)
{
  // Compute the reinitialization distance
  RI_distance = vertex_distance / fe_degree * param_.factor_RI_distance;
}

template <int dim, int fe_degree>
void ReinitilizationOperator<dim, fe_degree>::compute_penalty_parameter()
{
  RI_diffusion_operator.compute_penalty_parameter();
}

template <int dim, int fe_degree>
void ReinitilizationOperator<dim, fe_degree>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, Number> &data,
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
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
void ReinitilizationOperator<dim, fe_degree>::Godunov_Hamiltonian(
    const LinearAlgebra::distributed::Vector<Number> &solution)
{
  {

    // The upwind and downwind gradients are already calculated for the smoothed signum function.
    // The user has the choice if he want's to update the gradient vectors in every Runge-Kutta stage.

    // Hint: Especially for explicit reinitialization time stepping use_const_gradient_in_RI = is useful
    // to save computational effort. As rule of thumb, up to 10 explicit reinitialization time steps, the
    // accuracy is not deteriorated due to this simplification.
    if (param_.use_const_gradient_in_RI == false)
    {
      double const dummy_time = 0.0;
      // x-direction
      RI_grad_operator.template apply_operator<false, 0>(dummy_time, grad_x_l, solution);
      RI_grad_operator.template apply_operator<true, 0>(dummy_time, grad_x_r, solution);
      // y-direction
      RI_grad_operator.template apply_operator<false, 1>(dummy_time, grad_y_l, solution);
      RI_grad_operator.template apply_operator<true, 1>(dummy_time, grad_y_r, solution);

      if (dim == 3)
      {
        // z-direction
        RI_grad_operator.template apply_operator<false, 2>(dummy_time, grad_z_l, solution);
        RI_grad_operator.template apply_operator<true, 2>(dummy_time, grad_z_r, solution);
      }
    }
  }

  {

    // calculate the numerical Hamiltonian with Godunov's method
    apply_Hamiltonian(num_Hamiltonian, Signum_smoothed, solution);
  }
}

template <int dim, int fe_degree>
void ReinitilizationOperator<dim, fe_degree>::apply_Hamiltonian(
    LinearAlgebra::distributed::Vector<Number> &num_Hamiltonian,
    const LinearAlgebra::distributed::Vector<Number> &Signum_smoothed,
    const LinearAlgebra::distributed::Vector<Number> &solution) const
{
  auto &data = this->operator_data_.data_;

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

        if (param_.do_edge_rounding)
        {
          // definition of the target gradient
          gradient_goal = 1. * 4. / ((std::exp(6. * (1. - std::abs(phi_sol.get_dof_value(q)) / RI_distance)) + std::exp(-6. * (1. - std::abs(phi_sol.get_dof_value(q)) / RI_distance))) * (std::exp(6. * (1. - std::abs(phi_sol.get_dof_value(q)) / RI_distance)) + std::exp(-6. * (1. - std::abs(phi_sol.get_dof_value(q)) / RI_distance))));

          gradient_goal = compare_and_apply_mask<SIMDComparison::less_than>(std::abs(phi_sol.get_dof_value(q)),
                                                                            RI_distance,
                                                                            1.,
                                                                            gradient_goal);
        }

        auto u = compare_and_apply_mask<SIMDComparison::greater_than>(phi_sign_mod.get_dof_value(q),
                                                                      0.,

                                                                      (std::sqrt(

                                                                           std::max((std::min(phi_grad_x_l.get_dof_value(q), zero_vector)) * (std::min(phi_grad_x_l.get_dof_value(q), zero_vector)),
                                                                                    (std::max(phi_grad_x_r.get_dof_value(q), zero_vector)) * (std::max(phi_grad_x_r.get_dof_value(q), zero_vector)))

                                                                           + std::max((std::min(phi_grad_y_l.get_dof_value(q), zero_vector)) * (std::min(phi_grad_y_l.get_dof_value(q), zero_vector)),
                                                                                      (std::max(phi_grad_y_r.get_dof_value(q), zero_vector)) * (std::max(phi_grad_y_r.get_dof_value(q), zero_vector)))) -
                                                                       gradient_goal) *
                                                                          phi_sign_mod.get_dof_value(q),
                                                                      (std::sqrt(

                                                                           std::max((std::max(phi_grad_x_l.get_dof_value(q), zero_vector)) * (std::max(phi_grad_x_l.get_dof_value(q), zero_vector)),
                                                                                    (std::min(phi_grad_x_r.get_dof_value(q), zero_vector)) * (std::min(phi_grad_x_r.get_dof_value(q), zero_vector)))

                                                                           + std::max((std::max(phi_grad_y_l.get_dof_value(q), zero_vector)) * (std::max(phi_grad_y_l.get_dof_value(q), zero_vector)),
                                                                                      (std::min(phi_grad_y_r.get_dof_value(q), zero_vector)) * (std::min(phi_grad_y_r.get_dof_value(q), zero_vector)))) -
                                                                       gradient_goal) *
                                                                          phi_sign_mod.get_dof_value(q));

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

        if (param_.do_edge_rounding)
        {
          // definition of the target gradient
          gradient_goal = 1. * 4. / ((std::exp(6. * (1. - std::abs(phi_sol.get_dof_value(q)) / RI_distance)) + std::exp(-6. * (1. - std::abs(phi_sol.get_dof_value(q)) / RI_distance))) * (std::exp(6. * (1. - std::abs(phi_sol.get_dof_value(q)) / RI_distance)) + std::exp(-6. * (1. - std::abs(phi_sol.get_dof_value(q)) / RI_distance))));

          gradient_goal = compare_and_apply_mask<SIMDComparison::less_than>(std::abs(phi_sol.get_dof_value(q)),
                                                                            RI_distance,
                                                                            1.,
                                                                            gradient_goal);
        }

        auto u = compare_and_apply_mask<SIMDComparison::greater_than>(phi_sign_mod.get_dof_value(q), 0.,

                                                                      (std::sqrt(

                                                                           std::max((std::min(phi_grad_x_l.get_dof_value(q), zero_vector)) * (std::min(phi_grad_x_l.get_dof_value(q), zero_vector)),
                                                                                    (std::max(phi_grad_x_r.get_dof_value(q), zero_vector)) * (std::max(phi_grad_x_r.get_dof_value(q), zero_vector)))

                                                                           + std::max((std::min(phi_grad_y_l.get_dof_value(q), zero_vector)) * (std::min(phi_grad_y_l.get_dof_value(q), zero_vector)),
                                                                                      (std::max(phi_grad_y_r.get_dof_value(q), zero_vector)) * (std::max(phi_grad_y_r.get_dof_value(q), zero_vector)))

                                                                           + std::max((std::min(phi_grad_z_l.get_dof_value(q), zero_vector)) * (std::min(phi_grad_z_l.get_dof_value(q), zero_vector)),
                                                                                      (std::max(phi_grad_z_r.get_dof_value(q), zero_vector)) * (std::max(phi_grad_z_r.get_dof_value(q), zero_vector)))) -
                                                                       gradient_goal) *
                                                                          phi_sign_mod.get_dof_value(q),
                                                                      (std::sqrt(

                                                                           std::max((std::max(phi_grad_x_l.get_dof_value(q), zero_vector)) * (std::max(phi_grad_x_l.get_dof_value(q), zero_vector)),
                                                                                    (std::min(phi_grad_x_r.get_dof_value(q), zero_vector)) * (std::min(phi_grad_x_r.get_dof_value(q), zero_vector)))

                                                                           + std::max((std::max(phi_grad_y_l.get_dof_value(q), zero_vector)) * (std::max(phi_grad_y_l.get_dof_value(q), zero_vector)),
                                                                                      (std::min(phi_grad_y_r.get_dof_value(q), zero_vector)) * (std::min(phi_grad_y_r.get_dof_value(q), zero_vector)))

                                                                           + std::max((std::max(phi_grad_z_l.get_dof_value(q), zero_vector)) * (std::max(phi_grad_z_l.get_dof_value(q), zero_vector)),
                                                                                      (std::min(phi_grad_z_r.get_dof_value(q), zero_vector)) * (std::min(phi_grad_z_r.get_dof_value(q), zero_vector)))) -
                                                                       gradient_goal) *
                                                                          phi_sign_mod.get_dof_value(q));

        phi_grad_x_l.submit_dof_value(u, q);
      }

      phi_grad_x_l.set_dof_values(num_Hamiltonian);
    }
  }
}

template <int dim, int fe_degree>
void ReinitilizationOperator<dim, fe_degree>::local_apply_domain_num_Hamiltonian(
    const MatrixFree<dim, Number> &data,
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
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
void ReinitilizationOperator<dim, fe_degree>::Godunov_gradient(
    const LinearAlgebra::distributed::Vector<Number> &solution)
{
  auto &data = this->operator_data_.data_;

  {

    double const dummy_time = 0.0;
    // compute local upwind and downwind gradients
    //  x-direction
    RI_grad_operator.template apply_operator<false, 0>(dummy_time, grad_x_l, solution);
    RI_grad_operator.template apply_operator<true, 0>(dummy_time, grad_x_r, solution);
    // y-direction
    RI_grad_operator.template apply_operator<false, 1>(dummy_time, grad_y_l, solution);
    RI_grad_operator.template apply_operator<true, 1>(dummy_time, grad_y_r, solution);

    if (dim == 3)
    {
      // z-direction
      RI_grad_operator.template apply_operator<false, 2>(dummy_time, grad_z_l, solution);
      RI_grad_operator.template apply_operator<true, 2>(dummy_time, grad_z_r, solution);
    }
  }

  {

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

                                                                            std::max((std::min(phi_grad_x_l.get_dof_value(q), zero_vector)) * (std::min(phi_grad_x_l.get_dof_value(q), zero_vector)),
                                                                                     (std::max(phi_grad_x_r.get_dof_value(q), zero_vector)) * (std::max(phi_grad_x_r.get_dof_value(q), zero_vector)))

                                                                            + std::max((std::min(phi_grad_y_l.get_dof_value(q), zero_vector)) * (std::min(phi_grad_y_l.get_dof_value(q), zero_vector)),
                                                                                       (std::max(phi_grad_y_r.get_dof_value(q), zero_vector)) * (std::max(phi_grad_y_r.get_dof_value(q), zero_vector)))),
                                                                        std::sqrt(

                                                                            std::max((std::max(phi_grad_x_l.get_dof_value(q), zero_vector)) * (std::max(phi_grad_x_l.get_dof_value(q), zero_vector)),
                                                                                     (std::min(phi_grad_x_r.get_dof_value(q), zero_vector)) * (std::min(phi_grad_x_r.get_dof_value(q), zero_vector)))

                                                                            + std::max((std::max(phi_grad_y_l.get_dof_value(q), zero_vector)) * (std::max(phi_grad_y_l.get_dof_value(q), zero_vector)),
                                                                                       (std::min(phi_grad_y_r.get_dof_value(q), zero_vector)) * (std::min(phi_grad_y_r.get_dof_value(q), zero_vector)))));
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

                                                                            std::max((std::min(phi_grad_x_l.get_dof_value(q), zero_vector)) * (std::min(phi_grad_x_l.get_dof_value(q), zero_vector)),
                                                                                     (std::max(phi_grad_x_r.get_dof_value(q), zero_vector)) * (std::max(phi_grad_x_r.get_dof_value(q), zero_vector)))

                                                                            + std::max((std::min(phi_grad_y_l.get_dof_value(q), zero_vector)) * (std::min(phi_grad_y_l.get_dof_value(q), zero_vector)),
                                                                                       (std::max(phi_grad_y_r.get_dof_value(q), zero_vector)) * (std::max(phi_grad_y_r.get_dof_value(q), zero_vector)))

                                                                            + std::max((std::min(phi_grad_z_l.get_dof_value(q), zero_vector)) * (std::min(phi_grad_z_l.get_dof_value(q), zero_vector)),
                                                                                       (std::max(phi_grad_z_r.get_dof_value(q), zero_vector)) * (std::max(phi_grad_z_r.get_dof_value(q), zero_vector)))),
                                                                        std::sqrt(

                                                                            std::max((std::max(phi_grad_x_l.get_dof_value(q), zero_vector)) * (std::max(phi_grad_x_l.get_dof_value(q), zero_vector)),
                                                                                     (std::min(phi_grad_x_r.get_dof_value(q), zero_vector)) * (std::min(phi_grad_x_r.get_dof_value(q), zero_vector)))

                                                                            + std::max((std::max(phi_grad_y_l.get_dof_value(q), zero_vector)) * (std::max(phi_grad_y_l.get_dof_value(q), zero_vector)),
                                                                                       (std::min(phi_grad_y_r.get_dof_value(q), zero_vector)) * (std::min(phi_grad_y_r.get_dof_value(q), zero_vector)))

                                                                            + std::max((std::max(phi_grad_z_l.get_dof_value(q), zero_vector)) * (std::max(phi_grad_z_l.get_dof_value(q), zero_vector)),
                                                                                       (std::min(phi_grad_z_r.get_dof_value(q), zero_vector)) * (std::min(phi_grad_z_r.get_dof_value(q), zero_vector)))));
          phi_grad_x_l.submit_dof_value(u, q);
        }

        phi_grad_x_l.set_dof_values(God_grad);
      }
    }
  }
}

template <int dim, int fe_degree>
void ReinitilizationOperator<dim, fe_degree>::Smoothed_signum(
    const LinearAlgebra::distributed::Vector<Number> &solution,
    const uint max_vertex_distance) const
{
  auto &data = this->operator_data_.data_;
  {

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
        // tanh(x)=(e^x-e^(-x))/(e^x+e^(-x)), tanh(x) is not supported for VectorizedArray
        const auto u = (std::exp(arg) - std::exp(-arg)) / (std::exp(arg) + std::exp(-arg));

        source.submit_dof_value(u, q);
      }

      source.set_dof_values(Signum_smoothed);
    }
  }
}

#endif // REINIZILIZATION_OPERATOR_H