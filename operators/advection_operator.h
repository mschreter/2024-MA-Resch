#ifndef ADVECTION_OPERATOR_H
#define ADVECTION_OPERATOR_H

#include "operator_data.h"
#include "../levelset_problem_parameters.h"
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/matrix_free/operators.h>
#include "operator_base.h"

// Analytical function for the transport speed
template <int dim>
class TransportSpeed : public Function<dim>
{
public:
  TransportSpeed(const double time, const LevelSetProblemParameters &param)
      : Function<dim>(dim, time), param(param)
  {
  }

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

private:
  const LevelSetProblemParameters &param;
};

template <int dim>
double TransportSpeed<dim>::value(const Point<dim> &p,
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
        double const sin = std::sin(numbers::PI * p[0]);
        double const sin_squared = sin * sin;
        return factor * (std::sin(2 * numbers::PI * p[1]) *
                         sin_squared);
      }
      else if (component == 1)
      {
        double const sin = std::sin(numbers::PI * p[1]);
        double const sin_squared = sin * sin;
        return -factor * (std::sin(2 * numbers::PI * p[0]) *
                          sin_squared);
      }

      break;
    }

    case 1:
    {
      // slotted disk test case
      if (component == 0)
      {
        return numbers::PI * (0.5 - p[1]) / numbers::PI;
      }
      else if (component == 1)
      {
        return numbers::PI * (p[0] - 0.5) / numbers::PI;
      }

      break;
    }

    case 2:
    {
      // Poiseuille flow test case (channel flow) with modified periodic velocity field
      if (component == 0)
      {
        const double factor = std::sin(numbers::PI * t / param.FINAL_TIME);
        return 1. * (1. - (1. - factor) * (p[1] - 0.5) * (p[1] - 0.5) / (0.5 * 0.5));
      }
      else if (component == 1)
      {
        return 0.;
      }

      break;
    }

    case 8:
    {
      // Bubble Rotated
      if (component == 0)
      {
        return numbers::PI * 2.0 * (0.5 - p[1]);
      }
      else if (component == 1)
      {
        return numbers::PI * 2.0 * (p[0] - 0.5);
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
        return 2. * factor * (std::sin(2 * numbers::PI * p[1]) * std::sin(2 * numbers::PI * p[2]) * std::sin(numbers::PI * p[0]) * std::sin(numbers::PI * p[0]));
      }
      else if (component == 1)
      {
        return -factor * (std::sin(2 * numbers::PI * p[0]) *
                          std::sin(2 * numbers::PI * p[2]) *
                          std::sin(numbers::PI * p[1]) *
                          std::sin(numbers::PI * p[1]));
      }
      else if (component == 2)
      {
        return -factor * (std::sin(2 * numbers::PI * p[0]) *
                          std::sin(2 * numbers::PI * p[1]) *
                          std::sin(numbers::PI * p[2]) *
                          std::sin(numbers::PI * p[2]));
      }

      break;
    }

    case 7:
    {
      // Hagen-Poiseuille flow test case (channel flow), (domain: [0,1]x[0,1])
      double const x0 = 0.5;
      double const y0 = 0.5;

      double const R = 0.15;
      double const nu = 0.01;
      double const dpdx = 0.5;

      if (component == 0)
      {
        return 0.0;
      }

      if (component == 1)
      {
        return 0.0;
      }

      if (component == 2)
      {
        double const r2 = (p[0] - x0) * (p[0] - x0) + (p[1] - y0) * (p[1] - y0);

        if (r2 <= R * R)
        {
          return -R * R * dpdx / (4.0 * nu) * (1.0 - r2 / R / R);
        }
        else
        {
          return 0.0;
        }

        break;
      }
    }

    default:
      AssertThrow(false, ExcNotImplemented());
    }
  }

  return 0;
}
using Number = double;
template <int dim, int fe_degree>
class AdvectionOperator : public OperatorBase<dim, fe_degree>
{
public:
  using Number = double;

  AdvectionOperator(OperatorData<dim, fe_degree> &operator_data, LevelSetProblemParameters const &param) : OperatorBase<dim, fe_degree>(operator_data), param_(param)
  {
  }

  void reinit();
  void set_velocity_operator(double const time) const
  {

    TransportSpeed<dim> transport_speed(time, param_);

    VectorTools::interpolate(this->operator_data_.dof_handler_vel_,
                             transport_speed,
                             velocity_operator_);

    velocity_operator_.update_ghost_values();
  }

  void apply_operator(double const time, LinearAlgebra::distributed::Vector<Number> &dst, LinearAlgebra::distributed::Vector<Number> const &src) const;

  void
  compute_RI_distance(const double vertex_distance)
  {
    // Compute the reinitialization distance
    RI_distance = vertex_distance / fe_degree * param_.factor_RI_distance;
  }

      private :

      mutable LinearAlgebra::distributed::Vector<Number>
          velocity_operator_;
  LevelSetProblemParameters const &param_;

      void
      local_apply_domain(
          const MatrixFree<dim, Number> &data,
          LinearAlgebra::distributed::Vector<Number> &dst,
          const LinearAlgebra::distributed::Vector<Number> &src,
          const std::pair<unsigned int, unsigned int> &cell_range) const;

  void
  local_apply_inner_face(
      const MatrixFree<dim, Number> &data,
      LinearAlgebra::distributed::Vector<Number> &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

  void
  local_apply_boundary_face(
      const MatrixFree<dim, Number> &data_,
      LinearAlgebra::distributed::Vector<Number> &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

  void local_apply_inverse_mass_matrix(
      const MatrixFree<dim, Number> &data_,
      LinearAlgebra::distributed::Vector<Number> &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

      double RI_distance = 0.0;
};

template <int dim, int fe_degree>
void AdvectionOperator<dim, fe_degree>::reinit()
{
  this->operator_data_.data_.initialize_dof_vector(velocity_operator_, 2);
}

template <int dim, int fe_degree>
void AdvectionOperator<dim, fe_degree>::local_apply_domain(
    const MatrixFree<dim, Number> &data,
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
{
  FEEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval(data);
  FEEvaluation<dim, fe_degree, fe_degree + 1, dim, Number> eval_vel(data, 2);

  for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    eval.reinit(cell);
    eval_vel.reinit(cell);

    eval.gather_evaluate(src, EvaluationFlags::values);
    eval_vel.gather_evaluate(velocity_operator_, EvaluationFlags::values);

    for (unsigned int q = 0; q < eval.n_q_points; ++q)
    {
      const auto speed = eval_vel.get_value(q);
      const auto u = eval.get_value(q);
      const auto flux = speed * u;
      eval.submit_gradient(flux, q);
    }

    eval.integrate_scatter(EvaluationFlags::gradients, dst);
  }
}

template <int dim, int fe_degree>
void AdvectionOperator<dim, fe_degree>::local_apply_inner_face(
    const MatrixFree<dim, Number> &data,
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
{
  FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_minus(data, true);
  FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_plus(data, false);
  FEFaceEvaluation<dim, fe_degree, fe_degree + 1, dim, Number> eval_vel(data, true, 2);

  for (unsigned int face = face_range.first; face < face_range.second; face++)
  {
    eval_minus.reinit(face);
    eval_plus.reinit(face);
    eval_vel.reinit(face);

    eval_minus.gather_evaluate(src, EvaluationFlags::values);
    eval_plus.gather_evaluate(src, EvaluationFlags::values);
    eval_vel.gather_evaluate(velocity_operator_, EvaluationFlags::values);

    for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
    {
      const auto speed = eval_vel.get_value(q);
      const auto u_minus = eval_minus.get_value(q);
      const auto u_plus = eval_plus.get_value(q);
      const auto normal_vector_minus = eval_minus.get_normal_vector(q);

      const auto normal_times_speed = speed * normal_vector_minus;
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
void AdvectionOperator<dim, fe_degree>::local_apply_boundary_face(
    const MatrixFree<dim, Number> &data_,
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
{
  FEFaceEvaluation<dim, fe_degree, fe_degree + 1, 1, Number> eval_minus(data_, true);
  FEFaceEvaluation<dim, fe_degree, fe_degree + 1, dim, Number> eval_vel(data_, true, 2);

  for (unsigned int face = face_range.first; face < face_range.second; face++)
  {
    eval_minus.reinit(face);
    eval_minus.gather_evaluate(src, EvaluationFlags::values);
    eval_vel.reinit(face);
    eval_vel.gather_evaluate(velocity_operator_, EvaluationFlags::values);

    for (unsigned int q = 0; q < eval_minus.n_q_points; ++q)
    {
      const auto speed = eval_vel.get_value(q);
      // Dirichlet/Neumann boundary
      const auto u_minus = eval_minus.get_value(q);
      const auto normal_vector = eval_minus.get_normal_vector(q);
const auto u_plus = 1.2*RI_distance;
     // const auto u_plus = u_minus;

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
void AdvectionOperator<dim, fe_degree>::local_apply_inverse_mass_matrix(
    const MatrixFree<dim, Number> &data_,
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

template <int dim, int fe_degree>
void AdvectionOperator<dim, fe_degree>::apply_operator(double const time, LinearAlgebra::distributed::Vector<Number> &dst, LinearAlgebra::distributed::Vector<Number> const &src) const
{

  if (this->update_velocity_)
  {
    set_velocity_operator(time);
  }

  this->operator_data_.data_.loop(&AdvectionOperator<dim, fe_degree>::local_apply_domain,
                                  &AdvectionOperator<dim, fe_degree>::local_apply_inner_face,
                                  &AdvectionOperator<dim, fe_degree>::local_apply_boundary_face, this,
                                  dst, src, true, MatrixFree<dim, Number>::DataAccessOnFaces::values,
                                  MatrixFree<dim, Number>::DataAccessOnFaces::values);
}

#endif // ADVECTION_OPERATOR_H