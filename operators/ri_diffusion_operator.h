#ifndef RI_DIFFUSION_OPERATOR_H
#define RI_DIFFUSION_OPERATOR_H

#include "operator_data.h"
#include "../levelset_problem_parameters.h"
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/matrix_free/operators.h>
#include "operator_base.h"

using Number = double;
template <int dim, int fe_degree>
class RIDiffusionOperator : public OperatorBase<dim, fe_degree>
{
public:
  using Number = double;

  RIDiffusionOperator(OperatorData<dim, fe_degree> &operator_data, LevelSetProblemParameters const &param) : OperatorBase<dim, fe_degree>(operator_data), param_(param)
  {
  }

  void compute_viscosity_value(const double vertex_distance);
  void compute_penalty_parameter();

  void apply_operator(double const time, LinearAlgebra::distributed::Vector<Number> &dst, LinearAlgebra::distributed::Vector<Number> const &src) const;

private:
  LevelSetProblemParameters const &param_;

  mutable Number viscosity = 1.;
    mutable AlignedVector<VectorizedArray<Number>> array_penalty_parameter;

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
      const MatrixFree<dim, Number> &data,
      LinearAlgebra::distributed::Vector<Number> &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

  void local_apply_inverse_mass_matrix(
      const MatrixFree<dim, Number> &data,
      LinearAlgebra::distributed::Vector<Number> &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;
};

template <int dim, int fe_degree>
    void
    RIDiffusionOperator<dim, fe_degree>::compute_viscosity_value(const double vertex_distance)
    {
      // The value for the artificial viscosity is determined by the smallest enabled element size.
      if (param_.use_adaptive_mesh_refinement)
        viscosity = param_.factor_diffusivity * vertex_distance / std::pow(2., param_.n_refinement_levels) / fe_degree;
      else
        viscosity = param_.factor_diffusivity * vertex_distance / fe_degree;
    }

    template <int dim, int fe_degree>
    void
    RIDiffusionOperator<dim, fe_degree>::compute_penalty_parameter()
    {
      // Resize
      const unsigned int n_macro_cells =  this->operator_data_.data_.n_cell_batches() +  this->operator_data_.data_.n_ghost_cell_batches();
      array_penalty_parameter.resize(n_macro_cells);
      for (uint macro_cells = 0; macro_cells < n_macro_cells; ++macro_cells)
      {
        // Depending on the cell number, there might be empty lanes
        const unsigned int n_lanes_filled = this->operator_data_.data_.n_active_entries_per_cell_batch(macro_cells);
        for (uint lane = 0; lane < n_lanes_filled; ++lane)
        {
          auto cell = this->operator_data_.data_.get_cell_iterator(macro_cells, lane);
          array_penalty_parameter[macro_cells][lane] = 1. / cell->minimum_vertex_distance() *
                                                       (param_.fe_degree + 1) * (param_.fe_degree + 1) *
                                                       param_.IP_diffusion;
        }
      }
    }

template <int dim, int fe_degree>
void RIDiffusionOperator<dim, fe_degree>::local_apply_domain(
    const MatrixFree<dim, Number> &data,
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
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
void RIDiffusionOperator<dim, fe_degree>::local_apply_inner_face(
    const MatrixFree<dim, Number> &data,
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
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
void RIDiffusionOperator<dim, fe_degree>::local_apply_boundary_face(
    const MatrixFree<dim, Number> &data,
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
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
              //const auto u_plus  = RI_distance * 1.2;
            const auto u_plus =  u_minus;

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
void RIDiffusionOperator<dim, fe_degree>::local_apply_inverse_mass_matrix(
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
void RIDiffusionOperator<dim, fe_degree>::apply_operator(double const time, LinearAlgebra::distributed::Vector<Number> &dst, LinearAlgebra::distributed::Vector<Number> const &src) const
{

  this->operator_data_.data_.loop(&RIDiffusionOperator<dim, fe_degree>::local_apply_domain,
                                  &RIDiffusionOperator<dim, fe_degree>::local_apply_inner_face,
                                  &RIDiffusionOperator<dim, fe_degree>::local_apply_boundary_face, this,
                                  dst, src, true, MatrixFree<dim, Number>::DataAccessOnFaces::values,
                                  MatrixFree<dim, Number>::DataAccessOnFaces::values);
}

#endif // RI_DIFFUSION_OPERATOR_H