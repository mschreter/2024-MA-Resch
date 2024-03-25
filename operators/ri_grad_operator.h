#ifndef RI_GRAD_OPERATOR_H
#define RI_GRAD_OPERATOR_H

#include "operator_data.h"
#include "../levelset_problem_parameters.h"
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/matrix_free/operators.h>
#include "operator_base.h"

using Number = double;
template <int dim, int fe_degree>
class RIGradOperator : public OperatorBase<dim, fe_degree>
{
public:
  using Number = double;

  RIGradOperator(OperatorData<dim, fe_degree> &operator_data) : OperatorBase<dim, fe_degree>(operator_data)
  {
  }

  void compute_viscosity_value(const double vertex_distance);
  void compute_penalty_parameter();

  template <bool is_right, uint component>
  void apply_operator(double const time, LinearAlgebra::distributed::Vector<Number> &dst, LinearAlgebra::distributed::Vector<Number> const &src) const;

private:

  mutable Number viscosity = 1.;
    mutable AlignedVector<VectorizedArray<Number>> array_penalty_parameter;

  template <uint component>
  void
  local_apply_domain(
      const MatrixFree<dim, Number> &data,
      LinearAlgebra::distributed::Vector<Number> &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

template <bool is_right, uint component>  void
  local_apply_inner_face(
      const MatrixFree<dim, Number> &data,
      LinearAlgebra::distributed::Vector<Number> &dst,
      const LinearAlgebra::distributed::Vector<Number> &src,
      const std::pair<unsigned int, unsigned int> &cell_range) const;

  template <bool is_right, uint component>
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
template <uint component>
void RIGradOperator<dim, fe_degree>::local_apply_domain(
    const MatrixFree<dim, Number> &data,
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
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
template <bool is_right, uint component>

void RIGradOperator<dim, fe_degree>::local_apply_inner_face(
    const MatrixFree<dim, Number> &data,
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
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
    template <bool is_right, uint component>
void RIGradOperator<dim, fe_degree>::local_apply_boundary_face(
    const MatrixFree<dim, Number> &data,
    LinearAlgebra::distributed::Vector<Number> &dst,
    const LinearAlgebra::distributed::Vector<Number> &src,
    const std::pair<unsigned int, unsigned int> &face_range) const
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
                //const auto u_plus  =  RI_distance * 1.2;
            const auto u_plus =  u_minus;

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
                //const auto u_plus  = RI_distance * 1.2;
            const auto u_plus =  u_minus;

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
void RIGradOperator<dim, fe_degree>::local_apply_inverse_mass_matrix(
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
  template <bool is_right, uint component>
void RIGradOperator<dim, fe_degree>::apply_operator(double const time, LinearAlgebra::distributed::Vector<Number> &dst, LinearAlgebra::distributed::Vector<Number> const &src) const
{

  this->operator_data_.data_.loop(&RIGradOperator<dim, fe_degree>::local_apply_domain<component>,
              &RIGradOperator<dim, fe_degree>::local_apply_inner_face<is_right, component>,
              &RIGradOperator<dim, fe_degree>::local_apply_boundary_face<is_right, component>,
              this,
              dst,
              src,
              true,
              MatrixFree<dim, Number>::DataAccessOnFaces::values,
              MatrixFree<dim, Number>::DataAccessOnFaces::values);

    this->operator_data_.data_.cell_loop(
      &RIGradOperator<dim, fe_degree>::local_apply_inverse_mass_matrix,
      this,
      dst,
      dst);
}

#endif // RI_DIFFUSION_OPERATOR_H