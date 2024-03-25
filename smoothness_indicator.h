#ifndef smoothness_indicator_h
#define smoothness_indicator_h

#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/base/vectorization.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

DEAL_II_NAMESPACE_OPEN


template <int dim, int fe_degree>
class SmoothnessIndicator
{
public:

  using Number = double;
  static const unsigned int n_components = 1;

  SmoothnessIndicator(const unsigned int      dof_handler_dgq = 0,
                      const unsigned int      dof_handler_leg = 1,
                      const unsigned int      quad_index      = 0,
                      const FiniteElement<1> &fe = FE_DGQ<1>(degree));

  /*
   Computes the smoothness indicator according to Persson and Peraire (2006)
   as \frac{(û,û)_K}{(u,u)_K}, where û is a truncated (degree-1)
   solution by using Legendre polynomials. This results in a const value per
   element.*/
  void
  compute_smoothness_indicator(
    const MatrixFree<dim, Number>                       &data,
    const LinearAlgebra::distributed::Vector<Number>    &solution) const;

  /*
   Returns the smoothness indicator previously calculated by
   compute_smoothness_indicator as it is usually used by MatrixFree*/
  const AlignedVector<VectorizedArray<Number>> &
  get_smoothness_indicator() const;

  //Releases previously allocated memory in the smoothness_indicator
  void
  clear();

private:

  VectorizedArray<Number>
  square(VectorizedArray<Number>  val,
         const unsigned int       n_lanes_filled) const;

  AlignedVector<Number>                           linearized_matrix;
  mutable AlignedVector<VectorizedArray<Number>>  smoothness_indicator;

  static const unsigned int degree = fe_degree;
  static const unsigned int n_points_1d = fe_degree + 1;
  const unsigned int        dof_handler_dgq;
  const unsigned int        dof_handler_leg;
  const unsigned int        quad_index;
};



template <int dim, int fe_degree>
SmoothnessIndicator<dim, fe_degree>::SmoothnessIndicator(const unsigned int      dof_handler_dgq,
                                                         const unsigned int      dof_handler_leg,
                                                         const unsigned int      quad_index,
                                                         const FiniteElement<1> &fe)
  : dof_handler_dgq(dof_handler_dgq)
  , dof_handler_leg(dof_handler_leg)
  , quad_index(quad_index)
{
  Assert(
    fe.degree == degree,
    ExcMessage("Finite-Element degree needs to be consistent with the degree this class has been initialized with."));

  // Initializ projection matrix, degree+1=dofs_per_cell (1d)
  FullMatrix<Number> transformation_matrix(degree + 1);

  // Get the projection matrix to transform from dgq to leg basis
  FETools::get_projection_matrix(fe, FE_DGQLegendre<1>(degree), transformation_matrix);

  linearized_matrix.resize((degree + 1) * (degree + 1));
  for (uint i = 0; i < degree + 1; ++i)
    for (uint j = 0; j < degree + 1; ++j)
      linearized_matrix[j + ((degree + 1) * i)] = transformation_matrix[j][i];
}



template <int dim, int fe_degree>
void
SmoothnessIndicator<dim, fe_degree>::
  compute_smoothness_indicator(
    const MatrixFree<dim, Number, VectorizedArray<Number>>  &data,
    const LinearAlgebra::distributed::Vector<Number>        &solution) const
{
  smoothness_indicator.resize(data.n_cell_batches());
  int  dummy;
  bool division_by_zero = false;

  data.template cell_loop<int, LinearAlgebra::distributed::Vector<Number>>(
    [this,
     &division_by_zero](
      const auto &data, auto &, const auto &src, const auto &cell_range) {
      // Initialize one evaluation object underlying a polynomial basis and
      // one underlying a modal basis:
      FEEvaluation<dim, fe_degree> phi_dgq(data, dof_handler_dgq, quad_index);
      FEEvaluation<dim, fe_degree> phi_leg(data, dof_handler_leg, quad_index);

      Assert(
        phi_dgq.dofs_per_cell == phi_leg.dofs_per_cell,
        ExcMessage(
          "DoFHandler 0 (F_DGQ) and DoFHandler 1 (FE_DGQLegendre) needs to have the same polynomial degree"));

      VectorizedArray<Number> u_hat;
      VectorizedArray<Number> u;

      // Iterate overa all cell batches
      for (unsigned int cell = cell_range.first; cell < cell_range.second;
           ++cell)
        {
          phi_dgq.reinit(cell);
          phi_leg.reinit(cell);

          u_hat = Number(0);
          u     = Number(0);

          // Read in the solution vector
          phi_dgq.read_dof_values(src);

          // Transform from the solution vector given on the polynomial basis
          // to a modal basis
          ::dealii::internal::FEEvaluationImplBasisChange<
            ::dealii::internal::evaluate_general,
            ::dealii::internal::EvaluatorQuantity::value,
            dim,
            /*basis_size_1*/ degree + 1,
            /*basis_size_2*/ degree + 1
            >::do_forward(n_components,
                                linearized_matrix,
                                phi_dgq.begin_dof_values(),
                                phi_leg.begin_dof_values(),
                                numbers::invalid_unsigned_int,
                                numbers::invalid_unsigned_int);

          // Used to reset the values. Constructors default values are 0
          typename FEEvaluation<dim, fe_degree>::value_type null_data;

          // This is essentially (u-û), where û is the truncated (with p-1)
          // solution. Instead of explicitly calculate (u-û), all entries,
          // which are the same in both representations, are set to zero.
          // In one dimension, these are all apart from the last entry
          // resulting in an iterator, which goes to `degree`, since the basis
          // is of size degree+1. For higher dimensions, every basis differs
          // through its last entry. The whole basis is then represented by
          // (degree+1)^dim and we need to keep every multiplication with any
          // 'last entry' (which is the coefficient for the respective
          // (highest) mode). Expanding the term (degree+1)^dim could be
          // performed straightforward and results in the following scheme:

          if (dim == 1)
            for (uint i = 0; i < degree; ++i)
              phi_leg.submit_dof_value(null_data, i);
          else if (dim == 2)
            for (uint j = 0; j < degree; ++j)
              for (uint i = 0; i < degree; ++i)
                phi_leg.submit_dof_value(null_data, ((degree + 1) * j) + i);
          else
            for (uint k = 0; k < degree; ++k)
              for (uint j = 0; j < degree; ++j)
                for (uint i = 0; i < degree; ++i)
                  phi_leg.submit_dof_value(null_data,
                                           (((degree + 1) * j) + i) +
                                             (k * (degree + 1) * (degree + 1)));

          // Polynomial basis phi_dgq is used for the denominator, where the
          // whole solution is needed
          phi_dgq.evaluate(EvaluationFlags::values);
          // Used to evaluate the truncated solution
          phi_leg.evaluate(EvaluationFlags::values);

          for (unsigned int q = 0; q < phi_dgq.n_q_points; ++q)
            {
              u += square(phi_dgq.get_value(q),
                          data.n_active_entries_per_cell_batch(cell));
              u_hat += square(phi_leg.get_value(q),
                              data.n_active_entries_per_cell_batch(cell));
            }

          // We do not Assert here, weather the denominator is zero or not. In
          // general, the selected component should be non-zero. But skipping
          // the assertion here allows undershoots without terminating the
          // simulation, which might occur at e.g. intial states. In case out
          // indicator value is nan, the viscosity becomes the highest possible
          // value anyway. Instead, just raise a warning (later).
          division_by_zero = * std::min_element(u.begin(),
                              u.begin() + data.n_active_entries_per_cell_batch(cell)) 
                              < 1e-15;

          // Evaluate indicator, which is log10(u_hat/u)
          for (uint i = 0; i < data.n_active_entries_per_cell_batch(cell); ++i)
            smoothness_indicator[cell][i] = std::log10(u_hat[i] / u[i]);
        }
    },
    dummy,
    solution);

    if (division_by_zero)
      std::cout << "Warning: Division by zero in the smoothness indicator!" << std::endl;
}



template <int dim, int fe_degree>
const AlignedVector<VectorizedArray<double>> &
SmoothnessIndicator<dim, fe_degree>::get_smoothness_indicator() const
{
  return smoothness_indicator;
}



template <int dim, int fe_degree>
void
SmoothnessIndicator<dim, fe_degree>::clear()
{
  smoothness_indicator.clear();
}



template <int dim, int fe_degree>
VectorizedArray<double>
SmoothnessIndicator<dim, fe_degree>::square(VectorizedArray<Number> val,
                                            const uint              n_lanes_filled) const
{
  for (uint v = 0; v < n_lanes_filled; ++v)
    val[v] *= val[v];
  return val;
}


DEAL_II_NAMESPACE_CLOSE

#endif
