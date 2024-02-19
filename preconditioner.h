#ifndef preconditioner_h
#define preconditioner_h

#include <deal.II/matrix_free/operators.h>

DEAL_II_NAMESPACE_OPEN

  // For preconditioning, the inverse mass matrix is applied
  template <int dim, int fe_degree>
  class Preconditioner
  {
  public:
    using Number = double;
    using VectorType = LinearAlgebra::distributed::Vector<Number>;

    Preconditioner(
      const MatrixFree<dim, Number> &matrix_free)
      : matrix_free(matrix_free)
    {}

    void
    vmult(VectorType       &dst, 
          const VectorType &src) const
    {
      FEEvaluation<dim,fe_degree,fe_degree + 1, 1,Number> phi(matrix_free, 0, 1);

      MatrixFreeOperators::CellwiseInverseMassMatrix<dim,fe_degree,1,Number> inverse(phi);

      matrix_free.template cell_loop<VectorType, VectorType>(
        [&](const auto &, auto &dst, const auto &src, auto &range) {
          for (auto cell = range.first; cell < range.second; ++cell)
            {
              phi.reinit(cell);
              phi.read_dof_values(src);
              inverse.apply(phi.begin_dof_values(), phi.begin_dof_values());
              phi.set_dof_values(dst);
            }
        }
        ,
        dst,
        src,
        true);
    }

  private:
    const MatrixFree<dim, Number> &matrix_free;
  };

DEAL_II_NAMESPACE_CLOSE

#endif
