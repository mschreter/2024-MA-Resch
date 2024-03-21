#ifndef OPERATOR_DATA_H
#define OPERATOR_DATA_H
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q_generic.h>
#include <deal.II/lac/affine_constraints.h>

/*This class holds data_ which is shared between different classes.
 */
template <int dim, int fe_degree>
class OperatorData
{
public:
    using Number = double;
    MatrixFree<dim, Number> data_;
    DoFHandler<dim>  &dof_handler_;
    DoFHandler<dim>  &dof_handler_legendre_;
    DoFHandler<dim>  &dof_handler_vel_;
    OperatorData(DoFHandler<dim>  &dof_handler,
                                           DoFHandler<dim>  &dof_handler_legendre,
                                               DoFHandler<dim>  &dof_handler_vel);
    void reinit();
};
template <int dim, int fe_degree>
OperatorData<dim, fe_degree>::OperatorData(DoFHandler<dim>  &dof_handler,
                                           DoFHandler<dim>  &dof_handler_legendre,
                                               DoFHandler<dim>  &dof_handler_vel) : dof_handler_(dof_handler),
                                                                                         dof_handler_legendre_(dof_handler_legendre),
                                                                                         dof_handler_vel_(dof_handler_vel){}
template <int dim, int fe_degree>
void OperatorData<dim, fe_degree>::reinit()
{
    std::vector<const DoFHandler<dim> *> dof_handlers(
        {&dof_handler_, &dof_handler_legendre_, &dof_handler_vel_});
    MappingQGeneric<dim> mapping(fe_degree);
    Quadrature<dim> quadrature = QGauss<dim>(fe_degree + 1);
    Quadrature<dim> quadrature_mass = QGauss<dim>(fe_degree + 1);
    // QGauss<1>(fe_degree + 1) gives inaccurate results for the norm computation.
    // Use overintegration or GaussLobatto quadrature for norm computation.
    Quadrature<dim> quadrature_norm = QGauss<dim>(fe_degree + 2);
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
    data_.reinit(mapping,
                 dof_handlers,
                 constraints,
                 std::vector<Quadrature<dim>>{{quadrature, quadrature_mass, quadrature_norm}},
                 additional_data);
}
#endif // OPERATOR_DATA_H