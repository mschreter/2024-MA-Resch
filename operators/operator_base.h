#ifndef OPERATOR_BASE_h
#define OPERATOR_BASE_h
#include "operator_data.h"

template <int dim, int fe_degree>
class OperatorBase
{
public:
    using Number = double;
    OperatorData<dim, fe_degree> &operator_data_;
    OperatorBase(OperatorData<dim, fe_degree> &operator_data)
        : operator_data_(operator_data),
        update_velocity_(true)
    {}
    
    //Can not be virtual because derived functions are templated
    void apply_operator(LinearAlgebra::distributed::Vector<Number> &dst, LinearAlgebra::distributed::Vector<Number> const &src) const;

void set_velocity_operator(double const time) const{}
    bool update_velocity_;
private:
    // must me implemented in derived classes
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
};
#endif