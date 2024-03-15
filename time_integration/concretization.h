#ifndef CONCRETIZATION_H
#define CONCRETIZATION_H

#include "time_integration_setup.h"
#include "low_storage_runge_kutta.h"
#include "one_step_theta.h"

// Namespace for Time Integrator Concretization
namespace TimeIntegratorConcretization
{
    /**
     * @brief This function returns the typedef of a TimeIntegrator based on a constexpr template.
     * @tparam TimeIntegrators The constexpr template parameter to specify the exact TimeIntegrator type.
     */
    template <typename Operator, int dim, int fe_degree, TimeIntegrators>
    struct Concretize;

    /**
     * @brief Specialization of Concretize for stage_3_order_3 TimeIntegrator.
     */
    template <typename Operator, int dim, int fe_degree>
    struct Concretize<Operator, dim,  fe_degree,  TimeIntegrators::stage_2_order_2>
    {
        typedef LowStorageRungeKuttaIntegrator<Operator, dim, fe_degree, stage_2_order_2> type;
    };

    /**
     * @brief Specialization of Concretize for stage_3_order_3 TimeIntegrator.
     */
    template <typename Operator, int dim, int fe_degree>
    struct Concretize<Operator, dim,  fe_degree,  TimeIntegrators::stage_3_order_3>
    {
        typedef LowStorageRungeKuttaIntegrator<Operator, dim, fe_degree, stage_3_order_3> type;
    };

    /**
     * @brief Specialization of Concretize for stage_5_order_4 TimeIntegrator.
     */
    template <typename Operator, int dim, int fe_degree>
    struct Concretize<Operator, dim,  fe_degree,  TimeIntegrators::stage_5_order_4>
    {
        typedef LowStorageRungeKuttaIntegrator<Operator, dim, fe_degree, stage_5_order_4> type;
    };

    /**
     * @brief Specialization of Concretize for stage_7_order_4 TimeIntegrator.
     */
    template <typename Operator, int dim, int fe_degree>
    struct Concretize<Operator, dim,  fe_degree,  TimeIntegrators::stage_7_order_4>
    {
        typedef LowStorageRungeKuttaIntegrator<Operator, dim, fe_degree, stage_7_order_4> type;
    };

    /**
     * @brief Specialization of Concretize for stage_9_order_5 TimeIntegrator.
     */
    template <typename Operator, int dim, int fe_degree>
    struct Concretize<Operator, dim,  fe_degree,  TimeIntegrators::stage_9_order_5>
    {
        typedef LowStorageRungeKuttaIntegrator<Operator, dim, fe_degree, stage_9_order_5> type;
    };

    /**
     * @brief Specialization of Concretize for one_step_theta TimeIntegrator.
     */
    template <typename Operator, int dim, int fe_degree>
    struct Concretize<Operator, dim, fe_degree, TimeIntegrators::one_step_theta>
    {
        typedef OneStepTheta<Operator, dim,fe_degree> type;
    };
}

#endif