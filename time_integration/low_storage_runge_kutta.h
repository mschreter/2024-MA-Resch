#ifndef LOW_STORAGE_RUNGE_KUTTA_H
#define LOW_STORAGE_RUNGE_KUTTA_H

#include "time_integration_setup.h"
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <iostream>
// Runge-Kutta time integrator schemes


template <typename Operator, int dim, int fe_degree, TimeIntegrators scheme>
class LowStorageRungeKuttaIntegrator
{
public:
  using Number = double;
  LowStorageRungeKuttaIntegrator(Operator &pde_operator,
                                 const LevelSetProblemParameters &param,
                                 LinearAlgebra::distributed::Vector<Number> const &solution
                                 ) : pde_operator_(pde_operator),
                                                                       ri(solution),
                                                                       ki(solution)
  {
    switch (scheme)
    {

      case stage_2_order_2:
    {
      bi = {{0.5, 0.5}};
      ai = {{1.0}};
      break;
    }

    case stage_3_order_3:
    {
      bi = {{0.245170287303492, 0.184896052186740, 0.569933660509768}};
      ai = {{0.755726351946097, 0.386954477304099}};
      break;
    }
    case stage_5_order_4:
      {
        bi = {{1153189308089. / 22510343858157.,
               1772645290293. / 4653164025191.,
               -1672844663538. / 4480602732383.,
               2114624349019. / 3568978502595.,
               5198255086312. / 14908931495163.}};
        ai = {{970286171893. / 4311952581923.,
               6584761158862. / 12103376702013.,
               2251764453980. / 15575788980749.,
               26877169314380. / 34165994151039.}};

        break;
      }

    case stage_7_order_4:
    {
      bi = {{0.0941840925477795334,
             0.149683694803496998,
             0.285204742060440058,
             -0.122201846148053668,
             0.0605151571191401122,
             0.345986987898399296,
             0.186627171718797670}};
      ai = {{0.241566650129646868 + bi[0],
             0.0423866513027719953 + bi[1],
             0.215602732678803776 + bi[2],
             0.232328007537583987 + bi[3],
             0.256223412574146438 + bi[4],
             0.0978694102142697230 + bi[5]}};
      break;
    }
    case stage_9_order_5:
    {
      bi = {{2274579626619. / 23610510767302.,
             693987741272. / 12394497460941.,
             -347131529483. / 15096185902911.,
             1144057200723. / 32081666971178.,
             1562491064753. / 11797114684756.,
             13113619727965. / 44346030145118.,
             393957816125. / 7825732611452.,
             720647959663. / 6565743875477.,
             3559252274877. / 14424734981077.}};
      ai = {{1107026461565. / 5417078080134.,
             38141181049399. / 41724347789894.,
             493273079041. / 11940823631197.,
             1851571280403. / 6147804934346.,
             11782306865191. / 62590030070788.,
             9452544825720. / 13648368537481.,
             4435885630781. / 26285702406235.,
             2357909744247. / 11371140753790.}};
      break;
    }
    default:
      AssertThrow(false, ExcNotImplemented());
    }
  }
  // perform_time_step method advances the solution by one time step using the chosen Runge-Kutta scheme.
  template <typename VectorType>
  void
  perform_time_step(const double current_time,
                    const double time_step,
                    VectorType &solution,
                    VectorType &vec_ri,
                    VectorType &vec_ki,
                    TimerOutput &computing_timer) const
  {
    AssertDimension(ai.size() + 1, bi.size());
    ri = solution;  
    perform_stage(current_time,
                  bi[0] * time_step,
                  ai[0] * time_step,
                  solution,
                  solution,
                  computing_timer);
    double sum_previous_bi = 0;


    for (unsigned int stage = 1; stage < bi.size(); ++stage)
    {
      const double c_i = sum_previous_bi + ai[stage - 1];
      perform_stage(current_time + c_i * time_step,
                    bi[stage] * time_step,
                    (stage == bi.size() - 1 ? 0 : ai[stage] * time_step),
                    ri,
                    solution,
                    computing_timer);
      sum_previous_bi += bi[stage - 1];
    }
  }

  //void prepare_buffer_interpolation() const {}
  //template <typename VectorType>
  //void interpolate_buffers(VectorType const &solution) const {}

private:
  // Coefficients for the Runge-Kutta scheme.
  std::vector<double> bi;
  std::vector<double> ai;

  mutable LinearAlgebra::distributed::Vector<Number> ki;
  mutable LinearAlgebra::distributed::Vector<Number> ri;
  Operator &pde_operator_;

void local_apply_inverse_mass_matrix(
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
  void perform_stage(const Number time,
                     const Number factor_solution,
                     const Number factor_ai,
                     LinearAlgebra::distributed::Vector<Number> &current_ri,
                     LinearAlgebra::distributed::Vector<Number> &solution,
                     TimerOutput &computing_timer) const
  {
    
    
    pde_operator_.apply_operator(time, ki, ri);

    {
    TimerOutput::Scope t(computing_timer, "RI: apply inverse mass matrix");
    
    pde_operator_.operator_data_.data_.cell_loop(
      &LowStorageRungeKuttaIntegrator::local_apply_inverse_mass_matrix,
      this,
      ri,
      ki,
      std::function<void(const unsigned int, const unsigned int)>(),
      [&](const unsigned int start_range, const unsigned int end_range) {
        
        const Number ai = factor_ai;
        const Number bi = factor_solution;
        if (ai == Number())
          {
            DEAL_II_OPENMP_SIMD_PRAGMA
            for (unsigned int i = start_range; i < end_range; ++i)
              {
                const Number k_i          = ri.local_element(i);
                const Number sol_i        = solution.local_element(i);
                solution.local_element(i) = sol_i + bi * k_i;
              }
          }
        else
          {
            DEAL_II_OPENMP_SIMD_PRAGMA
            for (unsigned int i = start_range; i < end_range; ++i)
              {
                const Number k_i          = ri.local_element(i);
                const Number sol_i        = solution.local_element(i);
                solution.local_element(i) = sol_i + bi * k_i;
                ri.local_element(i)  = sol_i + ai * k_i;
              }
          }
      });
    }
  }
};
#endif // ONE_STEP_THETA