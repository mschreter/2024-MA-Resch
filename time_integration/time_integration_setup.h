#ifndef TIME_INTEGRATOR_SETUP_H
#define TIME_INTEGRATOR_SETUP_H

//time integration schemes
enum TimeIntegrators
{
    stage_2_order_2,
    stage_3_order_3, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_5_order_4, /* Kennedy, Carpenter, Lewis, 2000 */
    stage_7_order_4, /* Tselios, Simos, 2007 */
    stage_9_order_5, /* Kennedy, Carpenter, Lewis, 2000 */
    one_step_theta,
};

constexpr TimeIntegrators advection_integrator = TimeIntegrators::stage_5_order_4;
constexpr TimeIntegrators reinitilization_integrator = TimeIntegrators::stage_5_order_4;

#endif // TIME_INTEGRATOR_SETUP_H