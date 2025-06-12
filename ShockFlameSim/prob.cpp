#include "prob.H"

void
pc_prob_close()
{
}

extern "C" {
void
amrex_probinit(
  const int* /*init*/,
  const int* /*name*/,
  const int* /*namelen*/,
  const amrex::Real* /*problo*/,
  const amrex::Real* /*probhi*/)
{
  // Parse params
  {
    amrex::ParmParse pp("prob");
    
    // Reactante 1
    pp.query("p_1", PeleC::h_prob_parm_device->p_1);
    pp.query("T_1", PeleC::h_prob_parm_device->T_1);

    // Reactante 2
    pp.query("p_2", PeleC::h_prob_parm_device->p_2);
    pp.query("T_2", PeleC::h_prob_parm_device->T_2);

    // Producto 3
    pp.query("T_3", PeleC::h_prob_parm_device->T_3);

    // Parámetros de la onda de choque
    pp.query("shock_position", PeleC::h_prob_parm_device->shock_position);

    // Parámetros de la llama sinusoidal
    pp.query("perturb_amplitude", PeleC::h_prob_parm_device->perturb_amplitude);
    pp.query("perturb_wavelength", PeleC::h_prob_parm_device->perturb_wavelength);
    pp.query("flame_y_base", PeleC::h_prob_parm_device->flame_y_base);
    
    // Parámetros de especias reactantes
    pp.query("Y_init_H2", PeleC::h_prob_parm_device->Y_init_H2);
    pp.query("Y_init_O2", PeleC::h_prob_parm_device->Y_init_O2);
    pp.query("Y_init_N2", PeleC::h_prob_parm_device->Y_init_N2);
   
  }
  
  // Initial values
  PeleC::h_prob_parm_device->massfrac[H2_ID] =
    PeleC::h_prob_parm_device->Y_init_H2;
  PeleC::h_prob_parm_device->massfrac[O2_ID] =
    PeleC::h_prob_parm_device->Y_init_O2;
  PeleC::h_prob_parm_device->massfrac[N2_ID] =
    PeleC::h_prob_parm_device->Y_init_N2;
  
  auto eos = pele::physics::PhysicsType::eos();
  // Región 1
  eos.PYT2RE(
      PeleC::h_prob_parm_device->p_1,
      PeleC::h_prob_parm_device->massfrac.begin(),
      PeleC::h_prob_parm_device->T_1, 
      PeleC::h_prob_parm_device->rho_1,
      PeleC::h_prob_parm_device->e_1);

  // Región 2
  eos.PYT2RE(
      PeleC::h_prob_parm_device->p_2,
      PeleC::h_prob_parm_device->massfrac.begin(),
      PeleC::h_prob_parm_device->T_2, 
      PeleC::h_prob_parm_device->rho_2,
      PeleC::h_prob_parm_device->e_2);

  // Región 3
  eos.PYT2RE(
      PeleC::h_prob_parm_device->p_2,
      PeleC::h_prob_parm_device->massfrac.begin(),
      PeleC::h_prob_parm_device->T_3, 
      PeleC::h_prob_parm_device->rho_3,
      PeleC::h_prob_parm_device->e_3);
}
}	

void
PeleC::problem_post_timestep()
{
}

void
PeleC::problem_post_init()
{
}

void
PeleC::problem_post_restart()
{
}
