#ifndef PROB_PARM_H
#define PROB_PARM_H

#include <AMReX_REAL.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_GpuMemory.H>

struct ProbParmDevice
{
  // Región 1
  //amrex::Real p_1 = 42.59e5; Con lambda: γ=1.396.
  amrex::Real p_1 = 42.21716e4; //Con lambda: γ=1.354.
  amrex::Real T_1 = 293.0;
  amrex::Real rho_1 = 0.0;
  amrex::Real e_1 = 0.0;

  // Región 2 
  amrex::Real p_2 = 13.3e4;
  amrex::Real T_2 = 293.0;
  amrex::Real rho_2 = 0.0; 
  amrex::Real e_2 = 0.0;

  // Región 3
  amrex::Real T_3 = 1800.0;
  amrex::Real rho_3 = 0.0;
  amrex::Real e_3 = 0.0;
  
  //Parametros de la llama sinusoidal
  amrex::Real perturb_amplitude = 0.05;
  amrex::Real perturb_wavelength = 2.0;
  amrex::Real flame_y_base = 12.0;
  
  //Parametros de la onda de choque
  amrex::Real shock_position  = 11.0;
  
  //Parametros especies reactante
  amrex::Real Y_init_H2 = 0.0293;
  amrex::Real Y_init_O2 = 0.2465;
  amrex::Real Y_init_N2 = 0.7242;
  
  amrex::GpuArray<amrex::Real, NUM_SPECIES> massfrac = {0.0};
};

struct ProbParmHost
{
  ProbParmHost() {}
};

#endif
