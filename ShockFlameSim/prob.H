#ifndef PROB_H
#define PROB_H

#include <AMReX_Print.H>
#include <AMReX_ParmParse.H>
#include <AMReX_Geometry.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_REAL.H>
#include <AMReX_GpuMemory.H>

#include "mechanism.H"

#include "PeleC.H"
#include "IndexDefines.H"
#include "Constants.H"
#include "PelePhysics.H"
#include "Tagging.H"
#include "ProblemDerive.H"
#include "prob_parm.H"	

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
pc_initdata(
  int i,
  int j,
  int k,
  amrex::Array4<amrex::Real> const& state,
  amrex::GeometryData const& geomdata,
  ProbParmDevice const& prob_parm)
{
  // Geometry
  const amrex::Real* prob_lo = geomdata.ProbLo();
  const amrex::Real* dx = geomdata.CellSize();

  const amrex::Real x = prob_lo[0] + (i + 0.5) * dx[0];
  const amrex::Real y = prob_lo[1] + (j + 0.5) * dx[1];
  
  // Llama sinusoidal
  const amrex::Real y_flame = prob_parm.flame_y_base +
  			      prob_parm.perturb_amplitude *
  			      std::sin(2.0 * M_PI * x / prob_parm.perturb_wavelength 
  			      + M_PI /2.0);
  // Onda de choque
  const amrex::Real y_shock = prob_parm.shock_position;
  
  // Estados
  if (y < y_shock) {  // Región 1
    state(i, j, k, URHO) = prob_parm.rho_1;
    state(i, j, k, UMX)  = 0.0;
    state(i, j, k, UMY)  = 0.0;
    state(i, j, k, UMZ)  = 0.0;
    state(i, j, k, UEINT) = prob_parm.rho_1 * prob_parm.e_1;
    state(i, j, k, UEDEN) = prob_parm.rho_1 * prob_parm.e_1;
    state(i, j, k, UTEMP) = prob_parm.T_1;
    for (int n = 0; n < NUM_SPECIES; n++) {
      state(i, j, k, UFS + n) = prob_parm.rho_1 * prob_parm.massfrac[n];
    }
  }
  else if (y > y_shock && y < y_flame) {  // Región 2
    state(i, j, k, URHO) = prob_parm.rho_2;
    state(i, j, k, UMX)  = 0.0;
    state(i, j, k, UMY)  = 0.0;
    state(i, j, k, UMZ)  = 0.0;
    state(i, j, k, UEINT) = prob_parm.rho_2 * prob_parm.e_2;
    state(i, j, k, UEDEN) = prob_parm.rho_2 * prob_parm.e_2;
    state(i, j, k, UTEMP) = prob_parm.T_2;
    for (int n = 0; n < NUM_SPECIES; n++) {
      state(i, j, k, UFS + n) = prob_parm.rho_2 * prob_parm.massfrac[n];
    }
  }
  else {  // Región 3
    state(i, j, k, URHO) = prob_parm.rho_3;
    state(i, j, k, UMX)  = 0.0;
    state(i, j, k, UMY)  = 0.0;
    state(i, j, k, UMZ)  = 0.0;
    state(i, j, k, UEINT) = prob_parm.rho_3 * prob_parm.e_3;
    state(i, j, k, UEDEN) = prob_parm.rho_3 * prob_parm.e_3;
    state(i, j, k, UTEMP) = prob_parm.T_3;
    for (int n = 0; n < NUM_SPECIES; n++) {
      state(i, j, k, UFS + n) = prob_parm.rho_3 * prob_parm.massfrac[n];
    }
  }
}

AMREX_GPU_DEVICE
AMREX_FORCE_INLINE
void
bcnormal(
  const amrex::Real* /*x[AMREX_SPACEDIM]*/,
  const amrex::Real* /*s_int[NVAR]*/,
  amrex::Real* /*s_ext[NVAR]*/,
  const int /*idir*/,
  const int /*sgn*/,
  const amrex::Real /*time*/,
  amrex::GeometryData const& /*geomdata*/,
  ProbParmDevice const& /*prob_parm*/,
  const amrex::GpuArray<amrex::Real, AMREX_SPACEDIM>& /*turb_fluc*/)
{
}

void pc_prob_close();

using ProblemTags = EmptyProbTagStruct;
using ProblemDerives = EmptyProbDeriveStruct;

#endif
