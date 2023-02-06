#!/usr/bin/env python
# encoding: utf-8


"""
g_wave.py

Created by Chris Stevens 2022
Copyright (c) 2022 University of Canterbury. All rights reserved.
"""

import numpy as np
import math
from mpi4py import MPI
from functools import partial

from coffee.ibvp import IBVP
from coffee.grid import UniformCart
from coffee.solvers import RungeKutta4
from coffee.tslices import tslices
from coffee.system import System
from coffee.diffop.sbp import sbp
from coffee.diffop.fd import ghost_point_processor
from coffee import actions

rt2 = np.sqrt(2.)
r2 = 1./np.sqrt(2.)

def kpbump_pol(a, t, p):
	area   = 1.
	b      = 35.*np.pi*a / (128.*area)
	nbumps = len(p)

	for i in range(0,len(p)):
		if (b*t < (i+1)*np.pi):
			return np.exp(2.j*np.pi*p[i])*a*(np.sin(b*t)**8) * (b*t < (i+1)*np.pi)
	return 0.

class G_wave(System):

	################################
	# Constructor
	################################

	def __init__(self, D, tau, global_z, CFL = 0.5, amplitude = 1.0, \
				A = 1, initialCond = False, \
				numvar=15, constraints = None, pl = None, pr = None):
		super(G_wave, self).__init__()
		self.CFL = CFL
		self.D = D
		self.tau = tau
		self.name = "Colliding G-waves"
		self.amplitude = amplitude
		self.numvar=numvar
		self.F = None
		self.A = A

		self.constraints = constraints

		self.global_z = global_z

		self.pol_l = pl
		self.pol_r = pr

		# Place holders not used for anything
		self.max_char = 1
		self.regrid = False
		self.reparam_indices = None
		self.regridtype = -1

	def updateMaxChar(self, U):
		return

	def left(self, t, U):# u=0 psi4
		u = t / rt2 # Minkowski

		return u, kpbump_pol(self.amplitude, u, self.pol_l)

	def right(self, t, U): # v=0 psi0
		v = t / rt2 # Minkowski
		return v, kpbump_pol(self.amplitude, v, self.pol_r)

	def evaluate(self, t, U, intStep = None):

		# Define useful variables
		A, B, mu, rho, rhop, sigma, sigmap, psi0, psi4, u, v, xi, eta = U.data

		z  = U.domain.axes[0]
		dz = U.domain.step_sizes[0]
		mub     = np.conjugate(mu)
		sigmab  = np.conjugate(sigma)
		sigmapb = np.conjugate(sigmap)
		etab    = np.conjugate(eta)
		xib     = np.conjugate(xi)
		pi = 0.
		psi2 = sigma*sigmap - rho*rhop

		########################################################################
		# F = 0 START
		########################################################################

		# F  = self.F[0]
		# Fb = self.F[1]
		# Fd = self.F[2]
		# Fp = self.F[3]

		########################################################################
		# F = 0 STOP
		########################################################################

		########################################################################
		# F where dB/dt = 0 START
		########################################################################

		f = fp = fd = 0.

		chi  = rhop - rho
		chip = self.D(rhop,dz) - self.D(rho,dz)
		chid = 2.**(-0.5) * ((-1.) * (2. * chi + 3. * rho + (-3.) * rhop) \
			* (rho + rhop) + (-1.) * sigma * sigmab + sigmapb * sigmap)

		F  = chi + 1.j*f
		Fb = chi - 1.j*f
		Fd = chid + 1.j*fd
		Fp = chip + 1.j*fp

		self.F = np.array([F, Fb, Fd, Fp])

		########################################################################
		# F where dB/dt = 0 STOP
		########################################################################

		########################################################################
		# F where F = \chi START
		########################################################################

		# one = np.ones_like(A)
		# zro = np.zeros_like(A)

		# f = fp = fd = 0.

		# chi  = zro
		# chip = zro
		# chid = zro

		# F  = chi + 1.j*f
		# Fb = chi - 1.j*f
		# Fd = chid + 1.j*fd
		# Fp = chip + 1.j*fp

		# self.F = np.array([F, Fb, Fd, Fp])

		################################################################
		# F where F = \chi STOP
		################################################################

		################################################################
		# F where \epsilon = 0
		################################################################

		# sigmab  = np.conj(sigma)
		# sigmapb = np.conj(sigmap)
		# mub     = np.conj(mu)

		# F  = mu - rho + rhop
		# Fb = np.conj(F)
		# Fd = 0.5 * (1. + B)**(-1.) * ((-2.) * A * self.D(F, dz) + 2. * \
		# 	np.sqrt(2.) * sigma * ((-1.) * sigmab + sigmap) + np.sqrt(2.) * \
		# 	((-1.) * F**2. + (-4.) * F * rho + (-6.) * (pi + rho**2.) + \
		# 	mu**2. + mu * rho) + np.sqrt(2.) * (((-1.) * F + (-2.) * rho) * Fb + \
		# 	(2. * F + 6. * rho + mu + mub) * rhop + rho * mub) + np.sqrt(2.) * mu \
		# 	* mub)
		# Fp = 0.5 * A**(-1.) * ((-1.) * np.sqrt(2.) * sigmapb * sigmap + \
		# 	np.sqrt(2.) * ((-1.) * B * sigmapb * sigmap + ((-1.) + B) * sigma * \
		# 	sigmab + rhop * ((-1.) * (1. + 3. * B) * rhop + 4. * rho + B * F + \
		# 	mu)) + np.sqrt(2.) * ((-6.) * pi + rho * ((-1.) * rho + 3. * B * \
		# 	rho + B * F + mu)) + np.sqrt(2.) * B * Fb * (rho + rhop) + \
		# 	np.sqrt(2.) * mub * (rho + rhop) + 2. * A * self.D(mu, dz))

		# self.F = np.array([F, Fb, Fd, Fp])

		################################################################
		# F where \epsilon = 0 STOP
		################################################################

		# Evaluate right hand side
		dA = (A*(mu + mub)) / rt2
		dB = (F + Fb + B*(mu + mub) + 2.*(rho - rhop)) / rt2
		dmu = (-np.power(F,2.) + np.power(mu,2.) - 6.*pi - 3.*F*rho + mu*rho - \
					3.*np.power(rho,2.) - \
					sigma*sigmab - Fb*(F + rho - rhop) + \
					rhop*(3.*F + mu + 6.*rho - 3.*rhop) + \
					mub*(mu + rho + rhop) + 2.*sigma*sigmap - \
					sigmapb*sigmap - rt2*B*Fd - \
					rt2*A*Fp) / rt2
		drho = (rho*(F + Fb + 3.*rho) + sigma*sigmab - 3.*pi) / rt2
		drhop = (-rhop*(F + Fb - 3.*rhop) + \
					  sigmapb*sigmap - 3.*pi) / rt2
		dsigma = (3.*F*sigma + 4.*rho*sigma + rho*sigmapb + psi0 - \
					  sigma*(Fb + rhop)) / rt2
		dsigmap = (psi4 + sigmab*rhop + \
					   sigmap*(-3.*F - rho + Fb + \
									4.*rhop)) / rt2

		dzpsi0 = self.D(psi0, dz)

		dpsi0 = (-6.*sigma*psi2 + 2.*psi0*(-2.*(F + mu + rho) + rhop) - \
					  rt2*A*dzpsi0) / (rt2*(B - 1.))
		dpsi4 = (-2.*psi4*(2.*F - 2.*mu + rho - 2.*rhop) + \
					  6.*psi2*sigmap - rt2*A*self.D(psi4,dz)) / (rt2*(B + 1.))

		deta = (eta*(F - Fb + rho + rhop) + \
				   etab*(sigma + sigmapb)) / rt2
		dxi  = (xi*(F - Fb + rho + rhop) + \
				  xib*(sigma + sigmapb)) / rt2
		
		detab = np.conjugate(deta)
		dxib  = np.conjugate(dxi)

		u_prime = self.D(u, dz)
		v_prime = self.D(v, dz)

		du = -A*u_prime / (1. + B) 
		dv =  A*v_prime / (1. - B) 

		# Impose boundary conditions
		new_derivatives, _ = U.communicate(
			partial(ghost_point_processor),
			data=np.array([
				dA, dB, dmu, drho, drhop, dsigma, dsigmap, dpsi0, dpsi4, \
				du, dv, dxi, deta
			])
		)
		dA, dB, dmu, drho, drhop, dsigma, dsigmap, dpsi0, dpsi4, du, dv, \
			dxi, deta = new_derivatives

		pt_r = self.D.penalty_boundary(dz, "right")
		pt_l = self.D.penalty_boundary(dz, "left")
		pt_r_shape = pt_r.size
		pt_l_shape = pt_l.size

		C_left  = self.tau * pt_l * (-A[0] / (1. + B[0]))
		C_right = self.tau * pt_r * (A[-1] / (1. - B[-1]))

		l_u , l_psi4 = self.left(t,U)
		r_v , r_psi0 = self.right(t,U)

		b_data = U.external_slices()
		for dim, direction, d_slice in b_data:
			if direction == 1:
				dpsi0[-pt_r_shape:] -= C_right * (psi0[-1] - r_psi0)
				dv[-pt_r_shape:]    -= C_right * (v[-1] - r_v)
			else:
				dpsi4[:pt_l_shape]  -= C_left * (psi4[0] - l_psi4)
				du[:pt_l_shape]     -= C_left * (u[0] - l_u)

		# now all time derivatives are computed
		# package them into a time slice and return
		# if (z[-1] == 1):
			# test = u[200] - v[200]
			# print("t=%.16f %.16f %.16f %.16f" % (t, u[200], v[200], test))
			# print("t=%.16f" % t)

		return tslices.TimeSlice([dA, dB, dmu, drho, drhop, dsigma, \
									  dsigmap, dpsi0, dpsi4, du, dv, dxi, \
									  deta], \
									  U.domain, time = t)

	def initial_data(self, t0, grid):

		pi  = 0.
		z   = grid.meshes[0]
		dz  = np.fabs(z[1]-z[0])
		one = np.ones_like(z)
		zro = np.zeros_like(z)

		A           = self.A*one + 0.j
		B           = zro + 0.j
		sigma       = zro + 0.j
		sigmap      = zro + 0.j
		psi0        = zro + 0.j
		psi4        = zro + 0.j

		u = ((t0 - (1./A)*(z - self.global_z[0])) / rt2) + 0.j
		v = ((t0 + (1./A)*(z - self.global_z[-1])) / rt2) + 0.j

		mu     = zro + 0.j
		rho    = zro + 0.j
		rhop   = zro + 0.j
		self.F = np.array([0., 0., 0., 0.]) + 0.j
		xi     = (A / rt2) + 0.j
		eta    = A*1.j / rt2
		
		################################################################
		# F where dB/dt = 0 START
		################################################################

		f = fp = fd = 0.

		sigmab  = np.conjugate(sigma)
		sigmapb = np.conjugate(sigmap)

		chi  = rhop - rho
		chip = self.D(rhop,dz) - self.D(rho,dz)
		chid = 2.**(-0.5) * ((-1.) * (2. * chi + 3. * rho + (-3.) * \
			rhop) * (rho + rhop) + (-1.) * sigma * sigmab + \
			sigmapb * sigmap)

		F  = chi + 1.j*f
		Fb = chi - 1.j*f
		Fd = chid + 1.j*fd
		Fp = chip + 1.j*fp

		self.F = np.array([F, Fb, Fd, Fp])

		################################################################
		# F where dB/dt = 0 STOP
		################################################################

		################################################################
		# F where F = \chi START
		################################################################

		# f = fp = fd = 0.

		# chi  = zro
		# chip = zro
		# chid = zro

		# F  = chi + 1.j*f
		# Fb = chi - 1.j*f
		# Fd = chid + 1.j*fd
		# Fp = chip + 1.j*fp

		# self.F = np.array([F, Fb, Fd, Fp])

		################################################################
		# F where F = \chi STOP
		################################################################

		################################################################
		# F where \epsilon = 0
		################################################################

		# sigmab  = np.conj(sigma)
		# sigmapb = np.conj(sigmap)
		# mub     = np.conj(mu)

		# F  = mu - rho + rhop
		# Fb = np.conj(F)
		# Fd = 0.5 * (1. + B)**(-1.) * ((-2.) * A * self.D(F, dz) + 2. * \
		# 	np.sqrt(2.) * sigma * ((-1.) * sigmab + sigmap) + np.sqrt(2.) * \
		# 	((-1.) * F**2. + (-4.) * F * rho + (-6.) * (pi + rho**2.) + \
		# 	mu**2. + mu * rho) + np.sqrt(2.) * (((-1.) * F + (-2.) * rho) * Fb + \
		# 	(2. * F + 6. * rho + mu + mub) * rhop + rho * mub) + np.sqrt(2.) * mu \
		# 	* mub)
		# Fp = 0.5 * A**(-1.) * ((-1.) * np.sqrt(2.) * sigmapb * sigmap + \
		# 	np.sqrt(2.) * ((-1.) * B * sigmapb * sigmap + ((-1.) + B) * sigma * \
		# 	sigmab + rhop * ((-1.) * (1. + 3. * B) * rhop + 4. * rho + B * F + \
		# 	mu)) + np.sqrt(2.) * ((-6.) * pi + rho * ((-1.) * rho + 3. * B * \
		# 	rho + B * F + mu)) + np.sqrt(2.) * B * Fb * (rho + rhop) + \
		# 	np.sqrt(2.) * mub * (rho + rhop) + 2. * A * self.D(mu, dz))

		# self.F = np.array([F, Fb, Fd, Fp])

		################################################################
		# F where \epsilon = 0 STOP
		################################################################

		data = [A, B, mu, rho, rhop, sigma, sigmap, psi0, psi4, u, v, xi, eta]

		temp_tslice = tslices.TimeSlice(data, grid, t0)
		if self.constraints:
			cv          = self.constraint_violation(temp_tslice)
			tslice      = tslices.TimeSlice(data, grid, time = t0, cv = cv)
			return tslice
		else:
			return temp_tslice

	def timestep(self, U):

		return self.fixed_timestep(U)
		# return self.max_characteristic_timestep(U)

	def max_characteristic_timestep(self,u):
		if u.domain.mpi is not None:
			charspeed1 = np.abs(max(u.data[0]) / min(1.+ u.data[1]))
			charspeed2 = np.abs(max(u.data[0]) / min(1.- u.data[1]))
			maxspeed = max(charspeed1, charspeed2)
			maxspeed_array = np.array([maxspeed])
			maxspeed_return = np.ones_like(maxspeed_array)
			u.domain.mpi.comm.Allreduce(maxspeed_array, maxspeed_return, \
				MPI.MAX)
			return u.domain.step_sizes[0] / maxspeed_return[0]
		else:
			charspeed1 = np.abs(max(u.data[0]) / min(1.+ u.data[1]))
			charspeed2 = np.abs(max(u.data[0]) / min(1.- u.data[1]))
			maxspeed = max(charspeed1, charspeed2)
		return u.domain.step_sizes[0] / maxspeed

	def fixed_timestep(self,u):
		return self.CFL * u.domain.step_sizes[0]

	def constraint_violation(self, U):
		# Define useful variables
		A, B, mu, rho, rhop, sigma, sigmap, psi0, psi4, u, v, xi, eta = U.data

		z = U.domain.axes[0]
		dz = U.domain.step_sizes[0]

		pi = 0.

		mub     = np.conjugate(mu)
		sigmab  = np.conjugate(sigma)
		sigmapb = np.conjugate(sigmap)

		etab = np.conjugate(eta)
		xib  = np.conjugate(xi)

		# F  = self.F[0]
		# Fb = self.F[1]
		# Fd = self.F[2]
		# Fp = self.F[3]

		f = fp = fd = 0.

		chi  = rhop - rho
		chip = self.D(rhop,dz) - self.D(rho,dz)
		chid = 2.**(-0.5) * ((-1.) * (2. * chi + 3. * rho + (-3.) * \
			rhop) * (rho + rhop) + (-1.) * sigma * sigmab + \
			sigmapb * sigmap)

		F  = chi + 1.j*f
		Fb = chi - 1.j*f
		Fd = chid + 1.j*fd
		Fp = chip + 1.j*fp

		# F  = np.zeros_like(A)
		# Fb = np.zeros_like(A)
		# Fd = np.zeros_like(A)
		# Fp = np.zeros_like(A)

		Dr_rho    = self.D(rho, dz)
		Dr_rhop   = self.D(rhop, dz)
		Dr_sigma  = self.D(sigma, dz)
		Dr_sigmap = self.D(sigmap, dz)
		Dr_eta    = self.D(eta, dz)
		Dr_xi     = self.D(xi, dz)

		# Communicate first derivatives
		new_derivatives, _ = U.communicate(
			partial(ghost_point_processor),
			data=np.array([
				Dr_rho, Dr_rhop, Dr_sigma, Dr_sigmap, Dr_eta, Dr_xi
			])
		)

		Dr_rho, Dr_rhop, Dr_sigma, Dr_sigmap, Dr_eta, Dr_xi = new_derivatives

		# Calculate constraint quantities
		C1 = ((3.*pi*(B+1.) - rho*(mu - rho + B*(F + 3.*rho)) - \
				   B*rho*Fb - sigma*sigmab*(B-1.) - \
				   rho*(2.*rhop + mub)) / (rt2*A)) - Dr_rho
		C2 = ((-sigma*(3.*B*F + 3.*mu - 2.*rho + 4.*B*rho) + \
					B*Fb*sigma - rho*sigmapb*(B+1.) - \
					psi0*(B-1) + sigma*(mub + rhop*(B-1.))) / (rt2*A)) - \
					Dr_sigma
		C3 = ((3.*pi*(B-1) + rhop*(mu + mub + 2.*rho + B*(F + Fb)) - \
					  (rhop**2)*(3.*B+1) - \
					  sigmap*sigmapb*(B+1)) / (rt2*A)) - Dr_rhop
		C4 = ((-psi4*(B+1) - rhop*sigmab*(B-1) + \
					   sigmap*(3.*B*F + 3.*mu + rho + B*rho - B*Fb - \
					   mub - 2.*rhop*(1+2.*B))) / (rt2*A)) - Dr_sigmap
		
		C5 = (r2 * A**(-1.) * ((-1.) * B * eta * Fb + eta * ((-1.) * mub \
					+ (-1.) * rho + B * F + B * rho + B * rhop + mu + rhop) \
					+ etab * (((-1.) + B) * sigma + (1. + B) * sigmapb)) \
					+ Dr_eta)
		C6 = (r2 * A**(-1.) * ((-1.) * B * xi * Fb + (-1.) * sigma * xib \
				+ (-1.) * xi * mub + (-1.) * xi * rho + B * F * xi + B * sigma \
				* xib + B * xib * sigmapb + B * xi * rho + B * xi * rhop + mu \
				* xi + xib * sigmapb + xi * rhop) + Dr_xi)

		return np.array([C1, C2, C3, C4, C5, C6])