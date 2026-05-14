"""
Dedalus script simulating 2D horizontally-periodic Rayleigh-Benard convection.
Adapted from example on dedalus project website.

**************************
THIS IS THE FREE SLIP SCRIPT
**************************

This script specifically is designed to create a large data set of RB convection simulations in order to use as
training data for ML.

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 free_slip_data_creator.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5
"""

import numpy as np
import dedalus.public as d3
import logging
import pathlib 
logger = logging.getLogger(__name__)

def free_slip_rb(file_name, rayleigh_number, sim_time):
    # Parameters
    file_handler_name = file_name
    Lx, Lz = 1, 1
    Nx, Nz = 64, 64
    Rayleigh = rayleigh_number
    Prandtl = 1
    dealias = 3/2
    stop_sim_time = sim_time
    timestepper = d3.RK222
    max_timestep = 0.01
    dtype = np.float64
    dtmax=max_timestep

    # Bases
    coords = d3.CartesianCoordinates('x', 'z')
    dist = d3.Distributor(coords, dtype=dtype)
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
    zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

    # Fields
    p = dist.Field(name='p', bases=(xbasis,zbasis))
    b = dist.Field(name='b', bases=(xbasis,zbasis))
    u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
    tau_p = dist.Field(name='tau_p')
    tau_b1 = dist.Field(name='tau_b1', bases=xbasis)
    tau_b2 = dist.Field(name='tau_b2', bases=xbasis)
    tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
    tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)

    # Substitutions
    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)
    x, z = dist.local_grids(xbasis, zbasis)
    ex, ez = coords.unit_vector_fields(dist)
    lift_basis = zbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
    grad_b = d3.grad(b) + ez*lift(tau_b1) # First-order reduction
    w = d3.DotProduct(u,ez)
    bz = d3.DotProduct(grad_b, ez)

    x_hat, z_hat = coords.unit_vector_fields(dist)
    u_x = u @ x_hat
    u_z = u @ z_hat
    dzu_x = d3.Differentiate(u_x, coords["z"])

    # Problem
    # First-order form: "div(f)" becomes "trace(grad_f)"
    # First-order form: "lap(f)" becomes "div(grad_f)"
    problem = d3.IVP([p, b, u, tau_p, tau_b1, tau_b2, tau_u1, tau_u2], namespace=locals())
    problem.add_equation("trace(grad_u) + tau_p = 0")
    problem.add_equation("dt(b) - kappa*div(grad_b) + lift(tau_b2) = - u@grad(b)")
    problem.add_equation("dt(u) - nu*div(grad_u) + grad(p) - b*ez + lift(tau_u2) = - u@grad(u)")

    problem.add_equation("b(z=0) = Lz")   # Boundary condition for buoyancy
    problem.add_equation("b(z=Lz) = 0")   # Boundary condition for buoyancy

    problem.add_equation("dzu_x(z=0) = 0")
    problem.add_equation("dzu_x(z=Lz) = 0")

    problem.add_equation("u_z(z=0) = 0")
    problem.add_equation("u_z(z=Lz) = 0")

    problem.add_equation("integ(p) = 0") # Pressure gauge

    # Solver
    solver = problem.build_solver(timestepper)


    # Initial conditions or restart
    if not pathlib.Path('restart.h5').exists():

        b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
        b['g'] *= z * (Lz - z) # Damp noise at walls
        b['g'] += Lz - z # Add linear background

        fh_mode ='overwrite'
    else:
        # Restart
        write, last_dt = solver.load_state('restart.h5', -1)

        # Timestepping and output
        dtmax = last_dt
        stop_sim_time = 5000
        fh_mode = 'append'    
    solver.stop_sim_time = stop_sim_time

    # checkpointing (snapshots)


    snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.25, max_writes=50, mode=fh_mode)
    snapshots.add_tasks(solver.state)

    # other analysis tasks


    analysis = solver.evaluator.add_file_handler(file_handler_name, sim_dt=0.25, max_writes=50000)
    # Mean Re
    analysis.add_task(d3.Integrate(np.sqrt(u@u)/nu,  coords)/(Lx*Lz), layout='g', name='Re')
    analysis.add_task(d3.Average(np.sqrt(u@u)/nu , ('x', 'z')), layout='g', name='Reavg')

    # Nusselt
    analysis.add_task( 1.0 + d3.Integrate(b*w, coords)/(kappa*Lx*Lz), layout='g', name='Nusselt')

    # CFL
    CFL = d3.CFL(solver, initial_dt=dtmax, cadence=10, safety=0.5, threshold=0.05,
                max_change=1.5, min_change=0.5, max_dt=max_timestep)
    CFL.add_velocity(u)

    # Flow properties
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(np.sqrt(u@u)/nu, name='Re')

    # Main loop
    startup_iter = 10
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            timestep = CFL.compute_timestep()
            solver.step(timestep)
            if (solver.iteration-1) % 10 == 0:
                max_Re = flow.max('Re')
                logger.info('Iteration=%i, Time=%e, dt=%e, max(Re)=%f' %(solver.iteration, solver.sim_time, timestep, max_Re))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()

values = [1, 3, 5, 8]
rayleigh_nums = []
file_names = []
for power in range(8, 9):  # From 10^4 to 10^9
    base = 10**power
    for v in values:
        rayleigh_nums.append(v * base)
        file_names.append(f"FS_analysis3_{v * base:.0e}")

for i, x in enumerate(rayleigh_nums):
    free_slip_rb(file_names[i], x, 400)

