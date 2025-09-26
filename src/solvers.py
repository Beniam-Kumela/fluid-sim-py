import numpy as np

def diffuse(b, x, x0, diff, dt, N, boundary_mask):
    '''
    Solves the diffusion equation of the following form:
    dx/dt = nu * del^2 * x
    where x is the property vector, nu is the viscosity and del^2 represents the Laplacian operator. In this case, the Gauss-Seidel Solver is solving a linear equation of the following form:
    x - a * del^2(x) = x_0.

    Args:
        b (int): boundary condition choices (static = 0, x = 1, y = 2).
        x (NxN matrix): current property matrix.
        x0 (NxN matrix): property matrix from previous time step.
        diff (float): diffusion coefficient (for current property like velocity or density).
        dt (float): time interval.
        N (int): grid dimension.
        boundary_mask (NxN matrix - bool): stores position of drawn boundaries.
    '''

    a = dt * diff * (N - 2) ** 2 # (N-2)^2 comes from normalizing unit grid - not included in Laplacian operation within lin_solve().
    lin_solve(b, x, x0, a, c=1+6*a, N=N, boundary_mask=boundary_mask) # use 1+6*a instead of 1+4*a for convergence stability (over-relaxation).

def lin_solve(b, x, x0, a, c, N, boundary_mask, iter=16):
    '''
    Solves Poisson-like linear system of the form: x - a * del^2(x) = b. This discretizes into sparse linear system of form: Ax = b. The solution is given by Gauss-Seidel relaxation which scales O(N), compare this to the true Gauss-Seidel method which scales O(N^2). This is done using the following formula:
    x_new(i, j) = (b(i, j) + a * (x(i-1, j) + x(i+1, j) + x(i, j-1) + x(i, j+1))) / c.
    
    Args:
        b (int): boundary condition choices (static = 0, x = 1, y = 2).
        x (NxN matrix): current property matrix.
        x0 (NxN matrix): property matrix from previous time step.
        a (float): dampening factor (depends on time interval, diffusion coefficient, and grid dimension).
        c (float): normalization factor (depends on a).
        N (int): grid dimension.
        boundary_mask (NxN matrix - bool): stores position of drawn boundaries.
    '''
    
    cRecip = 1.0 / c
    for i in range(iter):
        x[1:-1, 1:-1] = (x0[1:-1, 1:-1] + a * (
            x[:-2, 1:-1] + 
            x[2:, 1:-1] +
            x[1:-1, :-2] + 
            x[1:-1, 2:]
        )) * cRecip
        set_bnd(b, x, N, boundary_mask)

def project(velocX, velocY, p, div, N, boundary_mask):
    '''
    Maintains incompressibility condition using Helmholtz-Hodge decomposition.

    Args:
        velocX (NxN matrix): x-components of velocity matrix.
        velocY (NxN matrix): y-components of velocity matrix.
        p (NXN matrix): pressure matrix.
        div (NxN matrix): divergence matrix.
        N (int): grid dimension.
        boundary_mask (NxN matrix - bool): stores position of drawn boundaries.
    '''

    # Compute divergence from velocity fields.
    div[1:-1, 1:-1] = -0.5 * (
        velocX[2:, 1:-1] - velocX[:-2, 1:-1]+ 
        velocY[1:-1, 2:] - velocY[1:-1, :-2]
    ) / N
    p[1:-1, 1:-1] = 0 # Initialize pressure field to 0.

    # Enforce boundary conditions for divergence and pressure fields. 
    set_bnd(0, div, N, boundary_mask)
    set_bnd(0, p, N, boundary_mask)

    # Solve del^2(p) = div
    lin_solve(0, p, div, 1, 6, N, boundary_mask=boundary_mask)

    # Remove divergent component from velocity: u = u - del(p).
    velocX[1:-1, 1:-1] -= 0.5 * (p[2:, 1:-1] - p[:-2, 1:-1]) * N
    velocY[1:-1, 1:-1] -= 0.5 * (p[1:-1, 2:] - p[1:-1, :-2]) * N
    '''
    for j in range(1, N - 1):
        for i in range(1, N - 1):
            velocX[i, j] -= 0.5 * (p[i + 1, j] - p[i - 1, j]) * N
            velocY[i, j] -= 0.5 * (p[i, j + 1] - p[i, j - 1]) * N
    '''
    
    # Reinforce boundary conditions for new velocity fields.
    set_bnd(1, velocX, N, boundary_mask)
    set_bnd(2, velocY, N, boundary_mask)

def advect(b, d, d0, velocX, velocY, dt, N, boundary_mask):
    '''
    Calculates current velocity matrix through a linear interpolation of neighboring velocities from previous time step.

    Args:
        b (int): boundary condition choices (static = 0, x = 1, y = 2).
        d (NxN matrix): current velocity matrix.
        d0 (NxN matrix): velocity matrix from previous time step.
        velocX (NxN matrix): x-components of velocity matrix.
        velocY (NxN matrix): y-components of velocity matrix.
        dt (float): time interval.
        N (int): grid dimension.
        boundary_mask (NxN matrix - bool): stores position of drawn boundaries.
    '''

    # Scale factors for moving through the grid based on time step.
    dtx = dt * (N - 2)
    dty = dt * (N - 2)
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")

    # Backtrace particle positions.
    x = i - dtx * velocX
    y = j - dty * velocY
    
    # Clamp backtraced positions to stay in domain.
    x = np.clip(x, 0.5, N - 0.5)
    y = np.clip(y, 0.5, N - 0.5)

    # Get integer neighbor indices around each backtraced position.
    i0 = np.floor(x).astype(np.int32)
    i1 = i0 + 1
    j0 = np.floor(y).astype(np.int32)
    j1 = j0 + 1

    # Clamp neighboring indices to valid range.
    i0 = np.clip(i0, 0, N - 1)
    i1 = np.clip(i1, 0, N - 1)
    j0 = np.clip(j0, 0, N - 1)
    j1 = np.clip(j1, 0, N - 1)

    # Compute interpolation weights.
    s1 = x - i0 # right
    s0 = 1 - s1 # left
    t1 = y - j0 # top
    t0 = 1 - t1 # bottom

    # Bilinear interpolation.
    d[...] = (
        s0 * (t0 * d0[i0, j0] + t1 * d0[i0, j1]) +
        s1 * (t0 * d0[i1, j0] + t1 * d0[i1, j1])
    )

    set_bnd(b, d, N, boundary_mask)

def set_bnd(b, x, N, boundary_mask):
    '''
    Reinforces wall boundary conditions.

    Args:
        b (int): boundary condition choices (static = 0, x = 1, y = 2).
        x (NxN matrix): current property matrix.
        N (int): grid dimension.
        boundary_mask (NxN matrix - bool): stores position of drawn boundaries.
    '''

    # Reverse direction of fluid one pixel away from wall if reaches boundary.
    x[1:-1, 0] = -x[1:-1, 1] if b == 2 else x[1:-1, 1] # bottom
    x[1:-1, N - 1] = -x[1:-1, N - 2] if b == 2 else x[1:-1, N - 2] # top
    x[0, 1:-1] = -x[1, 1:-1] if b == 1 else x[1, 1:-1] # left
    x[N - 1, 1:-1] = -x[N - 2, 1:-1] if b == 1 else x[N - 2, 1:-1] # right

    # If fluid reaches a corner, average nearest neighbor pixels.
    x[0, 0] = 0.5 * (x[1, 0] + x[0, 1]) # bottom left
    x[0, N - 1] = 0.5 * (x[1, N - 1] + x[0, N - 2]) # top left
    x[N - 1, 0] = 0.5 * (x[N - 2, 0] + x[N - 1, 1]) # bottom right
    x[N - 1, N - 1] = 0.5 * (x[N - 2, N - 1] + x[N - 1, N - 2]) # top right

    # Zero velocities for points on boundary mask (no-slip condition, see: https://en.wikipedia.org/wiki/No-slip_condition).
    if boundary_mask is not None:
        x[boundary_mask] = 0.0