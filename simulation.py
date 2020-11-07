import numpy as np
import matplotlib.pyplot as plt
import time
import math

# Simulation parameters
N = 100
TIME_STEP = 0.0001
MAX_TIME = 10
RE = 100

# Relaxation parameters
MAX_ITERATIONS = 10
MAX_ERROR = 10 ** -6

# Successive Over Relaxation (SOR) parameters
OMEGA = 1.8
MAX_ITERATIONS_SOR = 1000
MAX_ERROR_SOR = 0.0001

# Velocities of walls in the positive x direction
VELOCITY_TOP = 1
VELOCITY_BOTTOM = 1
VELOCITY_LEFT = VELOCITY_RIGHT = 0


def sor_stream_function(sf, vt, h):
    """
        Update stream function (sf) using Successive Over Relaxation (SOR) to solve the system of linear equations

        Args:
            sf: stream function
            vt: vorticity
            h: grid cell length, 1 / grid max length
    """

    for _ in range(MAX_ITERATIONS_SOR):
        sf_old = sf.copy()

        for i in range(1, N - 1):
            for j in range(1, N - 1):
                sf[i, j] = (0.25 * OMEGA * (sf[i + 1, j] + sf[i - 1, j] + sf[i, j + 1] + sf[i, j - 1]
                                            + h * h * vt[i, j])
                            + (1 - OMEGA) * sf[i, j])

        residual = np.sum(np.abs(sf_old - sf))
        if residual <= MAX_ERROR_SOR:
            break

    return sf


def relax_stream_function(sf, vt, h):
    """
        Update stream function (sf) using ordinary relaxation to solve the system of linear equations
        This function does not use Successive Over Relaxation (SOR), which enables it to use
        numpy vectors for a significant speed-up

        Args:
            sf: stream function
            vt: vorticity
            h: grid cell length, 1 / grid max length
    """

    for _ in range(MAX_ITERATIONS):
        sf_old = sf.copy()

        sf[1:-1, 1:-1] = (0.25 * (sf[2:, 1:-1] + sf[0:-2, 1:-1] + sf[1:-1, 2:] + sf[1:-1, 0:-2]
                                  + h * h * vt[1:-1, 1:-1]))

        residual = np.sum(np.abs(sf_old - sf))
        if residual <= MAX_ERROR:
            break

    return sf


def update_vorticity_boundaries(sf, vt, h):
    """
        Update vorticity on the boundaries

        Args:
            sf: stream function
            vt: vorticity
            h: grid cell length, 1 / grid max length
    """

    # Top wall (moving lid)
    vt[N - 1, 1:N - 1] = -2 * sf[N - 2, 1:N - 1] / (h * h) + VELOCITY_TOP * 2 / h
    # Bottom wall
    vt[0, 1:N - 1] = -2 * sf[1, 1:N - 1] / (h * h) - VELOCITY_BOTTOM * 2 / h
    # Left wall
    vt[1:N - 1, 0] = -2 * sf[1:N - 1, 1] / (h * h) + VELOCITY_LEFT * 2 / h
    # Right wall
    vt[1:N - 1, N - 1] = -2 * sf[1:N - 1, N - 2] / (h * h) - VELOCITY_RIGHT * 2 / h

    return vt


def update_vorticity_interior(sf, vt, w, h):
    """
        Update vorticity on the interior

        Args:
            sf: stream function
            vt: vorticity
            w: old right hand side
            h: grid cell length, 1 / grid max length
    """

    # Calculate right hand side
    w[1:-1, 1:-1] = (-(((sf[1:-1, 2:] - sf[1:-1, 0:-2]) * (vt[2:, 1:-1] - vt[0:-2, 1:-1])
                        - (sf[2:, 1:-1] - sf[0:-2, 1:-1]) * (vt[1:-1, 2:] - vt[1:-1, 0:-2])) / (4 * h * h))
                     + (1 / RE) * ((vt[2:, 1:-1] + vt[0:-2, 1:-1] + vt[1:-1, 2:] + vt[1:-1, 0:-2]
                                    - 4 * vt[1:-1, 1:-1]) / (h * h)))

    # Update interior vorticity
    vt += TIME_STEP * w

    return w, vt


def calculate_velocity(sf, h):
    velocity_x = np.zeros((N, N), dtype='float64')
    velocity_y = np.zeros((N, N), dtype='float64')

    # Set horizontal velocity at the boundaries
    velocity_x[N - 1, :] = VELOCITY_TOP
    velocity_x[0, :] = VELOCITY_BOTTOM
    velocity_y[:, 0] = VELOCITY_LEFT
    velocity_y[:, N - 1] = VELOCITY_RIGHT

    # Calculate velocity for the interior
    # Origin is at bottom left instead of top left, so the x and y velocities are swapped
    velocity_x[1:-1, 1:-1] = -(sf[2:, 1:-1] - sf[0:-2, 1:-1]) / (2 * h)
    velocity_y[1:-1, 1:-1] = (sf[1:-1, 2:] - sf[1:-1, 0:-2]) / (2 * h)
    velocity_total = np.sqrt(np.absolute(velocity_x + velocity_y))

    return velocity_x, velocity_y, velocity_total


def sor_pressure_field(sf, h):
    """
        Update pressure field (pf) using Successive Over Relaxation (SOR) to solve the system of linear equations

        Args:
            sf: stream function
            h: grid cell length, 1 / grid max length
    """

    pf = np.zeros((N, N), dtype='float64')

    for _ in range(MAX_ITERATIONS_SOR):
        pf_old = pf.copy()

        for i in range(1, N - 1):
            for j in range(1, N - 1):
                rhs = calculate_pressure_field(sf, h)

                pf[i, j] = (OMEGA * 0.25 * (pf_old[i + 1, j] + pf[i - 1, j]
                                            + pf_old[i, j + 1] + pf[i, j - 1] - h * h * rhs[i, j])
                            + (1 - OMEGA) * pf_old[i, j])

        if np.sum(np.abs(pf_old - pf)) <= MAX_ERROR_SOR:
            break

    return pf


def relax_pressure_field(sf, h):
    """
        Update pressure field (pf) using ordinary relaxation to solve the system of linear equations
        This function does not use Successive Over Relaxation (SOR), which enables it to use
        numpy vectors for a significant speed-up

        Args:
            sf: stream function
            h: grid cell length, 1 / grid max length
    """

    pf = np.zeros((N, N), dtype='float64')

    for _ in range(MAX_ITERATIONS):
        pf_old = pf.copy()
        rhs = calculate_pressure_field(sf, h)

        pf[1:-1, 1:-1] = 0.25 * (pf_old[2:, 1:-1] + pf[0:-2, 1:-1]
                                 + pf_old[1:-1, 2:] + pf[1:-1, 0:-2] - h * h * rhs[1:-1, 1:-1])

        if np.sum(np.abs(pf_old - pf)) <= MAX_ERROR:
            break

    return pf


def calculate_pressure_field(sf, h):
    """
        Calculate pressure field (pf) by solving the Poisson equation

        Args:
            sf: stream function
            h: grid cell length, 1 / grid max length
    """

    pressure_field = np.zeros((N, N), dtype='float64')
    pressure_field[1:-1, 1:-1] = 2 * (((sf[2:, 1:-1] + sf[0:-2, 1:-1] - 2 * sf[1:-1, 1:-1]) / (h * h))
                                      * ((sf[1:-1, 2:] + sf[1:-1, 0:-2] - 2 * sf[1:-1, 1:-1]) / (h * h))
                                      - ((sf[2:, 2:] - sf[2:, 0:-2] - sf[0:-2, 2:] + sf[0:-2, 0:-2])
                                         / (4 * h * h)) ** 2)

    return pressure_field


def plot_results(velocity_x, velocity_y, velocity_total, pressure_field, sf, vt, horizontal_velocities_center):
    plt.title('Horizontal velocity')
    velocity_plot = plt.contourf(velocity_x)
    clb = plt.colorbar(velocity_plot)
    clb.ax.set_title('m/s')
    plt.show()

    plt.title('Vertical velocity')
    velocity_plot = plt.contourf(velocity_y)
    clb = plt.colorbar(velocity_plot)
    clb.ax.set_title('m/s')
    plt.show()

    plt.title('Total velocity')
    velocity_plot = plt.contourf(velocity_total)
    clb = plt.colorbar(velocity_plot)
    clb.ax.set_title('m/s')
    plt.show()

    plt.title('Velocity vectors')
    plt.quiver(velocity_x, velocity_y)
    plt.show()

    plt.title('Pressure field')
    pressure_field_plot = plt.contourf(pressure_field)
    clb = plt.colorbar(pressure_field_plot)
    clb.ax.set_title('Pa')
    plt.show()

    plt.title('Velocity vectors overlaid on the pressure field')
    pressure_field_plot = plt.contourf(pressure_field)
    clb = plt.colorbar(pressure_field_plot)
    clb.ax.set_title('Pa')
    plt.quiver(velocity_x, velocity_y, color='k')
    plt.show()

    plt.title('Stream function')
    sf_plot = plt.contourf(sf)
    clb = plt.colorbar(sf_plot)
    clb.ax.set_title(r'$\mathregular{m^2/s}$')
    plt.show()

    plt.title('Velocity vectors overlaid on the stream function')
    sf_plot = plt.contourf(sf)
    clb = plt.colorbar(sf_plot)
    clb.ax.set_title(r'$\mathregular{m^2/s}$')
    plt.quiver(velocity_x, velocity_y, color='k')
    plt.show()

    plt.title('Vorticity')
    vt_plot = plt.contourf(vt)
    plt.colorbar(vt_plot)
    plt.show()

    plt.title('Velocity vectors overlaid on the vorticity')
    vt_plot = plt.contourf(vt)
    plt.colorbar(vt_plot)
    plt.quiver(velocity_x, velocity_y, color='k')
    plt.show()

    plt.title('Horizontal velocity over time')
    plt.ylabel('Horizontal velocity (m/s)')
    plt.xlabel('Time (s)')
    plt.plot(horizontal_velocities_center)
    plt.show()


def main():
    start = time.time()

    # Initialize stream function (sf), vorticity (vt) and a matrix w
    # that is used to hold the rhs of the vorticity equation (1b)
    sf = np.zeros((N, N), dtype='float64')
    w = np.zeros((N, N), dtype='float64')
    vt = np.zeros((N, N), dtype='float64')

    # Grid cell length
    h = 1 / N

    velocity_center_old = 0
    horizontal_velocities_center = []

    max_steps = math.floor(MAX_TIME / TIME_STEP)
    for i in range(max_steps):
        # Calculate time on each iteration to prevent accumulating floating point precision errors
        t = i * TIME_STEP
        if i % 1000 == 0:
            print(t)
            np.save(f'saves/SF_N={N}_RE={RE}_TIME_STEP={TIME_STEP}', sf)
            np.save(f'saves/VT_N={N}_RE={RE}_TIME_STEP={TIME_STEP}', vt)
            np.save(f'saves/velocities_N={N}_RE={RE}_TIME_STEP={TIME_STEP}', horizontal_velocities_center)

        sf = relax_stream_function(sf, vt, h)
        vt = update_vorticity_boundaries(sf, vt, h)
        w, vt = update_vorticity_interior(sf, vt, w, h)

        # Probe center for horizontal velocity
        velocity_center = -(sf[N // 2 + 1, N // 2] - sf[N // 2 - 1, N // 2]) / (2 * h)
        horizontal_velocities_center.append(velocity_center)

    print(f'Elapsed time: {time.time() - start}')

    # Save results for future plotting/post-processing
    np.save(f'saves/SF_N={N}_RE={RE}_TIME_STEP={TIME_STEP}', sf)
    np.save(f'saves/VT_N={N}_RE={RE}_TIME_STEP={TIME_STEP}', vt)
    np.save(f'saves/velocities_N={N}_RE={RE}_TIME_STEP={TIME_STEP}', horizontal_velocities_center)

    # Compute pressure and velocity from stream function
    velocity_x, velocity_y, velocity_total = calculate_velocity(sf, h)
    pressure_field = relax_pressure_field(sf, h)

    # Plot results
    plot_results(velocity_x, velocity_y, velocity_total, pressure_field, sf, vt, horizontal_velocities_center)


main()
