.. flashlight documentation master file, created by
   sphinx-quickstart on Tue Sep 27 16:33:20 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2

Welcome to Flashlight
=====================

Flashlight is a lightweight Python library for analyzing and solving quadrotor control problems.
Flashlight enables you to easily solve for minimum snap trajectories that go through a sequence of waypoints, compute the required control forces along trajectories, execute the trajectories in a physics simulator, and visualize the simulation results.
Flashlight also makes it easy to simulate external disturbances, and to recover from those disturbances using time-varying LQR feedback control.
Flashlight includes physical models for 2D quadrotors, 3D quadrotors, and 3D quadrotor cameras.

The following code snippet shows how easy it is to start analyzing quadrotor trajectories using Flashlight.
In this code snippet, we generate the control forces required for a 2D quadrotor to follow a simple trajectory, and simulate the results::

    from pylab import *; import scipy.integrate

    import flashlight.interpolate_utils as interpolate_utils
    import flashlight.quadrotor_2d      as quadrotor_2d

    # Define a simple position trajectory in 2D.
    num_samples = 200
    t_begin     = 0
    t_end       = pi
    dt          = (t_end - t_begin) / (num_samples - 1)

    t = linspace(t_begin, t_end, num_samples)
    p = c_[ sin(2*t) + t, t**2 ]

    # Compute the corresponding state space trajectory and control trajectories for a 2D quadrotor.
    q_qdot_qdotdot = quadrotor_2d.compute_state_space_trajectory_and_derivatives(p, dt)
    u              = quadrotor_2d.compute_control_trajectory(q_qdot_qdotdot)

    # Define a function that interpolates the control trajectory in between time samples.
    u_interp_func = interpolate_utils.interp1d_vector_wrt_scalar(t, u, kind="cubic")

    # Define a simulation loop.
    def compute_x_dot(x_t, t):

        # Get the current control vector.
        u_t = u_interp_func(clip(t, t_begin, t_end))
        
        # Compute the state derivative from the current state and current control vectors.
        x_dot_t = quadrotor_2d.compute_x_dot(x_t, u_t).A1

        return x_dot_t

    # Simulate.
    x_nominal, _, _, _ = quadrotor_2d.pack_state_space_trajectory_and_derivatives(q_qdot_qdotdot)
    x_0                = x_nominal[0]
    x_sim              = scipy.integrate.odeint(compute_x_dot, x_0, t)

    # Plot the results.
    quadrotor_2d.draw(t, x_sim, t_nominal=t, x_nominal=x_nominal, inline=True)

This code example produces the following animation, which shows our 2D quadrotor correctly following the intended trajectory:

.. raw:: html

    <video width="648.0" height="432.0" controls autoplay loop>
      <source type="video/mp4" src="_static/flashlight/welcome_to_flashlight.mp4" />
    </video>

Flashlight is designed and implemented by `Mike Roberts <http://graphics.stanford.edu/~mlrobert>`_.

Installing Flashlight
=====================

The steps for installing Flashlight are as follows.

1. Install all of Flashlight's dependencies.
The core functionality in Flashlight depends on the following Python libraries:

    * `IPython <https://ipython.org>`_
    * `NumPy <http://www.numpy.org>`_
    * `Matplotlib <http://matplotlib.org>`_
    * `scikit-learn <http://scikit-learn.org>`_
    * `SciPy <http://scipy.org>`_
    * `SymPy <http://www.sympy.org>`_

  Many of the example notebooks, and some of the debug rendering functions in Flashlight, depend on the following Python libraries:

    * `Mayavi <http://code.enthought.com/projects/mayavi>`_
    * `OpenCV <http://opencv.org>`_ (specifically the ``cv2`` Python module)
    * `The Python Control Systems Library <https://pypi.python.org/pypi/control/0.7.0>`_
    * `VTK <http://www.vtk.org>`_ (specifically the ``vtk`` Python module)

  Each of these dependencies comes pre-installed with `Enthought Canopy <https://www.enthought.com/products/canopy>`_, or can be installed very easily using the using the Enthought Canopy package manager, or `pip <https://pypi.python.org/pypi/pip>`_.

2. Download the Flashlight source code from our `GitHub repository <http://github.com/mikeroberts3000/flashlight>`_.

3. In any folder you'd like to use Flashlight, copy our ``path/to/flashlight/code/utils/path_utils.py`` file into that folder.

4. Include the following code snippet in your Python code before importing Flashlight::

    import path_utils
    path_utils.add_relative_to_current_source_file_path_to_sys_path("path/to/flashlight/code/lib")

5. Verify that you can import Flashlight::

    import flashlight
    print flashlight.__version__

After completing these steps, you're ready to start using Flashlight.

Examples
========

These example notebooks describe how to use Flashlight to analyze and solve a variety of quadrotor control problems:

* `Creating minimum snap splines in 1D <http://nbviewer.jupyter.org/github/mikeroberts3000/flashlight/blob/master/code/examples/jupyter/01%20-%20Creating%20minimum%20snap%20splines%20in%201D.ipynb>`_
* `Creating minimum snap splines in 2D and 3D <http://nbviewer.jupyter.org/github/mikeroberts3000/flashlight/blob/master/code/examples/jupyter/02%20-%20Creating%20minimum%20snap%20splines%20in%202D%20and%203D.ipynb>`_
* `Re-timing the progress along a parametric curve <http://nbviewer.jupyter.org/github/mikeroberts3000/flashlight/blob/master/code/examples/jupyter/03%20-%20Re-timing%20the%20progress%20along%20a%20parametric%20curve.ipynb>`_
* `Computing the control forces required for a 2D quadrotor to follow a trajectory <http://nbviewer.jupyter.org/github/mikeroberts3000/flashlight/blob/master/code/examples/jupyter/04%20-%20Computing%20the%20control%20forces%20required%20for%20a%202D%20quadrotor%20to%20follow%20a%20trajectory.ipynb>`_
* `Applying LQR feedback control to stabilize a 2D quadrotor at a point <http://nbviewer.jupyter.org/github/mikeroberts3000/flashlight/blob/master/code/examples/jupyter/05%20-%20Applying%20LQR%20feedback%20control%20to%20stabilize%20a%202D%20quadrotor%20at%20a%20point.ipynb>`_
* `Applying time-varying LQR feedback control to stabilize a 2D quadrotor along a trajectory <http://nbviewer.jupyter.org/github/mikeroberts3000/flashlight/blob/master/code/examples/jupyter/06%20-%20Applying%20time-varying%20LQR%20feedback%20control%20to%20stabilize%20a%202D%20quadrotor%20along%20a%20trajectory.ipynb>`_
* `Applying LQR feedback control in 2D, handling angular wrapping correctly <http://nbviewer.jupyter.org/github/mikeroberts3000/flashlight/blob/master/code/examples/jupyter/07%20-%20Applying%20LQR%20feedback%20control%20in%202D%2C%20handling%20angular%20wrapping%20correctly.ipynb>`_
* `Computing the control forces required for a 3D quadrotor to follow a trajectory <http://nbviewer.jupyter.org/github/mikeroberts3000/flashlight/blob/master/code/examples/jupyter/08%20-%20Computing%20the%20control%20forces%20required%20for%20a%203D%20quadrotor%20to%20follow%20a%20trajectory.ipynb>`_
* `Applying LQR feedback control to stabilize a 3D quadrotor at a point <http://nbviewer.jupyter.org/github/mikeroberts3000/flashlight/blob/master/code/examples/jupyter/09%20-%20Applying%20LQR%20feedback%20control%20to%20stabilize%20a%203D%20quadrotor%20at%20a%20point.ipynb>`_
* `Applying time-varying LQR feedback control to stabilize a 3D quadrotor along a trajectory <http://nbviewer.jupyter.org/github/mikeroberts3000/flashlight/blob/master/code/examples/jupyter/10%20-%20Applying%20time-varying%20LQR%20feedback%20control%20to%20stabilize%20a%203D%20quadrotor%20along%20a%20trajectory.ipynb>`_
* `Applying LQR feedback control in 3D, handling angular wrapping correctly <http://nbviewer.jupyter.org/github/mikeroberts3000/flashlight/blob/master/code/examples/jupyter/11%20-%20Applying%20LQR%20feedback%20control%20in%203D%2C%20handling%20angular%20wrapping%20correctly.ipynb>`_
* `Computing the control forces required for a 3D quadrotor camera to follow a look-from look-at trajectory <http://nbviewer.jupyter.org/github/mikeroberts3000/flashlight/blob/master/code/examples/jupyter/12%20-%20Computing%20the%20control%20forces%20required%20for%20a%203D%20quadrotor%20camera%20to%20follow%20a%20look-from%20look-at%20trajectory.ipynb>`_

Reproducible Research
=====================

As part of the Flashlight source code, we include notebooks that reproduce the experimental results from our research papers.

| `An Interactive Tool for Designing Quadrotor Camera Shots <http://stanford-gfx.github.io/Horus>`_
| Niels Joubert, Mike Roberts, Anh Truong, Floraine Berthouzoz, Pat Hanrahan
| *ACM Transactions on Graphics 35(4) (SIGGRAPH 2016)*

* `Evaluating different interpolation methods <http://nbviewer.jupyter.org/github/mikeroberts3000/flashlight/blob/master/code/experiments/00_siggraph_asia_2015/jupyter/00%20-%20Evaluating%20different%20interpolation%20methods.ipynb>`_
* `Evaluating different spline degrees and derivatives <http://nbviewer.jupyter.org/github/mikeroberts3000/flashlight/blob/master/code/experiments/00_siggraph_asia_2015/jupyter/01%20-%20Evaluating%20different%20spline%20degrees%20and%20derivatives.ipynb>`_

Citing Flashlight
=================

If you use Flashlight for published work, we encourage you to cite it as follows::

    @misc{flashlight:2016,
        author = {Mike Roberts},
        title  = {Flashlight: A Python Library for Analyzing and Solving Quadrotor Control Problems},
        year   = {2016},
        url    = {http://mikeroberts3000.github.io/flashlight}
    }

Additionally, if you use any of the functions in ``quadrotor_3d``, ``quadrotor_camera_3d``, or ``spline_utils`` for published work, we encourage you to cite the following paper::

    @article{joubert:2015,
        author  = {Niels Joubert AND Mike Roberts AND Anh Truong AND Floraine Berthouzoz AND Pat Hanrahan},
        title   = {An Interactive Tool for Designing Quadrotor Camera Shots},
        journal = {ACM Transactions on Graphics (SIGGRAPH Asia 2015)},
        year    = {2015}
    }
