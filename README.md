# EP-PINNs

Implementation of Physics-Informed Neural Networks (PINNs) for the solution of electrophysiology (EP) problems in forward and inverse mode. EP-PINNs is currently implemented with the Aliev-Panfilov EP model (Aliev & Panfilov, Chaos, 1996) and the monodomain equation in isotropic 1D and 2D settings. EP-PINNs use the DeepXDE library (Lu et al, SIAM, 2021).

Both forward and inverse modes provide a high-resolution spatio-temporal representation of the action potential. The inverse mode can additionally estimate global model parameters: a; b; D; a and D; b and D and spatial heterogeneities in D.

We also include Matlab files to simulate the cardiac (ventricular) Aliev-Panfilov electrophysiology model in a cable (1D) and rectangular (2D) domain with Neumann boundary conditions and several initial conditions (single ectopic focus, planar wave, spiral wave). The code also supports the inclusion of heterogeneities in D (which are rectangular, for the time being). Centred finite differences and a 4-step explicit Runge-Kutta solvers are used throughout. The Matlab code can be used to provide ground truth data to the EP-PINNs solver.

Further details can be found in a soon to be published article. In the meantime, email marta.varela (at) imperial.ac.uk with any comments or questions.


<img width="259" alt="ExampleAlievPanfilov" src="https://user-images.githubusercontent.com/83647272/139454087-c50c4ddb-342d-454c-8332-9aada8c4967c.jpeg">
