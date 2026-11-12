# Research basis and reconstruction scope

The supplied recording (`output/recording.twin`) contains source video, image
landmarks and a schema-2 skeletal fit. Its subject is upper body only and its
dimensions are in MediaPipe-estimated meters, not calibrated metric measurements.
The reconstruction must retain that qualification.

The [UVA Baek lab page](https://datascience.virginia.edu/labs/baek-lab)
identifies physics-aware learning and sports biomechanics as research areas.
Baek's [2025 UVA interview](https://datascience.virginia.edu/news/qa-stephen-baek-ai-extreme-physics-and-future-innovation)
discusses governing equations in learning models. These establish relevant research
context, but do not specify a reverse-FEM human reconstruction algorithm. The exact
paper intended by the user remains unresolved; this implementation must not be
represented as a reproduction of it.

[Ichim's 2017 thesis](https://infoscience.epfl.ch/entities/publication/c8e08f4c-260c-4927-9ef9-bd5d54b11a91)
describes inverse physics for anatomical reconstruction from actor scans and
subsequent animation. Its scan input is materially richer than monocular joint
estimates. [PhysTwin](https://arxiv.org/abs/2503.17973) reconstructs deformable
objects using inverse modeling and a spring-mass representation. It is related
context, not the FEM constitutive model used here or a human anatomy validation.

## Implemented inverse problem

For reference tetrahedron matrices Dm and deformed matrices Ds, F = Ds Dm^-1.
Let R be the nearest proper rotation from polar decomposition. The elastic energy
is integrated over tetrahedron rest volumes using a corotated isotropic density:

    psi(F) = mu ||F - R||_F^2 + lambda/2 (trace(R^T F) - 3)^2.

Add weighted squared distances to skeletal skinning targets and a log barrier
that becomes infinite at det(F)=0.05. The unknowns are deformed vertex positions.
Material parameters remain priors; reference radii are priors except where
supported video widths replace them. This regularizes a volumetric displacement
field; it does not estimate internal anatomy or material parameters. Corotation
handles large rigid rotation. Feasible initialization and determinant backtracking
keep the returned tetrahedra above the barrier threshold. A sparse projected
Newton method uses finite differences of analytic element stresses to construct
an element Hessian, clamps its negative eigenvalues, and assembles a sparse search
system. Energy and gradient remain analytic. No collision, dynamic equilibrium,
muscle activation, contact force, or injury inference is included.

## Accuracy interpretation

Small target error means consistency with the saved skeleton. It cannot demonstrate
agreement with unseen body surfaces or true 3D joint positions. The existing image
reprojection diagnostic is preserved as a separate upstream statistic. A useful
next stage for shape identification would add calibrated multiview surface or
depth evidence, measured body scale, and independent validation. Material fitting
would additionally need informative load/boundary data. These are conclusions
about the inverse problem's missing constraints, not results established by this
recording.

## Paper selection after the user's follow-up

"Best" here means relevant to the requested inverse mechanics and compatible with
the available measurements, rather than a ranking of all human reconstruction
benchmarks.

| Primary work | Relevant contribution | What is used here / missing |
| --- | --- | --- |
| [Kadlecek et al., SIGGRAPH Asia 2016](https://users.cs.utah.edu/~ladislav/kadlecek16reconstructing/kadlecek16reconstructing.pdf), *Reconstructing Personalized Anatomical Models for Physics-based Body Animation* | Inverse physics fits subject and pose parameters to 3D surface scans | Closest conceptual match. We implement a smaller displacement inverse problem and conservative visible surface-width fitting; we have no actor scans or anatomical template and do not implement the paper's full anatomy reconstruction. |
| [Smith, de Goes, Kim, TOG 2018](https://www.tkim.graphics/NEO/StableNeoHookean2018.pdf), *Stable Neo-Hookean Flesh Simulation* | Robust large-deformation hyperelastic modeling and Hessian projection | We adopt spectral positive-semidefinite projection of element Hessians as a numerical technique. Our energy is corotated elasticity plus a determinant barrier; our Hessian is estimated from stress differences. We do not implement this paper's Neo-Hookean energy or closed-form eigensystem. |
| [Yang et al., ICCV 2023](https://gengshan-y.github.io/ppr/), *PPR: Physically Plausible Reconstruction from Monocular Videos* | Alternates visual reconstruction and differentiable dynamics/contact optimization | Demonstrates the value of coupling image and mechanics objectives. Its full-body contact/dynamics system is not adopted because this clip does not observe feet, the floor, or applied loads. |
| [Wang et al., CGF 2020](https://diglib.eg.org/items/f22c248d-b84f-49d9-a37b-81e405f8f550), *Monocular Human Pose and Shape Reconstruction using Part Differentiable Rendering* | Uses body part segmentation to constrain shape and occlusion | Supports adding surface evidence. We use the installed pose model's foreground mask for conservative arm-width estimates, not its learned part renderer or full model. |

The video extension derives widths from repeated high-confidence, untruncated,
isolated arm cross-sections. These constrain the **visible clothed silhouette**;
they do not reveal internal anatomy or hidden depth. Sampling, rejection, and
fallback-to-prior counts are exported so unsupported girths remain explicit.
