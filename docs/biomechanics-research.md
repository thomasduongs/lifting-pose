# Research basis for the biomechanical human model

The user's selected priority is biomechanical simulation. This workflow therefore
uses an articulated reference skeleton and sourced muscle actuators, geometric
musculotendon paths, joint constraints and explicit passive-tissue coverage.
It supersedes the earlier coarse FEM scaffold as the simulation foundation.

## Starting with UVA M3

The [M3 lab](https://engineering.virginia.edu/labs-groups/multiscale-muscle-mechanophysiology-m3-lab)
studies muscle across tissue, muscle and whole-body scales. Its selected work
includes MRI-based continuum muscle models. These are relevant precisely because
they distinguish internal fiber mechanics from the simpler line-of-action models
commonly used for whole-body motion. A monocular motion recording does not provide
the MRI geometry, fiber architecture, or loads needed to identify those tissues.

| Primary research | Contribution and implementation decision |
| --- | --- |
| [Blemker, Pinsky & Delp (2005)](https://pubmed.ncbi.nlm.nih.gov/15713285/), DOI10.1016/j.jbiomech.2004.04.009 | Biceps continuum mechanics explains nonuniform tissue strain. This guides the accuracy boundary: our whole-body Hill-type actuators do not reproduce those MRI-based fiber strains. |
| [Blemker & Delp (2005)](https://pubmed.ncbi.nlm.nih.gov/15981866/), DOI10.1007/s10439-005-1433-7 | MRI-derived 3D muscle architecture and variable fiber moment arms. We preserve named reference geometry/routes rather than inventing muscle bellies from skeleton capsules; full continuum geometry reconstruction is not claimed. |
| [Handsfield et al. (2014)](https://pubmed.ncbi.nlm.nih.gov/24368144/), DOI10.1016/j.jbiomech.2013.12.002 | MRI muscle-size relationships use measured body size. Without supplied mass/height or imaging, muscle volumes/strength remain reference parameters, not video measurements. |
| [Holzbaur, Murray & Delp (2005)](https://doi.org/10.1007/s10439-005-3320-7) | Upper-extremity musculoskeletal anatomy and joint mechanics underlying the MoBL model used by the selected MyoArm component. |
| [Murray, Delp & Buchanan (1995)](https://pubmed.ncbi.nlm.nih.gov/7775488/), DOI10.1016/0021-9290(94)00114-j | Experimental elbow/forearm moment-arm variation provides an independent qualitative comparison for the force-production sweep. Checking our Jacobian against a finite difference establishes numerical consistency, not agreement with cadaver measurements. |
| [Rajagopal et al. (2016)](https://doi.org/10.1109/TBME.2016.2586891) | Muscle-driven gait model underlying the lower-limb component. The recording observes upper body only, so lower-body posture remains an explicitly unobserved reference. |
| [Caggiano et al. (2022), MyoSuite](https://arxiv.org/abs/2205.13600), and [MyoSim model sources](https://github.com/MyoHub/myo_sim) | Reusable MuJoCo musculoskeletal models, physiological actuation and geometry. We load the pinned package and run native path wrapping, joint couplings, force-length/velocity and activation mechanics. |
| [Seth et al. (2019)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6856649/) | Demonstrates why shoulder-girdle assumptions matter. The selected MyoArm's prescribed shoulder couplings remain a limitation; we do not call them independently measured scapular motion. |
| [Sahara et al. (2024)](https://arxiv.org/html/2410.05931v1) | Represents shoulder ligaments as elastic routes in MuJoCo. We use existing anatomical ligament routes in the source package, with optional tension-only spring mechanics and declared parameter priors; we do not reproduce its entire shoulder model or invent all-body ligament attachments. |
| [Boardman et al. (1996)](https://pubmed.ncbi.nlm.nih.gov/8872921/), DOI10.1016/s1058-2746(96)80050-4 | Cadaver linear stiffness measurements support an optional reference profile for SGHL (17.4 N/mm) and CHL (36.7 N/mm). Four bilateral routes use these values; MGHL/IGHL stay disabled in this profile rather than borrowing unrelated parameters. A sensitivity bench varies assumed slack/stiffness. |

## Chosen computational model

Pinned **myo-sim 0.2.5** and **MuJoCo 3.17.0**, in `.venv-biomech` under Python
3.13. `myofullbody` actually compiled locally with 104 bodies (including world,
phantoms and consolidated anatomy), 123 native joints, 416 muscle actuators and
424 spatial tendon routes. These are computational component counts, not a count
of all distinct human bones, muscles, or independently mobile anatomical joints.
Dependencies, source notices and model inventory accompany exported artifacts.

Eight unactuated glenohumeral/coracohumeral ligament routes exist in the source
model, but their native stiffness is zero. Geometric presence does not constitute
functional ligament mechanics. Optional explicit tension-only springs provide
an exploratory passive-mechanics model on the same routes. Stiffness and slack
are labeled as assumptions and tested for slack/taut behavior.
The additional Boardman profile uses sourced aggregate stiffness for four of the
eight routes. The measurements are not subject-specific, and linear springs omit
nonlinear toe behavior and strain-rate effects. Published parameter provenance
does not by itself validate the simplified attachment geometry or slack lengths.

MuJoCo's [muscle documentation](https://mujoco.readthedocs.io/en/3.17.0/modeling.html)
distinguishes an abstract muscle force generator from its spatial tendon
transmission. This model's stiff-series tendon assumption is retained. A routed
line is not a volumetric scan of a tendon or muscle. Ligament/cartilage anatomy
outside the modeled routes and tissue-resolved deformation remain unavailable.

## What the recording contributes

The saved observations can constrain pose and approximate segment proportions,
with confidence-weighted limits and continuity for underobserved coordinates.
Fitting the observed arms to reference anatomy should avoid copying the original
finger quaternion flips into the new model. Scale/pose alignment is distinct from
personalized bone shape, attachment locations, fiber lengths, PCSA, tendon
compliance, ligament stiffness and muscle strength. The latter require additional
measurements; kinematic fit error alone cannot validate them.

Force-production experiments will use **controlled, declared activations and
postures**. Any inverse-dynamics demand is conditional on mass, inertia, prescribed
motion and contact assumptions. It is not direct evidence of actual individual
muscle forces or neural activation in the recording.
