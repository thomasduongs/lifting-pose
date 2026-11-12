# biomech-lite data notices

Model and fixtures are compiled from MyoSim 0.2.5 `myofullbody` (Apache-2.0, https://github.com/MyoHub/myo_sim) with MuJoCo 3.17.0 (Apache-2.0, https://github.com/google-deepmind/mujoco/tree/3.17.0). Components: MoBL arm (Holzbaur et al. 2005), Rajagopal et al. 2016 legs, MyoBack torso (Walia et al. 2025), SimTK lumbarspine. Derived wrapping/muscle code in the Rust crate ports MuJoCo `engine_util_misc.c` and `engine_core_smooth.c`.
Original LICENSE/NOTICE/third-party files are in `provenance/`.

Reference (generic) anatomy only; not person-specific. Fitted motion is a monocular kinematic fit, not measured forces.
