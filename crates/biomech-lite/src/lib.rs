//! biomech-lite: a standalone Rust subset of the MuJoCo 3.17.0 smooth-dynamics pipeline,
//! specialised to the MyoSim 0.2.5 `myofullbody` reference model exported to
//! `assets/biomech-lite/model.json` (see `docs/rust-biomech-contract.md`).
//!
//! Implemented subset (anything else is rejected by [`Model::validate`]):
//! * joints: free (root only), slide, hinge (ball is rejected);
//! * native `mjEQ_JOINT` quartic couplings as exact parameterizations ([`Model::project`]);
//! * spatial tendons made of sites, sphere wraps and cylinder wraps (with optional side sites);
//! * MuJoCo muscle actuators acting on tendons with unit gear (gain/bias/dynamics `muscle`).
//!
//! Attribution: the forward kinematics (`kinematics.rs`), spatial tendon traversal and
//! wrapping (`wrapping.rs`) and muscle curves/activation dynamics (`muscle.rs`) are Rust ports
//! of MuJoCo 3.17.0 `src/engine/engine_core_smooth.c` and `src/engine/engine_util_misc.c`,
//! Copyright 2021 DeepMind Technologies Limited, licensed under the Apache License 2.0
//! (<https://github.com/google-deepmind/mujoco/tree/3.17.0>). Model data is derived from
//! MyoSim (Apache-2.0) and its cited source models; see the exported provenance.

pub mod dynamics;
pub mod kinematics;
pub mod hand_fit;
pub mod inverse;
pub mod live_fit;
pub mod model;
pub mod muscle;
pub mod wrapping;

pub use kinematics::Kinematics;
pub use live_fit::{FitResult, Landmark, LiveFitter, Observation};
pub use model::{Body, Coupling, Geom, Joint, Model, Muscle, Site, Tendon, WrapPoint};
pub use muscle::activation_dot;

/// MuJoCo `mjMINVAL`.
pub const MJ_MINVAL: f64 = 1e-15;

/// Native `mjtJoint` values.
pub const JNT_FREE: i32 = 0;
pub const JNT_BALL: i32 = 1;
pub const JNT_SLIDE: i32 = 2;
pub const JNT_HINGE: i32 = 3;

/// Native `mjtWrap` values used by spatial tendons.
pub const WRAP_SITE: i32 = 3;
pub const WRAP_SPHERE: i32 = 4;
pub const WRAP_CYLINDER: i32 = 5;

/// Native `mjtGeom` values for wrapping geoms.
pub const GEOM_SPHERE: i32 = 2;
pub const GEOM_CYLINDER: i32 = 5;
