//! biomech-viewer: native viewer for the biomech-lite reference model.
//!
//! Anatomy meshes are sourced from MyoSim 0.2.5 (Apache-2.0); kinematics, wrapping and
//! dynamics come from `biomech-lite` (ports of MuJoCo 3.17.0, Apache-2.0, DeepMind).

pub mod app;
pub mod camera;
pub mod render;
pub mod scene;
pub mod session;
