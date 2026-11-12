//! MuJoCo muscle model: length/velocity gain, passive bias and activation dynamics.
//!
//! Rust port of MuJoCo 3.17.0 `mju_muscleGainLength`, `mju_muscleGain`, `mju_muscleBias`,
//! `mju_muscleDynamicsTimescale`, `mju_muscleDynamics`, `mju_sigmoid`
//! (`src/engine/engine_util_misc.c`), Copyright 2021 DeepMind Technologies Limited, licensed
//! under the Apache License, Version 2.0. Modified: translated to Rust. Native sign convention
//! (force <= 0) is kept internally; the public `Model::muscle_forces` returns tension = -force.
//! Actuator gear is assumed 1 (exporter-verified), so actuator length/velocity = tendon's.

use crate::MJ_MINVAL;
use crate::model::Model;

/// Normalized active force-length curve (`mju_muscleGainLength`).
pub fn gain_length(length: f64, lmin: f64, lmax: f64) -> f64 {
    if lmin <= length && length <= lmax {
        let a = 0.5 * (lmin + 1.0);
        let b = 0.5 * (1.0 + lmax);
        if length <= a {
            let x = (length - lmin) / MJ_MINVAL.max(a - lmin);
            0.5 * x * x
        } else if length <= 1.0 {
            let x = (1.0 - length) / MJ_MINVAL.max(1.0 - a);
            1.0 - 0.5 * x * x
        } else if length <= b {
            let x = (length - 1.0) / MJ_MINVAL.max(b - 1.0);
            1.0 - 0.5 * x * x
        } else {
            let x = (lmax - length) / MJ_MINVAL.max(lmax - b);
            0.5 * x * x
        }
    } else {
        0.0
    }
}

fn peak_force(prm: &[f64; 9], acc0: f64) -> f64 {
    if prm[2] < 0.0 { prm[3] / MJ_MINVAL.max(acc0) } else { prm[2] }
}

fn normalized_length(len: f64, lengthrange: &[f64; 2], prm: &[f64; 9]) -> (f64, f64) {
    let l0 = (lengthrange[1] - lengthrange[0]) / MJ_MINVAL.max(prm[1] - prm[0]);
    (prm[0] + (len - lengthrange[0]) / MJ_MINVAL.max(l0), l0)
}

/// Active gain (`mju_muscleGain`, native sign: <= 0). prm = gainprm.
pub fn muscle_gain(len: f64, vel: f64, lengthrange: &[f64; 2], acc0: f64, prm: &[f64; 9]) -> f64 {
    let force = peak_force(prm, acc0);
    let (lmin, lmax, vmax, fvmax) = (prm[4], prm[5], prm[6], prm[8]);
    let (l, l0) = normalized_length(len, lengthrange, prm);
    let v = vel / MJ_MINVAL.max(l0 * vmax);
    let fl = gain_length(l, lmin, lmax);
    let y = fvmax - 1.0;
    let fv = if v <= -1.0 {
        0.0
    } else if v <= 0.0 {
        (v + 1.0) * (v + 1.0)
    } else if v <= y {
        fvmax - (y - v) * (y - v) / MJ_MINVAL.max(y)
    } else {
        fvmax
    };
    -force * fl * fv
}

/// Passive bias (`mju_muscleBias`, native sign: <= 0). prm = biasprm.
pub fn muscle_bias(len: f64, lengthrange: &[f64; 2], acc0: f64, prm: &[f64; 9]) -> f64 {
    let force = peak_force(prm, acc0);
    let (lmax, fpmax) = (prm[5], prm[7]);
    let (l, _) = normalized_length(len, lengthrange, prm);
    let b = 0.5 * (1.0 + lmax);
    if l <= 1.0 {
        0.0
    } else if l <= b {
        let x = (l - 1.0) / MJ_MINVAL.max(b - 1.0);
        -force * fpmax * 0.5 * x * x
    } else {
        let x = (l - b) / MJ_MINVAL.max(b - 1.0);
        -force * fpmax * (0.5 + x)
    }
}

/// Quintic sigmoid on [0, 1] (`mju_sigmoid`).
pub fn sigmoid(x: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else if x >= 1.0 {
        1.0
    } else {
        x * x * x * (3.0 * x * (2.0 * x - 5.0) + 10.0)
    }
}

fn dynamics_timescale(dctrl: f64, tau_act: f64, tau_deact: f64, width: f64) -> f64 {
    if width < MJ_MINVAL {
        if dctrl > 0.0 { tau_act } else { tau_deact }
    } else {
        tau_deact + (tau_act - tau_deact) * sigmoid(dctrl / width + 0.5)
    }
}

/// Muscle activation rate `d act / dt` (`mju_muscleDynamics`); prm = (tau_act, tau_deact,
/// smoothing width). Control and activation are clamped to [0, 1] as in MuJoCo.
pub fn activation_dot(ctrl: f64, act: f64, prm: [f64; 3]) -> f64 {
    let ctrlclamp = ctrl.clamp(0.0, 1.0);
    let actclamp = act.clamp(0.0, 1.0);
    let tau_act = prm[0] * (0.5 + 1.5 * actclamp);
    let tau_deact = prm[1] / (0.5 + 1.5 * actclamp);
    let dctrl = ctrlclamp - act;
    let tau = dynamics_timescale(dctrl, tau_act, tau_deact, prm[2]);
    dctrl / MJ_MINVAL.max(tau)
}

impl Model {
    /// Positive total muscle tensions (passive + active), muscle order.
    /// `lengths`, `velocities`: tendon order; `activation`: muscle order (used unclamped, as
    /// MuJoCo does without actlimited). Tension = -(gain(L, V) * act + bias(L)).
    pub fn muscle_forces(&self, lengths: &[f64], velocities: &[f64], activation: &[f64]) -> Result<Vec<f64>, String> {
        let nt = self.tendons.len();
        if lengths.len() != nt || velocities.len() != nt {
            return Err(format!("lengths/velocities must have {nt} tendon entries, got {}/{}",
                lengths.len(), velocities.len()));
        }
        if activation.len() != self.muscles.len() {
            return Err(format!("activation must have {} muscle entries, got {}", self.muscles.len(), activation.len()));
        }
        self.muscles.iter().zip(activation).enumerate().map(|(i, (m, &a))| {
            let (l, v) = (lengths[m.tendon], velocities[m.tendon]);
            if !(l.is_finite() && v.is_finite() && a.is_finite()) {
                return Err(format!("muscle {i} ({}): non-finite length/velocity/activation", m.name));
            }
            let gain = muscle_gain(l, v, &m.lengthrange, m.acc0, &m.gainprm);
            let bias = muscle_bias(l, &m.lengthrange, m.acc0, &m.biasprm);
            Ok(-(gain * a + bias))
        }).collect()
    }

    /// Passive (zero-activation) tensions, muscle order; velocity-independent in MuJoCo.
    pub fn passive_muscle_forces(&self, lengths: &[f64]) -> Result<Vec<f64>, String> {
        if lengths.len() != self.tendons.len() {
            return Err(format!("lengths must have {} tendon entries", self.tendons.len()));
        }
        Ok(self.muscles.iter()
            .map(|m| -muscle_bias(lengths[m.tendon], &m.lengthrange, m.acc0, &m.biasprm))
            .collect())
    }

    /// Activation rates for all muscles: control clamped to `ctrlrange` (MuJoCo ctrllimited)
    /// then [`activation_dot`].
    pub fn activation_rates(&self, control: &[f64], activation: &[f64]) -> Result<Vec<f64>, String> {
        let nu = self.muscles.len();
        if control.len() != nu || activation.len() != nu {
            return Err(format!("control/activation must have {nu} entries"));
        }
        Ok(self.muscles.iter().zip(control.iter().zip(activation)).map(|(m, (&c, &a))| {
            let (lo, hi) = (m.ctrlrange[0], m.ctrlrange[1]);
            let c = if lo < hi { c.clamp(lo, hi) } else { c };
            activation_dot(c, a, m.dynprm)
        }).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gain_length_peak_and_bounds() {
        assert_eq!(gain_length(1.0, 0.5, 1.6), 1.0);
        assert_eq!(gain_length(0.4, 0.5, 1.6), 0.0);
        assert_eq!(gain_length(1.7, 0.5, 1.6), 0.0);
        assert!((gain_length(0.75, 0.5, 1.6) - 0.5).abs() < 1e-15);
    }

    #[test]
    fn activation_dot_signs_and_sigmoid() {
        let prm = [0.01, 0.04, 0.0];
        assert!(activation_dot(1.0, 0.0, prm) > 0.0);
        assert!(activation_dot(0.0, 1.0, prm) < 0.0);
        assert!((activation_dot(1.0, 0.0, prm) - 1.0 / 0.005).abs() < 1e-9);
        assert_eq!(sigmoid(0.5), 0.5);
    }
}
