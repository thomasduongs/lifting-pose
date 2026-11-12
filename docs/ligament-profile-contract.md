# Sourced ligament parameter profile

Add an explicit optional profile to improve on a single arbitrary spring constant.
Do not modify `biomech_model.py` while Opus owns it. Own new
`scripts/biomech_ligaments.py`, tests, and the fitter integration.

Primary source verified by lead: Boardman et al.1996,
https://pubmed.ncbi.nlm.nih.gov/8872921/ DOI10.1016/s1058-2746(96)80050-4.
It reports cadaver bone-ligament-bone **linear stiffness** means:

* SGHL:17.4 N/mm, reported variation1.5 N/mm.
* CHL:36.7 N/mm, reported variation5.9 N/mm.

Store SI values17400 and36700 N/m. Reported variations are NOT established
confidence intervals. These are aggregate reference data, not this user's tissue
properties, nor modulus/stress. Applying linear stiffness to a simplified path
is an approximation: no toe region, tissue thickness, failure, rate dependence,
or patient-specific slack identification is modeled.

Implement `apply_profile(model,data,profile='boardman1996',slack_scale=1.0)`:
map only `s_glenohum_ligament` (+_l) and `coracohum_ligament` (+_l), four routes.
MGHL/IGHL have no matching stiffness established by this source and stay disabled
in this profile. Never extrapolate SGHL/CHL data to those tissues. Preserve original
source slack values; set tension-only springlength=[0,source_slack*slack_scale],
zero damping. Return JSON-safe profile metadata with DOI, route coverage,
exact source->SI conversion, approximation caveat and unsupported-route list.
Use source routes/geometry, no fabricated attachment points. Call mj_forward
after changes. Scalar stiffness remains a separate explicitly exploratory mode.

Fitter integration adds `--ligament-profile disabled|boardman1996` (default
disabled) alongside existing scalar option; reject nonzero scalar combined with
boardman. Apply profile before fitting/mechanics and saving model.mjb. Save exact
profile metadata and native inventory in fit_report. The saved replay model and
mechanics calculations must use the identical profile. Portable XML exported
elsewhere must not be claimed to have that profile if it does not.

CLI `scripts/biomech_ligaments.py --output DIR` runs a sensitivity bench on native
ligament routes. For each four supported routes use source slack multipliers
0.9,1.0,1.1 and optional stiffness low/mean/high (source mean +/- reported
variation, clearly not confidence bounds) at a reproducible small shoulder angle
sweep within model limits. Export route length, slack, tension, energy and
generalized passive ligament torque to CSV/JSON with SI units. Compare passive
generalized force change from springs enabled/disabled against -J^T*tension at
one pose, toleranced numerical equality, no unsupported force attribution.
Include source metadata and no patient-force claims.

Targeted tests: correct N/mm conversion/mapping, only4enabled/4unsupported,
tension-only slack/taut, passive Jacobian force consistency, incompatible options.
Use native model at small scale. Keep optional-dependency skip guard for global
Python without MuJoCo. Communicate fitter flag/output changes to Haiku.
