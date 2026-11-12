#!/usr/bin/env python3
"""Tetrahedral scaffold and corotated inverse displacement solver (NumPy + SciPy).

Scope (see docs/inverse-fem-contract.md): a coarse voxel-tetrahedral union of bone
capsules is deformed to follow skeletal skinning targets. Geometry, radii and
material parameters are assumed priors. The solve reconstructs a regularized
displacement field; it does not identify anatomy, tissue stiffness or forces.

Objective, normalized by total rest volume ``V`` and characteristic cell length ``h``::

    E(x) = sum_t (V_t / V) psi(F_t)
         + 0.5 * data_weight * sum_i w_i (V_i / V) |x_i - y_i|^2 / h^2

with the corotated density ``psi(F) = mu |F - R|_F^2 + lambda/2 (tr(R^T F) - 3)^2``,
``R`` the proper polar rotation of ``F`` and ``V_i`` the lumped vertex volume.
Because the gradient of ``tr(R^T F)`` is ``R``, ``P = 2 mu (F - R) + lambda (tr(R^T F) - 3) R``
is the exact first Piola-Kirchhoff stress of this density. By default an IPC-style
log barrier on det F (a declared regularization prior, see FemProblem) keeps every
element at det F > min_det; solves start from a feasible seed and use projected
Newton with feasibility-preserving line search. The normalization makes
``data_weight`` a nondimensional ratio of target penalty to elastic stiffness that
does not depend on mesh resolution or units.
"""
import numpy as np
from scipy import ndimage

# Kuhn (Freudenthal) decomposition of a unit cube into six tetrahedra along the
# (0,0,0)-(1,1,1) diagonal. Applying the same split in every cell yields matching
# face diagonals between neighbours, i.e. a conforming mesh.
_CUBE_CORNERS = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=np.int64)


def _corner_id(offset):
    return int(offset[0] * 4 + offset[1] * 2 + offset[2])


def _kuhn_tets():
    tets = []
    axes = np.eye(3, dtype=np.int64)
    for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
        p = np.zeros(3, dtype=np.int64)
        path = [_corner_id(p)]
        for a in perm:
            p = p + axes[a]
            path.append(_corner_id(p))
        tets.append(path)
    return np.array(tets, dtype=np.int64)


_KUHN = _kuhn_tets()
_FACE_LOCAL = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
_MIN_VOLUME_FRACTION = 1e-9


# --------------------------------------------------------------------------- helpers

def _finite_array(value, name, shape_tail=None, ndim=None):
    arr = np.asarray(value, dtype=np.float64)
    if ndim is not None and arr.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {arr.shape}")
    if shape_tail is not None and arr.shape[-len(shape_tail):] != tuple(shape_tail):
        raise ValueError(f"{name} must have trailing shape {shape_tail}, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _segment_point_distance(points, a, b):
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 0.0:
        return np.linalg.norm(points - a, axis=-1)
    t = np.clip((points - a) @ ab / denom, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[..., None] * ab), axis=-1)


def _resolve_radii(radii, active, joint_count):
    if isinstance(radii, dict):
        out = {}
        for child in active:
            if child not in radii:
                raise ValueError(f"radius missing for active bone {child}")
            out[child] = float(radii[child])
    else:
        arr = np.asarray(radii, dtype=np.float64)
        if arr.ndim == 0:
            out = {child: float(arr) for child in active}
        elif arr.shape == (len(active),):
            out = {child: float(r) for child, r in zip(active, arr)}
        elif arr.shape == (joint_count,):
            out = {child: float(arr[child]) for child in active}
        else:
            raise ValueError("radii must be a scalar, dict, or have length len(active_bones) or joint count")
    for child, r in out.items():
        if not np.isfinite(r) or r <= 0.0:
            raise ValueError(f"radius for bone {child} must be finite and positive")
    return out


def _validate_hierarchy(parents, joint_count):
    parents = np.asarray(parents, dtype=np.int64)
    if parents.shape != (joint_count,):
        raise ValueError("parents must have one entry per joint")
    for i, p in enumerate(parents):
        if p < -1 or p >= joint_count or p == i:
            raise ValueError(f"invalid parent {p} for joint {i}")
    for i in range(joint_count):  # cycle check
        seen, j = set(), i
        while j != -1:
            if j in seen:
                raise ValueError("parent hierarchy contains a cycle")
            seen.add(j)
            j = int(parents[j])
    return parents


def _signed_volumes(vertices, tets):
    x0 = vertices[tets[:, 0]]
    d = np.stack([vertices[tets[:, 1]] - x0, vertices[tets[:, 2]] - x0, vertices[tets[:, 3]] - x0], axis=2)
    return np.linalg.det(d) / 6.0, d


def _boundary_faces(tets):
    faces = tets[:, _FACE_LOCAL].reshape(-1, 3)
    key = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.reshape(-1)
    return faces[counts[inverse] == 1], counts


# --------------------------------------------------------------------------- mesh

def build_mesh(rest_joints, parents, active_bones, radii, spacing=0.06):
    """Build a conforming voxel-tetrahedral union of bone capsules.

    rest_joints: (J,3) global rest positions. parents: (J,) ints, root -1.
    active_bones: child joint indices whose bone (parent -> child) is meshed.
    radii: scalar, dict child->radius, array of len(active_bones) (same order),
    or array of length J indexed by child joint. Radii are geometric priors.

    Returns dict with vertices (N,3), tetrahedra (T,4) positively oriented,
    surface_triangles (S,3) outward-oriented, weights (N,J) rows summing to one
    (column j = bone whose child is j), plus spacing, active_bones, radii (J,)
    and parents (J,).
    """
    joints = _finite_array(rest_joints, "rest_joints", shape_tail=(3,), ndim=2)
    joint_count = joints.shape[0]
    parents = _validate_hierarchy(parents, joint_count)
    active = [int(c) for c in active_bones]
    if not active:
        raise ValueError("active_bones is empty")
    if len(set(active)) != len(active):
        raise ValueError("active_bones contains duplicates")
    for child in active:
        if child < 0 or child >= joint_count:
            raise ValueError(f"active bone {child} out of range")
        if parents[child] < 0:
            raise ValueError(f"active bone {child} is a root and has no parent bone")
    spacing = float(spacing)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be finite and positive")
    radius = _resolve_radii(radii, active, joint_count)

    segs = [(child, joints[parents[child]], joints[child], radius[child]) for child in active]
    lo = np.min([np.minimum(a, b) - r for _, a, b, r in segs], axis=0) - spacing
    hi = np.max([np.maximum(a, b) + r for _, a, b, r in segs], axis=0) + spacing
    origin = lo
    dims = np.maximum(np.ceil((hi - lo) / spacing).astype(np.int64), 1)
    if int(np.prod(dims)) > 20_000_000:
        raise ValueError("mesh grid too large; increase spacing")

    # Mark every cell whose closed box comes within `r` of a bone segment.
    # Segment-to-box distance is evaluated at dense samples along the segment
    # (step <= spacing/8) with the sampling error added as a conservative margin,
    # so all cells intersecting a capsule are included even for thin limbs.
    occupied = np.zeros(dims, dtype=bool)
    for _, a, b, r in segs:
        length = float(np.linalg.norm(b - a))
        step_count = max(1, int(np.ceil(length / (spacing / 8.0))))
        samples = a + np.linspace(0.0, 1.0, step_count + 1)[:, None] * (b - a)
        margin = 0.5 * length / step_count
        blo = np.clip(np.floor((np.minimum(a, b) - r - margin - origin) / spacing).astype(np.int64) - 1, 0, dims)
        bhi = np.clip(np.ceil((np.maximum(a, b) + r + margin - origin) / spacing).astype(np.int64) + 1, 0, dims)
        idx = np.stack(np.meshgrid(*[np.arange(blo[k], bhi[k]) for k in range(3)], indexing="ij"), axis=-1).reshape(-1, 3)
        if idx.size == 0:
            continue
        cmin = origin + idx * spacing
        cmax = cmin + spacing
        best = np.full(len(idx), np.inf)
        for chunk in np.array_split(samples, max(1, len(samples) // 64)):
            delta = np.maximum(np.maximum(cmin[:, None, :] - chunk[None], chunk[None] - cmax[:, None, :]), 0.0)
            best = np.minimum(best, np.linalg.norm(delta, axis=2).min(axis=1))
        hit = idx[best <= r + margin]
        occupied[hit[:, 0], hit[:, 1], hit[:, 2]] = True

    cells = np.argwhere(occupied)
    if len(cells) == 0:
        raise ValueError("no cells intersect the bone capsules")

    # Face connectivity of occupied cells (6-neighbourhood).
    _, components = ndimage.label(occupied)
    if components != 1:
        raise ValueError(f"active bones produce {components} disconnected mesh components; "
                         "check active_bones, radii and spacing")

    # Lattice vertices and Kuhn tetrahedra.
    stride = np.array([(dims[1] + 1) * (dims[2] + 1), dims[2] + 1, 1], dtype=np.int64)
    corner_lattice = cells[:, None, :] + _CUBE_CORNERS[None]  # (C,8,3)
    corner_keys = corner_lattice @ stride
    tet_keys = corner_keys[:, _KUHN].reshape(-1, 4)
    unique_keys, tets = np.unique(tet_keys, return_inverse=True)
    tets = tets.reshape(-1, 4).astype(np.int64)
    lattice = np.stack([unique_keys // stride[0], (unique_keys // stride[1]) % (dims[1] + 1),
                        unique_keys % (dims[2] + 1)], axis=1)
    vertices = origin + lattice * spacing
    volumes, _ = _signed_volumes(vertices, tets)
    flip = volumes < 0
    tets[flip] = tets[flip][:, [0, 2, 1, 3]]
    volumes, _ = _signed_volumes(vertices, tets)
    if np.any(volumes <= _MIN_VOLUME_FRACTION * spacing ** 3):
        raise RuntimeError("internal error: degenerate tetrahedron generated")
    surface, counts = _boundary_faces(tets)
    if np.any(counts > 2):
        raise RuntimeError("internal error: non-conforming tetrahedral faces")

    # Skinning weights: inverse power of radius-normalized distance to each bone.
    radius_by_joint = np.zeros(joint_count)
    weights = np.zeros((len(vertices), joint_count))
    for child, a, b, r in segs:
        radius_by_joint[child] = r
        s = _segment_point_distance(vertices, a, b) / r
        weights[:, child] = (s + 0.1) ** -6
    weights /= weights.sum(axis=1, keepdims=True)

    return {
        "vertices": vertices,
        "tetrahedra": tets,
        "surface_triangles": surface.astype(np.int64),
        "weights": weights,
        "spacing": spacing,
        "active_bones": np.array(active, dtype=np.int64),
        "radii": radius_by_joint,
        "parents": parents.copy(),
    }


# --------------------------------------------------------------------------- skinning

def _as_rotations(world_rotations, joint_count):
    rot = _finite_array(world_rotations, "world_rotations")
    if rot.shape == (joint_count, 4):  # xyzw quaternions
        norm = np.linalg.norm(rot, axis=1)
        if np.any(norm < 1e-12):
            raise ValueError("zero quaternion in world_rotations")
        x, y, z, w = (rot / norm[:, None]).T
        rot = np.stack([
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ], axis=1)
    if rot.shape != (joint_count, 3, 3):
        raise ValueError("world_rotations must be (J,3,3) matrices or (J,4) xyzw quaternions")
    return rot


def skin_targets(mesh, rest_joints, posed_joints, world_rotations, parents=None):
    """Linear-blend skinning targets (N,3).

    Bone with child j moves rigidly: x' = R_parent(j) (x - rest_j) + posed_j,
    where R is the parent's posed world rotation relative to the identity rest
    pose. world_rotations: (J,3,3) matrices or (J,4) xyzw quaternions, per joint.
    The hierarchy is read from mesh["parents"] (set by build_mesh) unless
    ``parents`` is given, e.g. for a mesh reloaded from an npz without it.
    """
    vertices = _finite_array(mesh["vertices"], "mesh vertices", shape_tail=(3,), ndim=2)
    weights = _finite_array(mesh["weights"], "mesh weights", ndim=2)
    joint_count = weights.shape[1]
    rest = _finite_array(rest_joints, "rest_joints", shape_tail=(3,), ndim=2)
    posed = _finite_array(posed_joints, "posed_joints", shape_tail=(3,), ndim=2)
    if rest.shape[0] != joint_count or posed.shape[0] != joint_count:
        raise ValueError("joint arrays must match mesh weight columns")
    rot = _as_rotations(world_rotations, joint_count)
    if parents is None:
        parents = mesh.get("parents") if hasattr(mesh, "get") else None
    if parents is None:
        raise ValueError("parents required: pass parents= or store mesh['parents']")
    parents = _validate_hierarchy(parents, joint_count)
    out = np.zeros_like(vertices)
    for child in np.nonzero(np.any(weights > 0.0, axis=0))[0]:
        p = parents[child]
        if p < 0:
            raise ValueError(f"weight column {child} is a root joint")
        moved = (vertices - rest[child]) @ rot[p].T + posed[child]
        out += weights[:, child, None] * moved
    return out


# --------------------------------------------------------------------------- solver

def _material(young, poisson):
    young = float(young)
    poisson = float(poisson)
    if not np.isfinite(young) or young <= 0.0:
        raise ValueError("young must be finite and positive")
    if not np.isfinite(poisson) or poisson < 0.0 or poisson > 0.49:
        # Negative lambda makes the volumetric term unbounded below; nu -> 0.5 locks.
        raise ValueError("poisson must be within [0, 0.49]")
    mu = young / (2.0 * (1.0 + poisson))
    lam = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    return mu, lam


def _cofactor(f):
    """d det(F) / dF, computed without inversion (valid for singular F)."""
    c0, c1, c2 = f[:, :, 0], f[:, :, 1], f[:, :, 2]
    return np.stack([np.cross(c1, c2), np.cross(c2, c0), np.cross(c0, c1)], axis=2)


class FemProblem:
    """Precomputed rest-state quantities for one mesh, material and barrier.

    Anti-inversion prior (``barrier_weight`` > 0): per element, with
    d = det F - min_det and activation distance D = 0.5, an IPC-style C2 barrier
    ``kappa * -(d - D)^2 log(d / D)`` for 0 < d < D (zero above, +inf for d <= 0),
    kappa = barrier_weight * mu, integrated with the same volume weights as psi.
    It keeps every element strictly un-inverted. It is a regularization prior,
    not a measured material property.
    """

    activation = 0.5

    def __init__(self, mesh, young=1.0, poisson=0.35, data_weight=100.0, barrier_weight=1.0, min_det=0.05):
        self.rest = _finite_array(mesh["vertices"], "mesh vertices", shape_tail=(3,), ndim=2)
        tets = np.asarray(mesh["tetrahedra"])
        if tets.ndim != 2 or tets.shape[1] != 4 or not np.issubdtype(tets.dtype, np.integer):
            raise ValueError("tetrahedra must be an integer (T,4) array")
        if len(tets) == 0:
            raise ValueError("mesh has no tetrahedra")
        if tets.min() < 0 or tets.max() >= len(self.rest):
            raise ValueError("tetrahedra reference missing vertices")
        if np.any(np.sort(tets, axis=1)[:, 1:] == np.sort(tets, axis=1)[:, :-1]):
            raise ValueError("tetrahedra contain repeated vertices")
        self.tets = tets.astype(np.int64)
        self.mu, self.lam = _material(young, poisson)
        self.data_weight = float(data_weight)
        if not np.isfinite(self.data_weight) or self.data_weight < 0.0:
            raise ValueError("data_weight must be finite and non-negative")
        self.barrier_weight = float(barrier_weight)
        if not np.isfinite(self.barrier_weight) or self.barrier_weight < 0.0:
            raise ValueError("barrier_weight must be finite and non-negative")
        self.min_det = float(min_det)
        if not np.isfinite(self.min_det) or not 0.0 <= self.min_det < 0.5:
            raise ValueError("min_det must be within [0, 0.5)")
        self.kappa = self.barrier_weight * self.mu
        volumes, dm = _signed_volumes(self.rest, self.tets)
        scale = np.ptp(self.rest, axis=0).max() if len(self.rest) > 1 else 0.0
        if scale <= 0.0 or np.any(volumes <= _MIN_VOLUME_FRACTION * scale ** 3):
            raise ValueError("mesh contains degenerate or inverted rest tetrahedra")
        self.volumes = volumes
        self.dm_inv = np.linalg.inv(dm)
        self.total_volume = float(volumes.sum())
        self.h = float((6.0 * volumes.mean()) ** (1.0 / 3.0))
        n = len(self.rest)
        self.vertex_volume = np.bincount(self.tets.reshape(-1), weights=np.repeat(volumes / 4.0, 4), minlength=n)
        self.n = n

    @property
    def barrier(self):
        return self.kappa > 0.0

    def deformation(self, x):
        x0 = x[self.tets[:, 0]]
        ds = np.stack([x[self.tets[:, 1]] - x0, x[self.tets[:, 2]] - x0, x[self.tets[:, 3]] - x0], axis=2)
        return ds @ self.dm_inv

    def feasible(self, x, margin=None):
        threshold = self.min_det if margin is None else margin
        return (not self.barrier) or bool(np.all(np.linalg.det(self.deformation(x)) > threshold))

    @staticmethod
    def polar_rotation(f):
        u, _, vt = np.linalg.svd(f)
        d = np.sign(np.linalg.det(u @ vt))
        d[d == 0] = 1.0
        u = u.copy()
        u[:, :, 2] *= d[:, None]
        return u @ vt

    def _scatter(self, pk1):
        """Volume-weighted first Piola-Kirchhoff stresses (T,3,3) -> nodal gradient (N,3)."""
        h = pk1 @ np.transpose(self.dm_inv, (0, 2, 1))  # dE/dDs, columns -> x1..x3
        local = np.concatenate([-h.sum(axis=2, keepdims=True), h], axis=2)  # (T,3,4)
        grad = np.empty((self.n, 3))
        flat = self.tets.reshape(-1)
        for k in range(3):
            grad[:, k] = np.bincount(flat, weights=local[:, k, :].reshape(-1), minlength=self.n)
        return grad

    def _corotated(self, f):
        r = self.polar_rotation(f)
        diff = f - r
        tr = np.einsum("tij,tij->t", r, f) - 3.0
        w = self.volumes / self.total_volume
        energy = float(np.sum(w * (self.mu * np.einsum("tij,tij->t", diff, diff) + 0.5 * self.lam * tr ** 2)))
        pk1 = 2.0 * self.mu * diff + (self.lam * tr)[:, None, None] * r
        return energy, pk1, w

    def terms(self, x, targets, target_weights):
        """Energies (elastic, barrier, data, total), gradient (N,3) and F.

        If the barrier is enabled and any element has det F <= min_det, the
        barrier and total are +inf and the gradient is None.
        """
        f = self.deformation(x)
        e_el, pk1, w = self._corotated(f)
        e_b = 0.0
        if self.barrier:
            d = np.linalg.det(f) - self.min_det
            if np.any(d <= 0.0):
                return {"elastic": e_el, "barrier": np.inf, "data": np.nan, "total": np.inf,
                        "gradient": None, "F": f}
            big_d = self.activation
            act = d < big_d
            if np.any(act):
                da = d[act]
                log = np.log(da / big_d)
                e_b = float(np.sum(w[act] * self.kappa * -(da - big_d) ** 2 * log))
                db = self.kappa * (-2.0 * (da - big_d) * log - (da - big_d) ** 2 / da)
                pk1[act] += db[:, None, None] * _cofactor(f[act])
        grad = self._scatter(w[:, None, None] * pk1)
        coeff = self.data_weight * target_weights * self.vertex_volume / (self.total_volume * self.h ** 2)
        resid = x - targets
        e_d = 0.5 * float(np.sum(coeff[:, None] * resid * resid))
        grad += coeff[:, None] * resid
        return {"elastic": e_el, "barrier": e_b, "data": e_d, "total": e_el + e_b + e_d,
                "gradient": grad, "F": f}

    def elastic(self, x):
        """Return (energy, gradient (N,3), F) of the corotated elastic term alone."""
        f = self.deformation(x)
        energy, pk1, w = self._corotated(f)
        return energy, self._scatter(w[:, None, None] * pk1), f

    def objective(self, x, targets, target_weights):
        out = self.terms(x, targets, target_weights)
        return out["total"], out["gradient"]

    def data_coefficients(self, target_weights):
        return self.data_weight * target_weights * self.vertex_volume / (self.total_volume * self.h ** 2)

    def _element_pk1(self, f):
        """Unweighted total stress (corotated + barrier) for a feasible batch of F."""
        r = self.polar_rotation(f)
        tr = np.einsum("tij,tij->t", r, f) - 3.0
        pk1 = 2.0 * self.mu * (f - r) + (self.lam * tr)[:, None, None] * r
        if self.barrier:
            big_d = self.activation
            d = np.maximum(np.linalg.det(f) - self.min_det, 1e-300)
            act = d < big_d
            if np.any(act):
                da = d[act]
                db = self.kappa * (-2.0 * (da - big_d) * np.log(da / big_d) - (da - big_d) ** 2 / da)
                pk1[act] += db[:, None, None] * _cofactor(f[act])
        return pk1

    def gradient_operator(self):
        """G (T,4,3): dF_ij / dx_(a,i) = G[a,j] for element vertices a = 0..3."""
        return np.concatenate([-self.dm_inv.sum(axis=1, keepdims=True), self.dm_inv], axis=1)

    def _vec_f_operator(self):
        """B (T,9,12): vec(F) (row-major) = B @ element coordinates (vertex-major)."""
        if getattr(self, "_b", None) is None:
            g = self.gradient_operator()  # (T,4,3)
            b = np.zeros((len(g), 3, 3, 4, 3))
            for c in range(3):
                b[:, c, :, :, c] = np.transpose(g, (0, 2, 1))
            self._b = b.reshape(len(g), 9, 12)
        return self._b

    def hessian(self, x, target_weights):
        """Sparse PSD-projected Hessian (3N,3N) of the total objective at feasible x.

        Per element, dP/dF (9x9) comes from central differences of the analytic
        stress, is symmetrized and projected to positive semidefinite by
        eigenvalue clamping, then mapped to the 12 vertex coordinates. The data
        term contributes its exact diagonal. Used only to choose Newton search
        directions; energy and gradient remain exact.
        """
        import scipy.sparse as sp

        f = self.deformation(x)
        t = len(f)
        eps = 1e-7
        basis = np.eye(9).reshape(9, 3, 3)
        batch = np.concatenate([f[None], f[None] + eps * basis[:, None]], axis=0).reshape(-1, 3, 3)
        pk1 = self._element_pk1(batch).reshape(10, t, 9)
        # hf[t, p, q] = dP_p / dF_q (forward differences; only steers directions).
        hf = np.transpose((pk1[1:] - pk1[0]) / eps, (1, 2, 0))
        hf = 0.5 * (hf + np.transpose(hf, (0, 2, 1)))
        evals, evecs = np.linalg.eigh(hf)
        hf = (evecs * np.maximum(evals, 0.0)[:, None, :]) @ np.transpose(evecs, (0, 2, 1))
        hf *= (self.volumes / self.total_volume)[:, None, None]
        b = self._vec_f_operator()
        hx = np.transpose(b, (0, 2, 1)) @ hf @ b
        dof = (3 * self.tets[:, :, None] + np.arange(3)[None, None, :]).reshape(t, 12)
        rows = np.repeat(dof, 12, axis=1).reshape(-1)
        cols = np.tile(dof, (1, 12)).reshape(-1)
        n3 = 3 * self.n
        mat = sp.coo_matrix((hx.reshape(-1), (rows, cols)), shape=(n3, n3)).tocsc()
        diag = np.repeat(self.data_coefficients(target_weights), 3)
        return mat + sp.diags(diag)


def fem_objective(mesh, vertices, targets, target_weights=None, young=1.0, poisson=0.35, data_weight=100.0,
                  barrier_weight=1.0, min_det=0.05):
    """Total normalized objective and analytic gradient (N,3) at ``vertices``.

    Returns (inf, None) if the barrier is enabled and some det F <= min_det.
    """
    problem = FemProblem(mesh, young, poisson, data_weight, barrier_weight, min_det)
    x = _finite_array(vertices, "vertices", shape_tail=(3,), ndim=2)
    t = _finite_array(targets, "targets", shape_tail=(3,), ndim=2)
    w = _target_weights(target_weights, problem.n)
    if x.shape != problem.rest.shape or t.shape != problem.rest.shape:
        raise ValueError("vertices and targets must match mesh vertices")
    return problem.objective(x, t, w)


def _target_weights(target_weights, n):
    if target_weights is None:
        return np.ones(n)
    w = _finite_array(target_weights, "target_weights")
    if w.shape != (n,):
        raise ValueError("target_weights must have one entry per vertex")
    if np.any(w < 0.0):
        raise ValueError("target_weights must be non-negative")
    return w


def _rigid_fit(source, target, weights):
    """Weighted Kabsch: proper rotation R and translation c with R s + c ~ t."""
    w = weights if weights.sum() > 0 else np.ones(len(source))
    w = w / w.sum()
    ms, mt = w @ source, w @ target
    cov = (target - mt).T @ (w[:, None] * (source - ms))
    u, _, vt = np.linalg.svd(cov)
    sign = np.sign(np.linalg.det(u @ vt))
    rot = u @ np.diag([1.0, 1.0, sign if sign != 0 else 1.0]) @ vt
    return rot, mt - rot @ ms


def _feasible_seed(problem, targets, weights, initial):
    """Pick a start with every det F > min_det (required by the barrier).

    Order: the caller's warm start, then the targets, then a blend from the warm
    start (if any) or the rigidly aligned rest shape toward the targets, pushed as
    far as stays feasible with margin (bisection on the blend factor).
    """
    if initial is not None and problem.feasible(initial):
        return initial, "initial_vertices"
    if problem.feasible(targets):
        return targets, "targets"
    margin = 0.5 * (problem.min_det + problem.activation)
    rot, shift = _rigid_fit(problem.rest, targets, weights)
    bases = [("rigid_fit", problem.rest @ rot.T + shift)]
    if initial is not None:
        bases.insert(0, ("initial_vertices", initial))
    for label, base in bases:
        if not problem.feasible(base, margin):
            continue
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (lo + hi)
            if problem.feasible(base + mid * (targets - base), margin):
                lo = mid
            else:
                hi = mid
        return base + lo * (targets - base), f"{label}+{lo:.3f}*(targets-{label})"
    raise RuntimeError("no feasible starting configuration found")  # rigid fit is always feasible


def _feasible_step(feasible, y, d, step, limit=60):
    """Halve ``step`` until y + step d passes the cheap det-only feasibility test."""
    for _ in range(limit):
        if feasible(y + step * d):
            return step
        step *= 0.5
    return step


def _newton(fun, hess, feasible, y, max_iterations, ftol=1e-12):
    """Projected Newton with feasibility-preserving Armijo backtracking.

    Directions solve H d = -g with the PSD-projected sparse Hessian (plus a tiny
    diagonal shift); the line search first shrinks the step until every element
    stays above the barrier's det F threshold, then enforces sufficient decrease.
    Converges when the Newton decrement or relative decrease is below ftol.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import spsolve

    f, g = fun(y)
    nfev, nit = 1, 0
    message = "iteration limit reached"
    converged = False
    while nit < max_iterations:
        h = hess(y)
        shift = 1e-10 * max(float(np.abs(h.diagonal()).mean()), 1e-300)
        d = spsolve((h + shift * sp.identity(h.shape[0], format="csc")).tocsc(), -g)
        slope = float(g @ d)
        if not np.all(np.isfinite(d)) or slope >= 0.0:
            d, slope = -g, -float(g @ g)
        if 0.5 * -slope <= ftol * max(abs(f), 1.0):
            converged, message = True, "Newton decrement below tolerance"
            break
        step = _feasible_step(feasible, y, d, 1.0)
        accepted = False
        for _ in range(60):
            f_new, g_new = fun(y + step * d)
            nfev += 1
            if np.isfinite(f_new) and f_new <= f + 1e-4 * step * slope:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            message = "line search failed to find a feasible decrease"
            break
        nit += 1
        decrease = (f - f_new) / max(abs(f), abs(f_new), 1.0)
        y, f, g = y + step * d, f_new, g_new
        if decrease <= ftol:
            converged, message = True, "relative objective decrease below tolerance"
            break
    return y, converged, message, nit, nfev


def _lbfgs(fun, y, max_iterations, memory=20, ftol=1e-12, gtol=1e-8, feasible=lambda y: True):
    """L-BFGS with Armijo backtracking; infeasible trial points (inf) are rejected."""
    f, g = fun(y)
    s_hist, y_hist = [], []
    nfev, nit = 1, 0
    message = "iteration limit reached"
    converged = False
    while nit < max_iterations:
        if np.max(np.abs(g)) <= gtol:
            converged, message = True, "gradient tolerance reached"
            break
        q = g.copy()
        alphas = []
        for s_k, y_k in zip(reversed(s_hist), reversed(y_hist)):
            a = (s_k @ q) / (y_k @ s_k)
            alphas.append(a)
            q -= a * y_k
        if s_hist:
            q *= (s_hist[-1] @ y_hist[-1]) / (y_hist[-1] @ y_hist[-1])
        for s_k, y_k, a in zip(s_hist, y_hist, reversed(alphas)):
            q += (a - (y_k @ q) / (y_k @ s_k)) * s_k
        d = -q
        slope = g @ d
        if not s_hist or slope >= 0.0:
            s_hist, y_hist = [], []
            d, slope = -g, -(g @ g)
        step = 1.0 if s_hist else min(1.0, 1.0 / max(np.linalg.norm(g), 1e-300))
        step = _feasible_step(feasible, y, d, step)
        accepted = False
        for _ in range(60):
            f_new, g_new = fun(y + step * d)
            nfev += 1
            if np.isfinite(f_new) and f_new <= f + 1e-4 * step * slope:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            message = "line search failed to find a feasible decrease"
            break
        nit += 1
        s = step * d
        yk = g_new - g
        if s @ yk > 1e-12 * np.sqrt((s @ s) * (yk @ yk)):
            s_hist.append(s)
            y_hist.append(yk)
            if len(s_hist) > memory:
                s_hist.pop(0)
                y_hist.pop(0)
        decrease = (f - f_new) / max(abs(f), abs(f_new), 1.0)
        y, f, g = y + s, f_new, g_new
        if decrease <= ftol:
            converged, message = True, "relative objective decrease below tolerance"
            break
    return y, converged, message, nit, nfev


def solve_frame(mesh, targets, target_weights=None, young=1.0, poisson=0.35,
                data_weight=100.0, max_iterations=60, initial_vertices=None,
                barrier_weight=1.0, min_det=0.05, method="newton"):
    """Minimize corotated FEM energy + inversion barrier + weighted target residual.

    method: "newton" (default; projected Newton, max_iterations Newton steps) or
    "lbfgs" (cheaper per step, much slower near the barrier).

    With the barrier on (default) every returned element has det F > min_det: the
    start is made feasible (see _feasible_seed) and every accepted step stays
    feasible. Linear-blend-skinning targets can themselves be inverted (e.g. a
    sharply bent elbow); then the solution cannot match them and target_rmse shows
    it (target_inverted_elements counts them). ``barrier_weight=0`` gives the
    unconstrained solve, whose inversions are only reported. Pass the previous
    frame's vertices as ``initial_vertices`` to warm start. Returns
    {"vertices", "diagnostics"}; ``converged`` is the optimizer's own tolerance
    test, False at the iteration budget or if the line search fails.
    """
    problem = FemProblem(mesh, young, poisson, data_weight, barrier_weight, min_det)
    t = _finite_array(targets, "targets", shape_tail=(3,), ndim=2)
    if t.shape != problem.rest.shape:
        raise ValueError("targets must match mesh vertices")
    w = _target_weights(target_weights, problem.n)
    max_iterations = int(max_iterations)
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if method not in ("newton", "lbfgs"):
        raise ValueError("method must be 'newton' or 'lbfgs'")
    initial = None
    if initial_vertices is not None:
        initial = _finite_array(initial_vertices, "initial_vertices", shape_tail=(3,), ndim=2)
        if initial.shape != problem.rest.shape:
            raise ValueError("initial_vertices must match mesh vertices")
    if problem.barrier:
        x0, seed = _feasible_seed(problem, t, w, initial)
    else:
        x0, seed = (t, "targets") if initial is None else (initial, "initial_vertices")

    h = problem.h

    def fun(y):  # optimize in cell-length units for conditioning
        total, grad = problem.objective(y.reshape(-1, 3) * h, t, w)
        if not np.isfinite(total):
            return np.inf, None
        return total, grad.reshape(-1) * h

    def feasible(y):
        return problem.feasible(y.reshape(-1, 3) * h)

    def hess(y):  # Hessian in the same scaled variables
        return problem.hessian(y.reshape(-1, 3) * h, w) * (h * h)

    initial_objective = float(fun(x0.reshape(-1) / h)[0])
    if method == "newton":
        y, success, message, nit, nfev = _newton(fun, hess, feasible, x0.reshape(-1) / h, max_iterations)
    else:
        y, success, message, nit, nfev = _lbfgs(fun, x0.reshape(-1) / h, max_iterations, feasible=feasible)
    if max_iterations == 0:
        message = "max_iterations=0"
    x = y.reshape(-1, 3) * h
    out = problem.terms(x, t, w)
    dets = np.linalg.det(out["F"])
    target_dets = np.linalg.det(problem.deformation(t))
    sq = np.sum((x - t) ** 2, axis=1)
    wsum = float(w.sum())
    grad = out["gradient"]
    threshold = problem.min_det if problem.barrier else 0.0
    if not np.all(np.isfinite(x)):
        valid, reason = False, "non-finite vertices"
    elif dets.min() <= threshold:
        valid, reason = False, f"{int(np.count_nonzero(dets <= threshold))} elements with det F <= {threshold:g}"
    else:
        valid, reason = True, f"all det F > {threshold:g}"
    diagnostics = {
        # valid: geometric validity of the returned volume (independent of convergence).
        "valid": bool(valid),
        "validity_reason": reason,
        "converged": bool(success),
        "message": message,
        "iterations": int(nit),
        "function_evaluations": int(nfev),
        "seed": seed,
        "objective": float(out["total"]),
        "initial_objective": initial_objective,
        "elastic_energy": float(out["elastic"]),
        "barrier_energy": float(out["barrier"]),
        "data_energy": float(out["data"]),
        "gradient_inf_norm": float(np.max(np.abs(grad * h))) if grad is not None else float("nan"),
        "target_rmse": float(np.sqrt(np.sum(w * sq) / wsum)) if wsum > 0 else float("nan"),
        "target_rmse_unweighted": float(np.sqrt(sq.mean())),
        "target_max_error": float(np.sqrt(sq.max())),
        "min_det_f": float(dets.min()),
        "max_det_f": float(dets.max()),
        "inverted_elements": int(np.count_nonzero(dets <= 0.0)),
        "target_min_det_f": float(target_dets.min()),
        "target_inverted_elements": int(np.count_nonzero(target_dets <= 0.0)),
        "element_count": int(len(dets)),
        "vertex_count": int(problem.n),
        "young": float(young),
        "poisson": float(poisson),
        "data_weight": float(data_weight),
        "barrier_weight": float(barrier_weight),
        "min_det": float(min_det),
        "method": method,
        "characteristic_length": h,
    }
    return {"vertices": x, "diagnostics": diagnostics}
