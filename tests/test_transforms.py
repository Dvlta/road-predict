"""Geometry tests (design doc §14.3 check 1, §27)."""

import numpy as np

from prs.data.transforms import LocalFrame
from prs.utils.geometry import wrap_angle

ORIGIN = np.array([24_620.9, -18_034.7])  # realistic WOMD magnitude


def test_target_to_local_origin():
    frame = LocalFrame(ORIGIN, 1.2)
    np.testing.assert_allclose(frame.points_to_local(ORIGIN), [0.0, 0.0], atol=1e-12)


def test_target_heading_alignment():
    # Facing north: the point 1 m north is straight ahead (+x); 1 m west is to the left (+y).
    frame = LocalFrame(ORIGIN, np.pi / 2)
    np.testing.assert_allclose(frame.points_to_local(ORIGIN + [0.0, 1.0]), [1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(frame.points_to_local(ORIGIN + [-1.0, 0.0]), [0.0, 1.0], atol=1e-9)
    # 10 m east and 10 m north of a north-facing target: 10 m ahead, 10 m to the right.
    np.testing.assert_allclose(
        frame.points_to_local(ORIGIN + [10.0, 10.0]), [10.0, -10.0], atol=1e-9
    )


def test_global_local_roundtrip():
    rng = np.random.default_rng(0)
    for theta in rng.uniform(-np.pi, np.pi, size=20):
        frame = LocalFrame(ORIGIN, theta)
        points = ORIGIN + rng.uniform(-80.0, 80.0, size=(500, 2))
        local32 = frame.points_to_local(points).astype(np.float32)  # what the cache stores
        error = np.abs(frame.points_to_global(local32) - points).max()
        assert error < 1e-4, f"round-trip error {error:.2e} m at theta={theta:.3f}"


def test_velocity_rotation():
    frame = LocalFrame(ORIGIN, np.pi / 2)
    # Walking north at 1.4 m/s while facing north is "forward" at 1.4 m/s; no translation.
    np.testing.assert_allclose(frame.vectors_to_local([0.0, 1.4]), [1.4, 0.0], atol=1e-12)
    np.testing.assert_allclose(frame.vectors_to_global([1.4, 0.0]), [0.0, 1.4], atol=1e-12)


def test_heading_wrap():
    frame = LocalFrame(ORIGIN, -3.1)
    local = float(frame.headings_to_local(3.1))
    assert np.isclose(local, 6.2 - 2 * np.pi)
    angles = wrap_angle(np.linspace(-10, 10, 1001))
    assert angles.min() >= -np.pi and angles.max() < np.pi
    assert float(wrap_angle(np.pi)) == -np.pi
