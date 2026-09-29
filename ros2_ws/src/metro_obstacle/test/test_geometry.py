import math

import numpy as np

from metro_obstacle.geometry import corridor_edges, corridor_length, yaw_at


def test_edges_of_straight_corridor_along_minus_y() -> None:
    c = np.stack([np.zeros(5), -np.arange(5.0) * 10, np.full(5, -1.8)], 1)
    left, right = corridor_edges(c, np.full(5, 1.05))
    # Едем в −Y, значит «влево» — это +X.
    np.testing.assert_allclose(left[:, 0], 1.05, atol=1e-9)
    np.testing.assert_allclose(right[:, 0], -1.05, atol=1e-9)
    np.testing.assert_allclose(left[:, 2], -1.8)
    assert math.isclose(corridor_length(c), 40.0)


def test_edges_follow_curve() -> None:
    a = np.linspace(0, math.pi / 2, 50)
    c = np.stack([100 * np.sin(a), 100 * (1 - np.cos(a)), np.zeros_like(a)], 1)
    left, right = corridor_edges(c, np.full(len(c), 1.0))
    np.testing.assert_allclose(np.linalg.norm(left - c, axis=1), 1.0, rtol=1e-6)
    # На повороте влево левая кромка ближе к центру поворота (0, 100).
    centre = np.array([0.0, 100.0, 0.0])
    assert (np.linalg.norm(left - centre, axis=1) < np.linalg.norm(right - centre, axis=1)).all()


def test_degenerate_inputs() -> None:
    empty = np.zeros((0, 3))
    left, right = corridor_edges(empty, np.zeros(0))
    assert left.shape == (0, 3) and right.shape == (0, 3)
    one = np.zeros((1, 3))
    assert corridor_edges(one, np.ones(1))[0].shape == (0, 3)
    assert yaw_at(empty, (5.0, 0.0, 0.0)) == 0.0
    # Повторяющиеся точки оси не дают NaN.
    dup = np.array([[0, 0, 0], [0, 0, 0], [10, 0, 0]], float)
    assert np.isfinite(corridor_edges(dup, np.ones(3))[0]).all()


def test_yaw_at_nearest_axis_point() -> None:
    c = np.stack([np.zeros(10), -np.arange(10.0), np.zeros(10)], 1)
    assert math.isclose(yaw_at(c, (0.1, -4.0, 0.0)), -math.pi / 2)
