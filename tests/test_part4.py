"""Independent P5 and diagonal-GD tests owned by Part 4."""
import unittest

import numpy as np
from numpy.testing import assert_allclose

from coordinate_geometry.contracts import RunConfig
from coordinate_geometry.preconditioning import effective_hessian, p5_condition_number
from coordinate_geometry.problems import QuadraticProblem, rotation_2d
from coordinate_geometry.runner import run


class Part4Tests(unittest.TestCase):
    def test_endpoints_and_curve(self):
        for kappa in (1, 10, 100, 1000, 10000):
            for angle in range(91):
                h = QuadraticProblem(np.array([1 / kappa, 1]),
                                     rotation_2d(np.deg2rad(angle))).hessian
                values = np.linalg.eigvalsh(effective_hessian(h))
                predicted = p5_condition_number(h)
                assert_allclose(predicted, values[-1] / values[0], rtol=1e-8)
                if angle in (0, 90):
                    assert_allclose(predicted, 1, atol=1e-10)
                if angle == 45:
                    assert_allclose(predicted, kappa, rtol=1e-8)

    def test_diagonal_one_step(self):
        p = QuadraticProblem(np.array([.0001, 1]), np.eye(2))
        result = run(p, np.array([2., -3.]), RunConfig("diagonal_gd", 1., 1))
        assert_allclose(result.final_position, 0, atol=1e-14)
        self.assertEqual(result.gradient_calls, 1)
        self.assertEqual(result.first_attainment, 1)

    def test_transformed_exact_recurrence(self):
        p = QuadraticProblem(np.array([.01, 1]), rotation_2d(.3))
        scale = np.sqrt(np.diag(p.hessian))
        w = np.array([2., -3.])
        rate, steps = .8, 25
        expected_y = np.linalg.matrix_power(np.eye(2) - rate * effective_hessian(p.hessian), steps) @ (scale * w)
        result = run(p, w, RunConfig("diagonal_gd", rate, steps))
        assert_allclose(result.final_position, expected_y / scale, atol=1e-12)

    def test_reject_non_spd(self):
        for h in (np.array([[1., 2.], [2., 1.]]), np.array([[1., 1.], [1., 1.]]),
                  np.array([[1., 1.], [0., 1.]])):
            with self.assertRaises(ValueError):
                effective_hessian(h)


if __name__ == "__main__":
    unittest.main()
