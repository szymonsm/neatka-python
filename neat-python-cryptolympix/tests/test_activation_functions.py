import unittest
import numpy as np
from activation_functions import step, sigmoid, tanh, relu, leaky_relu, prelu, elu, softmax, linear, swish


class TestActivationFunctions(unittest.TestCase):
    def runTest(self):
        self.test_step()
        self.test_sigmoid()
        self.test_tanh()
        self.test_relu()
        self.test_leaky_relu()
        self.test_prelu()
        self.test_elu()
        self.test_softmax()
        self.test_linear()
        self.test_swish()

    def test_step(self):
        self.assertEqual(step(0), 0)
        self.assertEqual(step(1), 1)
        self.assertEqual(step(-1), 0)

    def test_sigmoid(self):
        self.assertAlmostEqual(sigmoid(0), 0.5)
        self.assertAlmostEqual(sigmoid(1), 1 / (1 + np.exp(-1)))
        self.assertAlmostEqual(sigmoid(-1), 1 / (1 + np.exp(1)))

    def test_tanh(self):
        self.assertAlmostEqual(tanh(0), 0)
        self.assertAlmostEqual(tanh(1), np.tanh(1))
        self.assertAlmostEqual(tanh(-1), np.tanh(-1))

    def test_relu(self):
        self.assertEqual(relu(0), 0)
        self.assertEqual(relu(1), 1)
        self.assertEqual(relu(-1), 0)

    def test_leaky_relu(self):
        self.assertEqual(leaky_relu(0), 0)
        self.assertEqual(leaky_relu(1), 1)
        self.assertEqual(leaky_relu(-1), -0.01)

    def test_prelu(self):
        self.assertEqual(prelu(0, 0.1), 0)
        self.assertEqual(prelu(1, 0.1), 1)
        self.assertEqual(prelu(-1, 0.1), -0.1)

    def test_elu(self):
        self.assertEqual(elu(0), 0)
        self.assertEqual(elu(1), 1)
        self.assertAlmostEqual(elu(-1), -0.6321205588285577)

    def test_softmax(self):
        # Note: Softmax is applied to an array, testing with an array for expected behavior
        input_array = np.array([1, 2, 3])
        expected_output = np.exp(input_array) / np.sum(np.exp(input_array))
        np.testing.assert_array_almost_equal(
            softmax(input_array), expected_output)

    def test_linear(self):
        self.assertEqual(linear(0), 0)
        self.assertEqual(linear(1), 1)
        self.assertEqual(linear(-1), -1)

    def test_swish(self):
        self.assertAlmostEqual(swish(0), 0)
        self.assertAlmostEqual(swish(1), 1 / (1 + np.exp(-1)))
        self.assertAlmostEqual(swish(-1), -1 / (1 + np.exp(1)))
