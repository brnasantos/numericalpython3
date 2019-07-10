import unittest
import numpy as np
import sys
sys.path.insert(0,'../')
from Solvers import solvers
import math
import pandas as pd

class TestProblemSet2(unittest.TestCase):
    def test_getA(self):
        """
        This test is designed to check if all matrix entries are being read.
        """
        L = np.asarray([[1, 0, 0],
                     [1, 1, 0],
                     [1, 5 / 3, 1]], dtype=float)

        U = np.asarray([[1, 2, 4],
                        [0, 3, 21],
                        [0, 0, 0]], dtype = float)

        A = solvers.getA(self,L,U)

        Expected_A = np.asarray([[1, 2, 4],
                                 [1, 5, 25],
                                 [1, 7, 39]], dtype=float)

        rows = len(Expected_A.tolist())
        for row in range(rows):
            columns = len(Expected_A[row])
            for column in range(columns):
                self.assertAlmostEqual(Expected_A[row][column], A[row][column], delta=0.0001)

    def test_get_determinant_from_A(self):
        Expected_A = np.asarray([[1, 2, 4],
                                 [1, 5, 25],
                                 [1, 7, 39]], dtype=float)

        Expected_Determinant = 195

        determinant = solvers.get_determinant_from_A(self,Expected_A)

        self.assertAlmostEqual(determinant, Expected_Determinant, delta=0.0001)

    def test_get_determinant_from_L_and_U(self):
        L = np.asarray([[1, 0, 0],
                        [1, 1, 0],
                        [1, 5 / 3, 1]], dtype=float)

        U = np.asarray([[1, 2, 4],
                        [0, 3, 21],
                        [0, 0, 0]], dtype=float)

        Expected_Determinant = 195

        determinant = solvers.get_determinant_from_L_and_U(self,L,U)

        self.assertAlmostEqual(determinant, Expected_Determinant, delta = 0.0001)

    def test_byGaussElimin(self):
        A = np.asarray([[2, -3, -1],
                        [3, 2, -5],
                        [2, 4, -1]], dtype=float)

        B = np.asarray([[3],
                        [-9],
                        [-5]], dtype=float)

        expected_x = np.asarray([[32/49],[-8/7],[85/49]], dtype = float)

        x = solvers.byGaussElimin(self,A,B)

        n = len(expected_x.tolist())

        for i in range(n):
            self.assertAlmostEqual(expected_x[i], x[i], delta = 0.0001)

    def test_byLUdecomp(self):
        A = np.asarray([[4, -2, -3],
                        [12, 4, -10],
                        [-16, 28, 18]], dtype=float)

        B = np.asarray([[1.1],
                        [0],
                        [-2.3]], dtype=float)

        expected_x = np.asarray([[(1.1+(-3.3+8.7/8)/5+3*8.7/8)/4],
                              [(-3.3+8.7/8)/10],[8.7/8]],
                               dtype = float)

        x = solvers.byLUdecomp(self,A,B)

        n = len(expected_x.tolist())

        for i in range(n):
            self.assertAlmostEqual(expected_x[i], x[i], delta = 0.0001)

    def test_byCholeski(self):
        A = np.asarray([[1, 1, 1],
                        [1, 2, 2],
                        [1, 2, 3]], dtype=float)

        B = np.asarray([[1],
                        [3/2],
                        [3]], dtype=float)

        expected_x = np.asarray([[1 / 2],
                                 [-1],
                                 [3 / 2]], dtype=float)

        x = solvers.byCholeski(self,A,B)


        n = len(expected_x.tolist())

        for i in range(n):
            self.assertAlmostEqual(expected_x[i], x[i], delta = 0.0001)

    def test_byLUdecomp3(self):
        d = np.ones((5))*2.0
        c = np.ones((4))*(-1.0)
        b = np.array([5.0, -5.0, 4.0, -5.0, 5.0])
        e = c.copy()
        x = solvers.byLUdecomp3(self,c,d,e,b)
        expected_x = np.asarray([2., -1., 1., -1.,2.],dtype=float)
        n = len(expected_x.tolist())
        for i in range(n):
            self.assertAlmostEqual(expected_x[i], x[i], delta = 0.0001)

    def test_gaussPivot(self):
        A = np.array([[2, -2, 6],
                      [-2, 4, 3],
                      [-1, 8, 4]], dtype=float)

        B = np.array([[16],
                      [0],
                      [-1]], dtype=float)

        x = solvers.gaussPivot(self,A,B)

        expected_x = np.array([[1],
                               [-1],
                               [2]])

        n = len(expected_x.tolist())
        for i in range(n):
            self.assertAlmostEqual(expected_x[i], x[i], delta = 0.0001)

    def test_LUPivot(self):
        A = np.array([[2, -2, 6],
                      [-2, 4, 3],
                      [-1, 8, 4]], dtype=float)

        B = np.array([[16],
                      [0],
                      [-1]], dtype=float)

        x = solvers.LUpivot(self,A,B)

        expected_x = np.array([[1],
                               [-1],
                               [2]])

        n = len(expected_x.tolist())
        for i in range(n):
            self.assertAlmostEqual(expected_x[i], x[i], delta = 0.0001)

    def test_rootsearch(self):
        def f(x):
            return x**3 - 10.0*x**2 + 5.0
        x1 = 0.0
        x2 = 1.0
        for i in range(4):
            dx = (x2 - x1)/10.0
            x1,x2 = solvers.rootsearch(f,x1,x2,dx)
        x = []
        x.append((x1 + x2)/2.0)
        expected_x = []
        expected_x.append(0.7346)
        n = len(expected_x)
        for i in range(n):
            self.assertAlmostEqual(expected_x[i], x[i], delta = 0.05)

    def test_bisection(self):
        def f(x):
            return x**3 - 10.0*x**2 + 5.0
        x = []
        x.append(solvers.bisection(f, 0.0, 1.0, tol=1.0e-4))
        expected_x = []
        expected_x.append(0.7346)
        n = len(expected_x)
        for i in range(n):
            self.assertAlmostEqual(expected_x[i], x[i], delta = 0.05)

    def test_ridder(self):
        def f(x):
            a = (x - 0.3)**2 + 0.01
            b = (x - 0.8)**2 + 0.04
            return 1.0/a - 1.0/b
        x = []
        x.append(solvers.ridder(f,0.0,1.0))
        expected_x = []
        expected_x.append(0.5800000000000001)
        n = len(expected_x)
        for i in range(n):
            self.assertAlmostEqual(expected_x[i], x[i], delta = 0.000001)

    def test_newtonraphson(self):
        def f(x): return x**4 - 6.4*x**3 + 6.45*x**2 + 20.538*x - 31.752
        def df(x): return 4.0*x**3 - 19.2*x**2 + 12.9*x + 20.538
        x = []
        x.append(solvers.newtonRaphson(f,df,2.0))
        expected_x = []
        expected_x.append(2.09999998403)
        n = len(expected_x)
        for i in range(n):
            self.assertAlmostEqual(expected_x[i], x[i], delta = 0.000001)

    def test_newtonRaphson2(self):
        def f(x):
            f = np.zeros(len(x))
            f[0] = math.sin(x[0]) + x[1]**2 + math.log(x[2]) - 7.0
            f[1] = 3.0*x[0] + 2.0**x[1] - x[2]**3 + 1.0
            f[2] = x[0] + x[1] + x[2] - 5.0
            return f
        x = np.array([1.0, 1.0, 1.0])
        result = (solvers.newtonRaphson2(f,x))
        self.assertEqual(result,None)

    def test_central_finite_difference(self):
        f = {'x-2h': 0.,'x-h': 0.0819,'x': 0.1341,'x+h': 0.1646,'x+2h': 0.1797}
        n = 1
        h = 0.1
        expected_f1 = 0.4135
        f1 = solvers.first_central_difference(n, h, f)
        self.assertAlmostEqual(expected_f1, f1, delta = 0.0001)
        n = 2
        expected_f2 = -2.17
        f2 = solvers.first_central_difference(n, h, f)
        self.assertAlmostEqual(expected_f2, f2, delta = 0.0001)

    def test_first_foward_finite_difference(self):
        f = {'x': 0.,'x+h': 0.0819,'x+2h': 0.1341,'x+3h': 0.1646,'x+4h': 0.1797}
        n = 1
        h = 0.1
        expected_f1 = 0.819
        f1 = solvers.first_foward_finite_difference(n, h, f)
        self.assertAlmostEqual(expected_f1, f1, delta = 0.0001)
        n = 2
        expected_f2 = -2.97
        f2 = solvers.first_foward_finite_difference(n, h, f)
        self.assertAlmostEqual(expected_f2, f2, delta = 0.0001)

    def test_first_backward_finite_difference(self):
        f = {'x-4h': 0.,'x-3h': 0.0819,'x-2h': 0.1341,'x-h': 0.1646,'x': 0.1797}
        n = 1
        h = 0.1
        expected_f1 = 0.151
        f1 = solvers.first_backward_finite_difference(n, h, f)
        self.assertAlmostEqual(expected_f1, f1, delta = 0.0001)
        n = 2
        expected_f2 = -1.54
        f2 = solvers.first_backward_finite_difference(n, h, f)
        self.assertAlmostEqual(expected_f2, f2, delta = 0.0001)

    def test_second_foward_finite_difference(self):
        f = {'x': 0.,'x+h': 0.0819,'x+2h': 0.1341,'x+3h': 0.1646,'x+4h': 0.1797}
        n = 1
        h = 0.1
        expected_f1 = 0.9675
        f1 = solvers.second_foward_finite_difference(n, h, f)
        self.assertAlmostEqual(expected_f1, f1, delta = 0.0001)
        n = 2
        expected_f2 = -3.77
        f2 = solvers.second_foward_finite_difference(n, h, f)
        self.assertAlmostEqual(expected_f2, f2, delta = 0.0001)

    def test_second_backward_finite_difference(self):
        f = {'x-4h': 0.,'x-3h': 0.0819,'x-2h': 0.1341,'x-h': 0.1646,'x': 0.1797}
        n = 1
        h = 0.1
        expected_f1 = 0.074
        f1 = solvers.second_backward_finite_difference(n, h, f)
        self.assertAlmostEqual(expected_f1, f1, delta = 0.0001)
        n = 2
        expected_f2 = -0.91
        f2 = solvers.second_backward_finite_difference(n, h, f)
        self.assertAlmostEqual(expected_f2, f2, delta = 0.0001)




#richardson_extrapolation está funcionando, mas precisa de f1
#f2, h1 e h2. é bom tentar resolver isso e fazer o teste
