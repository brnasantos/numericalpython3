import unittest
import numpy as np
import sys
sys.path.insert(0,'../')
from Solvers import solvers

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
