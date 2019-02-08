import unittest
from numpy import *
import sys
sys.path.insert(0,'../')
from Problem_Set_2_1 import solvers


class Problem_Set_2_1_Test(unittest.TestCase):
    def test_getA(self):
        L = asarray([[1,0,0],[1,1,0],[1,5/3,1]], dtype = float)
        U = asarray([[1,2,4],[0,3,21],[0,0,0]], dtype = float)
        A = solvers.getA(self,L,U)
        Expected_A = asarray([[1,2,4],[1,5,25],[1,7,39]], dtype = float)
        self.assertAlmostEqual(A.all(),Expected_A.all(), delta = 0.0001)

    def test_get_determinant_from_A(self):
        Expected_A = asarray([[1,2,4],[1,5,25],[1,7,39]], dtype = float)
        Expected_Determinant = 195
        determinant = solvers.get_determinant_from_A(self,Expected_A)
        self.assertAlmostEqual(determinant, Expected_Determinant, delta = 0.0001)

    def test_get_determinant_from_L_and_U(self):
        L = asarray([[1,0,0],[1,1,0],[1,5/3,1]], dtype = float)
        U = asarray([[1,2,4],[0,3,21],[0,0,0]], dtype = float)
        Expected_Determinant = 195
        determinant = solvers.get_determinant_from_L_and_U(self,L,U)
        self.assertAlmostEqual(determinant, Expected_Determinant, delta = 0.0001)

    def test_byGaussElimin(self):
        A = asarray([[2,-3,-1],[3,2,-5],[2,4,-1]], dtype = float)
        B = asarray([[3],[-9],[-5]], dtype = float)
        Expected_x = asarray([[32/49],[-8/7],[85/49]], dtype = float)
        x = solvers.byGaussElimin(self,A,B)
        self.assertAlmostEqual(Expected_x.all(), x.all(), delta = 0.0001)

    def test_byLUdecomp(self):
        A = asarray([[4,-2,-3], [12,4,-10],[-16,28,18]], dtype = float)
        B = asarray([[1.1],[0],[-2.3]], dtype = float)
        Expected_x = asarray([[(1.1+(-3.3+8.7/8)/5+3*8.7/8)/4],[(-3.3+8.7/8)/10],[8.7/8]], dtype = float)
        x = solvers.byLUdecomp(self,A,B)
        self.assertAlmostEqual(Expected_x.all(), x.all(), delta = 0.0001)

    def test_byCholeski(self):
        A = asarray([[1,1,1],[1,2,2],[1,2,3]], dtype = float)
        B = asarray([[1],[3/2],[3]], dtype = float)
        Expected_x = asarray([[1/2],[-1],[3/2]], dtype = float)
        x = solvers.byCholeski(self,A,B)
        self.assertAlmostEqual(Expected_x.all(), x.all(), delta = 0.0001)

    def test_byLUdecomp3(self):
        d = ones((5))*2.0
        c = ones((4))*(-1.0)
        b = array([5.0, -5.0, 4.0, -5.0, 5.0])
        e = c.copy()
        x = solvers.byLUsolve3(self,c,d,e,b)
        expected_x = asarray([[2.], [-1.], [1.], [-1.],[2.]],dtype = float)
        self.assertAlmostEqual(Expected_x.all(), x.all(), delta = 0.0001)
        
        
        
