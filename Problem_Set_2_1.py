from numpy import asarray, prod, diag, dot
import LUdecomp as lu
import gaussElimin as gauss
import choleski as ch
import LUdecomp3 as lu3
import LUdecomp5 as lu5
import gaussPivot as gpivot
import LUpivot 

class solvers:
    def getA(self,L,U):
        A = dot(L,U)
        return A

    def get_determinant_from_L_and_U(self,L,U):
        A = solvers.getA(self,L,U)
        determinant = prod(diag(A))
        return determinant

    def get_determinant_from_A(self,A):
        determinant = prod(diag(A))
        return determinant

    def byLUdecomp(self,A,B):
        L = lu.LUdecomp(A)
        return lu.LUsolve(L,B)

    def byGaussElimin(self,A,B):
        return gauss.gaussElimin(A,B)

    def byCholeski(self,A,B):
        L = ch.choleski(A)
        return ch.choleskiSol(L,B)
    
    def byLUdecomp3(self,c,d,e,b):
        c,d,e= lu3.LUdecomp3(c,d,e)
        return lu3.LUsolve3(c,d,e,b)
    
    def byLUdecomp5(self,d,e,f,b):
        d,e,f = lu5.LUdecomp5(d,e,f,b)
        return lu5.LUsolve5(d,e,f,b)
    
    def gaussPivot(self,a,b):
        return gpivot.gaussPivot(a,b)
    
    def LUpivot(self,a,b):
        a,seq = LUpivot.LUdecomp(a)
        return LUpivot.LUsolve(a,b,seq)
    
