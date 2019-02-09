import numpy as np
import LUpivot
import math
import swap

class solvers:
    def getA(self,L,U):
        A = np.dot(L,U)
        return A

    def get_determinant_from_L_and_U(self,L,U):
        A = solvers.getA(self,L,U)
        determinant = np.prod(np.diag(A))
        return determinant

    def get_determinant_from_A(self,A):
        determinant = np.prod(np.diag(A))
        return determinant

    def byLUdecomp(self,A,B):
        '''Using LU decomposition to solve matrix problems.
        a = LUdecomp(a)
        LUdecomposition: [L][U] = [a]
        x = LUsolve(a,b)
        Solution phase: solves [L][U]{x} = {b}
        '''
        def LUdecomp(a):
            n = len(a)
            for k in range(0,n-1):
                for i in range(k+1,n):
                    if a[i,k] != 0.0:
                        lam = a [i,k]/a[k,k]
                        a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                        a[i,k] = lam
            return a

        def LUsolve(a,b):
            n = len(a)
            for k in range(1,n):
                b[k] = b[k] - np.dot(a[k,0:k],b[0:k])
            b[n-1] = b[n-1]/a[n-1,n-1]
            for k in range(n-2,-1,-1):
                b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
            return b
        # Solution of the parameters
        L = LUdecomp(A)
        return LUsolve(L,B)

    def byGaussElimin(self,A,B):
        ''' Using Gauss Elimination to solve matrix problems.
        x = gaussElimin(a,b).
        Solves [a]{b} = {x} by Gauss elimination.
        '''
        def gaussElimin(a,b):
            n = len(b)
        # Elimination Phase
            for k in range(0,n-1):
                for i in range(k+1,n):
                    if a[i,k] != 0.0:
                        lam = a [i,k]/a[k,k]
                        a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                        b[i] = b[i] - lam*b[k]
        # Back substitution
            for k in range(n-1,-1,-1):
                b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
            return b
        # Solving the parameters
        return gaussElimin(A,B)

    def byCholeski(self,A,B):
        ''' Using Choleski to solve matrix problems.
        L = choleski(a)
        Choleski decomposition: [L][L]transpose = [a]
        Solution phase of Choleski’s decomposition method
        '''
        def choleski(a):
          n = len(a)
          for k in range(n):
            try:
              a[k,k] = math.sqrt(a[k,k] \
              - np.dot(a[k,0:k],a[k,0:k]))
            except ValueError:
              error.err('Matrix is not positive definite')
            for i in range(k+1,n):
              a[i,k] = (a[i,k] - np.dot(a[i,0:k],a[k,0:k]))/a[k,k]
          for k in range(1,n): a[0:k,k] = 0.0
          return a

        def choleskiSol(L,b):
          n = len(b)
        # Solution of [L]{y} = {b}
          for k in range(n):
            b[k] = (b[k] - np.dot(L[k,0:k],b[0:k]))/L[k,k]
        # Solution of [L_transpose]{x} = {y}
          for k in range(n-1,-1,-1):
            b[k] = (b[k] - np.dot(L[k+1:n,k],b[k+1:n]))/L[k,k]
          return b
        # Solution of the paramaters
        L = choleski(A)
        return choleskiSol(L,B)

    def byLUdecomp3(self,c,d,e,b):

        '''Solvng matrix problems using LU decomposition with 3 main diagonals.
        c,d,e = LUdecomp3(c,d,e).
        LU decomposition of tridiagonal matrix [c\d\e]. On output
        {c},{d} and {e} are the diagonals of the decomposed matrix.
        x = LUsolve(c,d,e,b).
        Solves [c\d\e]{x} = {b}, where {c}, {d} and {e} are the
        vectors returned from LUdecomp3.'''

        def LUdecomp3(c,d,e):
            n = len(d)
            for k in range(1,n):
                lam = c[k-1]/d[k-1]
                d[k] = d[k] - lam*e[k-1]
                c[k-1] = lam
            return c,d,e

        def LUsolve3(c,d,e,b):
            n = len(d)
            for k in range(1,n):
                b[k] = b[k] - c[k-1]*b[k-1]
            b[n-1] = b[n-1]/d[n-1]
            for k in range(n-2,-1,-1):
                b[k] = (b[k] - e[k]*b[k+1])/d[k]
            return b
        # Solving the parameters problem
        c,d,e= LUdecomp3(c,d,e)
        return LUsolve3(c,d,e,b)

    def byLUdecomp5(self,d,e,f,b):
        '''Using LU decomposition to solve problems from 5 main diagonals matrixes.
        d,e,f = LUdecomp5(d,e,f).
        LU decomposition of symmetric pentadiagonal matrix [a], where
        {f}, {e} and {d} are the diagonals of [a]. On output
        {d},{e} and {f} are the diagonals of the decomposed matrix.
        x = LUsolve5(d,e,f,b).
        Solves [a]{x} = {b}, where {d}, {e} and {f} are the vectors
        returned from LUdecomp5.
        '''


        def LUdecomp5(d,e,f):
            n = len(d)
            for k in range(n-2):
                lam = e[k]/d[k]
                d[k+1] = d[k+1] - lam*e[k]
                e[k+1] = e[k+1] - lam*f[k]
                e[k] = lam
                lam = f[k]/d[k]
                d[k+2] = d[k+2] - lam*f[k]
                f[k] = lam
            lam = e[n-2]/d[n-2]
            d[n-1] = d[n-1] - lam*e[n-2]
            e[n-2] = lam
            return d,e,f

        def LUsolve5(d,e,f,b):
            n = len(d)
            b[1] = b[1] - e[0]*b[0]
            for k in range(2,n):
                b[k] = b[k] - e[k-1]*b[k-1] - f[k-2]*b[k-2]
            b[n-1] = b[n-1]/d[n-1]
            b[n-2] = b[n-2]/d[n-2] - e[n-2]*b[n-1]
            for k in range(n-3,-1,-1):
                b[k] = b[k]/d[k] - e[k]*b[k+1] - f[k]*b[k+2]
            return b
        # Solving the parameters problem.
        d,e,f = LUdecomp5(d,e,f,b)
        return LUsolve5(d,e,f,b)

    def gaussPivot(self,a,b):
        '''Solving matrix problems using Gauss Pivoting.
        x = gaussPivot(a,b,tol=1.0e-12).
        Solves [a]{x} = {b} by Gauss elimination with
        scaled row pivoting'''

        def gaussPivot(a,b,tol=1.0e-12):
            n = len(b)
        # Set up scale factors
            s = np.zeros(n)
            for i in range(n):
                s[i] = max(np.abs(a[i,:]))
            for k in range(0,n-1):
        # Row interchange, if needed
                p = np.argmax(np.abs(a[k:n,k])/s[k:n]) + k
                if abs(a[p,k]) < tol:
                    print ('matrix is singular')
                    sys.exit()
                if p != k:
                    swap.swapRows(b,k,p)
                    swap.swapRows(s,k,p)
                    swap.swapRows(a,k,p)
        # Elimination
                for i in range(k+1,n):
                    if a[i,k] != 0.0:
                        lam = a[i,k]/a[k,k]
                        a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                        b[i] = b[i] - lam*b[k]

            if abs(a[n-1,n-1]) < tol:
                print ('matrix is singular')
                sys.exit()

        # Back substitution
            b[n-1] = b[n-1]/a[n-1,n-1]
            for k in range(n-2,-1,-1):
                b[k] = (b[k] - np.dot(a[k,k+1:n],b[k+1:n]))/a[k,k]
            return b
        # Solving the parameters problem.
        return gaussPivot(a,b)

    def LUpivot(self,a,b):
    #    '''Using LU Pivoting to solve matrix problems.
    #    a,seq = LUdecomp(a,tol=1.0e-9).
    #    LU decomposition of matrix [a] using scaled row pivoting.
    #    The returned matrix [a] = [L\U] contains [U] in the upper
    #    triangle and the nondiagonal terms of [L] in the lower triangle.
    #    Note that [L][U] is a row-wise permutation of the original [a];
    #    the permutations are recorded in the vector {seq}.
    #    x = LUsolve(a,b,seq).
    #    Solves [L][U]{x} = {b}, where the matrix [a] = [L\U] and the
    #    permutation vector {seq} are returned from LUdecomp.

        def LUpivotdecomp(a,tol=1.0e-9):
            n = len(a)
            seq = np.array(range(n))

          # Set up scale factors
            s = np.zeros(n)
            for i in range(n):
                s[i] = max(abs(a[i,:]))

            for k in range(0,n-1):

              # Row interchange, if needed
                p = np.argmax(abs(a[k:n,k])/s[k:n]) + k
                if abs(a[p,k]) <  tol:
                    print ('matrix is singular')
                    sys.exit()
                if p != k:
                    swap.swapRows(s,k,p)
                    swap.swapRows(a,k,p)
                    swap.swapRows(seq,k,p)

              # Elimination
                for i in range(k+1,n):
                    if a[i,k] != 0.0:
                        lam = a[i,k]/a[k,k]
                        a[i,k+1:n] = a[i,k+1:n] - lam*a[k,k+1:n]
                        a[i,k] = lam
            return a,seq

        def LUpivotsolve(a,b,seq):
            n = len(a)

          # Rearrange constant vector; store it in [x]
            x = b.copy()
            for i in range(n):
                x[i] = b[seq[i]]

          # Solution
            for k in range(1,n):
                x[k] = x[k] - np.dot(a[k,0:k],x[0:k])
            x[n-1] = x[n-1]/a[n-1,n-1]
            for k in range(n-2,-1,-1):
               x[k] = (x[k] - np.dot(a[k,k+1:n],x[k+1:n]))/a[k,k]
            return x
        # Solving the parameters problem.
        a,seq = LUpivotdecomp(a)
        return LUpivotsolve(a,b,seq)
