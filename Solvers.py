import numpy as np
import math
import swap
import sys


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

        def gausspivot(a,b,tol=1.0e-12):
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
        return gausspivot(a,b)

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

    def rootsearch(f,a,b,dx):
        x1 = a
        f1 = f(a)
        x2 = a + dx
        f2 = f(x2)
        while np.sign(f1) == np.sign(f2):
            if x1 >= b:
                return None,None
            x1 = x2; f1 = f2
            x2 = x1 + dx; f2 = f(x2)
        else:
            return x1,x2

    def bisection(f,x1,x2,switch=1,tol=1.0e-9):
        f1 = f(x1)
        if f1 == 0.0:
            return x1
        f2 = f(x2)
        if f2 == 0.0:
            return x2
        if np.sign(f1) == np.sign(f2):
            print ('Root is not bracketed')
            sys.exit()
        n = int(math.ceil(math.log(abs(x2 - x1)/tol)/math.log(2.0)))
        for i in range(n):
            x3 = 0.5*(x1 + x2)
            f3 = f(x3)
        if (switch == 1) and (abs(f3) > abs(f1)) and (abs(f3) > abs(f2)):
            return None
        if f3 == 0.0:
            return x3
        if np.sign(f2)!= np.sign(f3):
            x1 = x3
            f1 = f3
        else:
            x2 = x3
            f2 = f3
        return (x1 + x2)/2.0

    def ridder(f,a,b,tol=1.0e-9):
        fa = f(a)
        if fa == 0.0:
            return a
        fb = f(b)
        if fb == 0.0:
            return b
        if fa*fb > 0.0:
            print('Root is not bracketed')
            sys.exit()
        for i in range(30):
            # Compute the improved root x from Ridder’s formula
            c = 0.5*(a + b); fc = f(c)
            s = math.sqrt(fc**2 - fa*fb)
            if s == 0.0:
                return None
            dx = (c - a)*fc/s
            if (fa - fb) < 0.0:
                dx = -dx
            x = c + dx
            fx = f(x)
            # Test for convergence
            if i > 0:
                if abs(x - xOld) < tol*max(abs(x),1.0):
                    return x
            xOld = x
            # Re-bracket the root as tightly as possible
            if fc*fx > 0.0:
                if fa*fx < 0.0:
                    b = x
                    fb = fx
                else:
                    a = x
                    fa = fx
            else:
                a = c
                b = x
                fa = fc
                fb = fx
        return None

    def newtonRaphson(f,df,x,tol=1.0e-9):
        for i in range(30):
            dx = -f(x)/df(x)
            x = x + dx
            if abs(dx) < tol:
                return x

    def newtonRaphson2(f,x,tol=1.0e-9):
        def jacobian(f,x):
            h = 1.0e-4
            n = len(x)
            jac = np.zeros((n,n))
            f0 = f(x)
            for i in range(n):
                temp = x[i]
                x[i] = temp + h
                f1 = f(x)
                x[i] = temp
                jac[:,i] = (f1 - f0)/h
                return jac,f0
            for i in range(30):
                    jac,f0 = jacobian(f,x)
                    if math.sqrt(np.dot(f0,f0)/len(x)) < tol: return x
                    dx = solvers.gaussPivot(jac,-f0)
                    x = x + dx
                    if math.sqrt(np.dot(dx,dx)) < tol*max(max(abs(x)),1.0):
                        return x

    def first_central_difference(n, h, f):
        ''' Being 'n' the degree of the desired derivative and f the dictionary
        containing the f(x) of each point. '''
        if n == 1:
            derivative = ((-f['x-h']+f['x+h'])/(h))/2

        elif n == 2:
            derivative = (f['x-h']-2*f['x']+f['x+h'])/h**2

        elif n == 3:
            derivative = ((-f['x-2h']+2*f['x-h']-2*f['x']+f['x+2h'])/h**3)/2

        elif n == 4:
            derivative = (f['x-2h']-4*f['x-h']+6*f['x']-4*f['x+h']+f['x+2h'])/h**4
        return derivative

    def first_foward_finite_difference(n, h, f):
        '''Similitar to the previous function, this one receives a 'n' as the
        degree of the desired derivative, and f, as the dictionary containing
        the f(x) of each points'''
        if n == 1:
            derivative = (-f['x']+f['x+h'])/(h)

        elif n == 2:
            derivative = (f['x']-2*f['x+h']+f['x+2h'])/h**2

        elif n == 3:
            derivative = (-f['x']+3*f['x+h']-3*f['x+2h']+f['x+3h'])/h**3

        elif n == 4:
            derivative = (f['x']-4*f['x+h']+6*f['x+2h']-4*f['x+3h']+f['x+4h'])/h**4
        return derivative

    def first_backward_finite_difference(n, h, f):
        '''Works the same way as the previous one, but it searches for the
        backward derivative.'''
        if n == 1:
            derivative = (f['x']-f['x-h'])/(h)

        elif n == 2:
            derivative = (f['x']-2*f['x-h']+f['x-2h'])/h**2

        elif n == 3:
            derivative = (f['x']-3*f['x-h']+3*f['x-2h']-f['x-3h'])/h**3

        elif n == 4:
            derivative = (f['x']-4*f['x-h']+6*f['x-2h']-4*f['x-3h']+f['x-4h'])/h**4
        return derivative

    def second_foward_finite_difference(n, h, f):
        '''Works as the first foward difference, but the truncation error
        remains at second degree. This one is more accurate than the firsts
        noncentral differences'''
        if n == 1:
            derivative = (-3*f['x']+4*f['x+h']-f['x+2h'])/(2*h)

        elif n == 2:
            derivative = (2*f['x']-5*f['x+h']+4*f['x+2h']-f['x+3h'])/h**2

        elif n == 3:
            derivative = (-5*f['x']+18*f['x+h']-24*f['x+2h']+14*f['x+3h']-3*f['x+4h'])/2*h**3

        elif n == 4:
            derivative = (3*f['x']-14*f['x+h']+26*f['x+2h']-24*f['x+3h']+11*f['x+4h']-2*f['x+5h'])/h**4
        return derivative

    def second_backward_finite_difference(n, h, f):
        '''Works the same way as the previous one, but it searches for the
        backward derivative.'''
        if n == 1:
            derivative = (3*f['x']-4*f['x-h']+f['x-2h'])/(2*h)

        elif n == 2:
            derivative = (2*f['x']-5*f['x-h']+4*f['x-2h']-f['x-3h'])/h**2

        elif n == 3:
            derivative = (5*f['x']-18*f['x-h']+24*f['x-2h']-14*f['x-3h']+3*f['x-4h'])/2*h**3

        elif n == 4:
            derivative = (3*f['x']-14*f['x-h']+26*f['x-2h']-24*f['x-3h']+11*f['x-4h']-2*f['x-5h'])/h**4
        return derivative

    def richardson_extrapolation(n, h1, h2, f1, f2):
        '''this method boosts the accuracy of certain difference,
        G is the most accurate approximation we could achieve, 'p'
        is a constant which we'll determine at this function as 2,
        'n' is the degree of the derivative, h1 and h2 the two 'steps'
        we take to approximate the derivative and f the dictionary
        containing the f(x) of each points'''
        p = 2
        if h1 > h2:
            h2 = h
            h2 = h1
            h1 = h2
        g1 = solvers.second_foward_finite_difference(n,h1,f1)
        g2 = solvers.second_foward_finite_difference(n,h2,f2)
        G = (((h2/h1)**p)*g1-g2)/((h2/h1)**p-1)
        return G
