import numpy as np

#along x,z equals dim = 1
#along y,z equals dim = 2
def calculateLine(P, Q, dim):
    #from P->Q
    if dim == 1:
        m = (Q[2]-P[2])/(Q[0] - P[0])
        b = -m*P[0] + P[2]
    elif dim ==2:
        m = (Q[2]-P[2])/(Q[1] - P[1])
        b = -m*P[1] + P[2]
    return m,b

def detrend3D(origSurf, coefficients, point):
    
    newSurface = origSurf
    sizeSurf = size(origSurf);
    for i in range(origSurf.shape[0]):
        for j in range(origSurf.shape[1]):
            newSurface[i][j] = origSurf[i][j] - (- (coefficients(0)*(i-point(0))-coefficients(1)*(j-point(1)))/coefficient(2)) +point(2)
    return newSurface
#Takes 3 points P, Q, R
def calculatePlane(P, Q, R):
    #find vector P->Q
    v1 = np.array([Q[0]- P[0], Q[1]-P[1], Q[2]-P[2]])
    #print("v1:...",v1)
    
    #find vector P->R
    v2 = np.array([R[0]- P[0], R[1]-P[1], R[2]-P[2]])
    #print("v2:...",v2)
    
    #cross product of two vectors to obtain perpendicular vector
    vc = np.cross(v1,v2)
    
    return vc

p = (1,0,2)
q = (-1,1,2)
r = (5,0,3)


planeEq= calculatePlane(p,q,r)
print(planeEq)
x = detrend3D(planeEq,p)


print(x)