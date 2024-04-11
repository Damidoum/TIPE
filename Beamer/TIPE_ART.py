from math import cos, sin, sqrt, pi
import numpy as np
import matplotlib.pyplot as plt

mat = np.random.randint(10, size=(20, 20))
l = mat.shape[0]  # nombre de ligne (coordonnee y)
L = mat.shape[1]  # nombre de colonne (coordonne x)
M = l * L
d = sqrt(l ** 2 + L ** 2)

# donne le point d'intersection de deux droites donnees en polaire
def distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# produit scalaire canonique
def ps(m1, m2):
    return float((sum([i * j for (i, j) in zip(m1, m2)])))


# coordonnees du pixel j
def pixel(j):
    r = (j - 1) % L  # reste dans la division euclidienne de j-1 par L
    q = (j - 1) // L  # quotient dans la division euclidienne de j-1 par L
    return (q, r)


# conversion de l'image en une matrice colonne :
def colonne():
    m = np.zeros((M, 1))
    for j in range(0, M):
        (q, r) = pixel(j + 1)
        m[j, 0] = mat[q, r]
    return m


# equation polaire de la droite
def rayon(u, t, theta):
    return (u * sin(theta) + t * cos(theta), u * cos(theta) - t * sin(theta))


# est ce que la droite de parametres (u,theta) passe par le pixel j

def isInPixel(u, theta, j):
    (a, b) = pixel(j)
    if theta != 0 and theta != pi / 2:
        # Cas 1 (haut) :
        t = (a - u * sin(theta)) / cos(theta)
        y2 = rayon(u, t, theta)[1]
        if b <= y2 <= b + 1:
            return True
        # Cas 2 (bas) :
        t = (a + 1 - u * sin(theta)) / cos(theta)
        y2 = rayon(u, t, theta)[1]
        if b <= y2 <= b + 1:
            return True
        # Cas 3 (gauche):
        t = (u * cos(theta) - b) / sin(theta)
        x2 = rayon(u, t, theta)[0]
        if a <= x2 <= a + 1:
            return True
        # Cas 4 (droite):
        t = (u * cos(theta) - b - 1) / sin(theta)
        y2 = rayon(u, t, theta)[0]
        if a <= x2 <= a + 1:
            return True
    elif theta == 0:
        if a <= u <= a + 1:
            return True
    elif theta == pi / 2:
        if b <= u <= b + 1:
            return True
    return False


def intersection(u, theta, j):
    long = 0
    if not isInPixel(u, theta, j):
        return False
    (a, b) = pixel(j)
    if theta != pi / 2 and theta != 0:
        lst = []
        lst2 = []
        # intersection avec le bord gauche :
        t = (u * cos(theta) - b) / sin(theta)
        lst.append(rayon(u, t, theta))

        # intersection avec le bord droit :
        t = (u * cos(theta) - b - 1) / sin(theta)
        lst.append((rayon(u, t, theta)))

        # intersection avec le bord haut :
        t = (a - u * sin(theta)) / cos(theta)
        lst.append((rayon(u, t, theta)))

        # intersection avec le bord bas :
        t = (a + 1 - u * sin(theta)) / cos(theta)
        lst.append((rayon(u, t, theta)))
        for point in lst[0:2]:
            if a <= point[0] <= a + 1:
                lst2.append(point)
        for point in lst[2:4]:
            if b <= point[1] <= b + 1:
                lst2.append(point)
        assert len(lst2) == 2
        long = distance(lst2[0], lst2[1])
    else:
        long = 1
    return long


# definition des rayons de projections
def projections(nb_dir, nb_droite):
    dir = np.linspace(0, pi / 2, nb_dir)
    droite = np.linspace(5, 15, nb_droite)
    proj = []
    for theta in dir:
        for u in droite:
            proj.append((u, theta))
    return proj


def matR_v1(proj):
    N = len(proj)
    R = np.zeros((N, M))
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if isInPixel(proj[i - 1][0], proj[i - 1][1], j):
                R[i - 1, j - 1] = 1
    return R


# definition de la matrice R - version 2
def matR_v2(proj):
    N = len(proj)
    R = np.zeros((N, M))
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            R[i - 1, j - 1] = intersection(proj[i - 1][0], proj[i - 1][1], j)
    return R


# construction de la matrice de projection
def matP(R):
    m = colonne()
    N = R.shape[0]
    P = np.zeros((N, 1))
    for i in range(0, N):
        for j in range(0, M):
            P[i, 0] += R[i, j] * m[j, 0]
    return P


# On peut alors tenter de resoudre le probleme inverse : RF = P (par n iteration)
def operateurs(R):
    # extractions des lignes de la matrice R
    N = R.shape[0]
    lignes = []
    transpo = []
    q = []
    for i in range(N):
        lignes.append(R[i, :])
        Ni = np.transpose(np.array([R[i, :]]))
        transpo.append(Ni)
        q.append(P[i, 0] * Ni / ps(Ni, Ni))

    # definition des operateurs de projections orthognales
    operator = []
    id = np.eye(M)
    for i in range(N):
        Ni = transpo[i]
        Ti = id - (1 / ps(Ni, Ni)) * np.dot(Ni, np.transpose(Ni))
        operator.append(Ti)
    return [q, operator]


def reconstruction(n, op, F_0=np.zeros((M, 1))):
    nb = len(op[0])
    T = op[1]
    q = op[0]
    F = F_0
    for k in range(1, n):
        F = q[(k - 1) % nb] + np.dot(T[(k - 1) % nb], F - q[(k - 1) % nb])
    return F


proj = projections(500, 100)
R = matR_v1(proj)
P = matP(R)
op = operateurs(R)
FC = reconstruction(500, op)


def columnIntoMatrix(C):
    A = np.zeros((l, L))
    for i in range(l):
        for j in range(L):
            A[i, j] = C[i * L + j, 0]
    return A
F = columnIntoMatrix(FC)
plt.imshow(F)
plt.show()
plt.imshow(mat)
plt.show()