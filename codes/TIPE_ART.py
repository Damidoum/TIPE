from math import cos, sin, sqrt, pi
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio


def agrandissement(im, n):
    x = im.shape[0]
    y = im.shape[1]
    img2 = np.zeros((x + 2 * n, y + 2 * n))
    img2[n: x + n, n: y + n] = im
    return img2


def saveAsPNG(im, name):
    """ Sauve un tableau numpy en png sous le nom 'name' """
    imageio.imwrite(name + '.png', im)


def imread(filename, greyscale=True):
    """ Transformation d'une image en tableau """
    if greyscale:
        pil_im = Image.open(filename).convert('L')  # converti en nuance de gris
    else:
        pil_im = Image.open(filename)
    return np.array(pil_im)


mat = np.random.randint(20, size=(20, 20))
l = mat.shape[0]  # nombre de ligne (coordonné y)
L = mat.shape[1]  # nombre de colonne (coordonné x)
M = l * L
d = sqrt(l ** 2 + L ** 2)


# le plan paramétré par la matrice est de la forme O --> y
#                                                  |                                                   |
#                                                  x

# donne le point d'intersection de deux droites données en polaire
def distance(point1, point2):
    return sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


# produit scalaire canonique
def ps(m1, m2):
    return float((sum([i * j for (i, j) in zip(m1, m2)])))


# coordonnées du pixel j
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


# équation polaire de la droite
def rayon(u, t, theta):
    return (u * sin(theta) + t * cos(theta), u * cos(theta) - t * sin(theta))


# est ce que la droite de paramètres (u,theta) passe par le pixel j

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


# définition des rayons de projections
def projections(nb_dir, nb_droite):
    dir = np.linspace(0, pi / 2, nb_dir)
    droite = np.linspace(0.1, l - 0.1, nb_droite)
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


# définition de la matrice R - version 2
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


# On peut alors tenter de résoudre le problème inverse : RF = P (par n itération)
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

    # définition des opérateurs de projections orthoognales
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


def columnIntoMatrix(C):
    A = np.zeros((l, L))
    for i in range(l):
        for j in range(L):
            A[i, j] = C[i * L + j, 0]
    return A


proj = projections(50, 20)
R = matR_v1(proj)
P = matP(R)
op = operateurs(R)
FC = reconstruction(10000000, op)

F = columnIntoMatrix(FC)
plt.imshow(F)
plt.show()
plt.imshow(mat)
plt.show()
error = 0
max = 0
for i in range(l - 3):
    for j in range(L - 3):
        error += abs(mat[i, j] - F[i, j])
        if abs(mat[i, j] - F[i, j]) >= max:
            max = abs(mat[i, j] - F[i, j])
print(error)
print(max)
saveAsPNG(F, 'matReconstruction10000')
saveAsPNG(mat, 'matOriginale10000')

# 3 : error : 679.4563049009196
#     max : 7.6729633974723335

# 4 : error : 626.4238403123719
#     max : 8.037156360731926

# 5 : error : 591.44307242218
#     max : 7.481272440724219

# 8 : error : 521.5618301563964
#     max : 7.116938276840337

# 15 : error : 464.70238900534827
#      max : 5.10045657475462

# 20 : error : 395.21433849560424
#     max : 5.106742306329753

# 25 : error : 299.5341543662293
#     max : 4.0766174363706345

# 50 : error : 246.31996624433555
#     max : 3.5438653295122675

# 100 : error : 191.0245806067057
#     max : 2.709169928559728

# 200 : error : 160.91769269954153
#     max : 3.0853778368814453

# 500 : error : 74.12011697547055
#     max : 1.5794650136150512

# 1000 : error : 28.243018034633636
#     max : 0.6177812243472234

# 2000 : error : 16.27973455562268
#     max : 0.3820856713033294

# 10000 : error : 0.2761238448059684
#     max : 0.006320163485931118

