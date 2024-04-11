from math import cos, sin, pi, sqrt 
# (x,y) = (u cos theta - v sin theta, u sin theta + v cos theta)  pour v dans R 


#(u cos theta - v sin theta, u sin theta + v cos theta) = (s cos x - v' sin x, s sin x + v' sin x )

# u cos theta - v sin theta = s cos x - v sin x 
# (u cos theta + s cos x)/(sin theta - sin x) = v 

# donne le point d'intersection de deux droites données en polaire
def distance(x1,y1, x2, y2): 
    return sqrt((x1 - x2)**2  + (y1 - y2)**2)

def intersection(u, s, alpha, theta):
    v = (s - u * cos(theta - alpha)) / sin(theta - alpha)
    return (u * cos(alpha) - v * sin(alpha), u * sin(alpha) + v * cos(alpha))


# renvoie les quatres droites qui délimite le pixel de coordonné (i,j)
def pixel(i, j):
    return [(j, 0), (j+1, 0), (i, pi/2), (i+1,pi)]


def interProjPixel(u, theta, i, j):
    res = []
    for k in range(0, 4):
        res.append(intersection(pixel(i, j)[k][0], u, pixel(i, j)[k][1], theta))
    return res

def isInPixel(i,j,a): 
    l = [False, False, False, False]
    if i - 0.2 < a[0][1] < i+1.2 : 
        l[0] = True 
    if i - 0.2 < a[1][1] < i+1.2: 
        l[1] = True
    if j - 0.2 < a[2][0] < j+1.2: 
        l[2] = True 
    if j - 0.2 < a[3][0] < j+1.2: 
        l[3] = True 
    return l 

# on peut alors définir la matrice R : 
def coefR(u,theta, i,j):
    a =interProjPixel(u,theta, i,j)
    l = isInPixel(i,j, a)
    l2 = []
    for k in range(0, len(l)):
        if l[k]: 
            l2.append(a[k][0]), l2.append(a[k][1])
    if len(l2) == 4: 
        return distance(l2[0], l2[1], l2[2], l2[3])
    else : 
        return 0




