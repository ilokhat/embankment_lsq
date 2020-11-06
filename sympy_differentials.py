from sympy import *

# xi, yi => x, y, | xi-1, yi-1 => xpp, ypp | xi+1, yi+1 => xss, yss
x, y, xpp, ypp, xss, yss = symbols('x y xpp ypp xss yss')
#jacobienne pour le produit vectoriel des vecteur (normés) qui se suivent dans un polyligne
res = Matrix([ ((x-xpp)*(yss-y) - (y-ypp)*(xss-x))/(((x-xpp)**2 + (y-ypp)**2) * ((xss-x)**2 + (yss-y)**2))**0.5  ]).jacobian([xpp, ypp, x, y, xss, yss ])

for i, var in enumerate(['xi-1', 'yi-1', 'xi', 'yi', 'xi+1', 'yi+1']):
    print(f'# df/d{var}')
    print(f'm[2*i + {i-2}] = {simplify(res[i])}')