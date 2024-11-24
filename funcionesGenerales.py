# deriv_RK implementa el método de Runge-Kutta.
# La derivada debe admitir los parámetros x, t y, opcionalmente, parámetros constantes
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


argLeastSquares = dict(ftol=1e-13, xtol=1e-13, gtol=1e-13,
                       verbose=True, kwargs={})

# Algoritmo Runge-Kutta 4º orden
def deriv_RK (fDeriv,x,t,deltaT,**paramDeriv):

    k1 = fDeriv ( x, t, **paramDeriv)
    k2 = fDeriv ( x+k1*deltaT/2, t+ deltaT/2, **paramDeriv)
    k3 = fDeriv ( x+k2*deltaT/2, t+ deltaT/2, **paramDeriv)
    k4 = fDeriv ( x+k3*deltaT, t+ deltaT, **paramDeriv)

    return (k1+2*k2+2*k3+k4)/6

def Bolzano (f, xMin, xMax, *param, epsilon = 1e-10 , **kwargs):
    fa = f(xMin, *param, **kwargs)
    if fa * f(xMax, *param, **kwargs) > 0:
        sol = np.nan
    else:
        while xMax - xMin > epsilon:
            xMed = (xMax + xMin)/2
            if f(xMed, *param, **kwargs) * fa > 0:
                xMin = xMed
            else:
                xMax = xMed
        sol = (xMax + xMin)/2
    return sol

def guarda(directorioOut, nombrFich, nombres, valores):
    fichero = open(directorioOut+'\\'+nombrFich+'.txt',"w",encoding='utf-8')
    for nombr in nombres:
        fichero.write(nombr+'\t')
    fichero.write('\n')
    for linea in valores:
        for nombr in linea:
            if isinstance (nombr, str):
                fichero.write('%s\t'%nombr)
            else:
                fichero.write('%g\t'%nombr)
        fichero.write('\n')
    fichero.close()

def derivada(x,y):
    return (y[2:]-y[:-2])/(x[2:]-x[:-2])

def ajusta (fResiduo, parVar, **argLS):
    nombrParVar, parFijos = [argLS['kwargs'][nombr] for nombr in ['nombrParVar','parFijos']]

    ajuste = least_squares(fun=fResiduo, x0=parVar, **argLS)

    parAjustados = dict(zip(nombrParVar, ajuste['x']), **parFijos)

    """
    Estimamos el error en los parámetros, siguiendo a 
    Gavin, H. P. (2019). The Levenberg-Marquardt algorithm for nonlinear least squares curve-fitting problems. 
    Department of civil and environmental engineering, Duke University, 19.
    ecuaciones 22 y 25
    """
    m = len(ajuste['fun'])
    n = len(ajuste['x'])
    #resVariance = np.sum(fResiduo(ajuste['x'], **argClave)**2)/(m-n+1)
    resVariance = np.linalg.norm(ajuste['fun'])**2 /(m-n+1)
    #print('ajuste: ', ajuste['fun'], np.linalg.norm(ajuste['fun']), m, n, resVariance)

    try:
        sd = np.sqrt(np.diag(np.linalg.inv(np.transpose(ajuste['jac']).dot(ajuste['jac'])) * resVariance))
    except:
        sd = [np.inf]*len(nombrParVar)
    print ('\n parAjustados:', parAjustados,'\n sd=', sd,'\n')
    """
    Cálculo del coeficiente de determinación R2:
    https: // www.mathworks.com / help / stats / coefficient - of - determination - r - squared_es.html
    """
    SSE = sum(ajuste['fun']**2)
    Y = argLS['kwargs']['Y']
    Y = Y - np.mean(Y)
    SST = sum(Y**2)
    R2 = 1 - SSE/SST

    sdPar = dict ( zip ([nombr+'_std' for nombr in nombrParVar],sd))
    return dict(parAjustados=parAjustados, sdPar = sdPar, R2=R2,detalles=ajuste)

def redefineALS (ALS, paramF,**dict):
    for nombr in paramF:
        ALS['kwargs'][nombr] = paramF[nombr]
    for nombr in dict:
        ALS[nombr] = dict[nombr]

def residualsLS (param, **kwargs):
    nombrParVar, parFijos, f, fKwargs,Y = \
        [kwargs[nombr] for nombr in ['nombrParVar','parFijos','f','fKwargs','Y']]
    parametros = dict ( zip(nombrParVar, param), **parFijos)
    sol = f (parametros, **fKwargs) - Y
    print('\t||residuals|| = ' + str(np.linalg.norm(sol)))
    #print(paramModelo)
    return sol

def procesa (**dictIn):
    argLeastSquares,nombrParVar, dictParEstim, f, fKwargs, Y= \
        [dictIn[nombr] for nombr in ['argLeastSquares','nombrParVar','dictParEstim','f','fKwargs','Y']]
    if 'bounds' in dictIn:
        bounds = dictIn['bounds']
    else:
        bounds = ([0 for nombr in nombrParVar], [np.inf for nombr in nombrParVar])

    nombrParFijos = [nombr for nombr in dictParEstim if nombr not in nombrParVar]
    valoresParFijos = [dictParEstim[nombr] for nombr in nombrParFijos]
    parFijos = dict(zip(nombrParFijos, valoresParFijos))
    dictResiduo =dict ( parFijos = parFijos,
                        nombrParVar= nombrParVar,
                        f = f,
                        fKwargs = fKwargs,
                        Y = Y)
    redefineALS(    argLeastSquares,
                    paramF=dictResiduo,
                    bounds=bounds
                )

    estim = [dictParEstim[nombr] for nombr in nombrParVar]
    sol = ajusta(residualsLS, estim, **argLeastSquares)
    return sol

