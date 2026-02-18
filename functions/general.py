import os

import numpy as np
from scipy.optimize import least_squares


argLeastSquares = dict(
    ftol=1e-13,
    xtol=1e-13,
    gtol=1e-13,
    verbose=True,
    kwargs={},
)


def deriv_RK(fDeriv, x, t, deltaT, paramDeriv) -> np.ndarray:
    """Computes the derivative of x at t using the 4th order Runge-Kutta method."""
    k1 = fDeriv(x, t, paramDeriv)
    k2 = fDeriv(x + k1 * deltaT / 2, t + deltaT / 2, paramDeriv)
    k3 = fDeriv(x + k2 * deltaT / 2, t + deltaT / 2, paramDeriv)
    k4 = fDeriv(x + k3 * deltaT, t + deltaT, paramDeriv)
    return (k1 + 2 * k2 + 2 * k3 + k4) / 6


def Bolzano(f, xMin, xMax, *param, epsilon=1e-10, **kwargs) -> float:
    fa = f(xMin, *param, **kwargs)
    if fa * f(xMax, *param, **kwargs) > 0:
        sol = np.nan
    else:
        while xMax - xMin > epsilon:
            xMed = (xMax + xMin) / 2
            if f(xMed, *param, **kwargs) * fa > 0:
                xMin = xMed
            else:
                xMax = xMed
        sol = (xMax + xMin) / 2
    return sol


def guarda(directorioOut, nombrFich, nombres, valores) -> None:
    fichero = open(os.path.join(directorioOut, f'{nombrFich}.txt'), "w", encoding='utf-8')
    for nombr in nombres:
        fichero.write(nombr + '\t')
    fichero.write('\n')
    for linea in valores:
        for nombr in linea:
            fichero.write(f"{nombr:g}\t" if isinstance(nombr, (int, float)) else f"{nombr}\t")
        fichero.write('\n')
    fichero.close()


def derivada(x, y) -> np.ndarray:
    return (y[2:] - y[:-2]) / (x[2:] - x[:-2])


def ajusta(fResiduo, parVar, **argLS) -> dict:
    nombrParVar, parFijos = [argLS['kwargs'][nombr] for nombr in ['nombrParVar', 'parFijos']]

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
    resVariance = np.linalg.norm(ajuste['fun'])**2 / (m - n + 1)

    try:
        sd = np.sqrt(np.diag(np.linalg.inv(np.transpose(ajuste['jac']).dot(ajuste['jac']))*resVariance))
    except:
        sd = [np.inf] * len(nombrParVar)
    print(f'\n parAjustados: {parAjustados} \n sd= {sd} \n')
    
    """
    Cálculo del coeficiente de determinación R2:
    https: // www.mathworks.com / help / stats / coefficient - of - determination - r - squared_es.html
    """
    SSE = sum(ajuste['fun']**2)
    Y = argLS['kwargs']['Y']
    Y = Y - np.mean(Y)
    SST = sum(Y**2)
    R2 = 1 - SSE / SST

    sdPar = dict(zip([nombr + '_std' for nombr in nombrParVar], sd))
    return dict(parAjustados=parAjustados, sdPar=sdPar, R2=R2, detalles=ajuste)


def redefineALS(ALS, paramF, **dict) -> None:
    for nombr in paramF:
        ALS['kwargs'][nombr] = paramF[nombr]
    for nombr in dict:
        ALS[nombr] = dict[nombr]


def residualsLS(param, **kwargs) -> np.ndarray:
    nombrParVar, parFijos, f, fKwargs,Y = \
        [kwargs[nombr] for nombr in ['nombrParVar','parFijos','f','fKwargs','Y']]
    parametros = dict(zip(nombrParVar, param), **parFijos)
    sol = f(parametros, **fKwargs) - Y
    
    # Check for NaNs or Infs and handle them
    if not np.isfinite(sol).all():
        print(f"Warning: Residuals contain non-finite values. Replacing with large penalty.")
        sol = np.nan_to_num(sol, nan=1e10, posinf=1e10, neginf=-1e10)
        
    print(f'\t||residuals|| = {np.linalg.norm(sol)}')
    return sol


def procesa(**dictIn) -> dict:
    argLeastSquares,nombrParVar, dictParEstim, f, fKwargs, Y= [dictIn[nombr] for nombr in ['argLeastSquares','nombrParVar','dictParEstim','f','fKwargs','Y']]
    if 'bounds' in dictIn:
        bounds = dictIn['bounds']
    else:
        bounds = ([0 for nombr in nombrParVar], [np.inf for nombr in nombrParVar])

    nombrParFijos = [nombr for nombr in dictParEstim if nombr not in nombrParVar]
    valoresParFijos = [dictParEstim[nombr] for nombr in nombrParFijos]
    parFijos = dict(zip(nombrParFijos, valoresParFijos))
    dictResiduo = dict(parFijos=parFijos,
                       nombrParVar=nombrParVar,
                       f=f,
                       fKwargs=fKwargs,
                       Y=Y)
    redefineALS(argLeastSquares, paramF=dictResiduo, bounds=bounds)

    estim = [dictParEstim[nombr] for nombr in nombrParVar]
    sol = ajusta(residualsLS, estim, **argLeastSquares)
    
    # ======== Convert sol information in dataframe =========

    parAjustados = sol['parAjustados']
    sdPar = sol['sdPar']

    df = pd.DataFrame({
        "Constant": nombrParVar,
        "Value": [parAjustados[n] for n in nombrParVar],
        "SD": [sdPar[n + "_std"] for n in nombrParVar]
    })

    # info extra útil
    df.attrs["R2"] = sol["R2"]

    return df

