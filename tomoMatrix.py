import numpy
from aotools.image import interp
from aotools import circle
from matplotlib import pyplot

from scipy.optimize import minimize, root

import numba
import time


def makeFourierCovMap( N, D, r0, subapDiam, L0=None, l0=None):
    """
    Makes the covariance map of Kolmogorov or Von Karman turbulence in Fourier
    space

    Uses either the Kolmogorov, Von Karman or modifies Von Karman spatial turbulence power spectrum to calculate the covariance of between points with different separation, represented in Fourier space. Adapted from a derivation given in Vidal 2010.

    Parameters:
        N (int): size of array which contains data.
        D (float): Maximum separation to calculate.
        r0 (float): Fried parameter of turbulent in metres
        subapDiam (float): Diameter of sup-aperture pupil
        L0 (float): Outer scale of turbulence in metres
        l0 (float): Inner scale of turbulence in metres

    Returns:
        ndarray: The covariance map in Fourier space
    """

    N_ap = (float(N)/(2*D)) * subapDiam
    delta = subapDiam/N_ap
    aperture = numpy.zeros((N, N))
    aperture[:N_ap, :N_ap] = 1
    fftAperture = abs(ft2(aperture, delta))

    covMaps = numpy.zeros((3, N, N))

    coords = numpy.linspace(-1./D, 1./D, N)
    kx, ky = numpy.meshgrid(coords, coords)

    k = numpy.sqrt(kx**2 + ky**2)

    if L0 == None and l0 == None:
        covMaps[0] = kx**2 * k**(-11./3) * r0**(-5./3)
        covMaps[1] = ky**2 * k**(-11./3) * r0**(-5./3)
        covMaps[2] = kx*ky * k**(-11./3) * r0**(-5./3)

    elif l0 == None and L0 != None:
        covMaps[0] = kx**2 * (k**2 + 2*numpy.pi * L0**(-2))**(-11./6) * r0**(-5/3)
        covMaps[1] = ky**2 * (k**2 + 2*numpy.pi * L0**(-2))**(-11./6) * r0**(-5/3)
        covMaps[2] = kx*ky * (k**2 + 2*numpy.pi * L0**(-2))**(-11./6) * r0**(-5/3)

    elif l0!=None and L0!=None:
        covMaps[0] = kx**2 * (k**2 + 2*numpy.pi * L0**(-2))**(-11./6) * r0**(-5/3) * numpy.exp(-((k**2)*(l0**2)/5.92))
        covMaps[1] = ky**2 * (k**2 + 2*numpy.pi * L0**(-2))**(-11./6) * r0**(-5/3) * numpy.exp(-((k**2)*(l0**2)/5.92))
        covMaps[2] = kx*ky * (k**2 + 2*numpy.pi * L0**(-2))**(-11./6) * r0**(-5/3) * numpy.exp(-((k**2)*(l0**2)/5.92))


    else:
        raise Exception("Sorry - I don't have a power spectrum for that!")

    covMaps[:, N/2, N/2] = 0

    covMaps *= fftAperture**2

    return covMaps


def makeCovMap(N, D, subapDiam, r0, wvl=500e-9, L0=None, l0=None):

    fCovMaps = makeFourierCovMap(N, D, r0, subapDiam,  L0, l0)
    delta = (float(N)/(2*D))
    # sampling = 3*N
    delta_f = 1. / (N*delta)
    covMaps = abs(ift2(fCovMaps, delta_f))
    # covMaps = abs(numpy.fft.fftshift(numpy.fft.fft2(fCovMaps, axes=(-1,-2), s=(sampling, sampling)), axes=(-1,-2) ))

    # covMaps = interp.binImgs(covMaps, 3)

    # Calculate theoretical max variance from Saint Jacques, 1998, Append A
    theoMaxVar = 0.179 * wvl**2 * r0**(-5./3) * subapDiam**(-1./3)
    covMaps /= covMaps.max()
    covMaps *= theoMaxVar
    
    return covMaps


def makeNgsCovMat(
        nxSubaps, subapDiam, pupilMask, r0, sep, wvl=500e-9, L0=None, l0=None,
        map_size=501):
    """
    Calculates the theoretical covariance matrix of SH WFSs through atmospheric turbulence with a given separation

    Parameters:
        nxSubaps (int): Number of x sub-apertures in WFS
        subapDiam (float): Size of each sub-aperture in metres
        pupilMask (ndarray): Array of 1 and 0 specifying pupil function
        r0 (float): Fried parameter in metres
        sep (ndarray): Separation between WFS meta-pupils at turbulence layer (m)
        L0 (float, optional): Outer scale in metres
        l0 (float, optional): Inner scale in metres

    Returns:
        ndarray: computed covariance matrix

    """
    t = time.time()
    
    sep = numpy.array(sep)
    
    max_sep = (1+2*nxSubaps)*subapDiam + max(abs(sep))
    
    print("map_size: {}".format(map_size))
    print("max_sep: {}".format(max_sep))
    print("r0: {}".format(r0))
    print("L0: {}".format(L0))

    covMaps = makeCovMap(
            map_size, max_sep, subapDiam, r0, wvl, L0, l0)
            
    #Total sub-apertures from mask
    totalSubaps = int(pupilMask.sum())
    
    #Loop over all sub-apertures and find seperation between them
    covMat = numpy.zeros((2*totalSubaps, 2*totalSubaps))

    print(time.time()-t)
    populateNgsCovMat(
            nxSubaps, sep, subapDiam, map_size, max_sep, pupilMask, totalSubaps, covMaps, covMat)
            
    print(time.time()-t)
    return covMat
                          
@numba.jit(nopython=True)         
def populateNgsCovMat(nxSubaps, sep, subapDiam, map_size, max_sep, pupilMask, totalSubaps, covMaps, covMat):
                         
    #Count valid sub-apertures
    i = 0
    #WFS #1
    for x1 in xrange(nxSubaps):
        for y1 in xrange(nxSubaps):
            
            #Only continue if a real sub-ap of WFS1
            if pupilMask[x1,y1]:
                #WFS #2
                j = 0
                for x2 in xrange(nxSubaps):
                    for y2 in xrange(nxSubaps):
                        
                        #Only continue if real sub-ap of WFS2
                        if pupilMask[x2,y2]:
                            
                            #Find the seperation between the 2 sub-aps
                            dx = numba.float64((x2-x1)*subapDiam + sep[0])
                            dy = numba.float64((y2-y1)*subapDiam + sep[1])
            
                            #Find the index in the covMap for seperations
                            ix = numba.int64(round(dx*(map_size/2.)/max_sep +map_size/2.))
                            iy = numba.int64(round(dy*(map_size/2.)/max_sep +map_size/2.))
            
                            #Get the index to put into the covMat
                            #i = x1*nxSubaps+y1
                            #j = x2*nxSubaps+y2

                            
                            # print("i: {}, j: {}".format(i,j))
#                             print("x1: {}, x2: {}, dx: {}, ix: {}".format(x1,x2,dx,ix))
#print("y1: {}, y2: {}, dy: {}, iy: {}".format(y1,y2,dy,iy))
                            
                            #Get x-x value
                            covMat[i+totalSubaps, j+totalSubaps] = covMaps[0, ix, iy]
                            
                            #y-y value
                            covMat[i,j] = covMaps[1, ix, iy]
                            
                            #x-y value
                            covMat[i, j+totalSubaps] = covMaps[2, ix, iy]
                            
                            #y-x value
                            covMat[i+totalSubaps, j] = covMaps[2, ix, iy]
                            
                            j+=1
                i+=1
    
    return covMat

def optNgsCovMat(params, rawCovMat, nxSubaps, subapDiam, pupilMask, ax=None):
    
#(nxSubaps, subapDiam, pupilMask, r0, sep, wvl=500e-9, L0=None, l0=None, map_size=80):

    print(params)
    theoCovMat = makeNgsCovMat(
            nxSubaps, subapDiam, pupilMask, params[2], numpy.array([params[0], params[1]]))

    rmsError = numpy.sqrt( ((rawCovMat - theoCovMat)**2).mean())

    if ax!=None:
        ax.imshow(theoCovMat, origin="lower")
        pyplot.pause(0.5)
        print("PLOT!")

    print("RMS Error:{}\n\n".format(rmsError))
    return rmsError

def optNgsCovMat2(params, rawCovMat, nxSubaps, subapDiam, pupilMask, ax=None):
    
#(nxSubaps, subapDiam, pupilMask, r0, sep, wvl=500e-9, L0=None, l0=None, map_size=80):

    print(params)
    theoCovMat = makeNgsCovMat(
            nxSubaps, subapDiam, pupilMask, params[2], numpy.array([params[0], params[1]]))

    rmsError = numpy.sqrt( ((rawCovMat - theoCovMat)**2).mean())

    print("ax:{}", ax)
    if ax!=None:
        ax.get_figure().clear()
        ax.imshow(theoCovMat, origin="lower")
        #img.colorbar()
        pyplot.pause(0.1)
        print("PLOT!")

    print("RMS Error: {}".format(rmsError))
    return ((rawCovMat - theoCovMat)**2).flatten()


def fitNgsCovMat_root(
        rawCovMat, nxSubaps, subapDiam, pupilMask, r01, sep1, L01, plot=False):
    
    pyplot.ion()
    # fig = pyplot.figure()
    # ax = fig.add_subplot(111)
    # img = ax.imshow(numpy.zeros(rawCovMat.shape))

    guess = numpy.array([sep1[0], sep1[1], r01])
    # opt_result = minimize(
    #         optNgsCovMat, guess, 
    #         args=(rawCovMat, nxSubaps, subapDiam, pupilMask),
    #         method="SLSQP",
    #         options={   "disp":True,
    #                     "eps":0.2},
    #         bounds=[(None,None), (None,None), (0,None)],
    #         tol=1e-30,
    #         )

    if plot:
        pyplot.ion()
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        opt_result = root(
                optNgsCovMat2, guess, 
                args=(rawCovMat, nxSubaps, subapDiam, pupilMask, ax),
                method="lm",
                # options={   "disp":True,
                #             "eps":0.2},
                # bounds=[(None,None), (None,None), (0,None)],
                tol=1e-30,
                )
    else:
        opt_result = root(
                optNgsCovMat2, guess, 
                args=(rawCovMat, nxSubaps, subapDiam, pupilMask),
                method="lm",
                # options={   "disp":True,
                #             "eps":0.2},
                # bounds=[(None,None), (None,None), (0,None)],
                tol=1e-30,
                )
            
    print(opt_result)
    
    fittedCovMat = makeNgsCovMat(
            nxSubaps, subapDiam, pupilMask, opt_result.x[2],
            numpy.array([opt_result.x[0], opt_result.x[1]])
            )
            
    return fittedCovMat

def fitNgsCovMat_minimise(
        rawCovMat, nxSubaps, subapDiam, pupilMask, r01, sep1, L01, plot=False):
    
    pyplot.ion()
    # fig = pyplot.figure()
    # ax = fig.add_subplot(111)
    # img = ax.imshow(numpy.zeros(rawCovMat.shape))

    guess = numpy.array([sep1[0], sep1[1], r01])
    # opt_result = minimize(
    #         optNgsCovMat, guess, 
    #         args=(rawCovMat, nxSubaps, subapDiam, pupilMask),
    #         method="SLSQP",
    #         options={   "disp":True,
    #                     "eps":0.2},
    #         bounds=[(None,None), (None,None), (0,None)],
    #         tol=1e-30,
    #         )

    if plot:
        pyplot.ion()
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        opt_result = minimize(
                optNgsCovMat, guess, 
                args=(rawCovMat, nxSubaps, subapDiam, pupilMask, ax),
                method="Nelder-Mead",
                # options={   "disp":True,
                #             "eps":0.2},
                # bounds=[(None,None), (None,None), (0,None)],
                tol=1e-30,
                )
    else:
        opt_result = root(
                optNgsCovMat, guess, 
                args=(rawCovMat, nxSubaps, subapDiam, pupilMask),
                method="Nelder-Mead",
                # options={   "disp":True,
                #             "eps":0.2},
                # bounds=[(None,None), (None,None), (0,None)],
                tol=1e-30,
                )
            
    print(opt_result)
    
    fittedCovMat = makeNgsCovMat(
            nxSubaps, subapDiam, pupilMask, opt_result.x[2],
            numpy.array([opt_result.x[0], opt_result.x[1]])
            )
            
    return fittedCovMat

#######################
#Old functions
#apr-3/5/15
######################################################################
def makeCovMat_NGS(
        subaps, samp, subapDiam, r0, mask, sep=(0,0), L0=None, l0=None):

    covMaps = makeCovMap(numpy.ceil(2*subaps+1 + max(sep))*samp, subapDiam*subaps, r0, subapDiam, L0, l0)
    print(covMaps.shape)
    covMaps = interp.binImgs(covMaps, samp)

    covMaps /= covMaps[0].max()

    
    totalSubaps = int(mask.sum())
    covMat = numpy.zeros( (2*totalSubaps, 2*totalSubaps))

    subapPos = []
    for x in range(subaps):
        for y in range(subaps):
            if mask[x,y]==1:
                subapPos.append([x,y])
     
    subapPos = numpy.array(subapPos)
    for i in range(totalSubaps):
        for j in range(totalSubaps):

            ix,iy = subapPos[i]
            jx,jy = subapPos[j] 
            dx = jx - ix + subaps + sep[1]
            dy = jy - iy + subaps + sep[0]

            #x-x terms
            covMat[i, j] = covMaps[0, dy, dx]
            #y-y terms
            covMat[i+totalSubaps, j+totalSubaps] = covMaps[1, dy,dx]
            #x-y terms
            covMat[i, j+totalSubaps] = covMaps[2, dy, dx]
            covMat[i+totalSubaps, j] = covMaps[2, dx, dy]

    #return covMaps
    return covMat

def makeCovMat_LGS(
        subaps, d, r0, h, H, mask, 
        samp=1000, pos1=(0,0), pos2=(0,0), L0=None, l0=None):
    """
    Parameters:
        subaps (int): No. of subaps in 1d on WFS.
        d (float): diameter of each subap.
        r0 (float): Fried param of layer.
        h (float): height of layer.
        H (float): height of Laser.
        mask (ndarray): 2d array of subaps, 1 if active, 0 if not.
        samp (int, opt): size of oversampled covariance map
        pos1 (tuple, opt): Angular Pos of LGS 1, arcsecs.
        pos2 (tuple, opt): Angular Pos of LGS 2, arcsecs.
        L0 (float, opt): Outer scale.
        l0 (float, opt): innterscale.

    Returns:
        ndarray: The covariance Matrix

    """
    #Size of the telescope
    D = d*subaps
    h = float(h)
    H = float(H)

    pos1 = numpy.array(pos1, dtype="float32") * (1./3600) * (numpy.pi/180.)
    pos2 = numpy.array(pos2, dtype="float32") * (1./3600) * (numpy.pi/180.)

    #Find the maximum seperation in metres
    maxSep = (pos2 - pos1).max() * h
    maxSep += d*(subaps+1)

    #Make the massively oversampled covaranace maps - make it an odd number
    #if samp%2==0:
     #   samp+=1
    covMaps = makeCovMap(samp, maxSep, r0, subapDiam, L0, l0)
    covMaps /= covMaps[0].max()

    #The subap size at this height
    dSize = d*(1.-(h/H))
    
    #And set to same scale as covMap array
    dSize *= (covMaps.shape[1]-1)/(2*maxSep)
    dSize = int(round(dSize))
    print(dSize)

    #Find total number of subaps, and their positions in subap space. 
    #Make a placeholder of the delicious covariance matrix
    totalSubaps = int(mask.sum())
    covMat = numpy.zeros( (2*totalSubaps, 2*totalSubaps))

    subapPos = []
    for x in range(subaps):
        for y in range(subaps):
            if mask[x,y]==1:
                subapPos.append([x,y])
     
    subapPos = numpy.array(subapPos)
    for i in range(totalSubaps):
        for j in range(totalSubaps):

            #This is the position in metres from the centre of the WFS field, i
            #to the start of the subap
            ix, iy = pos1*h + ((subapPos[i] * d) - D/2.)*(1.-(h/H))
            jx, jy = pos2*h + ((subapPos[j] * d) - D/2.)*(1.-(h/H))
           
            #The seperation between the two subap start positions in covMap pxls
            dx = (jx - ix) * (covMaps.shape[1]-1) / (2*maxSep)
            dy = (jy - iy) * (covMaps.shape[1]-1) / (2*maxSep)
            #print("ix:{}, jx:{}, dx:{}, iy:{}, jy:{},  dy:{}".format(ix, jx, dx, iy, jy, dy))

            #Adjust this for the array coords
            dx = int(round(dx + covMaps.shape[1]/2. - dSize/2.))
            dy = int(round(dy + covMaps.shape[1]/2. - dSize/2.))

            #print("i:{}, j:{}, ix:{}, jx:{}, dx:{}, iy:{}, jy:{},  dy:{}\n".format(i,j,ix, jx, dx, iy, jy, dy))
            
            #Get the sum of the covarance for the seperation of 
            #each point in the sub-apertures
            #x-x terms
            covMat[i, j] = covMaps[0, dy:dy+dSize, dx:dx+dSize].sum()
            #y-y terms
            covMat[i+totalSubaps, j+totalSubaps] = covMaps[1, dy:dy+dSize, dx:dx+dSize].sum()
            #x-y terms
            covMat[i, j+totalSubaps] = covMaps[2, dy:dy+dSize, dx:dx+dSize].sum()
            covMat[i+totalSubaps, j] = covMaps[2, dx:dx+dSize, dy:dy+dSize].sum()

           
    #return covMaps
    return covMat

 
def makeCovMat_LGSUplink(
        subaps, d, r0, h, H,  mask, 
        samp=1000, pos1=(0,0), pos2=(0,0), L0=None, l0=None):
    """
    Parameters:
        subaps (int): No. of subaps in 1d on WFS.
        d (float): diameter of each subap.
        r0 (float): Fried param of layer.
        h (float): height of layer.
        H (float): height of Laser.
        mask (ndarray): 2d array of subaps, 1 if active, 0 if not.
        samp (int, opt): size of oversampled covariance map
        pos1 (tuple, opt): Angular Pos of LGS 1, arcsecs.
        pos2 (tuple, opt): Angular Pos of LGS 2, arcsecs.
        L0 (float, opt): Outer scale.
        l0 (float, opt): innterscale.

    Returns:
        ndarray: The covariance Matrix

    """
    #Size of the telescope
    D = d*subaps
    h = float(h)
    H = float(H)

    pos1 = numpy.array(pos1, dtype="float32") * (1./3600) * (numpy.pi/180.)
    pos2 = numpy.array(pos2, dtype="float32") * (1./3600) * (numpy.pi/180.)

    #Find the maximum seperation in metres
    maxSep = (pos2 - pos1).max() * h
    maxSep += d*(subaps+1)

    #Make the massively oversampled covaranace maps - make it an odd number
    #if samp%2==0:
     #   samp+=1
    covMaps = makeCovMap(samp, maxSep, r0, subapDiam, L0, l0)
    covMaps /= covMaps[0].max()

    #The subap size at this height
    dSize = d*(1.-(h/H))
    
    #And set to same scale as covMap array
    dSize *= (covMaps.shape[1]-1)/(2*maxSep)
    dSize = int(round(dSize))
    print(dSize)

    #Find total number of subaps, and their positions in subap space. 
    #Make a placeholder of the delicious covariance matrix
    totalSubaps = int(mask.sum())
    covMat = numpy.zeros( (2*totalSubaps, 2*totalSubaps))
    
    #This is the start of the sub-ap seen by the LGS on the uplink, in metres
    lgs2Pos = (h * pos2)-d/2. #Would need to be changed for launch 

    #Make an LGS mask to determine the overlap with subaps, each element is a cm
    lgsScale = 100
    lgsMask = circle.circle(lgsScale*d/2., lgsScale*d*subaps, centre_offset=lgsScale*lgs2Pos)

    lgsSubapPos = []
    subapPos = []
    for x in range(subaps):
        for y in range(subaps):
            if mask[x,y]==1:
                subapPos.append([x,y])
                 
    subapPos = numpy.array(subapPos)
    lgsSubapPos = numpy.array(lgsSubapPos)

    for i in range(totalSubaps):
        for j in range(totalSubaps):
            #This is the position in metres from the centre of the WFS field, i
            #to the start of the subap
            maskVal = lgsMask[subapPos[j,0]*100:(subapPos[j,0]+1)*100, subapPos[j,1]*100:(subapPos[j,1]+1)*100].mean()
            if maskVal>0:

                ix, iy = pos1*h + ((subapPos[i] * d) - D/2.)*(1.-(h/H))
                jx, jy = pos2*h + ((subapPos[j] * d) - D/2.)*(1.-(h/H))

                #The seperation between the two subap start positions in covMap pxls
                dx = (jx - ix) * (covMaps.shape[1]-1) / (2*maxSep)
                dy = (jy - iy) * (covMaps.shape[1]-1) / (2*maxSep)
                #print("ix:{}, jx:{}, dx:{}, iy:{}, jy:{},  dy:{}".format(ix, jx, dx, iy, jy, dy))

                #Adjust this for the array coords
                dx = int(round(dx + covMaps.shape[1]/2. - dSize/2.))
                dy = int(round(dy + covMaps.shape[1]/2. - dSize/2.))

                #print("i:{}, j:{}, ix:{}, jx:{}, dx:{}, iy:{}, jy:{},  dy:{}\n".format(i,j,ix, jx, dx, iy, jy, dy))
                
                #Get the sum of the covarance for the seperation of 
                #each point in the sub-apertures
                #x-x terms
                covMat[i, :totalSubaps] += covMaps[0, dy:dy+dSize, dx:dx+dSize].sum()*maskVal
                #y-y terms
                covMat[i+totalSubaps, totalSubaps:] += covMaps[1, dy:dy+dSize, dx:dx+dSize].sum()*maskVal
                #x-y terms
                covMat[i, :totalSubaps] += covMaps[2, dy:dy+dSize, dx:dx+dSize].sum()*maskVal
                covMat[i+totalSubaps, totalSubaps:] += covMaps[2, dx:dx+dSize, dy:dy+dSize].sum()*maskVal

           
    #return covMaps
    return covMat



def ft2(g, delta, padFactor=1):
    padFactor = int(padFactor)
    G = numpy.fft.fftshift(
            numpy.fft.fft2(
                numpy.fft.fftshift(g, axes=(-1,-2)), axes=(-1,-2), s=(padFactor*g.shape[-2],padFactor*g.shape[-1])), axes = (-1,-2)) * (delta**2)
    return G

def ift2(G, delta_f, padFactor=1):
    N = G.shape[0]
    g = numpy.fft.ifftshift(
            numpy.fft.ifft2(
                    numpy.fft.ifftshift(G, axes=(-1,-2)),
                    s=( padFactor*G.shape[-2], padFactor*G.shape[-1])
                        ), axes=(-1,-2)
                    ) * ((N * delta_f)**2)

    return g