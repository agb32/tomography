# -*- coding: utf-8 -*-

import numpy

def matcov(
        nWfs, nSubaps, nxSubaps, subapDiam, subapPos, gsAlt, gsPos, nLayers,
        layerHeights, cn2, L0, covMatPart, data, pupilOffset=None, 
        gsMag=None, wfsRot=None):


    u, v = subap_position(
            nWfs, nSubaps, nxSubaps, gsAlt, gsPos, subapPos, nLayers, 
            layerHeights, pupilOffset, gsMag,
            wfsRot
            )

    # Compute u and v

    # Rescale the projected suze of all subapertures at the different altitudes
    subapSizes = numpy.zeros((nWfs, nLayers))
    for n in range(nWfs):
        for l in range(nLayers):
            subapSizes[n, l] = subapDiam[n] * (1. - gsAlt*layerHeights[l])

    # Computation of the covariance matrix
    #####################################

    # lambda2 = pow(206265.*0.5e-6/2./3.1415926535,2);
    lambda2 = 0.00026942094446267851 

    # Truth Sensor no
    ts = nWfs - 1

    ioff = joff = 0
    units = numpy.zeros(nLayers)

    # Find the total number of slopes
    totSlopes = 0
    for i in range(nWfs):
        totSlopes += nSubaps[i]

    if covMatPart==0:
        m0 = 0
        mf = nWfs
        n0 = 0

        NL = totSlopes

    elif covMatPart==1:
        m0 = 0
        mf = nWfs-1
        n0 = 0

        totSlopes *= 2
        NL = totSlopes - 2*nSubaps[ts]

    elif covMatPart==3:
        m0 = 0
        mf = nWfs - 1
        n0 = nWfs - 1
        nf = ts
        NL = 2*nSubaps[nWfs-1]

    # Loop over WFS 1
    for m in range(m0, mf):
        Ni = nSubaps[m] + ioff

        # Loop over WFS 2
        for n in range(n0, nf+1):

            off_XY = nSubaps[n]
            off_YX = nSubaps[m] * nLayers
            off_YY = off_XY + off_YX

            Nj = nSubaps[n] + joff

            kk = 1./( subapDiam[m] * subapDiam[n])


            for l in range(0, nLayers):
                units[l] = kk * lambda2 * cn2[l]

            # Loop through subap i on WFS 1   
            for i in range(ioff, Ni):
                # Loop through subap j on WFS 2
                for j in range(joff, Nj):

                    caa_xx = 0
                    caa_yy = 0
                    caa_xy = 0

                    # Loop through altitude layers
                    for l in range(0, nLayers):
                        # Check layer is not above LGS
                        if subapSizes[m, l]>0 and subapSizes[n,l]>0:
                            # Distances in x and y between the subapertures i and j
                            du = u[m, i-ioff, l] - u[n, j-joff, l];        
                            dv = v[m, i-ioff, l] - v[n, j-joff, l];

                            s1 = subapSizes[m, l] * 0.5;
                            s2 = subapSizes[n, l] * 0.5;

                            ac = s1 - s2;
                            ad = s1 + s2;
                            bc = -ad;
                            bd = -ac;

                            cov = compute_cov(
                                    du, dv, ac, ad, bc, bd, s1, s2, L0[l],
                                    units[l]);

                            caa_xx += cov[0]
                            caa_yy += cov[1]
                            caa_xy += cov[2]

                    i0 = i*NL + j
                    data[i0] = caa_xx
                    data[i0 + off_XY] = caa_xy
                    data[i0 + off_YX] = caa_xy
                    data[i0 + off_YY] = caa_yy
            
            joff = joff + 2*nSubaps[n]
        ioff = ioff + 2*nSubaps[m]
        joff = 0

    if covMatPart==0 or covMatPart==1:
        size = NL - 1
        matL = data[1:]
        matU = data[NL:]

    for j in range(size):
        matL += NL + 1
        matU += NL + 1
        size -= 1

        if size>0:
            break

    return data


def subap_position(
        nWfs, nSubaps, nxSubaps, gsAlt, gsPos, subapPos, 
        nLayers, layerHeights, pupilOffset=None, gsMag=1, wfsRot=0):
    
    rad = numpy.pi / 180.


    # u, v arrays, contain subap coordinates of all WFSs
    u = numpy.zeros((nWfs, nSubaps, nLayers))
    v = numpy.zeros((nWfs, nSubaps, nLayers))

    for l in range(0, nLayers):
        ioff = 0

        for n in range(0, nWfs):

            dX = gsPos[0,n] * layerHeights[l]
            dY = gsPos[1,n] * layerHeights[l]

            rr = 1. - layerHeights[l] * gsAlt[n]

            # Magnification 
            G = float(gsMag) / nxSubaps[n]

            # Rotation angle 
            th = wfsRot * rad

            for i in range(nSubaps[n]):
                xtp = subapPos[0][ioff + i] * G
                ytp = subapPos[1][ioff + i] * G

                uu = xtp * numpy.cos(th) - ytp * numpy.sin(th)
                vv = xtp * numpy.sin(th) + ytp * numpy.cos(th)

                if numpy.any(pupilOffset):
                    uu += pupilOffset[0, n]
                    vv += pupilOffset[1, n]

                u[n, i, l] = uu*rr + dX
                v[n, i, l] = vv*rr + dY
            ioff += nSubaps[n]
    return u, v


def compute_cov(du, dv, ac, ad, bc, bd, s1, s2, L0, units):
    """
    <du> & <dv>: X and Y coordinates of the distance between the two considered subapertures.
    <ac> & <ad> & <bc> & <bd>  : precomputed values
    <s1> & <s2>                : half size of each subapertures
    <L0>                  : 
    <units>                    :
    Computes the XX, XY and YY covariance values for two subapertures.
    """

    cov_xx = cov_yy = cov_xy = 0

    cov_xx = cov_XX(du, dv, ac, ad, bc, bd, L0)
    cov_xx *= 0.5

    cov_yy = cov_YY(du, dv, ac, ad, bc, bd, L0)
    cov_yy *= 0.5

    s0 = numpy.sqrt(s1**2 + s2**2)

    cov_xy = cov_XY(du, dv, s0, L0)
    cov_xy *= 0.25

    if s1>s2:
        cc = 1 - (float(s2)/s1)
    else:
        cc = 1 - (float(s1)/s2)

    cov_xy *= (1. - cc**2)

    cov_xx *= units
    cov_yy *= units
    cov_xy *= units

    cov = numpy.zeros(3, dtype="float64")

    cov[0] = cov_xx
    cov[1] = cov_yy
    cov[3] = cov_xy

    return cov



def cov_XX(du, dv, ac, ad, bc, bd, L0):
    """
    Compute the XX-covariance with the distance sqrt(du2+dv2). DPHI is 
    precomputed on tabDPHI.
    """

    cov =   (-1 * DPHI(du+ac, dv, L0)
            + DPHI(du+ad, dv, L0)
            + DPHI(du+bc, dv, L0)
            - DPHI(du+bd, dv, L0)
            )
    return cov

def cov_YY(du, dv, ac, ad, bc, bd, L0):
    """
    Compute the YY-covariance with the distance sqrt(du2+dv2). DPHI is 
    precomputed on tabDPHI.
    """

    cov =   (-1 * DPHI(du, dv+ac, L0)
            + DPHI(du, dv+ad, L0)
            + DPHI(du, dv+bc, L0)
            - DPHI(du, dv+bd, L0)
            )
    return cov

def cov_XY(du, dv, s0, L0):
    """
    Compute the XY-covariance with the distance sqrt(du2+dv2). DPHI is precomputed on tabDPHI.
   """

    cov = (  -1*DPHI(du + s0, dv - s0, L0)
            + DPHI(du + s0, dv + s0, L0)
            + DPHI(du - s0, dv - s0, L0)
            - DPHI(du - s0, dv + s0, L0)
            )
    return cov

def DPHI(x, y, L0):
    """
    Parameters:
        x (float): Seperation between apertures in X direction
        y (float): Separation between apertures in Y direction
        L0 (float): Outer scale

    Computes the phase structure function for a separation (x,y).
    The r0 is not taken into account : the final result of DPHI(x,y,L0)
    has to be scaled with r0^-5/3, with r0 expressed in meters, to get
    the right value.
    """
    r = numpy.sqrt(x**2 + y**2)

    return rodconan(r, L0, 10)

def rodconan(r, L0, k):
    """
    The phase structure function is computed from the expression
    Dphi(r) = k1  * L0^(5./3) * (k2 - (2.pi.r/L0)^5/6 K_{5/6}(2.pi.r/L0))

    For small r, the expression is computed from a development of
    K_5/6 near 0. The value of k2 is not used, as this same value
    appears in the series and cancels with k2.
    For large r, the expression is taken from an asymptotic form.

    """
    # k1 is the value of :
    # 2*gamma_R(11./6)*2^(-5./6)*pi^(-8./3)*(24*gamma_R(6./5)/5.)^(5./6);
    k1 = 0.1716613621245709486
    dprf0 = (2*numpy.pi/L0)*r

    if dprf0 > 4.71239:
    	res = asymp_macdo(dprf0)
    else:
    	res = -macdo_x56(dprf0, k)

    res *= k1 * L0**(5./3)

    return res

def asymp_macdo(x):
    """
    Computes a term involved in the computation of the phase struct
    function with a finite outer scale according to the Von-Karman
    model. The term involves the MacDonald function (modified bessel
    function of second kind) K_{5/6}(x), and the algorithm uses the
    asymptotic form for x ~ infinity.
    Warnings :
    - This function makes a doubleing point interrupt for x=0
    and should not be used in this case.
    - Works only for x>0.
    """

    # k2 is the value for
    # gamma_R(5./6)*2^(-1./6)
    k2 = 1.00563491799858928388289314170833
    k3 = 1.25331413731550012081   #  sqrt(pi/2)
    a1 = 0.22222222222222222222   #  2/9
    a2 = -0.08641975308641974829  #  -7/89
    a3 = 0.08001828989483310284   # 175/2187

    x1 = 1./x
    res = (	k2 
    		- k3 * numpy.exp(-x) * x**(1./3)
    		* (1.0 + x1*(a1 + x1*(a2 + x1*a3)))
    		)
    return res

def macdo_x56(x, k):
    """
    Computation of the function
    f(x) = x^(5/6)*K_{5/6}(x)
    using a series for the esimation of K_{5/6}, taken from Rod Conan thesis :
    K_a(x)=1/2 \sum_{n=0}^\infty \frac{(-1)^n}{n!}
    \left(\Gamma(-n-a) (x/2)^{2n+a} + \Gamma(-n+a) (x/2)^{2n-a} \right) ,
    with a = 5/6.

    Setting x22 = (x/2)^2, setting uda = (1/2)^a, and multiplying by x^a,
    this becomes :
    x^a * Ka(x) = 0.5 $ -1^n / n! [ G(-n-a).uda x22^(n+a) + G(-n+a)/uda x22^n ]
    Then we use the following recurrence formulae on the following quantities :
    G(-(n+1)-a) = G(-n-a) / -a-n-1
    G(-(n+1)+a) = G(-n+a) /  a-n-1
    (n+1)! = n! * (n+1)
    x22^(n+1) = x22^n * x22
    and at each iteration on n, one will use the values already computed at step (n-1).
    The values of G(a) and G(-a) are hardcoded instead of being computed.

    The first term of the series has also been skipped, as it
    vanishes with another term in the expression of Dphi.
    """

    a = 5./6
    x2a = x**(2.*a)
    x22 = x * x/4.

    Ga = [	0, 12.067619015983075, 5.17183672113560444, 0.795667187867016068, 
    		0.0628158306210802181, 0.00301515986981185091,
      		9.72632216068338833e-05, 2.25320204494595251e-06, 
      		3.93000356676612095e-08, 5.34694362825451923e-10, 
      		5.83302941264329804e-12 
      		]

    Gma = [ -3.74878707653729304, -2.04479295083852408,
      		-0.360845814853857083, -0.0313778969438136685,
      		-0.001622994669507603, -5.56455315259749673e-05,
      		-1.35720808599938951e-06, -2.47515152461894642e-08,
      		-3.50257291219662472e-10, -3.95770950530691961e-12,
      		-3.65327031259100284e-14  
  		]

    x2n = 0.5

    s = Gma[0] * x2a
    s*= x2n

    # Prepare recurrence iteration for next step
    x2n *= x22

    for n in range(1,11):
    	s += (Gma[n]*x2a + Ga[n]) * x2n

    	# Prepare recurrent iteration for next step
    	x2n *= x22

    return s