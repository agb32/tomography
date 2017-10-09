
#define USE_OPENMP 1
#ifdef USE_OPENMP
#include<omp.h>
#endif

#include"matcov_styc.h"
/*
 *
 *  __  __       _          _____           _   _                     
 * |  \/  | __ _(_)_ __    |  ___|__  _ __ | |_(_) ___  _ __  ___   
 * |      |/ _` | | '_ \   | |_ / _ \| '_ \| __| |/ _ \| '_ \/ __| 
 * | |\/| | (_| | | | | |  |  _| (_) | | | | |_| | (_) | | | \__ \ 
 * |_|  |_|\__,_|_|_| |_|  |_|  \___/|_| |_|\__|_|\___/|_| |_|___/  
 *
 */

/* %%%%%%%%%%%%%%%%%%%%%% Complete, Cpm, ou Cmm %%%%%%%%%%%%%%%%%%%%%%%% 
*/


void
matcov_styc(struct tomo_struct tomo, double *data) {
  
  const long cNw = tomo.Nw;
  const long cNlayer = tomo.Nlayer;
  long i; 
  

  // %%%%%%% Computation of the sub-apertures positions and sizes %%%%%%%%%%%
 // u, v :arrays containing all the sub-apertures coordinates of all WFS, one after the other
  // u[0][1][3] is the X-coordinate of subap number 3 of wfs number 0 at altitude 3
  printf("Allocating 3d arrays...\n");
  double*** u = arr3dAlloc( cNw, tomo.Nsubap, cNlayer);
  double*** v = arr3dAlloc( cNw, tomo.Nsubap, cNlayer);

  //Computes  u and v
   subap_position(tomo, u, v);
  
  //Rescale the projected size of all subapertures at the different altitude layer
  double sspSizeL[cNw][cNlayer];
  long l, n;

  for (n = 0; n < cNw; n++) {
    for (l = 0; l < cNlayer; l++) {
      sspSizeL[n][l] = tomo.sspSize[n] * (1. - tomo.GsAlt[n] * tomo.h[l]);
    }
  }

  
  // %%%%%%% Computation of the covariance matrix %%%%%%%%%%%
  const double lambda2 = 0.00026942094446267851; //lambda2 = pow(206265.*0.5e-6/2./3.1415926535,2);
  long m, j;
  long ioff = 0, joff = 0;
  double units[cNlayer];
  long Nslopes=0; // very total number of slopes


 for (i = 0; i < cNw; i++)
    Nslopes += tomo.Nsubap[i]*2;
  
  //  Initialization and limits of the loops on WFS, function of the computed parts of the matrix
  long m0=0;
  long mf=0;
  long n0=0;
  long *nf=0;
  long NL;
  long ts = cNw - 1;//Truth sensor : ts

  if (tomo.part == 0) { //Complete matrix
    m0 = 0;
    mf = cNw;
    n0 = 0.;
    nf = &m;

    NL=0; // very total number of slopes
    for (i = 0; i < cNw; i++)
      NL += tomo.Nsubap[i]*2;

  } else if (tomo.part == 1) { //Cmm
    m0 = 0;
    mf = cNw-1;
    n0 = 0; 
    nf = &m;

    Nslopes=0; // very total number of slopes
    for (i = 0; i < cNw; i++)
      Nslopes += tomo.Nsubap[i];
    Nslopes *= 2;
    NL = Nslopes-2.*tomo.Nsubap[ts];

  } else if  (tomo.part == 3) { //Cpm
    m0 = 0;
    mf = cNw - 1;
    n0 = cNw - 1;
    nf = &ts;
    NL = 2.*tomo.Nsubap[cNw - 1];
  }

  
  //WFS m
  for (m = m0; m < mf; m++) {

    const long Ni = tomo.Nsubap[m] + ioff;
    
    //WFS n
    for (n = n0; n < *nf+1; n++) {

      const long off_XY = tomo.Nsubap[n];
      const long off_YX = tomo.Nsubap[m] * NL;
      const long off_YY = off_XY+ off_YX;

      const long Nj = tomo.Nsubap[n] + joff;
      const double kk = 1. / (tomo.sspSize[m] * tomo.sspSize[n]);

      for (l = 0; l < cNlayer; l++) {
        units[l] = kk * lambda2 * tomo.cn2[l];
      }

      printf("Begin Covariance Matrix Calculation with %d threads\n", tomo.ncpu);
// #ifdef USE_OPENMP
// #pragma omp parallel private(j,l) num_threads(tomo.ncpu)
// #pragma omp for nowait
// #endif
      //Subaperture i
#pragma omp parallel private(j,l) num_threads(tomo.ncpu)
#pragma omp for nowait
      for (i = ioff; i < Ni; i++) {

        //Subaperture j
        for (j = joff; j < Nj; j++) {
	  double caa_xx = 0;
	  double caa_yy = 0;
	  double caa_xy = 0;

	  //Layer l
	   for (l = 0; l < cNlayer; l++) {

	    //test if the altitude layers is not higher than the LGS altitude
	    if ((sspSizeL[m][l] > 0) && (sspSizeL[n][l] > 0)) {
	      //Distances in x and y between the subapertures i and j
	      const double du = u[m][i-ioff][l] - u[n][j-joff][l];	      
	      const double dv =  v[m][i-ioff][l] - v[n][j-joff][l];

	      const double s1 = sspSizeL[m][l] * 0.5;
	      const double s2 = sspSizeL[n][l] * 0.5;

	      const double ac = s1 - s2;
	      const double ad = s1 + s2;
	      const double bc = -ad;   // initially -s1-s2;
	      const double bd = -ac;   // initially -s1+s2;

	      //Computation of the covariance on each layer
	      double *cov;
	      cov = compute_cov(du, dv, ac, ad, bc, bd, s1, s2,tomo.L0[l], units[l]);

	      caa_xx += cov[0];	    
	      caa_yy += cov[1];
	      caa_xy += cov[2];

	      free(cov);
	      }
	  }

	  const long i0 = i * NL + j;
	  data[i0] = caa_xx;   //xx
	  data[i0 + off_XY] = caa_xy;   //xy
	  data[i0 + off_YX] = caa_xy;   //yx
	  data[i0 + off_YY] = caa_yy; //yy

	}
      }
    printf("Done Parallel computation\n");
      joff = joff + 2 * tomo.Nsubap[n];
    }
    ioff = ioff + 2 * tomo.Nsubap[m];
    joff = 0;
  }

   // printf("Mirror matrix\n");
  //Recopie de la symétrie
  // if (tomo.part == 0 || tomo.part == 1) { //Complete matrix
  //   long size=NL-1;
  //   double *matL = (double*)data+1;
  //   double *matU = (double*)data+NL;
  //   do {
  //     for(int j=0; j<size; j++) 
  //   matL[j]=matU[j*NL];
  //     size--;
  //     matL+=NL+1;
  //     matU+=NL+1;
  //   } while (size>0);
  //  }
 
  printf("Cleanup...\n");
  arr3dFree(u, cNw, tomo.Nsubap);
  arr3dFree(v, cNw, tomo.Nsubap);
}




/* %%%%%%%%%%%%%%%%%%%%%% CPP %%%%%%%%%%%%%%%%%%%%%%%% */


void
matcov_cpp_styc(struct tomo_struct tomo, double *data)
/* DOCUMENT matcov_cpp_styc(struct tomo_struct tomo, double *data) 
Compute the covariance matrix of the TS. The TS must be at the last WFS.

   <tomo> :  structure with all the needed information ( see matcov_styc.h). Can 
             contain only the TS or all the WFS. In last case, the TS must be the last WFS.

 SEE ALSO:
*/
 {
  const long cNw = tomo.Nw;
  const long cNlayer = tomo.Nlayer;
  long i; 


  // %%%%%%% Computation of the covariance matrix %%%%%%%%%%%
  long l,j;
  const long ts = cNw - 1;
  const long cNsubapTS = tomo.Nsubap[ts];
  const long offXY = cNsubapTS* 2.*cNsubapTS;
  const long offYY = offXY+ cNsubapTS;
  const double lambda2 = 0.00026942094446267851; //lambda2 = pow(206265.*0.5e-6/2./3.1415926535,2);
  const double kk = 1. / (tomo.sspSize[ts] * tomo.sspSize[ts]);


  double units[cNlayer];
  for (l = 0; l < cNlayer; l++) {
    units[l] = kk * lambda2 * tomo.cn2[l];
  }


  long Nslopes=0; // very total number of slopes
  for (i = 0; i < cNw; i++)
    Nslopes += tomo.Nsubap[i];
  const long off = Nslopes - cNsubapTS;
  Nslopes *= 2;


  const double s = tomo.sspSize[ts] * 0.5;
  const double ac = 0.;
  const double ad = 2. * s;
  const double bc = -ad;   
  const double bd = 0.;  


#ifdef USE_OPENMP
#pragma omp parallel private(j,l) num_threads(tomo.ncpu)
#pragma omp for nowait
#endif
  //Subaperture i
  for (i = 0; i < cNsubapTS; i++) {

    //Subaperture j
    for (j = 0; j < cNsubapTS; j++) {
      double caa_xx = 0.;
      double caa_yy = 0.;
      double caa_xy = 0.;

      //Distances in x and y between the subapertures i and j
      const double du = tomo.X[i+off] - tomo.X[j+off];
      const double dv = tomo.Y[i+off] - tomo.Y[j+off];

      //Layer l
      for (l = 0; l < cNlayer; l++) {
        //test if the altitude layers is not higher than the LGS altitude
        if ((s > 0)) {
	  double *cov;
          //Computation of the covariance on each layer
	  cov = compute_cov(du, dv, ac, ad, bc, bd, s, s,tomo.L0[l], units[l]);
	  caa_xx += cov[0];	    
	  caa_yy += cov[1];
	  caa_xy += cov[2];
	  free(cov);
        }
      }

      const long i0 = i * 2.*cNsubapTS + j;

      data[i0] = caa_xx;          //xx

      data[i0 + cNsubapTS] =  caa_xy;          //xy

      data[i0 +  offXY] = caa_xy;          //yx
      
      data[i0 + offYY] = caa_yy; //yy
    }
  }
}




/*
 *
 *  ___                   _____           _   _                     
 * |  _ \ __   ___       |  ___|__  _ __ | |_(_) ___  _ __  ___   
 * | |_| / _ \/ __|      | |_ / _ \| '_ \| __| |/ _ \| '_ \/ __| 
 * |  __| (_) \__ \      |  _| (_) | | | | |_| | (_) | | | \__ \ 
 * |_|   \___/|___/      |_|  \___/|_| |_|\__|_|\___/|_| |_|___/  
 *
 */




void
subap_position(struct tomo_struct tomo, double ***u, double ***v) {
  /* DOCUMENT         subap_position(tomo, u, v)
   <tomo>                : structure with all the needed information.
   <u> and <v>           : 3d arrays containing the sub-apertures projected coordinates onto all the layers. u[0][2][1] is the X-coordinate of the subap 2 of the WFS 0 on the layer 1.

   Computes the projected coordinates of all subapertures  projected onto all the layer
   */
  long i;
  long n;
  long l;
  const double rad = 3.14159265358979323846 / 180.;

  for (l = 0; l < tomo.Nlayer; l++) {
    long ioff = 0;

    for (n = 0; n < tomo.Nw; n++) {

      const double dX = tomo.alphaX[n] * tomo.h[l];
      const double dY = tomo.alphaY[n] * tomo.h[l];

      const double rr = 1. - tomo.h[l] * tomo.GsAlt[n];

      const long Nsap = tomo.Nsubap[n];
      const long nssp = tomo.Nssp[n];

      //magnification factor
      const double G = tomo.diamPup[n] / (double) (nssp);

      //rotation angle
      const double th = tomo.thetaML[n] * rad;

      for (i = 0; i < Nsap; i++) {
        //taking magnification factor into account
        const double xtp = tomo.X[ioff + i] * G;
        const double ytp = tomo.Y[ioff + i] * G;

        //taking rotation into account
        double uu = xtp * cos(th) - ytp * sin(th);
        double vv = xtp * sin(th) + ytp * cos(th);

        //taking pupil offset into account
        uu += tomo.XPup[n];
        vv += tomo.YPup[n];

        //Projection onto  the layer
        u[n][i][l] = uu * rr + dX;
        v[n][i][l] = vv * rr + dY;
      }
      //index offset
      ioff += Nsap;
    }
  }
}

/*
 *  _____           _   _                       _        ____
 * |  ___|__  _ __ | |_(_) ___  _ __  ___    __| | ___  | __ )  __ _ ___  ___
 * | |_ / _ \| '_ \| __| |/ _ \| '_ \/ __|  / _` |/ _ \ |  _ \ / _` / __|/ _ \
 * |  _| (_) | | | | |_| | (_) | | | \__ \ | (_| |  __/ | |_) | (_| \__ \  __/
 * |_|  \___/|_| |_|\__|_|\___/|_| |_|___/  \__,_|\___| |____/ \__,_|___/\___|
 *
 */

double *compute_cov(double du, double dv, double ac, double ad, double bc, double bd,double s1, double s2, double L0, double units)
 /* DOCUMENT
   <du> & <dv>                 : X et Y coordinates of the distance between the deux considered subapertures.
   <ac> & <ad> & <bc> & <bd>  : precomputed values
   <s1> & <s2>                : half size of each subapertures
   <L0>                  : 
   <units>                    :
   Computes the XX, XY and YY covariance values for two subapertures.
   */
{        
  double *cov = (double*)malloc(3*sizeof(double));
  double cov_xx=0;
  double cov_yy=0;
  double cov_xy=0;


  //Computation of the covariance on each layer
  cov_xx = cov_XX(du,dv,ac,ad,bc,bd,L0);
  cov_xx *= 0.5;
  
  cov_yy = cov_YY(du,dv,ac,ad,bc,bd,L0);
  cov_yy *= 0.5;
  
  const double s0 = sqrt(s1 * s1 + s2 * s2); //half size of the subaperture equivalent to a convolution by s1 and s2
  cov_xy = cov_XY(du,dv,s0,L0);
  cov_xy *= 0.25;
  
  // double cc = 1.-fmin(s1,s2)/fmax(s1,s2);
  // when s1==s2, then cc=0 and the computation is just void,
  // so we perform it only when s1!=s2
  const double cc = (s1 > s2) ? 1. - s2 / s1 : 1. - s1 / s2; // Nono's style ....
  cov_xy *= (1. - cc * cc);
  
  //units
  cov_xx *= units;
  cov_yy *= units;
  cov_xy *= units;

  cov[0] = cov_xx;
  cov[1] = cov_yy;
  cov[2] = cov_xy;
  return cov;
}



double cov_XX(double du, double dv, double ac, double ad, double bc, double bd, double L0)
 /* DOCUMENT
   Compute the XX-covariance with the distance sqrt(du2+dv2). DPHI is precomputed on tabDPHI.
 */
{
  return -DPHI(du + ac, dv, L0)
    + DPHI(du + ad, dv, L0)
    + DPHI(du + bc, dv, L0)
    - DPHI(du + bd, dv,L0);
}

double cov_YY(double du, double dv, double ac, double ad, double bc, double bd, double L0)
/* DOCUMENT
   Compute the YY-covariance with the distance sqrt(du2+dv2). DPHI is precomputed on tabDPHI.
 */
{ 
  return  -DPHI(du, dv + ac, L0)
    + DPHI(du, dv + ad, L0)
    + DPHI(du, dv + bc, L0)
    - DPHI(du, dv + bd, L0);
}


double cov_XY(double du, double dv, double s0, double L0)
/* DOCUMENT
   Compute the XY-covariance with the distance sqrt(du2+dv2). DPHI is precomputed on tabDPHI.
 */
{
  return -DPHI(du + s0, dv - s0, L0)
    + DPHI(du + s0, dv + s0, L0)
    + DPHI(du - s0, dv - s0, L0)
    - DPHI(du - s0, dv + s0, L0);
}


double
macdo_x56(double x, int k)
/* DOCUMENT  macdo_x56(x)

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

 SEE ALSO:
 */
{
  const double a = 5. / 6.;
  const double x2a = pow(x, 2. * a), x22 = x * x / 4.;
  double x2n;               // x^2.a, etc
  double s = 0.0;
  int n;

  const double Ga[11] = { 0, 12.067619015983075, 5.17183672113560444,
      0.795667187867016068, 0.0628158306210802181, 0.00301515986981185091,
      9.72632216068338833e-05, 2.25320204494595251e-06, 3.93000356676612095e-08,
      5.34694362825451923e-10, 5.83302941264329804e-12 };

  const double Gma[11] = { -3.74878707653729304, -2.04479295083852408,
      -0.360845814853857083, -0.0313778969438136685, -0.001622994669507603,
      -5.56455315259749673e-05, -1.35720808599938951e-06,
      -2.47515152461894642e-08, -3.50257291219662472e-10,
      -3.95770950530691961e-12, -3.65327031259100284e-14 };

  x2n = 0.5;                           // init (1/2) * x^0

  s = Gma[0] * x2a;
  s *= x2n;

  // prepare recurrence iteration for next step
  x2n *= x22;    // x^n

  for (n = 1; n <= 10; n++) {

    s += (Gma[n] * x2a + Ga[n]) * x2n;
    // prepare recurrence iteration for next step
    x2n *= x22;    // x^n
  }
  return s;
}

double
asymp_macdo(double x)
/* DOCUMENT asymp_macdo(x)

 Computes a term involved in the computation of the phase struct
 function with a finite outer scale according to the Von-Karman
 model. The term involves the MacDonald function (modified bessel
 function of second kind) K_{5/6}(x), and the algorithm uses the
 asymptotic form for x ~ infinity.
 Warnings :
 - This function makes a doubleing point interrupt for x=0
 and should not be used in this case.
 - Works only for x>0.

 SEE ALSO:
 */
{
  // k2 is the value for
  // gamma_R(5./6)*2^(-1./6)
  const double k2 = 1.00563491799858928388289314170833;
  const double k3 = 1.25331413731550012081;   //  sqrt(pi/2)
  const double a1 = 0.22222222222222222222;   //  2/9
  const double a2 = -0.08641975308641974829;  //  -7/89
  const double a3 = 0.08001828989483310284;   // 175/2187
  double res;
  double x_1;

  x_1 = 1. / x;
  res = k2
      - k3 * exp(-x) * pow(x, 1 / 3.)
          * (1.0 + x_1 * (a1 + x_1 * (a2 + x_1 * a3)));
  return res;
}

double
rodconan(double r, double L0, int k)
/* DOCUMENT rodconan(r,L0,k=)
 The phase structure function is computed from the expression
 Dphi(r) = k1  * L0^(5./3) * (k2 - (2.pi.r/L0)^5/6 K_{5/6}(2.pi.r/L0))

 For small r, the expression is computed from a development of
 K_5/6 near 0. The value of k2 is not used, as this same value
 appears in the series and cancels with k2.
 For large r, the expression is taken from an asymptotic form.

 SEE ALSO:
 */
{

  const double pi = 3.1415926535897932384626433;
  double res = 0;

  // k1 is the value of :
  // 2*gamma_R(11./6)*2^(-5./6)*pi^(-8./3)*(24*gamma_R(6./5)/5.)^(5./6);
  const double k1 = 0.1716613621245709486;
  const double dprf0 = (2 * pi / L0) * r;
  // k2 is the value for gamma_R(5./6)*2^(-1./6),
  // but is now unused
  // k2 = 1.0056349179985892838;

  // Xlim = 0.75*2*pi;   // = 4.71239
  if (dprf0 > 4.71239) {
    res = asymp_macdo(dprf0);
  } else {
    res = -macdo_x56(dprf0, k);
  }
  res *= k1 * pow(L0, 5. / 3);
  return res;
}

double
DPHI(double x, double y, double L0)
/* DOCUMENT dphi = DPHI(x,y, L0) * r0^(-5./3)
 <x> & <y>         :  separation between apertures
 <L0>         :  outerscale

 Computes the phase structure function for a separation (x,y).
 The r0 is not taken into account : the final result of DPHI(x,y,L0)
 has to be scaled with r0^-5/3, with r0 expressed in meters, to get
 the right value.

 SEE ALSO:
 */
{
  double r = sqrt(x * x + y * y);

  return rodconan(r, L0,10);
}




/*
 *
 *  _____           _   _                    ___  _ _                     
 * |  ___|__  _ __ | |_(_) ___  _ __  ___   / _ \| | | ___   __ 
 * | |_ / _ \| '_ \| __| |/ _ \| '_ \/ __| | /_\ | | |/ _ \ / _|
 * |  _| (_) | | | | |_| | (_) | | | \__ \ | __  | | | (_) | (_| 
 * |_|  \___/|_| |_|\__|_|\___/|_| |_|___/ |_| |_|_|_|\___/ \__|
 *
 */

double***
arr3dAlloc(long Nw, long *Nsubap, long Nlayer) {
  /* DOCUMENT  array = arr3dAlloc(Nw,Nsubap,Nlayer)
   <Nw>                  :  number of WFS
   <Nsubap>              :  array of the number of subap of each WFS
   <Nlayer>              :  number of layer

   Allocates a 3d array in one big array with a non-constant size for the 2rd dimension.
   */

  double ***array = (double***) malloc(Nw * sizeof(double**));
  long n, i;

  for (n = 0; n < Nw; n++) {
    array[n] = (double**) malloc( Nsubap[n] * sizeof(double*));
    for (i = 0; i < Nsubap[n]; i++) {
      array[n][i] = (double*) malloc(Nlayer * sizeof(double));
    }
  }

  return array;

}

void
arr3dFree(double ***array, long Nw, long *Nsubap) {

  /* DOCUMENT  array = arr3dFree(array,Nw,Nsubap)
   <Nw>                  :  number of WFS
   <Nsubap>              :  array of the number of subap of each WFS

   Free a 3d array with a non-constant size for the  2nd dimension.
   */
  long n, i;
  for (n = 0; n < Nw; n++) {
    for (i = 0; i < Nsubap[n]; i++) {
      free(array[n][i]);
    }
    free(array[n]);
  }
  free(array);
}
