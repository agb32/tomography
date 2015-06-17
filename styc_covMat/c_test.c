#include "matcov_styc.h"

struct tomo_struct make_tomo_struct(long Nw, double *X, double *Y, double DiamTel, double obs, long* Nsubaps, long *Nssp, double *GsAlt, int *type, double *alphaX, double *alphaY, double *XPup, double *YPup,double *thetaML, double *diamPup, double *sspSize, long Nlayer, double *cn2, double *h, double *L0, int ncpu, int part)
	{
		struct tomo_struct tomo = {
			Nw, X, Y, DiamTel, obs, Nsubaps, Nssp, GsAlt, type, alphaX,
			alphaY, XPup, YPup, thetaML, diamPup, sspSize, Nlayer, cn2, h, 
			L0, ncpu, part
			};
		return tomo;
					
	}