from cffi import FFI

def getCovMatLib(lib_filename):
	ffiTomo = FFI()

	ffiTomo.cdef("""
		struct tomo_struct{
		long Nw;  // number of wavefront sensors

		// pointers on arrays containing corrdinates of sub-apertures
		// X and Y are biiiiig arrays from yorick containing all the subap
		// coordinates of all the WFSs, one after the other.
		double *X;
		double *Y;

		double DiamTel; // telescope Diameter

		double obs;//telescope obscuration

		// array of the number of subap of each WFS, contains Nw elements
		long *Nsubap;

		// array of the number of subap of each WFS along the telescop diameter, contains Nw elements
		long *Nssp;

		// array of the inverse of the guide star altitude (in 1/meters), contains Nw elements
		double *GsAlt;

		// type of WFS, 0, 1, 2 or 3. 0 is unused, 1=NGS, 2=LGS, 3=TipTilt-guide star
		int *type;

		// Pointing directions of WFS
		double *alphaX;           // pointing direction in X, arcseconds
		double *alphaY;           // pointing direction in Y, arcseconds

		// Deviations of WFSs
		double *XPup;             // pupil shift of the WFS, in meters
		double *YPup;             // pupil shift of the WFS, in meters
		double *thetaML;          // rotation of microlenses
		double *diamPup;          // magnification factor of this WFS
		double *sspSize;          // subaperture size of this WFS

		// PROFILE
		long Nlayer;              // number of layers in the profile
		double *cn2;              // profile strengh, units TBD ..
		double *h;                // altitude of layers (meters)
		double *L0;               // outer scale (meters)

		int ncpu;                 //Number of CPU used (only with openMP)
		int part;                 //Computed part of the cov. matrix. 0: complete 1: cmm 2: cpp 3: cpm
		};


		void matcov_styc(struct tomo_struct tomo, double *data);
		void matcov_cpp_styc(struct tomo_struct tomo, double *data);

		void subap_position(struct tomo_struct tomo,  double ***u, double ***v );

		double cov_XX(double du, double dv, double ac, double ad, double bc, double bd,double L0);
		double cov_YY(double du, double dv, double ac, double ad, double bc, double bd, double L0);
		double cov_XY(double du, double dv, double s0, double L0);
		double* compute_cov(double du, double dv,double ac, double ad, double bc, double bd,double s1, double s2, double L0, double units);

		double DPHI(double x, double y, double L0);


		double*** arr3dAlloc(long Nw, long *Nsubap, long Nlayer);
		void arr3dFree(double ***array, long Nw, long *Nsubap);

	"""
	)

	libTomo = ffiTomo.dlopen(lib_filename)

	return ffiTomo, libTomo