#include "matcov_styc.h"

void print_array_2d(double* data, int xSize, int ySize);
void print_array_1d(double* data, int size);
void print_array_3p(double ***data, int xSize, long* ySize, int zSize);

int main()
{
    double du, dv, ac, ad, bc, bd, s0, x, y;
    double L0;

    double covxx, covyy, covxy, dphi;

    du = 10;
    dv = 0.1;
    ac = 1;
    ad = 2;
    bc = 3.;
    bd = 1.;
    s0 = 1.;
    L0 = 100.;

    x = 1.;
    y = 1.;

    dphi = DPHI(x, y, L0);

    covxx = cov_XX(du, dv, ac, ad, bc, bd, L0);
    covyy = cov_YY(du, dv, ac, ad, bc, bd, L0);
    covxy = cov_XY(du, dv, s0, L0);

    printf("dphi: %0.8f \n", dphi);
    printf("Cov_xx: %.8f \n", covxx);
    printf("Cov_yy: %.8f \n", covyy);
    printf("Cov_xy: %.8f \n", covxy);

    // Now test the covMat stuff...

    long NWFS = 2;
    double OBS = 0.;
    double DIAMTEL = 1.;

    long NLAYER = 1;


    int NCPU = 1;
    int PART = 0;


    long* NSUBAP = (long*) malloc(sizeof(long)*NWFS);
    NSUBAP[0] = 4;
    NSUBAP[1] = 4;
    long totSubaps = NSUBAP[0]+NSUBAP[1];

    double* X = (double*) malloc(sizeof(double)*(totSubaps));
    double* Y = (double*) malloc(sizeof(double)*(totSubaps));
    X[0] = 0; X[1] = 0; X[2] = 0.5; X[3] = 0.5;
    X[4] = 0; X[5] = 0; X[6] = 0.5; X[7] = 0.5;
    Y[0] = 0; Y[1] = 0.5; Y[2] = 0; Y[3] = 0.5;
    Y[4] = 0; Y[5] = 0.5; Y[6] = 0; Y[7] = 0.5;

    long* NSSP = (long*) malloc(sizeof(long)*NWFS);
    NSSP[0] = 2;
    NSSP[1] = 2;

    double* GSALT = (double*) malloc(sizeof(double)*NWFS);
    GSALT[0] = 0;
    GSALT[1] = 0;

    int* TYPE = (int*) malloc(sizeof(int)*NWFS);
    TYPE[0] = 1;
    TYPE[1] = 1;

    double* ALPHAX = (double*) malloc(sizeof(double)*NWFS);
    ALPHAX[0] = 0;//(10./3600) * (3.142/180);
    ALPHAX[1] = 0;//(-10./3600) * (3.142/180);

    double* ALPHAY = (double*) malloc(sizeof(double)*NWFS);
    ALPHAY[0] = (10./3600)*(3.142/180);
    ALPHAY[1] = (-10./3600)*(3.142/180);

    double* XPUP = (double*) malloc(sizeof(double)*NWFS);
    XPUP[0] = 0;
    XPUP[1] = 0;

    double* YPUP = (double*) malloc(sizeof(double)*NWFS);
    YPUP[0] = 0;
    YPUP[1] = 0;

    double* THETAML = (double*) malloc(sizeof(double)*NWFS);
    THETAML[0] = 0;
    THETAML[1] = 0;

    double* DIAMPUP = (double*) malloc(sizeof(double)*NWFS);
    DIAMPUP[0] = 1;
    DIAMPUP[1] = 1;

    double* SSPSIZE = (double*) malloc(sizeof(double)*NWFS);
    SSPSIZE[0] = 0.5;
    SSPSIZE[1] = 0.5;

    double* CN2 = (double*) malloc(sizeof(double)*NLAYER);
    CN2[0] = 10.;

    double* H = (double*) malloc(sizeof(double)*NLAYER);
    H[0] = 5000.;

    double* L0_array = (double*) malloc(sizeof(double)*NLAYER);
    L0_array[0] = 100.;


    struct tomo_struct test_tomo = {NWFS, X, Y, DIAMTEL, OBS, NSUBAP, NSSP, GSALT, TYPE, ALPHAX, ALPHAY, XPUP, YPUP, THETAML, DIAMPUP, SSPSIZE, NLAYER, CN2, H, L0_array, NCPU, PART};

    // Test subap_position function
    double ***u = arr3dAlloc(NWFS, NSUBAP, NLAYER);
    double ***v = arr3dAlloc(NWFS, NSUBAP, NLAYER);

    subap_position(test_tomo, u, v);

    printf("\nTest subap_position function:\n");
    printf("***u array:\n");
    print_array_3p(u, NWFS, NSUBAP, NLAYER);
    printf("***v array:\n");
    print_array_3p(v, NWFS, NSUBAP, NLAYER);



    // Test covMat function

    printf("Test covmat_styc function:\n");
    double* covMat_Data = (double*) malloc(sizeof(double) * (2*totSubaps*2*totSubaps));
    matcov_styc(test_tomo, covMat_Data);

    printf("Resulting covmat:\n");
    print_array_2d(covMat_Data, (int) 2*totSubaps, (int) 2*totSubaps);

    return 0;

}

void print_array_2d(double* data, int xSize, int ySize)
{
    int x, y;

    for (x=0; x<xSize; x++)
    {
        printf("[");
        for (y=0; y<ySize; y++)
        {
            printf("%0.4f, \t", data[x*xSize+y]);
        }
        printf("],\n");
    }
}


void print_array_1d(double* data, int size)
{
    int i;

    printf("\n");
    for (i=0; i<size; i++)
    {
        printf("%0.4f  \t", data[i]);
    }
    printf("\n");
}


void print_array_3p(double ***data, int xSize, long* ySize, int zSize)
{
    int x, y, z;

    for (x=0; x<xSize; x++)
    {
        for (y=0; y<ySize[x]; y++)
        {
            for (z=0; z<zSize; z++)
            {
                printf("%0.4f\t", data[x][y][z]);
            }
            printf("\n");
        }
        printf("\n");
    }
}
