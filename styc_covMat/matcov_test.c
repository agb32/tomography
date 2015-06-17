#include "matcov_styc.h"


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

    return 0;


}