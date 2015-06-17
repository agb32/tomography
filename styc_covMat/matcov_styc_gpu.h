#ifndef MATCOV_STYC_GPU_H_
#define MATCOV_STYC_GPU_H_

#include <stdio.h>
#include <cuda_runtime.h>

struct gtomo_struct {
  double DiamTel; // Telescope diameter [m]
  double obs; // Central obstruction
  double pasDPHI; // DPHI step
  long *type; //type of WFS, 0, 1, 2 or 3. 0 is unused, 1=NGS, 2=LGS, 3=TipTilt-guide star
  long Nw; // Number of WFS
  long Nx; // Total number of valid ssp
  long *Nssp; // Number of nxsub for each wfs
  long *Nsubap; // Number of valid subap for each wfs
  double *diamPup; // Nssp
  double *XPup; // 0
  double *YPup; // 0
  double *thetaML; // 0
  double *alphaX; //wfs positions
  double *alphaY; //wfs positions
  double *GsAlt; //LGS altitude (0)
  double *h; //Layers altitudes
  double *cn2;
  double *L0;
  double *sspSize;
  double *X;
  double *Y;

  double lgs_cst; // 0
  double spot_width; // 1
  double lgs_depth; //10000
  double lgs_alt; // 90000.
  int   nlgs; // 0

  long  *ioff_d; // Cumulative number of nvalid ssp with ioff[0] = 0
  long  *Nssp_d; // Nssp
  long  *Nsubap_d;

  double *data_d;
  double *alphaX_d;
  double *alphaY_d;
  double *GsAlt_d;
  double *diamPup_d;
  double *thetaML_d;
  double *X_d;
  double *Y_d;
  double *XPup_d;
  double *YPup_d;

  long   Nlayer;
  long   max_Nl0;
  long  *indexL0_d;
  double *L0diff_d;
  double *h_d;
  double *cn2_d;
  double *tabDPHI_d;
  double *u_d;
  double *v_d;
  double *sspSizeL_d;

  cudaStream_t matcov_stream;
};

#ifdef __cplusplus
#define EXPORT_C extern "C"
#else
#define EXPORT_C
#endif

void process_err(cudaError_t e, const char* str);
EXPORT_C void init_tomo_gpu(struct gtomo_struct *tomo_gpu);
EXPORT_C void free_tomo_gpu(struct gtomo_struct *tomo_gpu);
EXPORT_C void update_tomo_atm_gpu(struct gtomo_struct *tomo_gpu);
EXPORT_C void update_tomo_sys_gpu(struct gtomo_struct *tomo_gpu);
EXPORT_C void matcov_gpu_4(double* data, int nrows, int ncols, int xoffset, int yoffset, int lda, struct gtomo_struct *tomo_gpu);
void generateXY(struct gtomo_struct *tomo_gpu);
void tab_dphi_gpu(double *tab_dphi, struct gtomo_struct *tomo_gpu, long Ndphi, double *L0diff_d, int Nl0, double convert);
void sub_pos_gpu(struct gtomo_struct *tomo_gpu, long Nlayer);

#endif // MATCOV_STYC_GPU_H_
