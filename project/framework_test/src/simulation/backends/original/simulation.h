#pragma once

namespace OriginalOptimized {

void computeTentativeVelocity(float **u, float **v, float **f, float **g,
                              char **flag, int imax, int jmax, float del_t, float delx, float dely,
                              float gamma, float Re);

void computeRhs(float **f, float **g, float **rhs, char **flag, int imax,
                int jmax, float del_t, float delx, float dely);

template<bool ErrorCheck>
int poissonSolver(float **p, float **p_red, float **p_black,
                  float **p_beta, float **p_beta_red, float **p_beta_black,
                  float **rhs, float **rhs_red, float **rhs_black,
                  int **fluidmask, int **surroundmask_black,
                  char **flag, int imax, int jmax,
                  float delx, float dely, float eps, int itermax, float omega,
                  int ifull);

void calculatePBeta(float **p_beta,
                    char **flag,
                    int imax, int jmax,
                    float delx, float dely, float eps, float omega);

void splitToRedBlack(float **joined, float **red, float **black,
                     int imax, int jmax);
void joinRedBlack(float **joined, float **red, float **black,
                  int imax, int jmax);

void updateVelocity(float **u, float **v, float **f, float **g, float **p,
                    char **flag, int imax, int jmax, float del_t, float delx, float dely);

void setTimestepInterval(float *del_t, int imax, int jmax, float delx,
                         float dely, float **u, float **v, float Re, float tau);

void applyBoundaryConditions(float **u, float **v, char **flag,
                             int imax, int jmax, float ui, float vi);

void calculateFluidmask(int **fluidmask, const char **flag,
                        int imax, int jmax);
void splitFluidmaskToSurroundedMask(const int **f, int **red, int **black, int imax, int jmax);

}
