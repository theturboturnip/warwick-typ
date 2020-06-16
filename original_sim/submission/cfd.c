#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include <sys/times.h>
#include <errno.h>
#include <string.h>
#include <math.h>
#include "utilities.h"
#include "simulation.h"
#include "constants.h"

void write_bin(float **u, float **v, float **p, char **flag,
     int imax, int jmax, float xlength, float ylength, char *file);

int read_bin(float **u, float **v, float **p, char **flag,
    int imax, int jmax, float xlength, float ylength, char *file);
    
int compareOutput();
long computeFlops(long imax, long jmax, long iterations);
double wtime();

#define RECORD_FUNC_TIMES 0
#if RECORD_FUNC_TIMES
#include <x86intrin.h>
uint64_t cycleCnt_div256() {
    // Shift down by 8 to divide by 256
    return __rdtsc() >> 8;
}
#endif

int main(int argc, char *argv[])
{
    int verbose = 1;          /* Verbosity level */
    float xlength = 22.0;     /* Width of simulated domain */
    float ylength = 4.1;      /* Height of simulated domain */
    int imax = 660;           /* Number of cells horizontally */
    int jmax = 120;           /* Number of cells vertically */

    char *infile;             /* Input raw initial conditions */
    char *outfile;            /* Output raw simulation results */

    float t_end = 10.0;        /* Simulation runtime */
    float del_t = 0.003;      /* Duration of each timestep */
    float tau = 0.5;          /* Safety factor for timestep control */

    int itermax = 100;        /* Maximum number of iterations in SOR */
    float eps = 0.001;        /* Stopping error threshold for SOR */
    float omega = 1.7;        /* Relaxation parameter for SOR */
    float gamma = 0.9;        /* Upwind differencing factor in PDE
                                 discretisation */

    float Re = 150.0;         /* Reynolds number */
    float ui = 1.0;           /* Initial X velocity */
    float vi = 0.0;           /* Initial Y velocity */

    float t, delx, dely;
    int itersor = 0, ifluid = 0, ibound = 0;
    float res;
    float **u, **v,
        **p, **p_red, **p_black,
        **p_beta, **p_beta_red, **p_beta_black,
        **rhs, **rhs_red, **rhs_black,
        **f, **g;
    char  **flag;
    int init_case, iters = 0;

    infile = strdup("initial.bin");    
    outfile = strdup("output.bin");

    delx = xlength/imax;
    dely = ylength/jmax;

    /* Allocate arrays */
    u    = allocFloatMatrix(imax+2, jmax+2);
    v    = allocFloatMatrix(imax+2, jmax+2);
    f    = allocFloatMatrix(imax+2, jmax+2);
    g    = allocFloatMatrix(imax+2, jmax+2);
    
    p    = allocFloatMatrix(imax+2, jmax+2);
    p_red = allocFloatMatrix(imax+2, (jmax+2)/2);
    p_black = allocFloatMatrix(imax+2, (jmax+2)/2);

    p_beta = allocFloatMatrix(imax+2, jmax+2);
    p_beta_red = allocFloatMatrix(imax+2, (jmax+2)/2);
    p_beta_black = allocFloatMatrix(imax+2, (jmax+2)/2);
    
    rhs  = allocFloatMatrix(imax+2, jmax+2);
    rhs_red = allocFloatMatrix(imax+2, (jmax+2)/2);
    rhs_black = allocFloatMatrix(imax+2, (jmax+2)/2);
    
    flag = allocCharMatrix(imax+2, jmax+2);                    

    if (!u || !v || !f || !g
        || !p || !p_red || !p_black
        || !p_beta || !p_beta_red || !p_beta_black
        || !rhs || !rhs_red || !rhs_black
        || !flag) {
        fprintf(stderr, "Couldn't allocate memory for matrices.\n");
        return 1;
    }

    /* Read in initial values from initial.bin */
    init_case = read_bin(u, v, p, flag, imax, jmax, xlength, ylength, infile);
        
    if (init_case > 0) {
        /* Error while reading file */
        return 1;
    }

    /* Check there are no non-zero values for obstacle squares */
    /* One of the optimizations relies on the initial p-value of obstacle squares being 0 */
    int i,j;
    int nonzero_count = 0;
    for (i = 0; i <= imax+1; i++) {
        for (j = 0; j <= jmax + 1; j++) {
            if (!(flag[i][j] & C_F)) {
                if (p[i][j] != 0) {
                    fprintf(stderr, "p[%d,%d] = %f != 0\n", i, j, p[i][j]);
                    nonzero_count++;
                }
                p[i][j] = 0.0f;
            }
        }
    }
    fprintf(stderr, "Found %d nonzero p-values for obstacle squares\n", nonzero_count);
    if (nonzero_count > 0) {
        fprintf(stderr, "Expected all p-values for obstacle squares to be zero\n");
        return 1;
    }

    /* Precalculate poisson beta values */
    calculatePBeta(p_beta, flag, imax, jmax, delx, dely, eps, omega);

    /* Split matrices */
    splitToRedBlack(p, p_red, p_black, imax, jmax);
    splitToRedBlack(p_beta, p_beta_red, p_beta_black, imax, jmax);

#if RECORD_FUNC_TIMES
    int64_t tentVelCnt_div256 = 0;
    int64_t rhs_div256 = 0;
    int64_t poisson_div256 = 0;
    int64_t updateVel_div256 = 0;
    int64_t bounds_div256 = 0;

    int64_t currCnt_div256 = 0;
    int64_t nextCnt_div256 = 0;

#define addToCnt(X) do {                                \
        nextCnt_div256 = cycleCnt_div256();             \
        if (nextCnt_div256 - currCnt_div256 > 0)        \
            X += (nextCnt_div256 - currCnt_div256);     \
        currCnt_div256 = nextCnt_div256;                \
    }    while (0);

#define updateCurrCnt() currCnt_div256 = cycleCnt_div256();
#else
#define addToCnt(X)
#define updateCurrCnt()
#endif
    
    double computeStart = wtime();
    /* Main loop */
    for (t = 0.0; t < t_end; t += del_t, iters++) {
        setTimestepInterval(&del_t, imax, jmax, delx, dely, u, v, Re, tau);

        ifluid = (imax * jmax) - ibound;

        updateCurrCnt();
        
        computeTentativeVelocity(u, v, f, g, flag, imax, jmax,
            del_t, delx, dely, gamma, Re);

        addToCnt(tentVelCnt_div256);
        
        computeRhs(f, g, rhs, flag, imax, jmax, del_t, delx, dely);

        addToCnt(rhs_div256);
        
        if (ifluid > 0) {
            itersor = poissonSolver(p, p_red, p_black,
                                    p_beta, p_beta_red, p_beta_black,
                                    rhs, rhs_red, rhs_black,
                                    flag, imax, jmax, delx, dely,
                        eps, itermax, omega, ifluid);
            addToCnt(poisson_div256);
        } else {
            itersor = 0;
            updateCurrCnt();
        }

        if (verbose > 1) {
            printf("%d t:%g, del_t:%g, SOR iters:%3d, res:%e, bcells:%d\n",
                iters, t+del_t, del_t, itersor, res, ibound);
        }

        updateCurrCnt();
        updateVelocity(u, v, f, g, p, flag, imax, jmax, del_t, delx, dely);
        addToCnt(updateVel_div256);

        applyBoundaryConditions(u, v, flag, imax, jmax, ui, vi);
        addToCnt(bounds_div256);
    } /* End of main loop */
    double computeEnd = wtime();
    
    write_bin(u, v, p, flag, imax, jmax, xlength, ylength, outfile);
    
    freeMatrix(u);
    freeMatrix(v);
    freeMatrix(f);
    freeMatrix(g);
    freeMatrix(p);
    freeMatrix(p_red);
    freeMatrix(p_black);
    freeMatrix(p_beta);
    freeMatrix(p_beta_red);
    freeMatrix(p_beta_black);
    freeMatrix(rhs);
    freeMatrix(rhs_red);
    freeMatrix(rhs_black);
    freeMatrix(flag);
    
    double computeTime = computeEnd - computeStart;
    
    //Print out performance metrics
    printf("==== Performance Metrics ====\n");
    printf("\tRuntime: %f seconds\n",computeTime);
    
    long flops = computeFlops(imax, jmax, iters);
    printf("\tGFLOP/s: \t%f\n", (float)flops/ 1000000000.0f / computeTime);
    
#if RECORD_FUNC_TIMES
    double tentVel = tentVelCnt_div256 * 256.0;
    double rhsTime = rhs_div256 * 256.0;
    double poisson = poisson_div256 * 256.0;
    double updateVel = updateVel_div256 * 256.0;
    double bounds = bounds_div256 * 256.0;

    double divisor = 1.0 * (1 << 23);
    
    double totalTime = tentVel + rhsTime + poisson + updateVel + bounds;

    fprintf(stderr, "==== Performance Breakdown ====\n");
    fprintf(stderr, "%-20s %-20s %-20s %-20s %-20s|%-20s\n", "tent vel", "rhs", "poisson", "update vel", "bounds", "total");
    fprintf(stderr, "%-20.2f %-20.2f %-20.2f %-20.2f %-20.2f|%-20.0f  (divided by 2^23, i.e. 8.4 million cycles)\n",
            tentVel / divisor,
            rhsTime / divisor,
            poisson / divisor,
            updateVel / divisor,
            bounds / divisor,
            totalTime);
    fprintf(stderr, "%4.1f%%%15c %4.1f%%%15c %4.1f%%%15c %4.1f%%%15c %4.1f%%%15c|%4.1f%%\n",
           tentVel*100.0/totalTime, ' ', 
           rhsTime*100.0/totalTime, ' ', 
           poisson*100.0/totalTime, ' ', 
           updateVel*100.0/totalTime, ' ', 
           bounds*100.0/totalTime, ' ', 
           totalTime*100.0/totalTime);
#endif
    
    //Compare the results
    int check = compareOutput(outfile, "target.bin");
    
    printf("==== Validating Program Output ====\n");
    if(check == 0) {
        printf("\tProgram output is " "\e[92m\e[1m" "validated" "\e[0m" ".\n");
    } else {
        printf("\tProgram output is " "\e[91m\e[1m" "incorrect" "\e[0m" ".\n");    
    }
    return 0;
}

/* Save the simulation state to a file */
void write_bin(float **u, float **v, float **p, char **flag,
    int imax, int jmax, float xlength, float ylength, char* file)
{
    int i;
    FILE *fp;

    fp = fopen(file, "wb"); 

    if (fp == NULL) {
        fprintf(stderr, "Could not open file '%s': %s\n", file,
            strerror(errno));
        return;
    }

    fwrite(&imax, sizeof(int), 1, fp);
    fwrite(&jmax, sizeof(int), 1, fp);
    fwrite(&xlength, sizeof(float), 1, fp);
    fwrite(&ylength, sizeof(float), 1, fp);

    for (i=0;i<imax+2;i++) {
        fwrite(u[i], sizeof(float), jmax+2, fp);
        fwrite(v[i], sizeof(float), jmax+2, fp);
        fwrite(p[i], sizeof(float), jmax+2, fp);
        fwrite(flag[i], sizeof(char), jmax+2, fp);
    }
    fclose(fp);
}

/* Read the simulation state from a file */
int read_bin(float **u, float **v, float **p, char **flag,
    int imax, int jmax, float xlength, float ylength, char* file)
{
    int i,j;
    FILE *fp;

    if (file == NULL) return -1;

    if ((fp = fopen(file, "rb")) == NULL) {
        fprintf(stderr, "Could not open file '%s': %s\n", file,
            strerror(errno));
        fprintf(stderr, "Generating default state instead.\n");
        return -1;
    }

    fread(&i, sizeof(int), 1, fp);
    fread(&j, sizeof(int), 1, fp);
    float xl, yl;
    fread(&xl, sizeof(float), 1, fp);
    fread(&yl, sizeof(float), 1, fp);

    if (i!=imax || j!=jmax) {
        fprintf(stderr, "Warning: imax/jmax have wrong values in %s\n", file);
        fprintf(stderr, "%s's imax = %d, jmax = %d\n", file, i, j);
        fprintf(stderr, "Program's imax = %d, jmax = %d\n", imax, jmax);
        return 1;
    }
    if (xl!=xlength || yl!=ylength) {
        fprintf(stderr, "Warning: xlength/ylength have wrong values in %s\n", file);
        fprintf(stderr, "%s's xlength = %g,  ylength = %g\n", file, xl, yl);
        fprintf(stderr, "Program's xlength = %g, ylength = %g\n", xlength,
            ylength);
        return 1;
    }

    for (i=0; i<imax+2; i++) {
        fread(u[i], sizeof(float), jmax+2, fp);
        fread(v[i], sizeof(float), jmax+2, fp);
        fread(p[i], sizeof(float), jmax+2, fp);
        fread(flag[i], sizeof(char), jmax+2, fp);
    }
    fclose(fp);
    return 0;
}

int compareOutput(char *student, char *target) 
{
    FILE *f1, *f2;
    int imax, jmax, i, j;

    float *u1, *u2, *v1, *v2, *p1, *p2;
    char *flags1, *flags2;
    float epsilon = 1e-7;

    if ((f1 = fopen(student, "rb"))  == NULL) {
        fprintf(stderr, "Could not open '%s': %s\n", student,
            strerror(errno));
        return 1;
    }
    if ((f2 = fopen(target, "rb"))  == NULL) {
        fprintf(stderr, "Could not open '%s': %s\n", target,
            strerror(errno));
        return 1;
    }

    fread(&imax, sizeof(int), 1, f1);
    fread(&jmax, sizeof(int), 1, f1);
    fread(&i, sizeof(int), 1, f2);
    fread(&j, sizeof(int), 1, f2);
    if (i != imax || j != jmax) {
        printf("Number of cells differ! (%dx%d vs %dx%d)\n", imax, jmax, i, j);
        return 1;
    }

    float xlength1, ylength1, xlength2, ylength2;
    fread(&xlength1, sizeof(float), 1, f1);
    fread(&ylength1, sizeof(float), 1, f1);
    fread(&xlength2, sizeof(float), 1, f2);
    fread(&ylength2, sizeof(float), 1, f2);
    if (xlength1 != xlength2 || ylength1 != ylength2) {
        printf("Image domain dimensions differ! (%gx%g vs %gx%g)\n",
            xlength1, ylength1, xlength2, ylength2);
        return 1;
    }

    u1 = malloc(sizeof(float) * (jmax + 2));
    u2 = malloc(sizeof(float) * (jmax + 2));
    v1 = malloc(sizeof(float) * (jmax + 2));
    v2 = malloc(sizeof(float) * (jmax + 2));
    p1 = malloc(sizeof(float) * (jmax + 2));
    p2 = malloc(sizeof(float) * (jmax + 2));
    flags1 = malloc(jmax + 2);
    flags2 = malloc(jmax + 2);
    if (!u1 || !u2 || !v1 || !v2 || !p1 || !p2 || !flags1 || !flags2) {
        fprintf(stderr, "Couldn't allocate enough memory.\n");
        return 1;
    }

    int diff_found = 0;
    for (i = 0; i < imax + 2 && !diff_found; i++) {
        fread(u1, sizeof(float), jmax + 2, f1);
        fread(v1, sizeof(float), jmax + 2, f1);
        fread(p1, sizeof(float), jmax + 2, f1);
        fread(flags1, 1, jmax + 2, f1);
        fread(u2, sizeof(float), jmax + 2, f2);
        fread(v2, sizeof(float), jmax + 2, f2);
        fread(p2, sizeof(float), jmax + 2, f2);
        fread(flags2, 1, jmax + 2, f2);
        for (j = 0; j < jmax + 2 && !diff_found; j++) {
            float du, dv, dp;
            int dflags;
            du = u1[j] - u2[j];
            dv = v1[j] - v2[j];
            dp = p1[j] - p2[j];
            dflags = flags1[j] - flags2[j];
            if(fpclassify(du) == FP_NAN ||
                fpclassify(dv) == FP_NAN ||
                fpclassify(dp) == FP_NAN ||
                fpclassify(du) == FP_INFINITE ||
                fpclassify(dv) == FP_INFINITE ||
                fpclassify(dp) == FP_INFINITE) {
                diff_found = 1;
                break;
            }
            
            if (fabs(du) > epsilon || fabs(dv) > epsilon ||
                fabs(dp) > epsilon || fabs(dflags) > epsilon) {
                diff_found = 1; 
                break;
            }   
        }
    }
    if (diff_found) {
        return 1;
    } else {
        return 0;
    }

}

long computeFlops(long imax, long jmax, long iterations) {
    
    long ctv = (78357*54) + (77816*54);
    long crhs = 78506*6;
    long uv = 78357*4 + 77816*4;
    long sti = 11;
    long abc = 406;
    long ps = 8+imax*jmax*2+2+(imax*jmax*20)+imax*jmax*16*100;
    
    return iterations * (ctv + crhs + uv +sti + abc + ps);
    
}

double wtime() {
	struct timeval t;
	gettimeofday( &t, (struct timezone *)0 );
	return t.tv_sec + t.tv_usec*1.0e-6;
}
