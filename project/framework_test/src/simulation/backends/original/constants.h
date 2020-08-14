#define C_B      0x0000u   /* This cell is an obstacle/boundary cell */
#define B_N      0x0001u   /* This obstacle cell has a fluid cell to the north */
#define B_S      0x0002u   /* This obstacle cell has a fluid cell to the south */
#define B_W      0x0004u   /* This obstacle cell has a fluid cell to the west */
#define B_E      0x0008u   /* This obstacle cell has a fluid cell to the east */
#define B_NW     (B_N | B_W)
#define B_SW     (B_S | B_W)
#define B_NE     (B_N | B_E)
#define B_SE     (B_S | B_E)
#define B_NSEW   (B_N | B_S | B_E | B_W)

#define C_F      0x0010u   /* This cell is a fluid cell */

/* Macros for poisson(), denoting whether there is an obstacle cell
 * adjacent to some direction
 */
#define eps_E ((flag[i+1][j] & C_F)?1:0)
#define eps_W ((flag[i-1][j] & C_F)?1:0)
#define eps_N ((flag[i][j+1] & C_F)?1:0)
#define eps_S ((flag[i][j-1] & C_F)?1:0)

//#define fluid_E_mask(X) ((float)(fluidmask[i+1][j] & (*(int*)&X) ))
//#define fluid_W_mask(X) ((float)(fluidmask[i-1][j] & (*(int*)&X) ))
//#define fluid_N_mask(X) ((float)(fluidmask[i][j+1] & (*(int*)&X) ))
//#define fluid_S_mask(X) ((float)(fluidmask[i][j-1] & (*(int*)&X) ))

#define fluid_E_mask(X) ((fluidmask[i+1][j]?X:0.0f))
#define fluid_W_mask(X) ((fluidmask[i-1][j]?X:0.0f))
#define fluid_N_mask(X) ((fluidmask[i][j+1]?X:0.0f))
#define fluid_S_mask(X) ((fluidmask[i][j-1]?X:0.0f))

#define eps_fromB_E ((flag[i][j] & B_E)?0:1)
#define eps_fromB_W ((flag[i][j] & B_W)?0:1)
#define eps_fromB_N ((flag[i][j] & B_N)?0:1)
#define eps_fromB_S ((flag[i][j] & B_S)?0:1)
