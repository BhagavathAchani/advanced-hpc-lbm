/*
** Code to implement a d2q9-bgk lattice boltzmann scheme.
** 'd2' inidates a 2-dimensional grid, and
** 'q9' indicates 9 velocities per grid cell.
** 'bgk' refers to the Bhatnagar-Gross-Krook collision step.
**
** The 'speeds' in each cell are numbered as follows:
**
** 6 2 5
**  \|/
** 3-0-1
**  /|\
** 7 4 8
**
** A 2D grid:
**
**           cols
**       --- --- ---
**      | D | E | F |
** rows  --- --- ---
**      | A | B | C |
**       --- --- ---
**
** 'unwrapped' in row major order to give a 1D array:
**
**  --- --- --- --- --- ---
** | A | B | C | D | E | F |
**  --- --- --- --- --- ---
**
** Grid indicies are:
**
**          ny
**          ^       cols(ii)
**          |  ----- ----- -----
**          | | ... | ... | etc |
**          |  ----- ----- -----
** rows(jj) | | 1,0 | 1,1 | 1,2 |
**          |  ----- ----- -----
**          | | 0,0 | 0,1 | 0,2 |
**          |  ----- ----- -----
**          ----------------------> nx
**
** Note the names of the input parameter and obstacle files
** are passed on the command line, e.g.:
**
**   ./d2q9-bgk input.params obstacles.dat
**
** Be sure to adjust the grid dimensions in the parameter file
** if you choose a different obstacle file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <omp.h>
#include <mpi.h>

#define NSPEEDS 9
#define FINALSTATEFILE "final_state.dat"
#define AVVELSFILE "av_vels.dat"
#define ROOT 0

/* struct to hold the parameter values */
typedef struct
{
    int nx;           /* no. of cells in x-direction */
    int ny;           /* no. of cells in y-direction */
    int maxIters;     /* no. of iterations */
    int reynolds_dim; /* dimension for Reynolds number */
    float density;    /* density per link */
    float accel;      /* density redistribution */
    float omega;      /* relaxation parameter */
} t_param;

/* struct to hold the 'speed' values */
// typedef struct
// {
//     float speeds[NSPEEDS];
// } t_speed;

typedef struct
{
    float *restrict speed_0; /* central cell, no movement */
    float *restrict speed_1; /* east */
    float *restrict speed_2; /* north */
    float *restrict speed_3; /* west */
    float *restrict speed_4; /* south */
    float *restrict speed_5; /* north-east */
    float *restrict speed_6; /* north-west */
    float *restrict speed_7; /* south-west */
    float *restrict speed_8; /* south-east */
} t_speed;

typedef struct
{
    int size;
    int rank;
    int start_row;
    int end_row;
    int top_rank;
    int bot_rank;
    int local_ny;
    int total_local_cells;
} rank_info;

/*
** function prototypes
*/

/* load params, allocate memory, load obstacles & initialise fluid particle densities */
int initialise(const char *paramfile, const char *obstaclefile,
               t_param *params, t_speed **cells_ptr, t_speed **tmp_cells_ptr,
               int **obstacles_ptr, float **av_vels_ptr, rank_info *rank_info);

/*
** The main calculation methods.
** timestep calls, in order, the functions:
** accelerate_flow(), propagate(), rebound() & collision()
*/
float timestep(const t_param params, rank_info rank_info, float *restrict speed_0, float *restrict speed_1, float *restrict speed_2, float *restrict speed_3,
               float *restrict speed_4, float *restrict speed_5, float *restrict speed_6, float *restrict speed_7, float *restrict speed_8,
               float *restrict tmp_cells_speed_0, float *restrict tmp_cells_speed_1, float *restrict tmp_cells_speed_2, float *restrict tmp_cells_speed_3,
               float *restrict tmp_cells_speed_4, float *restrict tmp_cells_speed_5, float *restrict tmp_cells_speed_6, float *restrict tmp_cells_speed_7,
               float *restrict tmp_cells_speed_8, int *obstacles);

int accelerate_flow(const t_param params, rank_info rank_info, float *restrict speed_0, float *restrict speed_1, float *restrict speed_2, float *restrict speed_3,
                    float *restrict speed_4, float *restrict speed_5, float *restrict speed_6, float *restrict speed_7, float *restrict speed_8,
                    int *obstacles);

int write_values(const t_param params, t_speed *cells, int *obstacles, float *av_vels);

/* finalise, including freeing up allocated memory */
int finalise(const t_param *params, t_speed **cells_ptr, t_speed **tmp_cells_ptr,
             int **obstacles_ptr, float **av_vels_ptr);

/* Sum all the densities in the grid.
** The total should remain constant from one timestep to the next. */
float total_density(const t_param params, t_speed *cells);

/* compute average velocity */
float av_velocity(const t_param params, t_speed *cells, int *obstacles);

/* calculate Reynolds number */
float calc_reynolds(const t_param params, t_speed *cells, int *obstacles);

/* utility functions */
void die(const char *message, const int line, const char *file);
void usage(const char *exe);

/*
** main program:
** initialise, timestep loop, finalise
*/
int main(int argc, char *argv[])
{
    // IDK yet why this
    //  MPI_Status status;

    MPI_Init(&argc, &argv);
    rank_info rank_info;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_info.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &rank_info.size);

    char *paramfile = NULL;                                                            /* name of the input parameter file */
    char *obstaclefile = NULL;                                                         /* name of a the input obstacle file */
    t_param params;                                                                    /* struct to hold parameter values */
    t_speed *cells = NULL;                                                             /* grid containing fluid densities */
    t_speed *tmp_cells = NULL;                                                         /* scratch space */
    int *obstacles = NULL;                                                             /* grid indicating which cells are blocked */
    float *av_vels = NULL;                                                             /* a record of the av. velocity computed for each timestep */
    struct timeval timstr;                                                             /* structure to hold elapsed time */
    double tot_tic, tot_toc, init_tic, init_toc, comp_tic, comp_toc, col_tic, col_toc; /* floating point numbers to calculate elapsed wallclock time */

    /* parse the command line */
    if (argc != 3)
    {
        usage(argv[0]);
    }
    else
    {
        paramfile = argv[1];
        obstaclefile = argv[2];
    }

    /* Total/init time starts here: initialise our data structures and load values from file */
    gettimeofday(&timstr, NULL);
    tot_tic = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    init_tic = tot_tic;
    initialise(paramfile, obstaclefile, &params, &cells, &tmp_cells, &obstacles, &av_vels, &rank_info);

    /* Init time stops here, compute time starts*/
    gettimeofday(&timstr, NULL);
    init_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    comp_tic = init_toc;

    for (int tt = 0; tt < params.maxIters; tt++)
    {
        av_vels[tt] = timestep(params, rank_info, cells->speed_0, cells->speed_1, cells->speed_2, cells->speed_3, cells->speed_4, cells->speed_5, cells->speed_6, cells->speed_7, cells->speed_8,
                               tmp_cells->speed_0, tmp_cells->speed_1, tmp_cells->speed_2, tmp_cells->speed_3, tmp_cells->speed_4,
                               tmp_cells->speed_5, tmp_cells->speed_6, tmp_cells->speed_7, tmp_cells->speed_8, obstacles);

        // Swap pointers
        t_speed *temp = cells;
        cells = tmp_cells;
        tmp_cells = temp;

        // av_vels[tt] = av_velocity(params, cells, obstacles);
#ifdef DEBUG
        printf("==timestep: %d==\n", tt);
        printf("av velocity: %.12E\n", av_vels[tt]);
        printf("tot density: %.12E\n", total_density(params, cells));
#endif
    }

    /* Compute time stops here, collate time starts*/
    gettimeofday(&timstr, NULL);
    comp_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    col_tic = comp_toc;

    // Collate data from ranks here
    // Allocate gathered data structures on root
    t_speed *gathered_cells = NULL;
    int *gathered_obstacles = NULL;

    if (rank_info.rank == ROOT)
    {
        // Allocate full grid for gathered data
        gathered_cells = malloc(sizeof(t_speed));
        gathered_cells->speed_0 = malloc(params.nx * params.ny * sizeof(float));
        gathered_cells->speed_1 = malloc(params.nx * params.ny * sizeof(float));
        gathered_cells->speed_2 = malloc(params.nx * params.ny * sizeof(float));
        gathered_cells->speed_3 = malloc(params.nx * params.ny * sizeof(float));
        gathered_cells->speed_4 = malloc(params.nx * params.ny * sizeof(float));
        gathered_cells->speed_5 = malloc(params.nx * params.ny * sizeof(float));
        gathered_cells->speed_6 = malloc(params.nx * params.ny * sizeof(float));
        gathered_cells->speed_7 = malloc(params.nx * params.ny * sizeof(float));
        gathered_cells->speed_8 = malloc(params.nx * params.ny * sizeof(float));
        gathered_obstacles = malloc(params.nx * params.ny * sizeof(int));
    }

    // Prepare counts and displacements for Gatherv
    int *recv_counts = malloc(rank_info.size * sizeof(int));
    int *displs = malloc(rank_info.size * sizeof(int));

    // Calculate counts and displacements for each rank
    int offset = 0;
    for (int rank = 0; rank < rank_info.size; rank++)
    {
        int local_rows = params.ny / rank_info.size + (rank < (params.ny % rank_info.size) ? 1 : 0);
        recv_counts[rank] = local_rows * params.nx;
        displs[rank] = offset;
        offset += recv_counts[rank];
    }

    // Gather obstacle data
    MPI_Gatherv(&obstacles[params.nx], // Skip halo row
                rank_info.local_ny * params.nx,
                MPI_INT,
                gathered_obstacles,
                recv_counts,
                displs,
                MPI_INT,
                ROOT,
                MPI_COMM_WORLD);

    // Gather each speed component
    float *speed_ptrs[NSPEEDS] = {
        cells->speed_0, cells->speed_1, cells->speed_2, cells->speed_3, cells->speed_4,
        cells->speed_5, cells->speed_6, cells->speed_7, cells->speed_8};

    float *gathered_speed_ptrs[NSPEEDS] = {
        gathered_cells ? gathered_cells->speed_0 : NULL,
        gathered_cells ? gathered_cells->speed_1 : NULL,
        gathered_cells ? gathered_cells->speed_2 : NULL,
        gathered_cells ? gathered_cells->speed_3 : NULL,
        gathered_cells ? gathered_cells->speed_4 : NULL,
        gathered_cells ? gathered_cells->speed_5 : NULL,
        gathered_cells ? gathered_cells->speed_6 : NULL,
        gathered_cells ? gathered_cells->speed_7 : NULL,
        gathered_cells ? gathered_cells->speed_8 : NULL};

    for (int s = 0; s < NSPEEDS; s++)
    {
        MPI_Gatherv(&speed_ptrs[s][params.nx], // Skip halo row
                    rank_info.local_ny * params.nx,
                    MPI_FLOAT,
                    gathered_speed_ptrs[s],
                    recv_counts,
                    displs,
                    MPI_FLOAT,
                    ROOT,
                    MPI_COMM_WORLD);
    }

    free(recv_counts);
    free(displs);

    /* Total/collate time stops here.*/
    gettimeofday(&timstr, NULL);
    col_toc = timstr.tv_sec + (timstr.tv_usec / 1000000.0);
    tot_toc = col_toc;

    /* write final values and free memory */
    if (rank_info.rank == ROOT)
    {
        printf("==done==\n");
        printf("Reynolds number:\t\t%.12E\n", calc_reynolds(params, gathered_cells, gathered_obstacles));
        printf("Elapsed Init time:\t\t\t%.6lf (s)\n", init_toc - init_tic);
        printf("Elapsed Compute time:\t\t\t%.6lf (s)\n", comp_toc - comp_tic);
        printf("Elapsed Collate time:\t\t\t%.6lf (s)\n", col_toc - col_tic);
        printf("Elapsed Total time:\t\t\t%.6lf (s)\n", tot_toc - tot_tic);
        write_values(params, gathered_cells, gathered_obstacles, av_vels);
    }

    // Free gathered data if it exists
    if (gathered_cells)
    {
        free(gathered_cells->speed_0);
        free(gathered_cells->speed_1);
        free(gathered_cells->speed_2);
        free(gathered_cells->speed_3);
        free(gathered_cells->speed_4);
        free(gathered_cells->speed_5);
        free(gathered_cells->speed_6);
        free(gathered_cells->speed_7);
        free(gathered_cells->speed_8);
        free(gathered_cells);
    }
    free(gathered_obstacles);

    finalise(&params, &cells, &tmp_cells, &obstacles, &av_vels);

    MPI_Finalize();
    return EXIT_SUCCESS;
}

float timestep(const t_param params, rank_info rank_info, float *restrict speed_0, float *restrict speed_1, float *restrict speed_2, float *restrict speed_3,
               float *restrict speed_4, float *restrict speed_5, float *restrict speed_6, float *restrict speed_7, float *restrict speed_8,
               float *restrict tmp_cells_speed_0, float *restrict tmp_cells_speed_1, float *restrict tmp_cells_speed_2, float *restrict tmp_cells_speed_3,
               float *restrict tmp_cells_speed_4, float *restrict tmp_cells_speed_5, float *restrict tmp_cells_speed_6, float *restrict tmp_cells_speed_7,
               float *restrict tmp_cells_speed_8, int *obstacles)
{

    int local_ny = rank_info.local_ny;
    int nx = params.nx;

    int top_row = 1;
    int bottom_row = local_ny;
    int top_halo = 0;
    int bottom_halo = local_ny + 1;

    float *speeds[9] = {speed_0, speed_1, speed_2, speed_3, speed_4, speed_5, speed_6, speed_7, speed_8};

    MPI_Request reqs[36];
    int req_count = 0;

    for (int i = 0; i < 9; i++)
    {
        int tag_up = i + 10 * 0;
        int tag_down = i + 10 * 1;

        // Send top row to top_rank, receive top halo from top_rank
        MPI_Isend(&speeds[i][top_row * nx], nx, MPI_FLOAT, rank_info.top_rank, tag_up, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Irecv(&speeds[i][top_halo * nx], nx, MPI_FLOAT, rank_info.top_rank, tag_down, MPI_COMM_WORLD, &reqs[req_count++]);

        // Send bottom row to bottom_rank, receive bottom halo from bottom_rank
        MPI_Isend(&speeds[i][bottom_row * nx], nx, MPI_FLOAT, rank_info.bot_rank, tag_down, MPI_COMM_WORLD, &reqs[req_count++]);
        MPI_Irecv(&speeds[i][bottom_halo * nx], nx, MPI_FLOAT, rank_info.bot_rank, tag_up, MPI_COMM_WORLD, &reqs[req_count++]);
    }

    MPI_Waitall(req_count, reqs, MPI_STATUSES_IGNORE);

    accelerate_flow(params, rank_info, speed_0, speed_1, speed_2, speed_3, speed_4, speed_5, speed_6, speed_7, speed_8, obstacles);
    // propagate(params, cells, tmp_cells);
    // rebound(params, cells, tmp_cells, obstacles);
    // collision(params, cells, tmp_cells, obstacles);

    // const float c_sq = 1.f / 3.f; /* square of speed of sound */
    const float w0 = 4.f / 9.f;  /* weighting factor */
    const float w1 = 1.f / 9.f;  /* weighting factor */
    const float w2 = 1.f / 36.f; /* weighting factor */

    const float inv_c_sq = 3.f;
    const float inv_2_c_sq = 0.5f * inv_c_sq;
    const float inv_c_sq_sq_half = 0.5f * inv_c_sq * inv_c_sq;

    int tot_cells = 0; /* no. of cells used in calculation */
    float tot_u = 0.f; /* accumulated magnitudes of velocity for each cell */
    int final_tot_cells = 0;
    float final_tot_u = 0.f;

    // #pragma omp parallel for reduction(+ : tot_u, tot_cells)
    for (int jj = 1; jj < rank_info.local_ny + 1; jj++)
    {
        int y_n = (jj + 1) % params.ny;
        int y_s = (jj == 0) ? (jj + params.ny - 1) : (jj - 1);

        // #pragma omp simd reduction(+ : tot_u, tot_cells) aligned(speed_0, speed_1, speed_2, speed_3, speed_4, speed_5, speed_6, speed_7, speed_8, \
//                                                              tmp_cells_speed_0, tmp_cells_speed_1, tmp_cells_speed_2, tmp_cells_speed_3,  \
//                                                              tmp_cells_speed_4, tmp_cells_speed_5, tmp_cells_speed_6, tmp_cells_speed_7,  \
//                                                              tmp_cells_speed_8 : 64) simdlen(8)
        for (int ii = 0; ii < params.nx; ii++)
        {
            /* determine indices of axis-direction neighbours
            ** respecting periodic boundary conditions (wrap around) */
            int x_e = (ii + 1) % params.nx;
            int x_w = (ii == 0) ? (ii + params.nx - 1) : (ii - 1);
            int index = ii + jj * params.nx;

            /* propagate densities from neighbouring cells, following
            ** appropriate directions of travel and writing into
            ** scratch space grid */

            float speed0 = speed_0[index];
            float speed1 = speed_1[x_w + jj * params.nx];
            float speed2 = speed_2[ii + y_s * params.nx];
            float speed3 = speed_3[x_e + jj * params.nx];
            float speed4 = speed_4[ii + y_n * params.nx];
            float speed5 = speed_5[x_w + y_s * params.nx];
            float speed6 = speed_6[x_e + y_s * params.nx];
            float speed7 = speed_7[x_e + y_n * params.nx];
            float speed8 = speed_8[x_w + y_n * params.nx];

            float is_obstacle = (float)(obstacles[index] != 0);
            float is_fluid = 1.0f - is_obstacle;

            /* compute local density total */
            float local_density = speed0 + speed1 + speed2 + speed3 + speed4 + speed5 + speed6 + speed7 + speed8;
            float inv_local_density = 1.f / local_density;

            /* compute x velocity component */
            float u_x = (speed1 + speed5 + speed8 - (speed3 + speed6 + speed7)) * inv_local_density;
            float u_y = (speed2 + speed5 + speed6 - (speed4 + speed7 + speed8)) * inv_local_density;
            float u_sq = u_x * u_x + u_y * u_y;
            float constant = u_sq * inv_2_c_sq;

            tot_u += is_fluid * sqrtf((u_x * u_x) + (u_y * u_y));
            tot_cells += (int)is_fluid;

            /* directional velocity components */
            float u[NSPEEDS];
            u[1] = u_x;        /* east */
            u[2] = u_y;        /* north */
            u[3] = -u_x;       /* west */
            u[4] = -u_y;       /* south */
            u[5] = u_x + u_y;  /* north-east */
            u[6] = -u_x + u_y; /* north-west */
            u[7] = -u_x - u_y; /* south-west */
            u[8] = u_x - u_y;  /* south-east */

            /* equilibrium densities */
            float d_equ[NSPEEDS];
            /* zero velocity density: weight w0 */
            d_equ[0] = w0 * local_density * (1.f - constant);

            /* axis speeds: weight w1 */
            d_equ[1] = w1 * local_density * (1.f + u[1] * inv_c_sq + (u[1] * u[1]) * inv_c_sq_sq_half - constant);
            d_equ[2] = w1 * local_density * (1.f + u[2] * inv_c_sq + (u[2] * u[2]) * inv_c_sq_sq_half - constant);
            d_equ[3] = w1 * local_density * (1.f + u[3] * inv_c_sq + (u[3] * u[3]) * inv_c_sq_sq_half - constant);
            d_equ[4] = w1 * local_density * (1.f + u[4] * inv_c_sq + (u[4] * u[4]) * inv_c_sq_sq_half - constant);
            /* diagonal speeds: weight w2 */
            d_equ[5] = w2 * local_density * (1.f + u[5] * inv_c_sq + (u[5] * u[5]) * inv_c_sq_sq_half - constant);
            d_equ[6] = w2 * local_density * (1.f + u[6] * inv_c_sq + (u[6] * u[6]) * inv_c_sq_sq_half - constant);
            d_equ[7] = w2 * local_density * (1.f + u[7] * inv_c_sq + (u[7] * u[7]) * inv_c_sq_sq_half - constant);
            d_equ[8] = w2 * local_density * (1.f + u[8] * inv_c_sq + (u[8] * u[8]) * inv_c_sq_sq_half - constant);

            /* relaxation step with mask-based blending between obstacle and fluid behavior */
            tmp_cells_speed_0[index] = is_obstacle * speed0 + is_fluid * (speed0 + params.omega * (d_equ[0] - speed0));
            tmp_cells_speed_1[index] = is_obstacle * speed3 + is_fluid * (speed1 + params.omega * (d_equ[1] - speed1));
            tmp_cells_speed_2[index] = is_obstacle * speed4 + is_fluid * (speed2 + params.omega * (d_equ[2] - speed2));
            tmp_cells_speed_3[index] = is_obstacle * speed1 + is_fluid * (speed3 + params.omega * (d_equ[3] - speed3));
            tmp_cells_speed_4[index] = is_obstacle * speed2 + is_fluid * (speed4 + params.omega * (d_equ[4] - speed4));
            tmp_cells_speed_5[index] = is_obstacle * speed7 + is_fluid * (speed5 + params.omega * (d_equ[5] - speed5));
            tmp_cells_speed_6[index] = is_obstacle * speed8 + is_fluid * (speed6 + params.omega * (d_equ[6] - speed6));
            tmp_cells_speed_7[index] = is_obstacle * speed5 + is_fluid * (speed7 + params.omega * (d_equ[7] - speed7));
            tmp_cells_speed_8[index] = is_obstacle * speed6 + is_fluid * (speed8 + params.omega * (d_equ[8] - speed8));
        }
    }

    MPI_Reduce(&tot_u, &final_tot_u, 1, MPI_FLOAT, MPI_SUM, ROOT, MPI_COMM_WORLD);
    MPI_Reduce(&tot_cells, &final_tot_cells, 1, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);

    return final_tot_u / (float)final_tot_cells;
}

int accelerate_flow(const t_param params, rank_info rank_info, float *restrict speed_0, float *restrict speed_1, float *restrict speed_2, float *restrict speed_3,
                    float *restrict speed_4, float *restrict speed_5, float *restrict speed_6, float *restrict speed_7, float *restrict speed_8,
                    int *restrict obstacles)
{
    /* compute weighting factors */
    float w1 = params.density * params.accel / 9.f;
    float w2 = params.density * params.accel / 36.f;

    /* modify the 2nd row of the grid */
    int jj = params.ny - 2;

    if (jj >= rank_info.start_row && jj <= rank_info.end_row)
    {
        int target = jj - rank_info.start_row + 1;

        // #pragma omp simd aligned(speed_0, speed_1, speed_2, speed_3, speed_4, speed_5, speed_6, speed_7, speed_8 : 64)
        for (int ii = 0; ii < params.nx; ii++)
        {
            int index = ii + target * params.nx;
            /* if the cell is not occupied and
            ** we don't send a negative density */
            if (!obstacles[index] && (speed_3[index] - w1) > 0.f && (speed_6[index] - w2) > 0.f && (speed_7[index] - w2) > 0.f)
            {
                /* increase 'east-side' densities */
                speed_1[index] += w1;
                speed_5[index] += w2;
                speed_8[index] += w2;
                /* decrease 'west-side' densities */
                speed_3[index] -= w1;
                speed_6[index] -= w2;
                speed_7[index] -= w2;
            }
        }
    }
    return EXIT_SUCCESS;
}

float av_velocity(const t_param params, t_speed *cells, int *obstacles)
{
    int tot_cells = 0; /* no. of cells used in calculation */
    float tot_u;       /* accumulated magnitudes of velocity for each cell */

    /* initialise */
    tot_u = 0.f;
    /* loop over all non-blocked cells */
    // #pragma omp for
    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            /* ignore occupied cells */
            if (!obstacles[ii + jj * params.nx])
            {
                int index = ii + jj * params.nx;
                /* local density total */
                float local_density = 0.f;
                local_density += cells->speed_0[index]; // speed[0]
                local_density += cells->speed_1[index]; // speed[1]
                local_density += cells->speed_2[index]; // speed[2]
                local_density += cells->speed_3[index]; // speed[3]
                local_density += cells->speed_4[index]; // speed[4]
                local_density += cells->speed_5[index]; // speed[5]
                local_density += cells->speed_6[index]; // speed[6]
                local_density += cells->speed_7[index]; // speed[7]
                local_density += cells->speed_8[index]; // speed[8]

                /* x-component of velocity */
                float u_x = (cells->speed_1[index] + cells->speed_5[index] + cells->speed_8[index] - (cells->speed_3[index] + cells->speed_6[index] + cells->speed_7[index])) / local_density;
                /* compute y velocity component */
                float u_y = (cells->speed_2[index] + cells->speed_5[index] + cells->speed_6[index] - (cells->speed_4[index] + cells->speed_7[index] + cells->speed_8[index])) / local_density;
                /* accumulate the norm of x- and y- velocity components */
                tot_u += sqrtf((u_x * u_x) + (u_y * u_y));
                /* increase counter of inspected cells */
                ++tot_cells;
            }
        }
    }

    return tot_u / (float)tot_cells;
}

int initialise(const char *paramfile, const char *obstaclefile,
               t_param *params, t_speed **cells_ptr, t_speed **tmp_cells_ptr,
               int **obstacles_ptr, float **av_vels_ptr, rank_info *rank_info)
{
    char message[1024]; /* message buffer */
    FILE *fp;           /* file pointer */
    int xx, yy;         /* generic array indices */
    int blocked;        /* indicates whether a cell is blocked by an obstacle */
    int retval;         /* to hold return value for checking */

    /* open the parameter file */
    fp = fopen(paramfile, "r");

    if (fp == NULL)
    {
        sprintf(message, "could not open input parameter file: %s", paramfile);
        die(message, __LINE__, __FILE__);
    }

    /* read in the parameter values */
    retval = fscanf(fp, "%d\n", &(params->nx));

    if (retval != 1)
        die("could not read param file: nx", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->ny));

    if (retval != 1)
        die("could not read param file: ny", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->maxIters));

    if (retval != 1)
        die("could not read param file: maxIters", __LINE__, __FILE__);

    retval = fscanf(fp, "%d\n", &(params->reynolds_dim));

    if (retval != 1)
        die("could not read param file: reynolds_dim", __LINE__, __FILE__);

    retval = fscanf(fp, "%f\n", &(params->density));

    if (retval != 1)
        die("could not read param file: density", __LINE__, __FILE__);

    retval = fscanf(fp, "%f\n", &(params->accel));

    if (retval != 1)
        die("could not read param file: accel", __LINE__, __FILE__);

    retval = fscanf(fp, "%f\n", &(params->omega));

    if (retval != 1)
        die("could not read param file: omega", __LINE__, __FILE__);

    /* and close up the file */
    fclose(fp);

    /*
    ** Allocate memory.
    **
    ** Remember C is pass-by-value, so we need to
    ** pass pointers into the initialise function.
    **
    ** NB we are allocating a 1D array, so that the
    ** memory will be contiguous.  We still want to
    ** index this memory as if it were a (row major
    ** ordered) 2D array, however.  We will perform
    ** some arithmetic using the row and column
    ** coordinates, inside the square brackets, when
    ** we want to access elements of this array.
    **
    ** Note also that we are using a structure to
    ** hold an array of 'speeds'.  We will allocate
    ** a 1D array of these structs.
    */

    int rows_per_rank = params->ny / rank_info->size;
    int extra_rows = params->ny % rank_info->size;

    // local number of rows
    rank_info->local_ny = rows_per_rank + (rank_info->rank < extra_rows ? 1 : 0);

    // Math makes sense. Tested
    rank_info->start_row = rank_info->rank * rows_per_rank + (rank_info->rank < extra_rows ? rank_info->rank : extra_rows);
    rank_info->end_row = rank_info->start_row + rank_info->local_ny - 1;

    rank_info->top_rank = (rank_info->rank == 0) ? (rank_info->size - 1) : (rank_info->rank - 1);
    rank_info->bot_rank = (rank_info->rank == rank_info->size - 1) ? 0 : (rank_info->rank + 1);

    // local rows plus 2 halo rows
    rank_info->total_local_cells = params->nx * (rank_info->local_ny + 2);

    /* main grid */
    *cells_ptr = (t_speed *)malloc(sizeof(t_speed) * (rank_info->total_local_cells));

    if (*cells_ptr == NULL)
        die("cannot allocate memory for cells", __LINE__, __FILE__);

    /* 'helper' grid, used as scratch space */
    *tmp_cells_ptr = (t_speed *)malloc(sizeof(t_speed) * (rank_info->total_local_cells));

    if (*tmp_cells_ptr == NULL)
        die("cannot allocate memory for tmp_cells", __LINE__, __FILE__);

    /* the map of obstacles */
    *obstacles_ptr = malloc(sizeof(int) * (rank_info->total_local_cells));

    if (*obstacles_ptr == NULL)
        die("cannot allocate column memory for obstacles", __LINE__, __FILE__);

    (*cells_ptr)->speed_0 = (float *)aligned_alloc(64, sizeof(float) * (rank_info->total_local_cells));
    (*cells_ptr)->speed_1 = (float *)aligned_alloc(64, sizeof(float) * (rank_info->total_local_cells));
    (*cells_ptr)->speed_2 = (float *)aligned_alloc(64, sizeof(float) * (rank_info->total_local_cells));
    (*cells_ptr)->speed_3 = (float *)aligned_alloc(64, sizeof(float) * (rank_info->total_local_cells));
    (*cells_ptr)->speed_4 = (float *)aligned_alloc(64, sizeof(float) * (rank_info->total_local_cells));
    (*cells_ptr)->speed_5 = (float *)aligned_alloc(64, sizeof(float) * (rank_info->total_local_cells));
    (*cells_ptr)->speed_6 = (float *)aligned_alloc(64, sizeof(float) * (rank_info->total_local_cells));
    (*cells_ptr)->speed_7 = (float *)aligned_alloc(64, sizeof(float) * (rank_info->total_local_cells));
    (*cells_ptr)->speed_8 = (float *)aligned_alloc(64, sizeof(float) * (rank_info->total_local_cells));

    (*tmp_cells_ptr)->speed_0 = (float *)aligned_alloc(64, sizeof(float) * (rank_info->total_local_cells));
    (*tmp_cells_ptr)->speed_1 = (float *)aligned_alloc(64, sizeof(float) * (rank_info->total_local_cells));
    (*tmp_cells_ptr)->speed_2 = (float *)aligned_alloc(64, sizeof(float) * (rank_info->total_local_cells));
    (*tmp_cells_ptr)->speed_3 = (float *)aligned_alloc(64, sizeof(float) * (rank_info->total_local_cells));
    (*tmp_cells_ptr)->speed_4 = (float *)aligned_alloc(64, sizeof(float) * (rank_info->total_local_cells));
    (*tmp_cells_ptr)->speed_5 = (float *)aligned_alloc(64, sizeof(float) * (rank_info->total_local_cells));
    (*tmp_cells_ptr)->speed_6 = (float *)aligned_alloc(64, sizeof(float) * (rank_info->total_local_cells));
    (*tmp_cells_ptr)->speed_7 = (float *)aligned_alloc(64, sizeof(float) * (rank_info->total_local_cells));
    (*tmp_cells_ptr)->speed_8 = (float *)aligned_alloc(64, sizeof(float) * (rank_info->total_local_cells));

    /* initialise densities */
    float w0 = params->density * 4.f / 9.f;
    float w1 = params->density / 9.f;
    float w2 = params->density / 36.f;

    for (int jj = 1; jj < rank_info->local_ny + 1; jj++)
    {
        // #pragma omp simd
        for (int ii = 0; ii < params->nx; ii++)
        {
            int index = ii + jj * params->nx;
            /* centre */
            (*cells_ptr)->speed_0[index] = w0;
            /* axis dire->speed_0ctions */
            (*cells_ptr)->speed_1[index] = w1;
            (*cells_ptr)->speed_2[index] = w1;
            (*cells_ptr)->speed_3[index] = w1;
            (*cells_ptr)->speed_4[index] = w1;
            /* diagonals->speed_0 */
            (*cells_ptr)->speed_5[index] = w2;
            (*cells_ptr)->speed_6[index] = w2;
            (*cells_ptr)->speed_7[index] = w2;
            (*cells_ptr)->speed_8[index] = w2;
            (*obstacles_ptr)[ii + jj * params->nx] = 0;
        }
    }

    /* open the obstacle data file */
    fp = fopen(obstaclefile, "r");

    if (fp == NULL)
    {
        sprintf(message, "could not open input obstacles file: %s", obstaclefile);
        die(message, __LINE__, __FILE__);
    }

    /* read-in the blocked cells list */
    while ((retval = fscanf(fp, "%d %d %d\n", &xx, &yy, &blocked)) != EOF)
    {
        /* some checks */
        if (retval != 3)
            die("expected 3 values per line in obstacle file", __LINE__, __FILE__);

        if (xx < 0 || xx > params->nx - 1)
            die("obstacle x-coord out of range", __LINE__, __FILE__);

        if (yy < 0 || yy > params->ny - 1)
            die("obstacle y-coord out of range", __LINE__, __FILE__);

        if (blocked != 1)
            die("obstacle blocked value should be 1", __LINE__, __FILE__);

        /* Only assign to this rank's array if the obstacle is in this rank's domain */
        /* Convert from global y-coordinate to local coordinate */
        if (yy >= rank_info->start_row && yy <= rank_info->end_row)
        {
            int local_y = yy - rank_info->start_row + 1; // +1 for halo
            (*obstacles_ptr)[xx + local_y * params->nx] = blocked;
        }
    }

    /* and close the file */
    fclose(fp);

    /*
    ** allocate space to hold a record of the avarage velocities computed
    ** at each timestep
    */
    *av_vels_ptr = (float *)malloc(sizeof(float) * params->maxIters);

    return EXIT_SUCCESS;
}

int finalise(const t_param *params, t_speed **cells_ptr, t_speed **tmp_cells_ptr,
             int **obstacles_ptr, float **av_vels_ptr)
{
    /*
    ** free up allocated memory
    */
    free((*cells_ptr)->speed_0);
    free((*cells_ptr)->speed_1);
    free((*cells_ptr)->speed_2);
    free((*cells_ptr)->speed_3);
    free((*cells_ptr)->speed_4);
    free((*cells_ptr)->speed_5);
    free((*cells_ptr)->speed_6);
    free((*cells_ptr)->speed_7);
    free((*cells_ptr)->speed_8);

    free((*tmp_cells_ptr)->speed_0);
    free((*tmp_cells_ptr)->speed_1);
    free((*tmp_cells_ptr)->speed_2);
    free((*tmp_cells_ptr)->speed_3);
    free((*tmp_cells_ptr)->speed_4);
    free((*tmp_cells_ptr)->speed_5);
    free((*tmp_cells_ptr)->speed_6);
    free((*tmp_cells_ptr)->speed_7);
    free((*tmp_cells_ptr)->speed_8);

    free(*cells_ptr);
    *cells_ptr = NULL;

    free(*tmp_cells_ptr);
    *tmp_cells_ptr = NULL;

    free(*obstacles_ptr);
    *obstacles_ptr = NULL;

    free(*av_vels_ptr);
    *av_vels_ptr = NULL;

    return EXIT_SUCCESS;
}

float calc_reynolds(const t_param params, t_speed *cells, int *obstacles)
{
    const float viscosity = 1.f / 6.f * (2.f / params.omega - 1.f);

    return av_velocity(params, cells, obstacles) * params.reynolds_dim / viscosity;
}

float total_density(const t_param params, t_speed *cells)
{
    float total = 0.f; /* accumulator */

    for (int jj = 0; jj < params.ny; jj++)
    {
        // #pragma omp simd
        for (int ii = 0; ii < params.nx; ii++)
        {
            int index = ii + jj * params.nx;
            // Sum all speed components for the current cell
            total += cells->speed_0[index]; // speed[0]
            total += cells->speed_1[index]; // speed[1]
            total += cells->speed_2[index]; // speed[2]
            total += cells->speed_3[index]; // speed[3]
            total += cells->speed_4[index]; // speed[4]
            total += cells->speed_5[index]; // speed[5]
            total += cells->speed_6[index]; // speed[6]
            total += cells->speed_7[index]; // speed[7]
            total += cells->speed_8[index]; // speed[8]
        }
    }

    return total;
}

int write_values(const t_param params, t_speed *cells, int *obstacles, float *av_vels)
{
    FILE *fp;                     /* file pointer */
    const float c_sq = 1.f / 3.f; /* sq. of speed of sound */
    float local_density;          /* per grid cell sum of densities */
    float pressure;               /* fluid pressure in grid cell */
    float u_x;                    /* x-component of velocity in grid cell */
    float u_y;                    /* y-component of velocity in grid cell */
    float u;                      /* norm--root of summed squares--of u_x and u_y */

    fp = fopen(FINALSTATEFILE, "w");

    if (fp == NULL)
    {
        die("could not open file output file", __LINE__, __FILE__);
    }

    for (int jj = 0; jj < params.ny; jj++)
    {
        for (int ii = 0; ii < params.nx; ii++)
        {
            int index = ii + jj * params.nx;
            /* an occupied cell */
            if (obstacles[index])
            {
                u_x = u_y = u = 0.f;
                pressure = params.density * c_sq;
            }
            /* no obstacle */
            else
            {
                local_density = 0.f;
                local_density += cells->speed_0[index]; // speed[0]
                local_density += cells->speed_1[index]; // speed[1]
                local_density += cells->speed_2[index]; // speed[2]
                local_density += cells->speed_3[index]; // speed[3]
                local_density += cells->speed_4[index]; // speed[4]
                local_density += cells->speed_5[index]; // speed[5]
                local_density += cells->speed_6[index]; // speed[6]
                local_density += cells->speed_7[index]; // speed[7]
                local_density += cells->speed_8[index]; // speed[8]

                /* compute x velocity component */
                u_x = (cells->speed_1[index] + cells->speed_5[index] + cells->speed_8[index] - (cells->speed_3[index] + cells->speed_6[index] + cells->speed_7[index])) / local_density;
                /* compute y velocity component */
                u_y = (cells->speed_2[index] + cells->speed_5[index] + cells->speed_6[index] - (cells->speed_4[index] + cells->speed_7[index] + cells->speed_8[index])) / local_density;
                /* compute norm of velocity */
                u = sqrtf((u_x * u_x) + (u_y * u_y));
                /* compute pressure */
                pressure = local_density * c_sq;
            }

            /* write to file */
            fprintf(fp, "%d %d %.12E %.12E %.12E %.12E %d\n", ii, jj, u_x, u_y, u, pressure, obstacles[ii + params.nx * jj]);
        }
    }

    fclose(fp);

    fp = fopen(AVVELSFILE, "w");

    if (fp == NULL)
    {
        die("could not open file output file", __LINE__, __FILE__);
    }

    for (int ii = 0; ii < params.maxIters; ii++)
    {
        fprintf(fp, "%d:\t%.12E\n", ii, av_vels[ii]);
    }

    fclose(fp);

    return EXIT_SUCCESS;
}

void die(const char *message, const int line, const char *file)
{
    fprintf(stderr, "Error at line %d of file %s:\n", line, file);
    fprintf(stderr, "%s\n", message);
    fflush(stderr);
    exit(EXIT_FAILURE);
}

void usage(const char *exe)
{
    fprintf(stderr, "Usage: %s <paramfile> <obstaclefile>\n", exe);
    exit(EXIT_FAILURE);
}