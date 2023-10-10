#ifndef DEF_PARTICULEAOS_KERNEL
#define DEF_PARTICULEAOS_KERNEL
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "particuleAoS.cuh"
#include "cell.h"

__global__ void forceOnPoint_global(float* p_x, float* p_y, float* p_vx, float* p_vy, int nbParticule, float dt, int x, int y, float intensite);
__global__ void force_global(float* p_x, float* p_y, float* p_lastx, float* p_lasty, float* p_vx, float* p_vy, int nbParticule, float dt);
__global__ void CalcPosition_global(float* p_x, float* p_y, float* p_lastx, float* p_lasty, float* p_vx, float* p_vy, int nbParticule, float dt);
__global__ void toCell_global(float* p_x, float* p_y, cell* p_cell, int nbParticule, int SIZECASEX, int SIZECASEY, int CASEMAXX, int CASEMAXY);
//__global__ void collision_global(float* p_x, float* p_y, int* p_radius, cell* p_cell, int nbParticule, int** dev_grilleP2D, int* dev_sizeTabs, int CASEMAXX, int CASEMAXY, int width, int height);
__global__ void borderCollision_global(float* p_x, float* p_y, int* p_radius, int width, int height, int nbParticule);
#endif