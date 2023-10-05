#ifndef DEF_PARTICULE_KERNEL
#define DEF_PARTICULE_KERNEL
#include <SDL.h>
#include <SDL_ttf.h>
#include <vector>
#include <math.h>
#include <iostream>
#include <string>
#include <atomic>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Particule.cuh"



__global__ void ApplyForce(float dt, int nbparts, Particule* dev_parts);
__global__ void ApplyForceOnPoint(float dt, int nbparts, Particule* dev_parts, int x, int y, int intensite);
__global__ void calcPos(float dt, int nbparts, Particule* dev_parts);
__global__ void borderCollision(int nbparts, Particule* dev_parts, int width, int height);
void borderCollisionCall(int nbparts, Particule* dev_parts, int width, int height);
__global__ void setColor(int nbparts, Particule* dev_parts, uint32_t color);
void setColorCall(int nbparts, Particule* dev_parts, uint32_t color);
__global__ void collision(int nbparts, Particule* dev_parts, int** dev_grilleP2D, int* dev_sizeTabs, int W, int H, int CASEMAXX, int CASEMAXY);

#endif