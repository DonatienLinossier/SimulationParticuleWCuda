#pragma once
#ifndef DEF_PARTICULEAOS
#define DEF_PARTICULEAOS
#include <SDL.h>
#include <SDL_ttf.h>
#include <vector>
#include <math.h>
#include <iostream>
#include <atomic>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "const.cpp"
#include "cell.h"
#include "ParticuleAoS_Kernel.cuh"





class particuleAoS {
public:
	int nbParticule;
	particuleAoS();
	particuleAoS(int gridSize, int w, int h, int cw, int wh, int wt, int ht);
	void addParticules(int nbNewParticules);

	//void addParticule(int x, int y);

	void force(float dt);
	void forceOnPoint(int x, int y, float dt, int intensite);
	void CalcPosition(float dt);
	void toCell(float SIZECASEX, float SIZECASEY, int CASEMAXX, int CASEMAXY);
	void borderCollision();
	__device__ cell dev_toCell(int index, float SIZECASEX, float SIZECASEY, int CASEMAXX, int CASEMAXY);
	void collision(int** dev_grilleP2D, int* dev_sizeTabs);
	void GPUdraw_point(uint32_t* buf, int width, int height);
	void GPUdraw_pointNew(uchar3* dev_gpuPixels, int width, int height);
	void GPUdraw_CircleNew(uchar3* dev_gpuPixels, int width, int height);
	void addParticules(int nbNewParticules, int x, int y, int vx, int vy);
	__device__ cell getCell(int index);
	__device__ void setChanged(int index, bool newValue);
	__device__ bool getChanged(int index);
	__device__ void setIndex(int index, int newValue);
	__device__ int getIndex(int index);
	__device__ void setRadius(int index, float newValue);
	__device__ int getRadius(int index);
	__device__ void setX(int index, float newValue);
	__device__ float getX(int index);
	__device__ void setY(int index, float newValue);
	__device__ float getY(int index);
	__device__ float getTension(int index);
	__device__ void setTension(int index, float newValue);
	void GPUdrawFilledCircle(uchar3* dev_gpuPixels, int width, int height);


	float* dev_x;
	float* dev_y; 
	int* dev_radius;
private:
	 
	
	int GRIDSIZE;
	float SIZECASEX;
	float SIZECASEY;
	int CASEMAXX;
	int CASEMAXY;
	int width;
	int height; 

	int nbThread;
	int nbBlock;

	void* alloc;
	void* dev_alloc;

	float* dev_lastx;
	float* dev_lasty;
	float* dev_vx;
	float* dev_vy;
	cell* dev_cell; 
	uchar3* dev_color;
	int* dev_id;

	int* dev_index;
	float* dev_tension;
	bool* dev_changed;
	





};

#endif
