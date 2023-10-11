#pragma once
#ifndef DEF_SYSTEM
#define DEF_SYSTEM
#include <cstdlib>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "particuleAoS.cuh"



class SystemCuda {
public:
	//int m_partsMax;
	int m_partsInit;
	int m_width;
	int m_height;
	int m_taillemaxballe;
	int m_nbCaseX;
	float m_sizeCaseX;
	int m_nbCaseY;
	float m_sizeCaseY;
	

	particuleAoS particules;


	double* dev_metaballs;
	int** dev_grilleP2D;
	int* dev_sizeTabs; 
	int* dev_previousSizeTabs;
	SDL_Renderer* pRenderer;
	SDL_Window* pWindow;
	SDL_Texture* pTexture;
	uchar3* dev_gpuPixels;
	uchar3* hostPixels;
	bool* tabChange; 

public:
	SystemCuda(int width, int height, int taillemaxballe, int partsInit);
	int init();
	void allocateGrilleP();
	int initSDL();
	int displaySDL(SDL_Texture* Message, SDL_Rect Message_rect);
	int getDisplayFromGpu();
};
#endif