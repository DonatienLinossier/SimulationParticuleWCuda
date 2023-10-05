#ifndef DEF_SYSTEM
#define DEF_SYSTEM
#include "Particule.cuh"
#include <cstdlib>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


class System {
public:
	int m_partsMax;
	int m_partsInit;
	int m_width;
	int m_height;
	int m_taillemaxballe;
	int m_nbCaseX;
	float m_sizeCaseX;
	int m_nbCaseY;
	float m_sizeCaseY;

	cudaPitchedPtr dev_grilleP; //pointeur vers un tabl 3D stck sur le GPU
	int* dev_sizeTabs;

	//int* dev_arr;

	std::vector<Particule> parts;
	std::vector<std::vector<std::vector<int>>> grilleP; //why static ??

public:
	System(int width, int height, int taillemaxballe, int partsMax, int partsInit);
	void init();
	void partsMoveAndRepartition(bool opti, float dt);
	void collisionAndDraw(SDL_Renderer* pRenderer);
	void collisionAndDrawGPU(SDL_Renderer* pRenderer);

	void addParticule(int x, int y);
};

#endif