#include "ParticuleAoS_Kernel.cuh"


__global__ void forceOnPoint_global(float* p_x, float* p_y, float* p_vx, float* p_vy, int nbParticule, float dt, int x, int y, float intensite) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= nbParticule) {
		return;
	}

	float distx = x - p_x[index];
	float disty = y - p_y[index];
	float dist = sqrt(pow(distx, 2) + pow(disty, 2));

	float acc = dist * dist / intensite;

	if (acc == 0) {
		return;
	}

	float forcex = distx / acc;
	float forcey = disty / acc;

	p_vx[index] += forcex;
	p_vy[index] += forcey;
}


__global__ void force_global(float* p_x, float* p_y, float* p_lastx, float* p_lasty, float* p_vx, float* p_vy, int nbParticule, float dt) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= nbParticule) {
		return;
	}

	p_vx[index] = (p_x[index] - p_lastx[index]) / dt;
	p_vy[index] = (p_y[index] - p_lasty[index]) / dt + GRAVITY;
}

__global__ void CalcPosition_global(float* p_x, float* p_y, float* p_lastx, float* p_lasty, float* p_vx, float* p_vy, int nbParticule, float dt) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= nbParticule) {
		return;
	}

	p_lastx[index] = p_x[index];
	p_lasty[index] = p_y[index];
	p_x[index] += p_vx[index] * dt;
	p_y[index] += p_vy[index] * dt;
}


/*
Laisser la cell de retour en argument ?? -> lourd en transfert
Ou creer carrement un tab contenant les last Cell; -> lourd en memoire
*/
__global__ void toCell_global(float* p_x, float* p_y, cell* p_cell, cell* p_lastCell, int nbParticule, int SIZECASEX, int SIZECASEY, int CASEMAXX, int CASEMAXY) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= nbParticule) {
		return;
	}

	int cx = p_x[index] / SIZECASEX;
	int cy = p_y[index] / SIZECASEY;


	//cell retoure;
	//retoure.x = p_cell[index].x;
	//retoure.y = p_cell[index].y;
	if (cx != p_cell[index].x || cy != p_cell[index].y) {
		//p_actualiazed = false;

		p_cell[index].x = cx;
		p_cell[index].y = cy;

		if (p_cell[index].x < 0) {
			p_cell[index].x = 0;
		}
		else if (p_cell[index].x >= CASEMAXX) {
			p_cell[index].x = CASEMAXX - 1;
		}

		if (p_cell[index].y < 0) {
			p_cell[index].y = 0;
		}
		else if (p_cell[index].y >= CASEMAXY) {
			p_cell[index].y = CASEMAXY - 1;
		}
	}
	p_lastCell[index] = p_cell[index];
	//retour[index] = retoure;
}



__global__ void borderCollision_global(float* p_x, float* p_y, int* p_radius, int width, int height, int nbParticule) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index >= nbParticule) {
		return;
	}

	if (p_x[index] + p_radius[index] > width) {
		p_x[index] = width - p_radius[index];
	}
	else if (p_x[index] - p_radius[index] < 0) {
		p_x[index] = p_radius[index];
	}

	if (p_y[index] + p_radius[index] > height) {
		p_y[index] = height - p_radius[index];
	}
	else if (p_y[index] - p_radius[index] < 0) {
		p_y[index] = p_radius[index];
	}
}