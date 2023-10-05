#include "particuleAoS.cuh"


particuleAoS::particuleAoS()
{
	nbThread = 0;
	nbBlock = 0;
	width = 0;
	height = 0;
	GRIDSIZE = 0;
	CASEMAXX = 0;
	CASEMAXY = 0;
	SIZECASEX = 0;
	SIZECASEY = 0;
	nbParticule = 0;
	dev_x = nullptr;
	dev_y = nullptr;
	dev_lastx = nullptr;
	dev_lasty = nullptr;
	dev_vx = nullptr;
	dev_vy = nullptr;
	dev_cell = nullptr;
	dev_color = nullptr;
	dev_index = nullptr;
	dev_radius = nullptr;
	dev_changed = nullptr;
};
particuleAoS::particuleAoS(int gridSize, int wt, int ht, int nbcaseX, int nbcaseY, int cw, int ch)
{
	nbThread = NBTHREAD;
	nbBlock = 0;
	width = wt;
	height = ht;
	GRIDSIZE = gridSize;
	CASEMAXX = nbcaseX;
	CASEMAXY = nbcaseY;
	SIZECASEX = cw;
	SIZECASEY = ch;
	nbParticule = 0;
	dev_alloc = nullptr;
	dev_x = nullptr;
	dev_y = nullptr;
	dev_lastx = nullptr;
	dev_lasty = nullptr;
	dev_vx = nullptr;
	dev_vy = nullptr;
	dev_cell = nullptr;
	dev_color = nullptr;
	dev_index = nullptr;
	dev_radius = nullptr;
	dev_changed = nullptr;
};


__device__ cell particuleAoS::dev_toCell(int index, float SIZECASEX, float SIZECASEY, int CASEMAXX, int CASEMAXY) {


	if (index >= nbParticule) {
		return;
	}

	dev_tension[index] = 0;

	int cx = dev_x[index] / SIZECASEX;
	int cy = dev_y[index] / SIZECASEY;


	cell retoure;
	retoure.x = dev_cell[index].x;
	retoure.y = dev_cell[index].y;
	if (cx != dev_cell[index].x || cy != dev_cell[index].y) {
		//p_actualiazed = false;

		dev_cell[index].x = cx;
		dev_cell[index].y = cy;

		if (dev_cell[index].x < 0) {
			dev_cell[index].x = 0;
		}
		else if (dev_cell[index].x >= CASEMAXX) {
			dev_cell[index].x = CASEMAXX - 1;
		}

		if (dev_cell[index].y < 0) {
			dev_cell[index].y = 0;
		}
		else if (dev_cell[index].y >= CASEMAXY) {
			dev_cell[index].y = CASEMAXY - 1;
		}
	}
	return retoure;



}

void particuleAoS::forceOnPoint(int x, int y, float dt, int intensite) {
	forceOnPoint_global << <nbBlock, nbThread >> > (dev_x, dev_y, dev_vx, dev_vy, nbParticule, dt, x, y, intensite);
}

void particuleAoS::force(float dt) {
	force_global << <nbBlock, nbThread >> > (dev_x, dev_y, dev_lastx, dev_lasty, dev_vx, dev_vy, nbParticule, dt);
}

void particuleAoS::CalcPosition(float dt) {
	CalcPosition_global << <nbBlock, nbThread >> > (dev_x, dev_y, dev_lastx, dev_lasty, dev_vx, dev_vy, nbParticule, dt);
}

void particuleAoS::toCell(float SIZECASEX, float SIZECASEY, int CASEMAXX, int CASEMAXY) {
	toCell_global<<<nbBlock, nbThread>>>(dev_x, dev_y, dev_cell, dev_lastCell, nbParticule, SIZECASEX, SIZECASEY, CASEMAXX, CASEMAXY);
}

void particuleAoS::borderCollision() {
	borderCollision_global << <nbBlock, nbThread >> > (dev_x, dev_y, dev_radius, width, height, nbParticule);
}

__global__ void displayData(float* data, float* data2, float* data3, float* data4, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index > size) {
		return;
	}
	printf("%lf %lf %lf %lf\n", data[index], data2[index], data3[index], data4[index]);
}

void particuleAoS::addParticules(int nbNewParticules) {


	void* cpy_alloc = dev_alloc;

	size_t floatSize = (nbParticule + nbNewParticules) * sizeof(float);
	size_t intSize = (nbParticule + nbNewParticules) * sizeof(int);
	size_t boolSize = (nbParticule + nbNewParticules) * sizeof(bool);
	size_t cellSize = (nbParticule + nbNewParticules) * sizeof(cell);
	size_t uchar4_tSize = (nbParticule + nbNewParticules) * sizeof(uchar4);

	cudaMalloc((void**)&dev_alloc,
		floatSize * 7 +  // 6 arrays of floats
		intSize * 2 +  // 3 arrays of ints
		cellSize * 2 +
		uchar4_tSize * 1 +
		boolSize);       // 1 array of bools

	// Assign pointers to different parts of the allocated memory
	dev_x = reinterpret_cast<float*>(dev_alloc);
	dev_y = dev_x + (nbParticule + nbNewParticules);
	dev_lastx = dev_y + (nbParticule + nbNewParticules);
	dev_lasty = dev_lastx + (nbParticule + nbNewParticules);
	dev_vx = dev_lasty + (nbParticule + nbNewParticules);
	dev_vy = dev_vx + (nbParticule + nbNewParticules);
	dev_tension = dev_vy + (nbParticule + nbNewParticules);
	dev_cell = reinterpret_cast<cell*>(dev_tension + (nbParticule + nbNewParticules));
	dev_lastCell = reinterpret_cast<cell*>(dev_lastCell + (nbParticule + nbNewParticules));
	dev_color = reinterpret_cast<uchar4*>(dev_cell + (nbParticule + nbNewParticules));
	dev_index = reinterpret_cast<int*>(dev_color + (nbParticule + nbNewParticules));
	dev_radius = dev_index + (nbParticule + nbNewParticules);
	dev_changed = reinterpret_cast<bool*>(dev_radius + (nbParticule + nbNewParticules));

	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("\n---------AddParticules before lalala - : %s\n", cudaGetErrorString(cudaStatus));
	}

	//Copie des anciens elements
	if (cpy_alloc != nullptr) {
		void* offset = cpy_alloc;
		cudaMemcpy(dev_x, offset, nbParticule * sizeof(float), cudaMemcpyDeviceToDevice);

		offset = reinterpret_cast<float*>(offset) + (nbParticule + nbNewParticules);
		cudaMemcpy(dev_y, offset, nbParticule * sizeof(float), cudaMemcpyDeviceToDevice);

		offset = reinterpret_cast<float*>(offset) + (nbParticule + nbNewParticules);
		cudaMemcpy(dev_lastx, offset, nbParticule * sizeof(float), cudaMemcpyDeviceToDevice);

		offset = reinterpret_cast<float*>(offset) + (nbParticule + nbNewParticules);
		cudaMemcpy(dev_lasty, offset, nbParticule * sizeof(float), cudaMemcpyDeviceToDevice);

		offset = reinterpret_cast<float*>(offset) + (nbParticule + nbNewParticules);
		cudaMemcpy(dev_vx, offset, nbParticule * sizeof(float), cudaMemcpyDeviceToDevice);

		offset = reinterpret_cast<float*>(offset) + (nbParticule + nbNewParticules);
		cudaMemcpy(dev_vy, offset, nbParticule * sizeof(float), cudaMemcpyDeviceToDevice);

		offset = reinterpret_cast<float*>(offset) + (nbParticule + nbNewParticules);
		cudaMemcpy(dev_tension, offset, nbParticule * sizeof(float), cudaMemcpyDeviceToDevice);

		offset = reinterpret_cast<float*>(offset) + (nbParticule + nbNewParticules);
		cudaMemcpy(dev_cell, offset, nbParticule * sizeof(cell), cudaMemcpyDeviceToDevice);

		offset = reinterpret_cast<cell*>(offset) + (nbParticule + nbNewParticules);
		cudaMemcpy(dev_lastCell, offset, nbParticule * sizeof(cell), cudaMemcpyDeviceToDevice);

		offset = reinterpret_cast<cell*>(offset) + (nbParticule + nbNewParticules);
		cudaMemcpy(dev_color, offset, nbParticule * sizeof(uchar4), cudaMemcpyDeviceToDevice);

		offset = reinterpret_cast<uchar4*>(offset) + (nbParticule + nbNewParticules);
		cudaMemcpy(dev_index, offset, nbParticule * sizeof(int), cudaMemcpyDeviceToDevice);

		offset = reinterpret_cast<int*>(offset) + (nbParticule + nbNewParticules);
		cudaMemcpy(dev_radius, offset, nbParticule * sizeof(int), cudaMemcpyDeviceToDevice);

		offset = reinterpret_cast<int*>(offset) + (nbParticule + nbNewParticules);
		cudaMemcpy(dev_changed, offset, nbParticule * sizeof(bool), cudaMemcpyDeviceToDevice);

		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("\n---------AddParticules in lalala - : %s\n", cudaGetErrorString(cudaStatus));
		}

		cudaFree(cpy_alloc);
	}



	//Creation des nouveaux elements
	float* new_x = (float*)malloc(nbNewParticules * sizeof(float));
	float* new_y = (float*)malloc(nbNewParticules * sizeof(float));
	float* new_lastx = (float*)malloc(nbNewParticules * sizeof(float));
	float* new_lasty = (float*)malloc(nbNewParticules * sizeof(float));
	float* new_vx = (float*)malloc(nbNewParticules * sizeof(float));
	float* new_vy = (float*)malloc(nbNewParticules * sizeof(float));
	cell* new_cell = (cell*)malloc(nbNewParticules * sizeof(cell));
	cell* new_lastCell = (cell*)malloc(nbNewParticules * sizeof(cell));
	uchar4* new_color = (uchar4*)malloc(nbNewParticules * sizeof(uchar4));
	int* new_index = (int*)malloc(nbNewParticules * sizeof(int));
	int* new_radius = (int*)malloc(nbNewParticules * sizeof(int));
	float* new_tension = (float*)malloc(nbNewParticules * sizeof(float));
	bool* new_bool = (bool*)malloc(nbNewParticules * sizeof(bool));
	for (int i = 0; i < nbNewParticules; ++i) {
		new_x[i] = (float) 5 + std::rand() % (width - 10);
		new_y[i] = (float) 5 + std::rand() % (height - 10);
		new_vx[i] = 0;
		new_vy[i] = 0;
		new_lastx[i] = new_x[i] - new_vx[i];
		new_lasty[i] = new_y[i] - new_vy[i];
		new_cell[i] = { -1, -1 };
		new_lastCell[i] = { -1, -1 }; 
		new_color[i] = { static_cast<unsigned char>((new_y[i] / height) * 255) , static_cast<unsigned char>(255-(new_y[i] / height) * 255), static_cast<unsigned char>(255 - (new_x[i] / width) * 255), 0};
		new_index[i] = -1;
		new_radius[i] = PARTICULE_SIZE;
		new_tension[i] = 0;
		new_bool[i] = false;
	}



	//Transfert des nouveaux elements sur le GPU
	cudaMemcpy(dev_x + nbParticule, new_x, nbNewParticules * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y + nbParticule, new_y, nbNewParticules * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_lastx + nbParticule, new_lastx, nbNewParticules * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_lasty + nbParticule, new_lasty, nbNewParticules * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vx + nbParticule, new_vx, nbNewParticules * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vy + nbParticule, new_vy, nbNewParticules * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_tension + nbParticule, new_tension, nbNewParticules * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_cell + nbParticule, new_cell, nbNewParticules * sizeof(cell), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_lastCell + nbParticule, new_lastCell, nbNewParticules * sizeof(cell), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_color + nbParticule, new_color, nbNewParticules * sizeof(uchar4), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_index + nbParticule, new_index, nbNewParticules * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_radius + nbParticule, new_radius, nbNewParticules * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_changed + nbParticule, new_bool, nbNewParticules * sizeof(bool), cudaMemcpyHostToDevice);
	//displayData << <1, 50 >> > (dev_x, dev_y, dev_vx, dev_vy, 50);
	free(new_x);
	free(new_y);
	free(new_lastx);
	free(new_lasty);
	free(new_vx);
	free(new_vy);
	free(new_cell);
	free(new_lastCell);
	free(new_color);
	free(new_index);
	free(new_radius);
	free(new_tension);
	free(new_bool);

	nbParticule += nbNewParticules;

	nbBlock = (nbParticule + nbThread - 1) / nbThread;
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("\n---------AddParticules - : %s\n", cudaGetErrorString(cudaStatus));
	} 

}



__global__ void global_drawDotsBis(uint32_t* buf, int width, int height, float* tab_x, float* tab_y, int* tab_color, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index > size) {
		return;
	}


	int x = (int) tab_x[index];
	int y = (int) tab_y[index];
	int color = tab_color[index];

	int pos = y * width + x;
	if (y >= height || y < 0 || x >= width || x < 0) {
		return;
	}

	buf[pos] = (uint32_t)color;
}

void particuleAoS::GPUdraw_point(uint32_t* buf, int width, int height) {
	//drawDots(system_, (int*) dev_x, (int*) dev_y, (int*) dev_color, nbParticule);
	int size = nbParticule;

	int nbthread = 1024;
	int numBlocks = (size + nbthread - 1) / nbthread;

	global_drawDotsBis << <numBlocks, nbthread >> > (buf, width, height, dev_x, dev_y, (int*)dev_color, size);
}

__global__ void global_drawDotsBisNew(uchar4* dev_gpuPixels, int width, int height, float* tab_x, float* tab_y, int* tab_color, int size) {
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= size) {
		return;
	}

	float x_d = tab_x[index];
	float y_d = tab_y[index];
	int color = tab_color[index];

	int x = static_cast<int>(x_d);
	int y = static_cast<int>(y_d);

	if (y >= height || y < 0 || x >= width || x < 0) {
		return;
	}

	//size_t pos = y * width + x;
	uchar4* element = dev_gpuPixels + y * width + x;
	element->x = 255; // Red channel
	element->y = 0;   // Green channel
	element->z = 0;   // Blue channel
	element->w = 255; // Alpha channel
}

void particuleAoS::GPUdraw_pointNew(uchar4* dev_gpuPixels, int width, int height) {
	//drawDots(system_, (int*) dev_x, (int*) dev_y, (int*) dev_color, nbParticule);
	int size = nbParticule;

	int nbthread = 1024;
	int numBlocks = (size + nbthread - 1) / nbthread;

	global_drawDotsBisNew << <numBlocks, nbthread >> > (dev_gpuPixels, width, height, dev_x, dev_y, (int*)dev_color, size);

	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();

	if (cudaStatus != cudaSuccess) {
		printf("Cuda error -> heuuu nouveau:%s\n", cudaGetErrorString(cudaStatus));
	}
}


__device__ void drawCircle_arcs(int xc, int yc, int x, int y, uchar4* dev_gpuPixels, int width, int height, uchar4 color)
{
	//printf("color %d", color);
	/*
	each block is an octant

	*/
	int yc_add_y = yc + y;
	int yc_sub_y = yc - y;
	int xc_add_x = xc + x;
	int xc_sub_x = xc - x;

	int yc_add_x = yc + x;
	int yc_sub_x = yc - x;
	int xc_add_y = xc + y;
	int xc_sub_y = xc - y;

	uchar4* limite = dev_gpuPixels + height * width;


	if (!(yc_add_y >= height)) {

		//int pos = yc_add_y * width + xc_add_x;

		if (!(xc_add_x >= width)) {
			uchar4* element = dev_gpuPixels + yc_add_y * width + xc_add_x;
			if (element >= limite)
				return;
			*element = color;
		}

		//pos = yc_add_y * width + xc_sub_x;
		if (!(xc_sub_x < 0)) {
			uchar4* element = dev_gpuPixels + yc_add_y * width + xc_sub_x;
			if (element >= limite)
				return;
			*element = color;
		}

	}

	if (!(yc_sub_y < 0)) {
		//int pos = yc_sub_y * width + xc_add_x;
		if (!(xc_add_x >= width)) {
			uchar4* element = dev_gpuPixels + yc_sub_y * width + xc_add_x;
			if (element >= limite)
				return;
			*element = color;
		}

		//pos = yc_sub_y * width + xc_sub_x;
		if (!(xc_sub_x < 0)) {
			uchar4* element = dev_gpuPixels + yc_sub_y * width + xc_sub_x;
			if (element >= limite)
				return;
			*element = color;
		}
	}




	if (!(yc_add_x >= height)) {
		//int pos = yc_add_x * width + xc_add_y;
		if (!(xc_add_y >= width)) {
			uchar4* element = dev_gpuPixels + yc_add_x * width + xc_add_y;
			if (element >= limite)
				return;
			*element = color;
		}

		//pos = yc_add_x * width + xc_sub_y;
		if (!(xc_sub_y < 0)) {
			uchar4* element = dev_gpuPixels + yc_add_x * width + xc_sub_y;
			if (element >= limite)
				return;
			*element = color;
		}

	}

	if (!(yc_sub_x < 0)) {
		//int pos = yc_sub_x * width + xc_add_y;
		if (!(xc_add_y >= width)) {
			uchar4* element = dev_gpuPixels + yc_sub_x * width + xc_add_y;
			if (element >= limite)
				return;
			*element = color;
		}

		//pos = yc_sub_x * width + xc_sub_y;
		if (!(xc_sub_y < 0)) {
			uchar4* element = dev_gpuPixels + yc_sub_x * width + xc_sub_y;
			if (element >= limite)
				return;
			*element = color;
		}
	}
}

__global__ void global_drawCircleNew(uchar4* dev_gpuPixels, int width, int height, float* tab_x, float* tab_y, int* tab_radius, uchar4* tab_color, float* dev_tension, int size) {

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= size) {
		return;
	}

	int xc = tab_x[index];
	int yc = tab_y[index];
	int r = tab_radius[index];
	//int color = tab_color[index];
	uchar4 color = tab_color[index];// (int)(dev_tension[index] * 255) % 255;

	int x = 0, y = r;
	int d = 3 - 2 * r;
	drawCircle_arcs(xc, yc, x, y, dev_gpuPixels, width, height, color);
	while (y >= x)
	{
		// for each pixel we will
		// draw all eight pixels

		x++;

		// check for decision parameter
		// and correspondingly
		// update d, x, y
		if (d > 0)
		{
			y--;
			d = d + 4 * (x - y) + 10;
		}
		else
			d = d + 4 * x + 6;
		drawCircle_arcs(xc, yc, x, y, dev_gpuPixels, width, height, color);
	}

}

void particuleAoS::GPUdraw_CircleNew(uchar4* dev_gpuPixels, int width, int height) {
	//drawDots(system_, (int*) dev_x, (int*) dev_y, (int*) dev_color, nbParticule);
	int size = nbParticule;

	int nbthread = 1024;
	int numBlocks = (size + nbthread - 1) / nbthread;

	global_drawCircleNew << <numBlocks, nbthread >> > (dev_gpuPixels, width, height, dev_x, dev_y, dev_radius, dev_color, dev_tension, size);

	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();

	if (cudaStatus != cudaSuccess) {
		printf("Cuda error -> heuuu nouveau:%s\n", cudaGetErrorString(cudaStatus));
	}
}

__global__ void global_GPUdrawFilledCircle(uchar4* dev_gpuPixels, int width, int height, float* dev_x, float* dev_y, int* dev_radius, uchar4* dev_color, int size)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= size) {
		return;
	}

	int xc = dev_x[index];
	int yc = dev_y[index];
	int radius = dev_radius[index];
	uchar4 color = dev_color[index];
	int rSquared = radius * radius;

	uchar4* limite = dev_gpuPixels + height * width;

	for (int y = -radius; y <= radius; y++) {
		for (int x = -radius; x <= radius; x++) {
			if (x * x + y * y <= rSquared) {
				int xpos = xc + x;
				int ypos = yc + y;

				if (xpos >= 0 && xpos < width && ypos >= 0 && ypos < height) {
					uchar4* element = dev_gpuPixels + ypos * width + xpos;
					if (element >= limite)
						continue;
					*element = color;
				}
			}
		}
	}
}

void particuleAoS::GPUdrawFilledCircle(uchar4* dev_gpuPixels, int width, int height) {
	int size = nbParticule;

	int nbthread = 1024;
	int numBlocks = (size + nbthread - 1) / nbthread;

	global_GPUdrawFilledCircle << <numBlocks, nbthread >> > (dev_gpuPixels, width, height, dev_x, dev_y, dev_radius, dev_color, size);

	cudaDeviceSynchronize();
	cudaError_t cudaStatus = cudaGetLastError();

	if (cudaStatus != cudaSuccess) {
		printf("Cuda error -> heuuu nouveau:%s\n", cudaGetErrorString(cudaStatus));
	}
}





__device__ cell particuleAoS::getCell(int index) {
	if (index >= nbParticule) {
		printf("\nmaaahaha");
		return { -1, -1 };
	}
	return dev_cell[index];
}

__device__ void particuleAoS::setChanged(int index, bool newValue) {
	if (index >= nbParticule) {
		return;
	}
	dev_changed[index] = newValue;
}

__device__ bool particuleAoS::getChanged(int index) {
	if (index >= nbParticule) {
		return false;
	}
	return dev_changed[index];
}

__device__ void particuleAoS::setIndex(int index, int newValue) {
	if (index >= nbParticule) {
		return;
	}
	dev_index[index] = newValue;
}

__device__ int particuleAoS::getIndex(int index) {
	if (index >= nbParticule) {
		return -1;
	}
	return dev_index[index];
}

__device__ void particuleAoS::setRadius(int index, float newValue) {
	if (index >= nbParticule) {
		return;
	}
	dev_radius[index] = newValue;
}

__device__ int particuleAoS::getRadius(int index) {
	if (index >= nbParticule) {
		return -1; // Return a default value or an error indicator if out of bounds.
	}
	return dev_radius[index];
}

__device__ void particuleAoS::setTension(int index, float newValue) {
	if (index >= nbParticule)
		return; 

	dev_tension[index] = newValue;
}

__device__ float particuleAoS::getTension(int index) {
	if (index >= nbParticule) {
		return -1.0f; // Return a default value or an error indicator if out of bounds.
	}
	return dev_tension[index];
}

__device__ void particuleAoS::setX(int index, float newValue) {
	if (index >= nbParticule) {
		return;
	}
	dev_x[index] = newValue;
}

__device__ float particuleAoS::getX(int index) {
	if (index >= nbParticule) {
		return -1.0; // Return a default value or an error indicator if out of bounds.
	}
	return dev_x[index];
}

__device__ void particuleAoS::setY(int index, float newValue) {
	if (index >= nbParticule) {
		return;
	}

	//printf("\n %d %f", index, newValue);
	dev_y[index] = newValue;
}

__device__ float particuleAoS::getY(int index) {
	if (index >= nbParticule) {
		return -1.0; // Return a default value or an error indicator if out of bounds.
	}
	return dev_y[index];
}