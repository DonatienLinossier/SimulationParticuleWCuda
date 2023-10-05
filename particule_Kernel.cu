#include "particule_Kernel.cuh"




__global__ void ApplyForce(float dt, int nbparts, Particule* dev_parts) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nbparts) {
        dev_parts[index].force(dt);
    }
}

__global__ void ApplyForceOnPoint(float dt, int nbparts, Particule* dev_parts, int x, int y, int intensite) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nbparts) {
        dev_parts[index].forceOnPoint(x, y, dt, intensite);
    }
}

__global__ void calcPos(float dt, int nbparts, Particule* dev_parts) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nbparts) {
        dev_parts[index].CalcPosition(dt);
    }
}

__global__ void borderCollision(int nbparts, Particule* dev_parts, int width, int height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nbparts) {
        dev_parts[index].borderCollision(width, height);
    }
}

void borderCollisionCall(int nbparts, Particule* dev_parts, int width, int height) {
    int nbthread = 1024;
    int numBlocks = (nbparts + nbthread - 1) / nbthread;

    borderCollision << < numBlocks, nbthread >> > (nbparts, dev_parts, width, height);
}

__global__ void setColor(int nbparts, Particule* dev_parts, uint32_t color) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nbparts) {
        dev_parts[index].setColor(color);
    }
}

void setColorCall(int nbparts, Particule* dev_parts, uint32_t color) {
    int nbthread = 1024;
    int numBlocks = (nbparts + nbthread - 1) / nbthread;

    setColor << < numBlocks, nbthread >> > (nbparts, dev_parts, color);
}

__global__ void collision(int nbparts, Particule* dev_parts, int** dev_grilleP2D, int* dev_sizeTabs, int W, int H, int CASEMAXX, int CASEMAXY) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nbparts) {
        //dev_parts[index].collisionNonOpti(dev_parts, nbparts, W, H);
        dev_parts[index].collision5GPU(dev_parts, nbparts, dev_grilleP2D, dev_sizeTabs, W, H, CASEMAXX, CASEMAXY);
    }
}






/*__global__ void getRepartition(int***& dev_grilleP, int* dev_sizeTabs, Particule* dev_parts, int nbparts, int SIZECASEX, int SIZECASEY, int CASEMAXX, int CASEMAXY, int* dev_retourGetCell) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nbparts) {
        dev_parts[index].toGrilleP(dev_grilleP, dev_sizeTabs, SIZECASEX, SIZECASEY, CASEMAXX, CASEMAXY, dev_retourGetCell);
    }
}*/

/*__global__ void getRects(int nbparts, Particule* dev_parts, SDL_Rect* dev_rects) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < nbparts) {
        dev_rects[index] = dev_parts[index].getRect();
    }
}*/