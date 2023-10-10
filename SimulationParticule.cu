#include <cstdlib>    
#include <iostream>
#include <ctime>
#include <SDL.h>   
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include "systemCuda.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "render_Kernel.cuh"
#include "const.cpp"
#include <string> 
#include <iomanip> 


 



using namespace std;


// Function wrapper to measure CUDA execution time using events
template <typename Func, typename... Args>
float measureCudaExecutionTime(Func func, Args&&... args) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    func(std::forward<Args>(args)...); // Call the wrapped CUDA kernel function with arguments

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return milliseconds;
}

__device__ void collisionAoS(particuleAoS particuleAoS, int nbparts, int p1, int** dev_grilleP2D, int* dev_sizeTabs, int W, int H, int CASEMAXX, int CASEMAXY) {
    if (p1 >= nbparts) {
        return;
    }

    int kmin = -1;
    int kmax = 2;
    int jmin = -1;
    int jmax = 2;

    cell m_cell = particuleAoS.getCell(p1);

    if (m_cell.x == CASEMAXX - 1) {
        kmax = 1;
    }

    if (m_cell.x == 0) {
        kmin = 0;
    }

    if (m_cell.y == 0) {
        jmin = 0;
    }

    if (m_cell.y == CASEMAXY - 1) {
        jmax = 1;
    }

    float tension = 0;
    float p1x = particuleAoS.getX(p1);
    float p1y = particuleAoS.getY(p1);
    int p1r = particuleAoS.getRadius(p1);
    for (int k = kmin; k < kmax; ++k) {
        int caseX = m_cell.x + k;

        for (int j = jmin; j < jmax; ++j) {
            int caseY = m_cell.y + j;
            int* ptr_indexParts = dev_grilleP2D[caseX + CASEMAXX * caseY];
            int size = dev_sizeTabs[caseX + CASEMAXX * caseY];

            for (int i = 0; i < size; ++i) {
                int p2 = ptr_indexParts[i];

                if (p1 == p2) {
                    continue;
                }

                int distAttendu = p1r + particuleAoS.getRadius(p2);

                float Xdist = p1x - particuleAoS.getX(p2);
                float Ydist = p1y - particuleAoS.getY(p2);
                double dist = sqrtf(Xdist * Xdist + Ydist * Ydist);

                if (dist < distAttendu) {
                    float delta = distAttendu - dist;
                    

                    if (dist == 0.0) {
                        // Handle the case where particles are exactly on top of each other
                        continue;
                    }

                    double nx = Xdist / dist;
                    double ny = Ydist / dist;
                    double multiple = 0.5 * delta;

                    float dtx = static_cast<float>(multiple * nx);
                    float dty = static_cast<float>(multiple * ny);


                    // Update the positions of the particles with atomicAdd using float casts
                    particuleAoS.setX(p2, particuleAoS.getX(p2) - dtx);
                    particuleAoS.setY(p2, particuleAoS.getY(p2) - dty);
                    particuleAoS.setTension(p2, particuleAoS.getTension(p2) + delta);
                    tension += delta;
                    p1x +=dtx;
                    p1y +=dty;

                }
            }
        }
    }
    particuleAoS.setX(p1, p1x);
    particuleAoS.setY(p1, p1y);
    particuleAoS.setTension(p1, particuleAoS.getTension(p1) + tension);
}


__device__ int getCaseFromBlockIndexAoSold(int index, int i, int j, int CASEMAXX) {
    int blockPerLine = CASEMAXX / 3; // nombre de block, et donc de thread, par ligne
    int sizeBlockLine = blockPerLine * 9; // taille d'une ligne de block
    if (CASEMAXX % 3 != 0) {
        int lastBlockSize = (CASEMAXX % 3) * 3;
        sizeBlockLine += lastBlockSize;
        blockPerLine += 1;
    }

    int line = index / blockPerLine;
    int Yoffset = line * sizeBlockLine;



    int column = index % blockPerLine;
    int Xoffset = 3 * column;



    int Case = Yoffset + j * CASEMAXX + Xoffset + i;
    return Case;
}

//A relire
__device__ int getCaseFromBlockIndexAoS(int index, int i, int j, int CASEMAXX) {
    // Calculate the number of blocks per line (as in the comment)
    int blockPerLine = CASEMAXX / 3;

    // Calculate the size of a line of blocks (as in the comment)
    int sizeBlockLine = blockPerLine * 3;

    // Check if there's a remainder when dividing CASEMAXX by 3
    if (CASEMAXX % 3 != 0) {
        int lastBlockSize = (CASEMAXX % 3) * 3;

        // Adjust the size of the line of blocks accordingly
        sizeBlockLine += lastBlockSize;

        // Increase the number of blocks per line to account for the remainder
        blockPerLine += 1;
    }

    // Calculate the line index
    int line = index / blockPerLine;

    // Calculate the Y offset based on the line index
    int Yoffset = line * sizeBlockLine;

    // Calculate the column index
    int column = index % blockPerLine;

    // Calculate the X offset based on the column index
    int Xoffset = 3 * column;

    // Calculate the final grid cell index
    int Case = Yoffset + j * CASEMAXX + Xoffset + i;

    return Case;
}

__global__  void global_collisionAoS(SystemCuda system_, int nbParts, int** dev_grilleP2D, int* dev_sizeTabs, int W, int H, int CASEMAXX, int CASEMAXY, int i, int j) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int blockPerLine = CASEMAXX / 3; // nombre de block, et donc de thread, par ligne
    if (CASEMAXX % 3 != 0) {
        blockPerLine += 1;
    }
    int nb_line = CASEMAXY / 3;
    if (CASEMAXY % 3 != 0) {
        nb_line += 1;
    }

    if (index >= nb_line * blockPerLine) {
        return;
    }


    
    int baseCase = getCaseFromBlockIndexAoSold(index, i, j, CASEMAXX);
    if (baseCase > CASEMAXX * CASEMAXY) {
        return;
    }

    for (int particule = 0; particule < dev_sizeTabs[baseCase]; particule++) {
        collisionAoS(system_.particules, nbParts, dev_grilleP2D[baseCase][particule], dev_grilleP2D, dev_sizeTabs, W, H, CASEMAXX, CASEMAXY);
    }

}

void call_collisionAoS(SystemCuda system_) {
    int nbthread = 512;
    int numBlocks = (((system_.m_nbCaseX / 3 + 1) * (system_.m_nbCaseY / 3 + 1)) + nbthread - 1) / nbthread;
    cudaThreadSynchronize();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            global_collisionAoS << <numBlocks, nbthread >> > (system_, system_.particules.nbParticule, system_.dev_grilleP2D, system_.dev_sizeTabs, system_.m_width, system_.m_height, system_.m_nbCaseX, system_.m_nbCaseY, i, j);
            cudaThreadSynchronize();
        }
    }
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("\ncall_collisionAoS - fin error: %s\n", cudaGetErrorString(cudaStatus));
    }
}

__device__ void GPU_drawSizeMiniMap_Tilenew(uchar3* dev_gpuPixels, int x, int y, int w, int h, int m_nbCaseX, int m_nbCaseY, int width, int height, int sizevalue) {
    int xsize = w / m_nbCaseX;
    int ysize = h / m_nbCaseY;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int linex = index % m_nbCaseX;
    int liney = index / m_nbCaseX;
    uchar3 color;//RGBA
    switch (sizevalue) {
    case 0:
        color = DEBUG_COLOR_0;
        break;
    case 1:
        color = DEBUG_COLOR_1;
        break;
    case 2:
        color = DEBUG_COLOR_2;
        break;
    default:
        color = DEBUG_COLOR_DEFAULT;
    }

    int rectx = x + linex * xsize;
    int recty = y + liney * ysize;

    for (int i = rectx; i < rectx + xsize - 1; ++i) {
        if (i<0 || i> width) {
            continue;
        }
        for (int j = recty; j < recty + ysize - 1; ++j) {
            if (j< 0 || j > height) {
                continue;
            }

            /*if (liney % 3 == 0 && linex % 3 == 0) {//show first case of each thread
                color = { 100, 100, 100, 255 };
            }*/
            uchar3* element = dev_gpuPixels + j * width + i;
            *element = color;

        }
    }
}

__global__ void GPU_drawSizeMiniMapNEw(uchar3* dev_gpuPixels, int x, int y, int w, int h, int m_nbCaseX, int m_nbCaseY, int width, int height, int* dev_SizeTab) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= m_nbCaseX * m_nbCaseY)
        return;

    int sizeValue = dev_SizeTab[index];
    GPU_drawSizeMiniMap_Tilenew(dev_gpuPixels, x, y, w, h, m_nbCaseX, m_nbCaseY, width, height, sizeValue);
}

__global__ void partitionAoS(int* dev_sizeTabs, int** dev_grilleP2D, int nbparts, particuleAoS particuleAoS, bool* tabChange, float SIZECASEX, float SIZECASEY, int CASEMAXX, int CASEMAXY) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= nbparts) {
        return;
    }

    cell previousCell = particuleAoS.dev_toCell(index, SIZECASEX, SIZECASEY, CASEMAXX,  CASEMAXY);
    cell actualCell = particuleAoS.getCell(index);

    if (previousCell.x != actualCell.x || previousCell.y != actualCell.y) {
        particuleAoS.setChanged(index, true);
        if (previousCell.x != -1) {
            atomicSub(&dev_sizeTabs[previousCell.x + CASEMAXX * previousCell.y], 1);
            dev_grilleP2D[previousCell.x + CASEMAXX * previousCell.y][particuleAoS.getIndex(index)] = -1;
            tabChange[previousCell.x + CASEMAXX * previousCell.y] = true;
        }
    }
}


__global__ void addSizeAndSetIndexAoS(int* dev_sizeTabs, int nbparts, particuleAoS particuleAoS, bool* tabChange, int CASEMAXX) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nbparts) {
        return;
    }


    if (particuleAoS.getChanged(index)) {
        cell actualCell = particuleAoS.getCell(index);
        int tabIndex = atomicAdd(&dev_sizeTabs[actualCell.x + CASEMAXX * actualCell.y], 1);
        particuleAoS.setIndex(index, tabIndex);
        tabChange[actualCell.x + CASEMAXX * actualCell.y] = true;
    }
}

/**
 * @brief [Kernel] resize or free memory of the partition array of particles in StructOfArray format.
 *
 * @param dev_sizeTabs   - Array containing new sizes for each cell.
 * @param dev_grilleP2D  - Array of pointers to particle data.
 * @param particuleAoS   - Object storing particles in StructOfArray and provinding functions
 * @param tabChange      - Array indicating whether a change is needed for each cell.
 * @param prevSizeTab    - Array storing the previous sizes for each cell.
 * @param CASEMAXX       - Maximum X dimension of the grid.
 * @param CASEMAXY       - Maximum Y dimension of the grid.
 */
__global__ void changeSizeTabSoA(int* dev_sizeTabs, int** dev_grilleP2D, particuleAoS particuleAoS, bool* tabChange, int* prevSizeTab, int CASEMAXX, int CASEMAXY) {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if (index >= CASEMAXX * CASEMAXY)
            return;

        //No change in the tab -> early return
        if (!tabChange[index])
            return;

        //Case of new size = 0 -> No Allocation nor index rearrangement needed, just freeing. -> early return;
        if (dev_sizeTabs[index] <= 0) {
            if (dev_grilleP2D[index] != nullptr)
                cudaFree(dev_grilleP2D[index]);
            dev_grilleP2D[index] = nullptr;
            prevSizeTab[index] = dev_sizeTabs[index];
            tabChange[index] = false;
            return;
        }


        //Store the last tab : Used for index rearrangement and for freeing it
        int* copie = dev_grilleP2D[index];

        //Case change size -> create a new tab
        // Case no change size -> reutilization of last allocation, no allocation needed
        if (dev_sizeTabs[index] != prevSizeTab[index]) {
            cudaError_t cudaStatus = cudaMalloc((void**)&dev_grilleP2D[index], dev_sizeTabs[index] * sizeof(int));
            if (cudaStatus != cudaSuccess) {
                    printf("mouap cudaMalloc of size %d failed: %s\n", dev_sizeTabs[index], cudaGetErrorString(cudaStatus));
            }
        }

        //Case change size -> Copy the indexs that are still in this case in the new cudaAlloc
        //Case no change size -> Rearrange the indexes to the begining of the tab
        int it = 0;
        for (int i = 0; i < prevSizeTab[index]; i++) {
            if (!(copie[i] == -1)) {
                dev_grilleP2D[index][it] = copie[i];
                particuleAoS.setIndex(copie[i], it);
                it++;
            }
        }


        //Case change size, free last alloc
        if (dev_sizeTabs[index] != prevSizeTab[index]) {
            if (copie != nullptr)
                cudaFree(copie);
        }


        // Update size and resetting tabChange
        prevSizeTab[index] = dev_sizeTabs[index];
        tabChange[index] = false;
}



__global__ void addNewElementsInTabAoS(int* dev_sizeTabs, int** dev_grilleP2D, int nbparts, particuleAoS particuleAoS, int CASEMAXX) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nbparts) {
        return;
    }

    if (!particuleAoS.getChanged(index))
        return;
    
    
    dev_grilleP2D[particuleAoS.getCell(index).x + particuleAoS.getCell(index).y * CASEMAXX][particuleAoS.getIndex(index)] = index;
    particuleAoS.setChanged(index, false);
}


void partitionFunctionsAoS(SystemCuda system_) {
    int nbthread = 1024;
    int numBlocks = (system_.particules.nbParticule + nbthread - 1) / nbthread;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    partitionAoS << <numBlocks, nbthread >> >(system_.dev_sizeTabs, system_.dev_grilleP2D, system_.particules.nbParticule, system_.particules, system_.tabChange, system_.m_sizeCaseX, system_.m_sizeCaseY, system_.m_nbCaseX, system_.m_nbCaseY);
    //printf("partition");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float partitionAoS_time = 0.0f;
    cudaEventElapsedTime(&partitionAoS_time, start, stop);



    cudaEventRecord(start);
    addSizeAndSetIndexAoS << <numBlocks, nbthread >> > (system_.dev_sizeTabs, system_.particules.nbParticule, system_.particules, system_.tabChange, system_.m_nbCaseX);
    //printf("Addsize");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float addSizeAndSetIndexAoS_time = 0.0f;
    cudaEventElapsedTime(&addSizeAndSetIndexAoS_time, start, stop);



    cudaThreadSynchronize();
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("post partition error: %s\n", cudaGetErrorString(cudaStatus));
        exit(1020);
    }

    nbthread = 1024;
    numBlocks = (system_.m_nbCaseX * system_.m_nbCaseY + nbthread - 1) / nbthread;


    cudaEventRecord(start);
    changeSizeTabSoA << <numBlocks, nbthread >> > (system_.dev_sizeTabs, system_.dev_grilleP2D, system_.particules, system_.tabChange, system_.dev_previousSizeTabs, system_.m_nbCaseX, system_.m_nbCaseY);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float changeSizeTabAoS_time = 0.0f;
    cudaEventElapsedTime(&changeSizeTabAoS_time, start, stop);




    cudaThreadSynchronize();
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("post change sizetab error: %s\n", cudaGetErrorString(cudaStatus));
        exit(1022);
    }

    nbthread = 1024;
    numBlocks = (system_.particules.nbParticule + nbthread - 1) / nbthread;


    cudaEventRecord(start);
    addNewElementsInTabAoS << <numBlocks, nbthread >> > (system_.dev_sizeTabs, system_.dev_grilleP2D, system_.particules.nbParticule, system_.particules, system_.m_nbCaseX);
    //printf("addToTab");
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float addNewElementsInTabAoS_time = 0.0f;
    cudaEventElapsedTime(&addNewElementsInTabAoS_time, start, stop);

    // Destroy the CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaThreadSynchronize();

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("bofire free error: %s\n", cudaGetErrorString(cudaStatus));
        exit(1024);
    }

    //printf("Part/Index/Sizetab/AddElmnt : %.2fms | %.2fms | %.2fms | %.2fms | %.2fms |", partitionAoS_time, addSizeAndSetIndexAoS_time, changeSizeTabAoS_time, addNewElementsInTabAoS_time, partitionAoS_time+ addSizeAndSetIndexAoS_time+ changeSizeTabAoS_time+ addNewElementsInTabAoS_time);
}













int main(int argc, char* argv[])
{
    SystemCuda system_(WIDTH, HEIGHT, PARTICULE_SIZE * SIZE_CASE_COEF, NB_PARTICULES);
    unsigned int nbFrame = 0;
    float sommeRender = 0;
    float sommeComputation = 0;
    float sommeinput = 0;



    srand((unsigned int)time(nullptr));


    SDL_Event events;
    bool isOpen{ true };
    SDL_Point MousePosition;
    time_t lastTime = time(NULL);
    clock_t lastClock = clock() - 1;
    clock_t clock_time;
    

    system_.init();

    TTF_Font* Sanss = TTF_OpenFont("OpenSans-Bold.ttf", 24);
    SDL_Surface* surfaceMessage = nullptr;
    SDL_Texture* Message = nullptr;
    SDL_Rect Message_rect = MESSAGE_RECT; //Rect of the FPS / nb of particules
    bool mouseDown = false;
    float dt = DELTA_TIME; //Delta time between each computation

    //Set CUDA DEVICE
    cudaError_t cudaStatus = cudaSetDevice(CUDA_DEVICE);
    if (cudaStatus != cudaSuccess) {
        printf("cudaSetDevice failed: %s\n", cudaGetErrorString(cudaStatus));
        return 1;
    }



    while (isOpen)
    {
        nbFrame++;
        clock_time = clock() - lastClock;
        lastClock = clock();
        lastTime = time(NULL);


        //###################################
        //  Inputs
        //###################################
        while (SDL_PollEvent(&events))
        {
            switch (events.type)
            {
            case SDL_QUIT:
                isOpen = false;
                break;

            case SDL_MOUSEMOTION:
                SDL_GetMouseState(&MousePosition.x, &MousePosition.y);
                break;
            case SDL_MOUSEWHEEL:
                SDL_GetMouseState(&MousePosition.x, &MousePosition.y);
                system_.particules.addParticules(1, MousePosition.x, MousePosition.y, 5, 0);
                break;
            case SDL_MOUSEBUTTONDOWN:
                mouseDown = true;
                break;
            case SDL_MOUSEBUTTONUP:
                mouseDown = false;
                break;
            case SDL_KEYDOWN:
                break;
            }

        }
        if (mouseDown) {
            SDL_GetMouseState(&MousePosition.x, &MousePosition.y);
            int x = MousePosition.x;
            int y = MousePosition.y;
            system_.particules.addParticules(1, x, y, 5, 0); 
        }





        //###################################
        //  Computation
        //###################################

        auto measureComputation = [&]() {
            system_.particules.force(dt);


            if (FORCEONMOUSEACTIVE) {
                system_.particules.forceOnPoint(MousePosition.x, MousePosition.y, dt, FORCEONMOUSESTRENGtH);
            }
            system_.particules.CalcPosition(dt);


            cudaDeviceSynchronize();


            if (PARTITIONACTIVE) {
                measureCudaExecutionTime(partitionFunctionsAoS, system_);
                cudaDeviceSynchronize();


                if (COLLISION) {
                    measureCudaExecutionTime(call_collisionAoS, system_);

                    cudaDeviceSynchronize();
                    cudaStatus = cudaGetLastError();

                    if (cudaStatus != cudaSuccess) {
                        printf("Cuda error -> during collision:%s\n", cudaGetErrorString(cudaStatus));
                    }
                }
            }

            if (BORDERCOLLISIONACTIVE) {
                system_.particules.borderCollision();
            }
        };

        float computationTime = measureCudaExecutionTime(measureComputation);
        


        //###################################
        //  RENDER
        //###################################

        auto measureRender = [&]() {
            clearScreenNEw(system_.dev_gpuPixels, system_.m_width, system_.m_height);
            cudaDeviceSynchronize();
            if (DRAWGPUDEBUG) {
                int nbthread = 1024;
                int numBlocks = (system_.m_nbCaseX * system_.m_nbCaseY + nbthread - 1) / nbthread;
                GPU_drawSizeMiniMapNEw << <numBlocks, nbthread >> > (system_.dev_gpuPixels, SizeMapX, SizeMapY, SizeMapW, SizeMapH, system_.m_nbCaseX, system_.m_nbCaseY, system_.m_width, system_.m_height, system_.dev_sizeTabs);
            }
            if(DRAW_CIRCLE_FILLED)
                system_.particules.GPUdrawFilledCircle(system_.dev_gpuPixels, system_.m_width, system_.m_height);
            if(DRAW_CIRCLE_EDGE)
                system_.particules.GPUdraw_CircleNew(system_.dev_gpuPixels, system_.m_width, system_.m_height);

            int nbthread = 1024;
            int numBlocks = (system_.m_width * system_.m_height + nbthread - 1) / nbthread;
            if (GAUSSIAN_BLUR) {
                gaussianBlurInPlace << <numBlocks, nbthread >> > (system_.dev_gpuPixels, system_.m_width, system_.m_height);
            }
            //bloom << <numBlocks, nbthread >> > (system_.dev_gpuPixels, system_.m_width, system_.m_height, 5, 2);
            cudaDeviceSynchronize();
            system_.getDisplayFromGpu();


            int FPS = (int)(1 / ((float)clock_time / 1000.f));
            surfaceMessage = TTF_RenderText_Solid(Sanss, std::string("FPS :" + std::to_string(FPS) + "; nb : " + std::to_string(system_.particules.nbParticule)).c_str(), MESSAGE_COLOR);

            Message = SDL_CreateTextureFromSurface(system_.pRenderer, surfaceMessage);

            SDL_FreeSurface(surfaceMessage);
            cudaDeviceSynchronize();
            system_.displaySDL(Message, Message_rect);
            SDL_DestroyTexture(Message);

        };


        float renderTime =  measureCudaExecutionTime(measureRender);
            
        if(SHOW_TIME)
            cout << "\nRender|Computation : "<< std::setw(10) << std::fixed << std::setprecision(4) << renderTime << " | " << computationTime;


        cudaStatus = cudaGetLastError();

        if (cudaStatus != cudaSuccess) {
            printf("End of boucle CUDA error: %s\n", cudaGetErrorString(cudaStatus));
        }
    }
    // Don't forget to free your surface and texturev

    /*SDL_DestroyRenderer(pRenderer);
    SDL_DestroyWindow(pWindow);
    TTF_Quit();
    SDL_Quit();*/

    cudaProfilerStop();
    return 0;
};