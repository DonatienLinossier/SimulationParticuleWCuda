#include "SystemCuda.cuh"

SystemCuda::SystemCuda(int width, int height, int taillemaxballe) {
    m_width = width;
    m_height = height;
    m_taillemaxballe = taillemaxballe;
    m_nbCaseX = (width / taillemaxballe) + 1;
    m_sizeCaseX = (float)width / (float)m_nbCaseX;
    m_nbCaseY = (height / taillemaxballe) + 1;
    m_sizeCaseY = (float)height / (float)m_nbCaseY;
    particules = particuleAoS(0, WIDTH, HEIGHT, m_nbCaseX, m_nbCaseY, m_sizeCaseX, m_nbCaseY);
    printf("\n NbCaseX : %d; \n  NbCaseY : %d", m_nbCaseX, m_nbCaseY);
    pRenderer = nullptr;
    pWindow = nullptr;
};


int SystemCuda::initSDL() {

    //SDL Initialisation
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    if (TTF_Init() < 0) {
        printf("SDL_ttf initialization failed: %s\n", TTF_GetError());
    }

    //Window Initialisation
    pWindow = SDL_CreateWindow(WINDOW_TITLE, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, m_width, m_height, SDL_WINDOW_SHOWN);
    if (pWindow == NULL) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    //Renderer Initialisation
    pRenderer = SDL_CreateRenderer(pWindow, -1, SDL_RENDERER_ACCELERATED);
    if (pRenderer == NULL) {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        return 1;
    }

    //Creation of the texture
    pTexture = SDL_CreateTexture(pRenderer, SDL_PIXELFORMAT_RGB24, SDL_TEXTUREACCESS_STREAMING, m_width, m_height);


    //Allocation of the Pixels on the gpu
    cudaMalloc(&dev_gpuPixels, m_width * sizeof(uchar3)* m_height);


    //Allocation of the Pixels on the cpu - The memory is pitched for faster transfert GPU->CPU
    cudaMallocHost(&hostPixels, m_width * m_height * sizeof(uchar3));

    //Tab for the metaballs computation
    cudaMalloc(&dev_metaballs, m_width * sizeof(double) * m_height);

    return 0;
}

int SystemCuda::displaySDL(SDL_Texture* Message, SDL_Rect Message_rect) {
    SDL_UpdateTexture(pTexture, NULL, hostPixels, m_width * sizeof(uchar3));
    SDL_RenderCopy(pRenderer, pTexture, NULL, NULL);
    SDL_RenderCopy(pRenderer, Message, NULL, &Message_rect);
    SDL_RenderPresent(pRenderer);
    return 0;
}


int SystemCuda::getDisplayFromGpu() {

    // Your CUDA code here
    cudaError_t err = cudaMemcpy(hostPixels, dev_gpuPixels, m_width * sizeof(uchar3) * m_height, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        printf("Erreur à l'interieur du blit: %s\n", cudaGetErrorString(err));
        return 1;
    }
    return 0;
}




void SystemCuda::allocateGrilleP() {
    const int width = m_nbCaseX;
    const int height = m_nbCaseY;

    //alloc grilleP
    int** dev_grilleP_3rdLayer_Vector;
    cudaMalloc((void***)&dev_grilleP_3rdLayer_Vector, width * height * sizeof(int*));
    printf("\nCUDA mem : tab Pointeur ->  Allocation of %f MB", (float) width * height * sizeof(int*) / (1024 * 1024));

    int** hostarray = new int*[width * height];

    for (int i = 0; i < width * height; i++) {
        hostarray[i] = nullptr;
    }

    cudaMemcpy(dev_grilleP_3rdLayer_Vector, hostarray, width * height * sizeof(int*), cudaMemcpyHostToDevice);
    
    dev_grilleP2D = dev_grilleP_3rdLayer_Vector;
    //host_grilleP2D = hostarray; we do not use the host anymore
    free(hostarray);



    //alloc tab size;
    cudaMalloc((void***)&dev_sizeTabs, width * height * sizeof(int));
    cudaMalloc((void***)&dev_previousSizeTabs, width * height * sizeof(int));
    cudaMalloc((void***)&tabChange, width * height * sizeof(bool));

    printf("\nCUDA mem : sizeTab 1 ->  Allocation of %f MB", (float) width * height * sizeof(int) / (1024 * 1024));
    printf("\nCUDA mem : sizeTab 2 ->  Allocation of %f MB", (float) width * height * sizeof(int) / (1024 * 1024));
    printf("\nCUDA mem : tabChange ->  Allocation of %f MB", (float) width * height * sizeof(bool) / (1024 * 1024));
    
    int* arrayt = new int[width * height];
    for (int i = 0; i < width * height; i++) {
        arrayt[i] = 0;
    }
    cudaMemcpy(dev_sizeTabs, arrayt, width * height * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_previousSizeTabs, arrayt, width * height * sizeof(int), cudaMemcpyHostToDevice);
    free(arrayt);

    cudaError_t cudaStatus = cudaGetLastError();

    if (cudaStatus != cudaSuccess) {
        printf("Cuda error -> allocate grilleP:%s\n", cudaGetErrorString(cudaStatus));
    }
}




/*void SystemCuda::addParticule(int x, int y) {
    nbparts += 1;

    // Allocate memory for the new array of particles on the device
    Particule* new_dev_parts;
    cudaError_t cudaStatus = cudaMalloc((void**)&new_dev_parts, nbparts * sizeof(Particule));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(1);
    }

    // Copy existing particles from the old array to the new array
    cudaMemcpy(new_dev_parts, dev_parts, sizeof(Particule) * (nbparts - 1), cudaMemcpyDeviceToDevice);

    // Free the old array of particles on the device
    cudaFree(dev_parts);

    // Update the device pointer to point to the new array of particles
    dev_parts = new_dev_parts;

    // Create a new particle on the host
    Particule newPart = Particule(x, y);

    // Copy the new particle to the device memory
    cudaMemcpy(&dev_parts[nbparts - 1], &newPart, sizeof(Particule), cudaMemcpyHostToDevice);
}*/

/*void SystemCuda::addParticules(int nb) {
    //printf("%d", nb);
    Particule* copie = dev_parts;

    // Allocate memory for the new particles on the device
    cudaError_t cudaStatus = cudaMalloc((void**)&dev_parts, (nbparts + nb) * sizeof(Particule));
    if (cudaStatus != cudaSuccess) {
        printf("cudaMalloc failed: %s\n", cudaGetErrorString(cudaStatus));
        exit(1);
    }
    printf("\nCUDA mem : Particules -> Allocation of %f MB", (float) (nbparts + nb) * sizeof(Particule) / (1024*1024));
    // Copy the old particles from copie to the newly allocated memory on the device
    cudaMemcpy(dev_parts, copie, sizeof(Particule) * nbparts, cudaMemcpyDeviceToDevice);

    // Allocate memory for new particles on the host (CPU) and initialize them
    Particule* newParts = (Particule*)malloc(nb * sizeof(Particule));
    for (int i = 0; i < nb; ++i) {
        newParts[i] = Particule(5 + std::rand() % (m_width-10), 5 + std::rand() % (m_height-10));
    }

    // Copy the new particles from the host to the device
    cudaMemcpy(&dev_parts[nbparts], newParts, sizeof(Particule) * nb, cudaMemcpyHostToDevice);

    nbparts += nb; // Update the total number of particles
    //printf("%d", nbparts);

    cudaFree(copie); // Free the old device memory
    free(newParts); // Free the host memory for new particles

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        printf("\nCuda error -> addParticles: %s\n", cudaGetErrorString(cudaStatus));
    }
}*/

int SystemCuda::init() {

    initSDL();
    allocateGrilleP();
    particules.addParticules(NBNEWPARTS);
    return 0;
}