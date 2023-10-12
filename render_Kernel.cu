#include "render_Kernel.cuh"

void GpuBufferToSDLSurface(SDL_Surface* screen, void* cuda_pixels, int width, int height) {
    cudaError_t err = cudaMemcpy(screen->pixels, cuda_pixels, width * height * 4, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess) {
        printf("Erreur à l'interieur du blit: %s\n", cudaGetErrorString(err));
        exit(1000);
    }
}

__global__ void gaussianBlurInPlace(uchar3* image, int width, int height)
{

    float gaussianKernel5x5[25] = {
        1.0f / 256,  4.0f / 256,  6.0f / 256,  4.0f / 256, 1.0f / 256,
        4.0f / 256, 16.0f / 256, 24.0f / 256, 16.0f / 256, 4.0f / 256,
        6.0f / 256, 24.0f / 256, 36.0f / 256, 24.0f / 256, 6.0f / 256,
        4.0f / 256, 16.0f / 256, 24.0f / 256, 16.0f / 256, 4.0f / 256,
        1.0f / 256,  4.0f / 256,  6.0f / 256,  4.0f / 256, 1.0f / 256
    };
    int kernelSize = 5;
    float* kernel = gaussianKernel5x5;
    int x = (blockIdx.x * blockDim.x + threadIdx.x) % width;
    int y = (blockIdx.x * blockDim.x + threadIdx.x) / width;
    if (x < width && y < height)
    {
        float4 result = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // Iterate over the Gaussian kernel
        for (int i = -kernelSize / 2; i <= kernelSize / 2; i++)
        {
            for (int j = -kernelSize / 2; j <= kernelSize / 2; j++)
            {
                int xOffset = x + i;
                int yOffset = y + j;

                if (xOffset >= 0 && xOffset < width && yOffset >= 0 && yOffset < height)
                {
                    int kernelIndex = (i + kernelSize / 2) * kernelSize + (j + kernelSize / 2);
                    uchar3 pixel = image[yOffset * width + xOffset];
                    result.x += static_cast<float>(pixel.x) * kernel[kernelIndex];
                    result.y += static_cast<float>(pixel.y) * kernel[kernelIndex];
                    result.z += static_cast<float>(pixel.z) * kernel[kernelIndex];
                }
            }
        }

        image[y * width + x] = make_uchar3(static_cast<unsigned char>(result.x),
            static_cast<unsigned char>(result.y),
            static_cast<unsigned char>(result.z));
    }
}


__global__ void global_clearScreenNEw(uchar3* dev_gpuPixels, int width, int height) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= width * height) {
        return; // Out-of-bounds thread
    }

    uchar3* element = dev_gpuPixels + index;

    #if (CLEAR_SCREEN_MODE == 0) {
        *element = { BACKGROUND_R , BACKGROUND_G, BACKGROUND_B};
        
    #elif (CLEAR_SCREEN_MODE == 1)
        if (element->x > 20)
            element->x -= element->x / 20;
        else 
            element->x = 0;

        if (element->y > 20)
            element->y -= element->y / 20;
        else
            element->y = 0;



        if (element->z > 20)
            element->z -= element->z / 20;
        else
            element->z = 0;    


    #elif (CLEAR_SCREEN_MODE == 2) 
        if (element->x > 0)
            element->x -= element->y / 20;

        if (element->y > 0)
            element->y -= element->z / 20;

        if (element->z > 0)
            element->z -= element->x / 20;
    #endif
}

void clearScreenNEw(uchar3* dev_gpuPixels, int width, int height) {
    int nbthread = 1024;
    int numBlocks = (width * height + nbthread - 1) / nbthread;

    // Launch your kernel
    global_clearScreenNEw << <numBlocks, nbthread >> > (dev_gpuPixels, width, height);
}



/*__global__ void global_drawDots(SystemCuda system_, int* tab_x, int* tab_y, int* tab_color, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > size) {
        return;
    }

    int x = tab_x[index];
    int y = tab_y[index];
    int color = tab_color[index];

    int pos = y * system_.m_width + x;
    if (y >= system_.m_height || y < 0 || x >= system_.m_width || x < 0) {
        return;
    }
    system_.dev_screen[pos] = color;
}

void drawDots(SystemCuda system_, int* x, int* y, int* color, int size) {
    int* dev_x = NULL;
    int* dev_y = NULL;
    int* dev_color = NULL;

    cudaMalloc((void**)&dev_x, size * sizeof(int));
    cudaMalloc((void**)&dev_y, size * sizeof(int));
    cudaMalloc((void**)&dev_color, size * sizeof(int));

    cudaMemcpy(dev_x, x, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_color, color, size * sizeof(int), cudaMemcpyHostToDevice);

    int nbthread = 1024;
    int numBlocks = (size + nbthread - 1) / nbthread;
    global_drawDots << <numBlocks, nbthread >> > (system_, dev_x, dev_y, dev_color, size);

    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_color);
}


__global__ void global_drawDot(SystemCuda system_, int x, int y, int color) {
    int pos = y * system_.m_width + x;
    if (y >= system_.m_height || y < 0 || x >= system_.m_width || x < 0) {
        return;
    }
    system_.dev_screen[pos] = color;
}

void drawDot(SystemCuda system_, int x, int y, int color) {
    global_drawDot << <1, 1 >> > (system_, x, y, color);
}


__global__ void global_drawDotNew(SystemCuda system_, int x, int y, int color) {
    int pos = y * system_.m_width + x;
    if (y >= system_.m_height || y < 0 || x >= system_.m_width || x < 0) {
        return;
    }
    system_.dev_screen[pos] = color;
}

void drawDotNew(SystemCuda system_, int x, int y, int color) {
    global_drawDotNew << <1, 1 >> > (system_, x, y, color);
}*/





/*__device__ void drawCircle_arcs(int xc, int yc, int x, int y, uint32_t* buf, int width, int height, uint32_t m_color)
{

    int yc_add_y = yc + y;
    int yc_sub_y = yc - y;
    int xc_add_x = xc + x;
    int xc_sub_x = xc - x;

    int yc_add_x = yc + x;
    int yc_sub_x = yc - x;
    int xc_add_y = xc + y;
    int xc_sub_y = xc - y;



    if (!(yc_add_y >= height)) {

        int pos = yc_add_y * width + xc_add_x;
        if (!(xc_add_x >= width)) {
            buf[pos] = (uint32_t)m_color;
        }

        pos = yc_add_y * width + xc_sub_x;
        if (!(xc_sub_x < 0)) {
            buf[pos] = (uint32_t)m_color;
        }

    }

    if (!(yc_sub_y < 0)) {
        int pos = yc_sub_y * width + xc_add_x;
        if (!(xc_add_x >= width)) {
            buf[pos] = (uint32_t)m_color;
        }

        pos = yc_sub_y * width + xc_sub_x;
        if (!(xc_sub_x < 0)) {
            buf[pos] = (uint32_t)m_color;
        }
    }




    if (!(yc_add_x >= height)) {
        int pos = yc_add_x * width + xc_add_y;
        if (!(xc_add_y >= width)) {
            buf[pos] = (uint32_t)m_color;
        }

        pos = yc_add_x * width + xc_sub_y;
        if (!(xc_sub_y < 0)) {
            buf[pos] = (uint32_t)m_color;
        }

    }

    if (!(yc_sub_x < 0)) {
        int pos = yc_sub_x * width + xc_add_y;
        if (!(xc_add_y >= width)) {
            buf[pos] = (uint32_t)m_color;
        }

        pos = yc_sub_x * width + xc_sub_y;
        if (!(xc_sub_y < 0)) {
            buf[pos] = (uint32_t)m_color;
        }
    }
}

__global__ void global_drawCircle(SystemCuda system_, int x_, int y_, int r_, int color) {

    int xc = x_;
    int yc = y_;
    int r = r_;


    int x = 0, y = r;
    int d = 3 - 2 * r;
    drawCircle_arcs(xc, yc, x, y, (uint32_t*) system_.dev_screen, system_.m_width, system_.m_height, color);
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
        drawCircle_arcs(xc, yc, x, y, (uint32_t*)system_.dev_screen, system_.m_width, system_.m_height, color);
    }

}

void drawCircle(SystemCuda system_, int x, int y, int r, int color) {
    global_drawCircle << <1, 1 >> > (system_, x, y, r, color);
}

__global__ void global_drawCircles(SystemCuda system_, int* x_, int* y_, int* r_, int* colors, int size) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > size) {
        return;
    }

    int xc = x_[index];
    int yc = y_[index];
    int r = r_[index];
    int color = colors[index];


    int x = 0, y = r;
    int d = 3 - 2 * r;
    drawCircle_arcs(xc, yc, x, y, (uint32_t*)system_.dev_screen, system_.m_width, system_.m_height, color);
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
        drawCircle_arcs(xc, yc, x, y, (uint32_t*)system_.dev_screen, system_.m_width, system_.m_height, color);
    }

}


void drawCircles(SystemCuda system_, int* x, int* y, int* r, int* color, int size) {
    int* dev_x = NULL;
    int* dev_y = NULL;
    int* dev_r = NULL;
    int* dev_color = NULL;

    cudaMalloc((void**)&dev_x, size * sizeof(int));
    cudaMalloc((void**)&dev_y, size * sizeof(int));
    cudaMalloc((void**)&dev_r, size * sizeof(int));
    cudaMalloc((void**)&dev_color, size * sizeof(int));

    cudaMemcpy(dev_x, x, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_r, r, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_color, color, size * sizeof(int), cudaMemcpyHostToDevice);

    int nbthread = 1024;
    int numBlocks = (size + nbthread - 1) / nbthread;
    global_drawCircles << <numBlocks, nbthread >> > (system_, dev_x, dev_y, dev_r, dev_color, size);

    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_r);
    cudaFree(dev_color);
}





__global__ void global_drawRect(SystemCuda system_, int x0, int y0, int w, int h, int color) {

    if (w < 0) {
        w *= -1;
        x0 -= w;

    }
    if (h < 0) {
        h *= -1;
        y0 -= h;

    }


    int limiteX = x0 + w;
    int limiteY = y0 + h;

    if (limiteX > system_.m_width) {
        limiteX = system_.m_width;
    }
    if (limiteY > system_.m_height) {
        limiteY = system_.m_height;
    }
    if (x0 < 0) {
        x0 = 0;
    }
    if (y0 < 0) {
        y0 = 0;
    }



    //Les deux lignes horizontales
    int pos1 = x0 + system_.m_width * y0;
    memset((void**)&system_.dev_screen[pos1], color, (limiteX - x0) * sizeof(int));
    int pos2 = x0 + system_.m_width * limiteY;
    memset((void**)&system_.dev_screen[pos2], color, (limiteX - x0) * sizeof(int));
    for (int i = x0; i < limiteX; ++i) {
        int pos1 = i + system_.m_width * y0;
        int pos2 = i + system_.m_width * limiteY;s
        system_.dev_screen[pos1] = color;
        system_.dev_screen[pos2] = color;
    }

    //Les deux lignes verticales
    for (int j = y0; j < limiteY; ++j) {
        int pos1 = x0 + system_.m_width * j;
        int pos2 = limiteX + system_.m_width * j;
        system_.dev_screen[pos1] = color;
        system_.dev_screen[pos2] = color;
    }

}

void drawRect(SystemCuda system_, int x, int y, int w, int h, int color) {
    global_drawRect << <1, 1 >> > (system_, x, y, w, h, color);
}

__global__ void global_drawRects(SystemCuda system_, int* tab_x0, int* tab_y0, int* tab_w, int* tab_h, int* tab_color, int size) {

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > size) {
        return;
    }

    int x0 = tab_x0[index];
    int y0 = tab_y0[index];
    int w = tab_w[index];
    int h = tab_h[index];
    int color = tab_color[index];

    if (w < 0) {
        w *= -1;
        x0 -= w;

    }
    if (h < 0) {
        h *= -1;
        y0 -= h;
    }


    int limiteX = x0 + w;
    int limiteY = y0 + h;

    if (limiteX > system_.m_width) {
        limiteX = system_.m_width;
    }
    if (limiteY > system_.m_height) {
        limiteY = system_.m_height;
    }
    if (x0 < 0) {
        x0 = 0;
    }
    if (y0 < 0) {
        y0 = 0;
    }


    //Les deux lignes horizontales
    for (int i = x0; i < limiteX; ++i) {
        int pos1 = i + system_.m_width * y0;
        int pos2 = i + system_.m_width * limiteY;
        system_.dev_screen[pos1] = color;
        system_.dev_screen[pos2] = color;
    }



    //Les deux lignes
    for (int j = y0; j < limiteY; ++j) {
        int pos1 = x0 + system_.m_width * j;
        int pos2 = limiteX + system_.m_width * j;
        system_.dev_screen[pos1] = color;
        system_.dev_screen[pos2] = color;
    }
}

void drawRects(SystemCuda system_, int* x, int* y, int* w, int* h, int* color, int size) {
    int* dev_x = NULL;
    int* dev_y = NULL;
    int* dev_w = NULL;
    int* dev_h = NULL;
    int* dev_color = NULL;

    cudaMalloc((void**)&dev_x, size * sizeof(int));
    cudaMalloc((void**)&dev_y, size * sizeof(int));
    cudaMalloc((void**)&dev_w, size * sizeof(int));
    cudaMalloc((void**)&dev_h, size * sizeof(int));
    cudaMalloc((void**)&dev_color, size * sizeof(int));

    cudaMemcpy(dev_x, x, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y, y, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_w, w, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_h, h, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_color, color, size * sizeof(int), cudaMemcpyHostToDevice);

    int nbthread = 1024;
    int numBlocks = (size + nbthread - 1) / nbthread;
    global_drawRects << <numBlocks, nbthread >> > (system_, dev_x, dev_y, dev_w, dev_h, dev_color, size);

    cudaFree(dev_x);
    cudaFree(dev_y);
    cudaFree(dev_w);
    cudaFree(dev_h);
    cudaFree(dev_color);
}

__global__ void global_drawline(SystemCuda system_, int x0, int y0, int x1, int y1, int color) {
    if (x0 > x1) {
        int tmp = x0;
        x0 = x1;
        x1 = tmp;

        tmp = y0;
        y0 = y1;
        y1 = tmp;
    }

    int dx = x1 - x0;
    int dy = y1 - y0;

    int limiteX = x1;
    if (limiteX > system_.m_width) {
        limiteX = system_.m_width;
    }

    for (int x = x0; x < limiteX; x++) {
        int y = y0 + dy * (x - x0) / dx;
        if (y > system_.m_height || y < 0) {
            continue;
        }
        int pos = y * system_.m_width + x;

        system_.dev_screen[pos] = (uint32_t)color;
    }
}

void drawline(SystemCuda system_, int x0, int y0, int x1, int y1, int color) {
    global_drawline << <1, 1 >> > (system_, x0, y0, x1, y1, color);
}

__global__ void global_drawlines(SystemCuda system_, int* tab_x0, int* tab_y0, int* tab_x1, int* tab_y1, int* tab_color, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index > size) {
        return;
    }

    int x0 = tab_x0[index];
    int y0 = tab_y0[index];
    int x1 = tab_x1[index];
    int y1 = tab_y1[index];
    int color = tab_color[index];


    if (x0 > x1) {
        int tmp = x0;
        x0 = x1;
        x1 = tmp;

        tmp = y0;
        y0 = y1;
        y1 = tmp;
    }

    int dx = x1 - x0;
    int dy = y1 - y0;

    for (int x = x0; x < x1; x++) {
        int y = y0 + dy * (x - x0) / dx;
        if (x > system_.m_width) {
            return;
        }
        if (y > system_.m_height || y < 0) {
            continue;
        }
        int pos = y * system_.m_width + x;

        system_.dev_screen[pos] = (uint32_t)color;
    }
}

void drawlines(SystemCuda system_, int* x0, int* y0, int* x1, int* y1, int* color, int size) {

    int* dev_x0 = NULL;
    int* dev_y0 = NULL;
    int* dev_x1 = NULL;
    int* dev_y1 = NULL;
    int* dev_color = NULL;

    cudaMalloc((void**)&dev_x0, size * sizeof(int));
    cudaMalloc((void**)&dev_y0, size * sizeof(int));
    cudaMalloc((void**)&dev_x1, size * sizeof(int));
    cudaMalloc((void**)&dev_y1, size * sizeof(int));
    cudaMalloc((void**)&dev_color, size * sizeof(int));

    cudaMemcpy(dev_x0, x0, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y0, y0, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x1, x1, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_y1, y1, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_color, color, size * sizeof(int), cudaMemcpyHostToDevice);

    int nbthread = 1024;
    int numBlocks = (size + nbthread - 1) / nbthread;
    global_drawlines << <numBlocks, nbthread>> > (system_, dev_x0, dev_y0, dev_x1, dev_y1, dev_color, size);

    cudaFree(dev_x0);
    cudaFree(dev_y0);
    cudaFree(dev_x1);
    cudaFree(dev_y1);
    cudaFree(dev_color);
}*/