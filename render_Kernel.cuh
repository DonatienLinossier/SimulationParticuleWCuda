#pragma once
#ifndef DEF_RENDER_KERNEL
#define DEF_RENDER_KERNEL
#include <SDL.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "SystemCuda.cuh"






void GpuBufferToSDLSurface(SDL_Surface* screen, void* cuda_pixels, int width, int height);

void clearScreen(void* cuda_pixels);
void clearScreenNEw(uchar3* dev_gpuPixels, int width, int height);

void drawline(SystemCuda system_, int x0, int y0, int x1, int y1, int color);
void drawlines(SystemCuda system_, int* x0, int* y0, int* x1, int* y1, int* color, int size);
void drawRect(SystemCuda system_, int x, int y, int w, int h, int color);
void drawRects(SystemCuda system_, int* x, int* y, int* w, int* h, int* color, int size);
void drawCircle(SystemCuda system_, int x, int y, int r, int color);
void drawCircles(SystemCuda system_, int* x, int* y, int* r, int* color, int size);
void drawDot(SystemCuda system_, int x, int y, int color);
void drawDots(SystemCuda system_, int* x, int* y, int* color, int size);
__global__ void gaussianBlurInPlace(uchar3* image, int width, int height);
void drawDotNew(SystemCuda system_, int x, int y, int color);
#endif