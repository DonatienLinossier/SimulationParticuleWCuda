#ifndef CONST
#define CONST

#define HEIGHT 1000
#define WIDTH 1800

#define NB_PARTICULES 1
#define NBNEWPARTS 35000
#define PARTICULE_SIZE 3
#define SIZE_CASE_COEF 3//Min 2, can be increased to reduce realloc of tabs, at the cost of heavier computation

#define FORCEONMOUSEACTIVE true
#define FORCEONMOUSESTRENGtH 500

#define GRAVITY 0

#define PARTITIONACTIVE true

#define COLLISION true 

#define BORDERCOLLISIONACTIVE true

#define NBTHREAD 1024

#define BACKGROUND_R 0
#define BACKGROUND_G 0
#define BACKGROUND_B 0
#define BACKGROUND_A 0

#define DRAW_CIRCLE_FILLED true
#define DRAW_CIRCLE_EDGE true

#define SHOW_TIME false
#define DRAWGPUDEBUG false
#define SizeMapX 50
#define SizeMapY 0 
#define SizeMapW 1000
#define SizeMapH 700
#define DEBUG_COLOR_0 { 20, 20, 20, 255 }
#define DEBUG_COLOR_1 { 0, 0, 255, 255 }
#define DEBUG_COLOR_2 { 0, 255, 0, 255 }
#define DEBUG_COLOR_DEFAULT { 255, 255, 255, 255 }

#define MESSAGE_RECT { 10, 10, 150, 50 }
#define MESSAGE_COLOR { 255, 255, 255 }

#define DELTA_TIME 0.001f

//CUDA

#define CUDA_DEVICE 0


#endif