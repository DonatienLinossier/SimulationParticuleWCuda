#ifndef CONST
#define CONST

#define HEIGHT 1000
#define WIDTH 1800

#define NB_PARTICULES 1
#define NBNEWPARTS 1
#define PARTICULE_SIZE 10
#define SIZE_CASE_COEF 3//Min 2, can be increased to reduce realloc of tabs, at the cost of heavier computation

#define FORCEONMOUSEACTIVE false
#define FORCEONMOUSESTRENGtH 500

#define GRAVITY 1

#define PARTITIONACTIVE true

#define COLLISION true 

#define BORDERCOLLISIONACTIVE true

#define NBTHREAD 1024

#define BACKGROUND_R 0
#define BACKGROUND_G 0
#define BACKGROUND_B 0
#define BACKGROUND_A 0

#define CLEAR_SCREEN_MODE 0//0-> black screen; 1-> {x-x/20, y-y/20, z-z/20, . }, 2-> {x-y/20, y-z/20, z-x/20, . }

#define DRAW_CIRCLE_FILLED false
#define DRAW_CIRCLE_EDGE false
#define GAUSSIAN_BLUR false


#define DEBUG_SHOW_MEMORY_ALLOCATION false
#define SHOW_TIME false
#define DRAWGPUDEBUG false
#define SizeMapX 50
#define SizeMapY 0 
#define SizeMapW 1000
#define SizeMapH 700
#define DEBUG_COLOR_0 { 20, 20, 20}
#define DEBUG_COLOR_1 { 0, 0, 255}
#define DEBUG_COLOR_2 { 0, 255, 0}
#define DEBUG_COLOR_DEFAULT { 255, 255, 255}

#define MESSAGE_RECT { 10, 10, 150, 50 }
#define MESSAGE_COLOR { 255, 255, 255 }

#define DELTA_TIME 0.0001f

#define METABALLS_RENDERING true
#define METABALLS_RADIUSCOMPAREDTOPARTICULESIZE 3 
#define METABALLS_INTENSITY 100
#define METABALLS_ATTENUATIONFACTOR 0.005
#define METABALLS_THRESHOLD_DEEP 75
#define METABALLS_THRESHOLD_MEDIUM 25
#define METABALLS_THRESHOLD_BORDER 15
//CUDA

#define CUDA_DEVICE 0


#endif