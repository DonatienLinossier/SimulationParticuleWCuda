#ifndef CONST
#define CONST

#define HEIGHT 1000
#define WIDTH 1800

#define WINDOW_TITLE "Particule Simulation with cuda"


#define NBNEWPARTS 10000 
#define PARTICULE_SIZE 5
#define SIZE_CASE_COEF 3//Min 2, can be increased to reduce realloc of tabs, at the cost of heavier computation

#define FORCEONMOUSEACTIVE true
#define FORCEONMOUSESTRENGtH 500

#define GRAVITY 1

#define PARTITIONACTIVE true

#define COLLISION true 

#define BORDERCOLLISIONACTIVE true



#define BACKGROUND_R 20
#define BACKGROUND_G 20
#define BACKGROUND_B 20
#define BACKGROUND_A 0

#define CLEAR_SCREEN_MODE 0//0-> black screen; 1-> {x-x/20, y-y/20, z-z/20, . }, 2-> {x-y/20, y-z/20, z-x/20, . }

#define DRAW_CIRCLE_FILLED false
#define DRAW_CIRCLE_EDGE false
#define GAUSSIAN_BLUR false




//Debug const
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



//Placement and color for the text showing the FPS and the nb of particules
#define MESSAGE_RECT { 10, 10, 150, 50 }
#define MESSAGE_COLOR { 255, 255, 255 }





#define DELTA_TIME 0.001f //Delta of the simulation. Higher speed up the simulation at the cost of stability and vice-versa                                                                              //TODO -> find a formula 






//Metaballs parameters
#define METABALLS_RENDERING true //True to render metaballs, else false
#define METABALLS_BORDER_ACTIVE true
#define METABALLS_RADIUSCOMPAREDTOPARTICULESIZE 2 //Directly affect metaballs quality, as the performances
#define METABALLS_INTENSITY 100
#define METABALLS_ATTENUATIONFACTOR 0.05 //Need to be manually change with PARTICULE_SIZE																												  //TODO -> find a formula 

#define METABALLS_BORDER_SIZEINPIXEL 1.5 //Size in pixel of the metaball border render
#define METABALLS_MEDIUM_SIZEINPIXEL 2 //To change if you want to extend the size of the metaball over the particule 
#define METABALLS_THRESHOLD_MEDIUM (float)METABALLS_INTENSITY*expf(-((float)METABALLS_ATTENUATIONFACTOR * (PARTICULE_SIZE+METABALLS_MEDIUM_SIZEINPIXEL)*(PARTICULE_SIZE+METABALLS_MEDIUM_SIZEINPIXEL)))   //TODO -> precalculate to avoid calculation in runtime
#define METABALLS_THRESHOLD_PARTICULE (float)METABALLS_INTENSITY*expf(-((float)METABALLS_ATTENUATIONFACTOR * PARTICULE_SIZE*PARTICULE_SIZE))															  //TODO -> precalculate to avoid calculation in runtime
#define METABALLS_THRESHOLD_BORDER (float)METABALLS_INTENSITY*expf(-((float)METABALLS_ATTENUATIONFACTOR * (PARTICULE_SIZE+METABALLS_MEDIUM_SIZEINPIXEL+METABALLS_BORDER_SIZEINPIXEL)*(PARTICULE_SIZE+METABALLS_MEDIUM_SIZEINPIXEL+METABALLS_BORDER_SIZEINPIXEL)))   //TODO -> precalculate to avoid calculation in runtime
#define METABALLS_THRESHOLD_DEEP (float)METABALLS_INTENSITY*expf(-((float)METABALLS_ATTENUATIONFACTOR * ((PARTICULE_SIZE/10)*(PARTICULE_SIZE/10))))														  //TODO -> precalculate to avoid calculation in runtime




//CUDA
#define CUDA_DEVICE 0 //0 if you wanna use your first GPU, 1 if you wanna use your second GPU, 2 if ...
#define NBTHREAD 1024 //Partialy used, to change if your card support higher or lower number of threads

#endif