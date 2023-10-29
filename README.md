# 🌌 CUDA Particle Simulation README

## Table of Contents
- [Introduction](#introduction)
- [Project Inspiration](#project-inspiration)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Configuration](#configuration)
- [Simulation Parameters](#simulation-parameters)
- [Future Features](#future-features)

## 🚀 Introduction
The CUDA Particle Simulation is a parallelized particle simulation program that utilizes CUDA to accelerate computations.

## 🌟 Project Inception
This project was born out of my inspiration from Pezza's work, which featured a particle simulation using Verlet integration. My initial Python attempt faced limitations, as it struggled to handle more than 5000 particles while maintaining a 60 FPS performance.
To overcome this, I embarked on the journey to recreate the simulation in C++ and optimize it for handling hundreds of thousands of particles while maintaining a consistent 60 FPS performance. This project served as my gateway to the world of GPU computing and CUDA, involving substantial research and hands-on learning in parallel computing on GPUs.

## 🔬 Features
- Particle simulation with collision detection and response.
- Visualization of particles using CUDA and SDL.
- Dynamic grid-based partitioning of particles for efficient collision detection.
- GPU computing of the simulation.
- Metaball rendering for visual representation of particles.
- User interaction with the simulation using the mouse.
- Adjustable simulation parameters and visual effects.

## ⚙️ Prerequisites
- CUDA-enabled GPU with compute capability 3.0 or higher.
- SDL library for visualization.

## 🖱 Usage
- Left-click to attract particles.
- Right-click to repulse particles.
- Middle-click to add particles at the cursor position.
- Scroll the mouse wheel to add more particles.

## 🛠️ Configuration
The behavior of the simulation can be configured by modifying the code. You can adjust parameters such as particle size, grid size, forces, and rendering options in the source code. These options are usually found in the relevant source files.

## 📊 Simulation Parameters
You can fine-tune the simulation by modifying parameters in the source code. Some important parameters include:
- Particle size and properties.
- Delta time between computations.
- Forces applied to particles.
- Several rendering parameters.

## 🔮 Future Features
In the pipeline for this project, we have plans to implement the following exciting features:
- **Add polygon for dynamic and more diverse environments:** Introducing polygons into the simulation to create dynamic and varied environments for particles to interact with.
- **Add new parameters for the particle to make the simulation more realistic:** Expanding the range of particle properties to enhance the realism of the simulation (Viscosity, Temperature, Surface Tension.. ).
- **More possibilities for rendering:** Providing additional rendering options for a richer visual experience and better understanding of the circonstances of the simulation. 
- **Possibility to import environments:** Allowing users to import custom environments for their simulations.
- **Possibility to batch render:** Add the possibility to batch rendering, and naviguate through them.
