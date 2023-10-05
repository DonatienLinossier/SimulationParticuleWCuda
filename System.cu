#include "System.cuh"

System::System(int width, int height, int taillemaxballe, int partsMax, int partsInit) {
    m_width = width;
    m_height = height;
    m_taillemaxballe = taillemaxballe;
    m_partsMax = partsMax;
    m_partsInit = partsInit;
    m_nbCaseX = (width / taillemaxballe) + 1;
    m_sizeCaseX = (float)width / (float)m_nbCaseX;
    m_nbCaseY = (height / taillemaxballe) + 1;
    m_sizeCaseY = (float)height / (float)m_nbCaseY;

};

void System::addParticule(int x, int y) {
    parts.push_back(Particule(x, y));
    //parts.back().toCell2();
    grilleP[parts.back().getCell().x][parts.back().getCell().y].push_back(parts.back().getId());
}

void System::init() {
    grilleP.resize(m_nbCaseX, std::vector<std::vector<int>>(m_nbCaseY));
    for (int i = 0; i < m_partsInit; i++) {
        parts.push_back(Particule(20 + std::rand() % (m_width - 40), 20 + std::rand() % (m_height - 40)));
        //parts[i].toCell2();
        grilleP[parts[i].getCell().x][parts[i].getCell().y].push_back(parts[i].getId());
    }
}


/*void System::partsMoveAndRepartition(bool opti, float dt) {
    int max = parts.size();
    for (int i = 0; i < max; i++) {
        parts[i].force(dt);

        cell retour = parts[i].toCell2();

        if (retour.x == parts[i].getCell().x && retour.y == parts[i].getCell().y && opti) {
            continue;

        }

        int idpart = parts[i].getId();
        auto& grilleCell = grilleP.at(retour.x).at(retour.y);
        for (auto it = grilleCell.begin(); it != grilleCell.end(); it++) {
            if (*it == parts[i].getId()) {
                // Erase the pointer from the vector
                grilleCell.erase(it);
                parts[i].m_actualiazed = true;
                break;  // Stop the loop since the object is found and deleted
            }
        }
        grilleP[parts[i].getCell().x][parts[i].getCell().y].push_back(parts[i].getId());
    }
}*/



/*void System::collisionAndDraw(SDL_Renderer* pRenderer) {
    for (int i = 0; i < parts.size(); i++) {
        parts[i].collision5(parts, grilleP, m_width, m_height, m_nbCaseX, m_nbCaseY);
        parts[i].draw(pRenderer);
    }
}*/


void System::collisionAndDrawGPU(SDL_Renderer* pRenderer) {
    Particule* arr = parts.data();
    Particule* dev_arr = nullptr;
    cudaError_t rc = cudaMalloc((void**)&dev_arr, parts.size() * sizeof(Particule));
    if (dev_arr == nullptr) {
        printf("Could not allocate memory: %d", rc);
        exit(0);
    }
    cudaMemcpy(dev_arr, arr, parts.size() * sizeof(Particule), cudaMemcpyHostToDevice);

    int threadsPerBlock = 1024;
    int nbBloc = (parts.size() + threadsPerBlock - 1) / threadsPerBlock;
    printf("\n launch with : (%d, %d) (%p, %p, %p, %d, %d, %d, %d, %d)", nbBloc, threadsPerBlock, dev_grilleP.ptr, dev_sizeTabs, dev_arr, m_width, m_height, m_nbCaseX, m_nbCaseY, m_partsMax);
    //collisionGPU<<<nbBloc, threadsPerBlock>>>(dev_grilleP, dev_sizeTabs, dev_arr, m_width, m_height, m_nbCaseX, m_nbCaseY, m_partsMax);
    std::cout << "Error: " << cudaGetErrorString(cudaGetLastError()) << '\n';
    cudaDeviceSynchronize();
    
    //cudaMemcpy(arr, dev_arr, parts.size() * sizeof(Particule), cudaMemcpyDeviceToHost);

    free(dev_sizeTabs);
    cudaFree(dev_arr);
    cudaFree(dev_grilleP.ptr);
    //parts.assign(arr, arr+parts.size());
    for (int i = 0; i < parts.size(); i++) {
        //parts[i].collision5(parts, grilleP, m_width, m_height, m_nbCaseX, m_nbCaseY);
        parts[i].draw(pRenderer);
    }
}






__global__ void collisionGPU(cudaPitchedPtr dev_grilleP, int* sizeArrayGrilleP, Particule* parts, int W, int H, int CASEMAXX, int CASEMAXY, int max) {
    printf("first block");
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    printf("\nindex : %d", index);
    if (index >= max) {
        return;
    }

    //creer le tab 3D;

    //push dans tab avec parts[index].getCell();
    int kmin = -1;
    int kmax = 2;

    if (parts[index].getCell().x == 0) {
        kmin = 0;
    }
    else if (parts[index].getCell().x == CASEMAXX - 1) {
        kmax = 1;
    }

    int jmin = -1;
    int jmax = 2;
    if (parts[index].getCell().y == 0) {
        jmin = 0;
    }
    else if (parts[index].getCell().y == CASEMAXY - 1) {
        jmax = 1;
    }


    Particule* OtherBall;
    char* data = static_cast<char*>(dev_grilleP.ptr);
    size_t pitch = dev_grilleP.pitch;
    size_t slicePitch = pitch * CASEMAXY;

    for (int k = kmin; k < kmax; ++k) {
        int caseX = parts[index].getCell().x + k;
        char* slice = data + caseX * slicePitch;
        for (int j = jmin; j < jmax; ++j) {
            int caseY = parts[index].getCell().y + j;
            int* test = reinterpret_cast<int*>(slice + caseY * pitch);
            int size = sizeArrayGrilleP[caseX * W + caseY];

            for (int i = 0; i < size; ++i) {
                OtherBall = &parts[test[i]];
                if (&parts[index] == OtherBall) {
                    continue;
                }

                int distAttendu = parts[index].getRadius() + OtherBall->getRadius();
                float Xdist = (float) parts[index].getX() - OtherBall->getX();
                float Ydist = (float)parts[index].getY() - OtherBall->getY();
                float dist = sqrt(Xdist * Xdist + Ydist * Ydist);


                if (dist < distAttendu) {
                    float delta = distAttendu - dist;
                    parts[index].setTension(parts[index].getTension() + delta);
                    OtherBall->setTension(OtherBall->getTension() + delta);

                    if (dist == 0) {
                        //décale légerment la boule, ce qui corrigera le pb le tour d'apres
                        //m_x += 1;
                        parts[index].setY(parts[index].getY() + 1);

                        //Permet de passer ce tour de boucle
                        dist = 1;
                    }


                    float nx = Xdist / dist;
                    float multiple = 0.5 * delta;
                    float dtx = multiple * nx;
                    OtherBall->setX(OtherBall->getX() - dtx);

                    float ny = Ydist / dist;
                    float dty = multiple * ny;
                    parts[index].setY(parts[index].getY() + dty);
                    OtherBall->setY(OtherBall->getY() - dty);

                }
            }
        }
    }

    if (parts[index].getX() + parts[index].getRadius() > W) {
        parts[index].setX(W - parts[index].getRadius());
    }
    else if (parts[index].getX() - parts[index].getRadius() < 0) {
        parts[index].setX(parts[index].getRadius());
    }

    if (parts[index].getY() + parts[index].getRadius() > H) {
        parts[index].setY(H - parts[index].getRadius());
    }
    else if (parts[index].getY() - parts[index].getRadius() < 0) {
        parts[index].setY(parts[index].getRadius());
    }
}