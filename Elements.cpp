#include "Elements.h"

#include <Eigen/Dense>
using namespace Eigen;

MatrixXd NodeCoordinates(const double dx, const int Nx, const int Ny) {
    MatrixXd nodeCoord((Nx+1)*(Ny+1),2);
    for (int i=0; i<Nx+1; ++i) {
        for (int j=0; j<Ny+1; ++j) {
            nodeCoord(i*(Ny+1)+j, 0) = i * dx;
            nodeCoord(i*(Ny+1)+j, 1) = j * dx;
        }
    }
    return nodeCoord;
}

MatrixXi ElementNodes(MatrixXd nodeCoord, const int Nx, const int Ny) {
    MatrixXi elementNodes(Nx*Ny,4);
    for (int i=0; i<Nx; ++i) {
        for (int j=0; j<Ny; ++j) {
            elementNodes(i*Ny+j,0) = i*(Ny+1) + j;
            elementNodes(i*Ny+j,1) = (i+1)*(Ny+1) + j;
            elementNodes(i*Ny+j,2) = (i+1)*(Ny+1) + j + 1;
            elementNodes(i*Ny+j,3) = i*(Ny+1) + j + 1;
        }
    }
    return elementNodes;
}



