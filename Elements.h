#ifndef ELEMENTS_H
#define ELEMENTS_H
#include <Eigen/Dense>

Eigen::MatrixXd NodeCoordinates(const double dx, const int Nx, const int Ny);

Eigen::MatrixXi ElementNodes(Eigen::MatrixXd nodeCoord, const int Nx, const int Ny);

#endif // ELEMENTS_H
