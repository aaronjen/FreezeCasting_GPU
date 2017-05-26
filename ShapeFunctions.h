#ifndef SHAPEFUNCTIONS_H
#define SHAPEFUNCTIONS_H
#include <Eigen/Dense>
#include <bitset>

Eigen::RowVectorXd ShapeFunction(double xi, double eta, std::bitset<8> bitElementType);

Eigen::MatrixXd NaturalDerivatives(double xi, double eta, std::bitset<8> bitElementType);

Eigen::Matrix2d Jacobian(const Eigen::MatrixXd& nodeCoord, const Eigen::MatrixXd& naturalDerivatives);

Eigen::Matrix2d invJacobian(const Eigen::MatrixXd& nodeCoord, const Eigen::MatrixXd& naturalDerivatives);

Eigen::MatrixXd XYDerivatives(const Eigen::MatrixXd& nodeCoord, const Eigen::MatrixXd& naturalDerivatives);

double detJacobian(const Eigen::MatrixXd& nodeCoord, const Eigen::MatrixXd& naturalDerivatives);

#endif
