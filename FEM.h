#ifndef FEM_H
#define FEM_H
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <memory>
#include <ctime>
#include "Quadtree.h"

void find_matrixs(Eigen::VectorXd& Theta, const Eigen::VectorXd& PHI, const Eigen::VectorXd& U, const Eigen::VectorXd& PHIvelocity, const std::vector<Coord>& vdNodeCoord, const std::vector<std::vector<int>>& vvEFT,
	const std::vector<std::shared_ptr<Element>>& vpFinalElementList,
	double epsilon, double lambda, double tloop, std::array<double, 9> model,
	Eigen::SparseMatrix<double>& mM11, Eigen::SparseMatrix<double>& mM12, Eigen::SparseMatrix<double>& mM21, Eigen::SparseMatrix<double>& mM22, Eigen::SparseMatrix<double>& mK11, Eigen::SparseMatrix<double>& mK12,
	Eigen::SparseMatrix<double>& mK21, Eigen::SparseMatrix<double>& mK22, Eigen::VectorXd& vF1);

void time_discretization(std::ofstream& fout_time,
	const unsigned tloop, Eigen::VectorXd& Theta, Eigen::VectorXd& PHI, Eigen::VectorXd& U, Eigen::VectorXd& PHIvelocity, Eigen::VectorXd& Uvelocity, double dt, double D,
	Eigen::SparseMatrix<double>& mM11, Eigen::SparseMatrix<double>& mM12, Eigen::SparseMatrix<double>& mM21, Eigen::SparseMatrix<double>& mM22, Eigen::SparseMatrix<double>& mK11, Eigen::SparseMatrix<double>& mK12,
	Eigen::SparseMatrix<double>& mK21, Eigen::SparseMatrix<double>& mK22, Eigen::VectorXd& vF1,
	const std::vector<Coord>& vdNodeCoord, const std::vector<std::vector<int>>& vvEFT,
	const std::vector<std::shared_ptr<Element>>& vpFinalElementList,
	double epsilon, double lambda);

Eigen::MatrixXd get_cotangent(const Eigen::VectorXd& phi, const Eigen::MatrixXd& B);

double f(double phi, double u, double theta, double lambda);

double q(double phi, double k);

void MeshRefinement(unsigned maxLv, double gamma, std::ofstream& fout_time,
	Eigen::VectorXd& Theta, Eigen::VectorXd& PHI, Eigen::VectorXd& U, Eigen::VectorXd& PHIvelocity, Eigen::VectorXd& Uvelocity,
	std::map<Coord, double>& PhiCoordinateList, std::map<Coord, double>& UCoordinateList,
	std::map<Coord, double>&PhiVelocityCoordinateList, std::map<Coord, double>&UVelocityCoordinateList,
	std::vector<Coord>& NodeCoordinates, std::vector<std::vector<int>>& EFT, std::vector<std::vector<std::shared_ptr<Element>>>& LevelElementList,
	std::map<Coord, unsigned>& NodeCoordinateList, std::vector<std::shared_ptr<Element>>& FinalElementList);

#endif
