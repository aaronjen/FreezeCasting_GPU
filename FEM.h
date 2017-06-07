#ifndef FEM_H
#define FEM_H
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <memory>
#include <ctime>
#include "Quadtree.h"

class FEM{
public:
	FEM(unsigned maxLv,
		double gamma,
		std::ofstream& fout_time,
		Eigen::VectorXd& Theta,
		Eigen::VectorXd& PHI,
		Eigen::VectorXd& U,
		Eigen::VectorXd& PHIvelocity,
		Eigen::VectorXd& Uvelocity,
		std::map<Coord, double>& PhiCoordinateList,
		std::map<Coord, double>& UCoordinateList,
		std::map<Coord, double>& PhiVelocityCoordinateList,
		std::map<Coord, double>& UVelocityCoordinateList,
		std::vector<Coord>& NodeCoordinates,
		std::vector<std::vector<int>>& EFT,
		std::vector<std::vector<std::shared_ptr<Element>>>& LevelElementList,
		std::map<Coord, unsigned>& NodeCoordinateList,
		std::vector<std::shared_ptr<Element>>& FinalElementList) : 
		maxLv(maxLv),
		gamma(gamma),
		fout_time(fout_time),
		Theta(Theta),
		PHI(PHI),
		U(U),
		PHIvelocity(PHIvelocity),
		Uvelocity(Uvelocity),
		PhiCoordinateList(PhiCoordinateList),
		UCoordinateList(UCoordinateList),
		PhiVelocityCoordinateList(PhiVelocityCoordinateList),
		UVelocityCoordinateList(UVelocityCoordinateList),
		NodeCoordinates(NodeCoordinates),
		EFT(EFT),
		LevelElementList(LevelElementList),
		NodeCoordinateList(NodeCoordinateList),
		FinalElementList(FinalElementList) {
	};

	void MeshRefinement();
	void find_matrixs(double lambda, double epsilon,unsigned tloop, double dt);
	void time_discretization(double lambda, double epsilon,unsigned tloop, double dt);

private:
	unsigned maxLv;
	double gamma;
	std::ofstream& fout_time;
	Eigen::VectorXd& Theta;
	Eigen::VectorXd& PHI;
	Eigen::VectorXd& U;
	Eigen::VectorXd& PHIvelocity;
	Eigen::VectorXd& Uvelocity;
	std::map<Coord, double>& PhiCoordinateList;
	std::map<Coord, double>& UCoordinateList;
	std::map<Coord, double>& PhiVelocityCoordinateList;
	std::map<Coord, double>& UVelocityCoordinateList;
	std::vector<Coord>& NodeCoordinates;
	std::vector<std::vector<int>>& EFT;
	std::vector<std::vector<std::shared_ptr<Element>>>& LevelElementList;
	std::map<Coord, unsigned>& NodeCoordinateList;
	std::vector<std::shared_ptr<Element>>& FinalElementList;


	// create by meshrefinment
	Eigen::SparseMatrix<double> mM11;
	Eigen::SparseMatrix<double> mM21;
	Eigen::SparseMatrix<double> mM22;
	Eigen::SparseMatrix<double> mK11;
	Eigen::SparseMatrix<double> mK21;
	Eigen::SparseMatrix<double> mK22;
	Eigen::VectorXd vF1;
};



Eigen::MatrixXd get_cotangent(const Eigen::VectorXd& phi, const Eigen::MatrixXd& B);

double f(double phi, double u, double theta, double lambda);

double q(double phi, double k);



#endif
