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
	void cu_find_matrixs(float lambda, float epsilon,unsigned tloop, float dt);
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
	int ncSize = 0;
	int elemSize = 0;

	double* aPHI = 0;
	double* aU = 0;
	int* aEFT = 0;
	int* aNodeNum = 0;
	unsigned char* elementType = 0;
	double* aCoordX = 0;
	double* aCoordY = 0;


	float* adM11 = 0;
	float* adM21 = 0;
	float* adM22 = 0;
	float* adK11 = 0;
	float* adK21 = 0;
	float* adK22 = 0;
	float* adF1 = 0;

	double* aM11 = 0;
	double* aM21 = 0;
	double* aM22 = 0;
	double* aK11 = 0;
	double* aK21 = 0;
	double* aK22 = 0;
	double* aF1 = 0;


};



Eigen::MatrixXd get_cotangent(const Eigen::VectorXd& phi, const Eigen::MatrixXd& B);

double f(double phi, double u, double theta, double lambda);

double q(double phi, double k);



#endif
