#ifndef VARIABLES_H
#define VARIABLES_H
#include <fstream>
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include <map>
#include "Quadtree.h"

void read_input(const double a_2, unsigned& maxLv, double& gamma, double& Nx, double& Ny, unsigned& file_skip, unsigned& mesh_skip, unsigned& tmax, double& dt,
    double& delta, double& lambda, double& alpha, double& a_12, double& a_s, double& ephilon,
	std::vector<std::vector<int>>& vvEFT, Eigen::VectorXd& Theta, Eigen::VectorXd& PHI, Eigen::VectorXd& U, Eigen::VectorXd& PHIvelocity, Eigen::VectorXd& Uvelocity,
	std::vector<Coord>& vcNodeCoordinates,	std::vector<std::shared_ptr<Element>>& vpFinalElementList,
	std::map<Coord, unsigned>& mNodeCoordinateList, std::vector<std::vector<std::shared_ptr<Element>>>& LevelElementList,
	std::map<Coord, double>& mPhiCoordinateList, std::map<Coord, double>& UCoordinateList,
	std::map<Coord, double>& mPhiVelocityCoordinateList, std::map<Coord, double>& mUVelocityCoordinateList);

std::ifstream& open_file(std::ifstream& fin, const std::string& fname); 

void PF_Initialization(double Nx, double Ny, std::vector<std::shared_ptr<Element>>& vpFinalElementList, 
					  std::vector<Coord>& vcNodeCoordinates, std::map<Coord, double>& mPhiCoordinateList, 
					  std::map<Coord, double>& mUCoordinateList, Eigen::VectorXd& Theta, Eigen::VectorXd& PHI, Eigen::VectorXd& U, Eigen::VectorXd& PHIvelocity, Eigen::VectorXd& Uvelocity, double delta);

void Output(unsigned tloop, std::ofstream& fout_plot, 
			std::ofstream& fout_PHI, std::ofstream& fout_U, std::ofstream& fout_X, std::ofstream& fout_Y,
			const Eigen::VectorXd& Theta, const Eigen::VectorXd& PHI, const Eigen::VectorXd& U, const std::vector<Coord> NodeCoordinates,
			std::ofstream& foutX, std::ofstream& foutY, std::vector<std::shared_ptr<Element>>& FinalElementList,
			std::vector<std::vector<std::shared_ptr<Element>>>& LevelElementList);

#endif // VARIABLES_H
