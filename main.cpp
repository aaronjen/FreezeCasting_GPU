#include <iostream>
#include <fstream>
#include <cmath>
#include <memory>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <ctime>
#include <cstdlib>
#include "Variables.h"
#include "ShapeFunctions.h"
#include "FEM.h"
#include "GaussPoints.h"
#include "Quadtree.h"
using namespace std;
using namespace Eigen;



int main() {
	clock_t t, t_init=clock();
	unsigned maxLv, tmax, file_skip, mesh_skip;
	double gamma, dt, delta, epsilon, lambda, D, Nx, Ny;
    double a_12, a_s;
    const double a_2 = 0.6267; 
	vector<Coord> NodeCoordinates;
	vector<vector<int>> EFT;
	vector<shared_ptr<Element>> FinalElementList;
    VectorXd PHI, U, Theta;
	VectorXd PHIvelocity, Uvelocity;
	map<Coord, unsigned> NodeCoordinateList;
	vector<vector<shared_ptr<Element>>> LevelElementList;
	map<Coord, double> PhiCoordinateList;
	map<Coord, double> UCoordinateList;
	map<Coord, double> PhiVelocityCoordinateList;
	map<Coord, double> UVelocityCoordinateList;
	ofstream fout_PHI, fout_U, fout_X, fout_Y, foutX, foutY, fout_plot;
	ofstream fout_time("time_step.txt");
	srand(2);
	
	// Read input
	read_input(a_2, maxLv, gamma, Nx, Ny, file_skip, mesh_skip, tmax, dt, delta, lambda, D, a_12, a_s, epsilon, EFT, Theta, PHI, U, PHIvelocity, Uvelocity,
			   NodeCoordinates, FinalElementList, NodeCoordinateList, LevelElementList, PhiCoordinateList, UCoordinateList, PhiVelocityCoordinateList, UVelocityCoordinateList);


	// time marching
	fout_time << "Start!" << endl;

	// input data
	FEM fem = FEM(maxLv, gamma, fout_time, Theta, PHI, U, PHIvelocity, Uvelocity, PhiCoordinateList, UCoordinateList, PhiVelocityCoordinateList, UVelocityCoordinateList,NodeCoordinates, EFT, LevelElementList, NodeCoordinateList, FinalElementList);
	for (unsigned tloop=0; tloop<tmax; tloop++) {
		//t = clock();

		
		// Mesh refinement
		if ((tloop) % mesh_skip == 0){
			fem.MeshRefinement();
		}

		t = clock();

		// Output initial condition
		if (tloop == 0)
			Output(0, fout_plot, fout_PHI, fout_U, fout_X, fout_Y, Theta, PHI, U, NodeCoordinates, foutX, foutY, FinalElementList, LevelElementList);

		// Output time step
       
		if ((tloop + 1) % (file_skip / 10) == 0) {
			fout_time << "time step = " << tloop + 1 << endl;
			cout << "time step = " << tloop + 1 << endl;
		}
	
		// FEM
		fem.time_discretization(lambda, epsilon, tloop, dt);

		// Output
		if ((tloop + 1) % file_skip == 0)
			Output(tloop + 1, fout_plot, fout_PHI, fout_U, fout_X, fout_Y, Theta, PHI, U, NodeCoordinates, foutX, foutY, FinalElementList, LevelElementList);

		t = clock() - t_init;
		//fout_time << "\ttotal:  " << 1.*t/CLOCKS_PER_SEC << " sec" << endl;
		if ((tloop + 1) % (file_skip / 10) == 0) {
			int sec = t / CLOCKS_PER_SEC;
			int min = sec / 60;
			sec %= 60;
			int hour = min / 60;
			min %= 60;
			int day = hour / 24;
			hour %= 24;
			fout_time << "\ttotal:  " << 1.*t / CLOCKS_PER_SEC << " sec" << endl;
			cout << "\ttotal: " << day << " days " << hour << " hr " << min << " min " << sec << " sec" << endl;
		}
	}
    cout << "All Done!" << endl;
    fout_time << "All Done!" << endl;
    
	return 0;
}
