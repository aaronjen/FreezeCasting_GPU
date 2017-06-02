#include "Variables.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <Eigen/Dense>
#include <vector>
#include <map>
#include "Quadtree.h"
using namespace std;
using namespace Eigen;

void read_input(const double a_2, unsigned& maxLv, double& gamma, double& Nx, double& Ny, unsigned& file_skip, unsigned& mesh_skip, unsigned& tmax, double& dt,
    double& delta, double& lambda, double& D, double& a_12, double& a_s, double& ephilon,
	vector<vector<int>>& vvEFT, VectorXd& Theta, VectorXd& PHI, VectorXd& U, VectorXd& PHIvelocity, VectorXd& Uvelocity, vector<Coord>& vcNodeCoordinates,
	vector<shared_ptr<Element>>& vpFinalElementList, map<Coord, unsigned>& mNodeCoordinateList, vector<vector<shared_ptr<Element>>>& LevelElementList,
	map<Coord, double>& mPhiCoordinateList, map<Coord, double>& mUCoordinateList,
	map<Coord, double>& mPhiVelocityCoordinateList, map<Coord, double>& mUVelocityCoordinateList) {

    ofstream fout("initial.txt");
    // read in initial parameters form file
    string fname = "input";
    ifstream fin;
    if (!open_file(fin, fname)) {
        cerr << "Complain: I cannot find the file \"input\"" << endl << endl;
        exit(0);
    }

    string dump;
	fin >> maxLv >> dump >> gamma >> dump >> dt >> dump >> file_skip >> dump >> mesh_skip >> dump >> tmax >> dump
        >> D >> dump >> delta >> dump >> ephilon;

	Nx = 0.8 * pow(2.0, maxLv);
	Ny = 0.8 * pow(2.0, maxLv);

	// Quadtree initialization
	mPhiCoordinateList[Coord(Nx, Ny)] = 0;
	Quadtree_Initialization(Nx, Ny, maxLv, gamma, LevelElementList, mNodeCoordinateList, vvEFT, 11, mPhiCoordinateList, mUCoordinateList, mPhiVelocityCoordinateList, mUVelocityCoordinateList); // case = 11
	Quadtree_AddNodes(LevelElementList, mNodeCoordinateList);
	ReportElement(LevelElementList, vpFinalElementList, mNodeCoordinateList, vvEFT, vcNodeCoordinates);

	// Initialize PHI & U
	PF_Initialization(Nx, Ny, vpFinalElementList, vcNodeCoordinates, mPhiCoordinateList, mUCoordinateList, Theta, PHI, U, PHIvelocity, Uvelocity, delta);
 
	lambda = D / a_2;
    a_12 = 4.0 * a_s * ephilon;

    fout << "----------------------" << endl;
    fout << " delta     = " << delta << endl;
    fout << " ephilon_4 = " << ephilon << endl;
    fout << " lambda    = " << lambda << endl;
	fout << " D         = " << D << endl;
    fout << " dt        = " << dt << endl; 
    fout << " tmax      = " << tmax << endl;
    fout << " file_skip = " << file_skip << endl;
	fout << " mesh_skip = " << mesh_skip << endl;
	fout << " maxLv     = " << maxLv << endl;
    fout << " Nx        = " << Nx << endl;
    fout << " Ny        = " << Ny << endl;
    fout << "----------------------" << endl;
    fout.close();
}

ifstream& open_file(ifstream& fin, const string& fname) {
    fin.close();
    fin.clear();
    fin.open(fname.c_str());
    return fin;
}

void PF_Initialization(double Nx, double Ny, vector<shared_ptr<Element>>& vpFinalElementList,
	vector<Coord>& vcNodeCoordinates, map<Coord, double>& mPhiCoordinateList, 
	map<Coord, double>& mUCoordinateList, VectorXd& Theta, VectorXd& PHI, VectorXd& U,
	VectorXd& PHIvelocity, VectorXd& Uvelocity, double delta) {
	mPhiCoordinateList.clear();
	mUCoordinateList.clear();
	double x_mid = Nx / 2.0;
	double y_mid = Ny / 2.0;
	double rad = x_mid / 256;
	double seeds = 4;

	double m = abs(-2.6);					// liquidus solpe (K/wt%)						//
	double C_inf = 3;						// alloy composition (wt%)						//
	double k = 0.14;						// partition coefficient						//
	double G = 140 * 40;					// thermal gradient (K/cm)						//
	double Ts = 273;						// solidus temperature (K)						//
	double dT0 = m * C_inf * (1 - k) / k;	// equilibrium freezing temperature range (K)	//
	double T0 = Ts - dT0 / 10;				// reference temperature (K)					//
	double Vp = 3000;						// pulling speed (1E-6m/s)						//
	double dTn = 8;							// nucleation undercooling (K)					//
	double d0 = 5E-3;						// chemical capillary length (1E-6m)			//
	double alpha = 3000;					// liquid diffusion coefficient (1E-6m2/s)		//
	double C;								// "mixture" concentration (wt%)				//

	for (const auto Node : vcNodeCoordinates) {
		double dist = sqrt(pow(Node.x - x_mid, 2.0) + pow(Node.y - y_mid, 2.0)) - rad;
		//double dist = sqrt(pow(Node.x,2.0) + pow(Node.y,2.0)) - rad;

		/*double dist = sqrt(pow(Node.x-x_mid*2, 2.0) + pow(Node.y, 2.0)) - rad;
		for (int s = 0; s < seeds; s++)
			if (sqrt(pow(Node.x - s * x_mid * 2.0 / seeds, 2.0) + pow(Node.y, 2.0)) - rad < dist)
				dist = sqrt(pow(Node.x - s * x_mid * 2.0 / seeds, 2.0) + pow(Node.y, 2.0)) - rad;*/

		//if (Node.x < 0 || Node.x > 1638.4 || Node.y > rad)
		//	mPhiCoordinateList[Node] = -1;
		//else
		//	mPhiCoordinateList[Node] = 1;
		//double dist = sqrt(pow(Node.x - 819.2, 2.0) + pow(Node.y, 2.0)) - rad;
		//double dist = Node.y - rad;
		mPhiCoordinateList[Node] = -tanh(dist / sqrt(2));
		C = -0.64 * mPhiCoordinateList[Node] / 2;
		mUCoordinateList[Node] = -delta * (mPhiCoordinateList[Node] / 2 - 0.5); // 0 --> delta
		//mUCoordinateList[Node] = ((2 * C * k / C_inf) / (1 + k - (1 - k) * mPhiCoordinateList[Node]) - 1) / (1 - k);
		//mUCoordinateList[Node] = -1;
	}
	PHI.resize(vcNodeCoordinates.size());
	U.resize(vcNodeCoordinates.size());
	Theta.resize(vcNodeCoordinates.size());
	PHIvelocity.resize(vcNodeCoordinates.size());
	Uvelocity.resize(vcNodeCoordinates.size());
	for (unsigned i = 0; i < vcNodeCoordinates.size(); i++) {
		PHI(i) = mPhiCoordinateList[vcNodeCoordinates[i]];
		U(i) = mUCoordinateList[vcNodeCoordinates[i]];
		Theta(i) = 0;
		PHIvelocity(i) = 0;
		Uvelocity(i) = 0;
	}
}

void Output(unsigned tloop, ofstream& fout_plot,
			ofstream& fout_PHI, ofstream& fout_U, ofstream& fout_X, ofstream& fout_Y,
	const VectorXd& Theta, const VectorXd& PHI, const VectorXd& U, const vector<Coord> NodeCoordinates,
			ofstream& foutX, ofstream& foutY, vector<shared_ptr<Element>>& FinalElementList, 
			vector<vector<shared_ptr<Element>>>& LevelElementList) {
	string tstep;
	stringstream num2str;
	num2str << tloop;
	num2str >> tstep;

	string outputPHI = "outPHI_";
	string outputU = "outU_";
	string outputX = "outX_";
	string outputY = "outY_";
	string outputPlot = "outPlot_";
	string outputXmesh = "meshX_";
	string outputYmesh = "meshY_";

	string outputPHI_tstep = outputPHI + tstep;
	string outputU_tstep = outputU + tstep;
	string outputX_tstep = outputX + tstep;
	string outputY_tstep = outputY + tstep;
	string outputPlot_tstep = outputPlot + tstep;
	outputPlot_tstep += ".dat";
	string outputXmesh_tstep = outputXmesh + tstep;
	string outputYmesh_tstep = outputYmesh + tstep;

	fout_PHI.open(outputPHI_tstep.c_str());
	fout_U.open(outputU_tstep.c_str());
	fout_X.open(outputX_tstep.c_str());
	fout_Y.open(outputY_tstep.c_str());
	fout_plot.open(outputPlot_tstep.c_str());
	foutX.open(outputXmesh_tstep.c_str());
	foutY.open(outputYmesh_tstep.c_str());

	fout_plot << "VARIABLES = \"X\", \"Y\", \"PHI\", \"U\", \"Theta\"" << endl;
	fout_plot << "ZONE T = \"" << tstep << "\"" << endl;
	fout_PHI.precision(15);
	fout_U.precision(15);
	for (int i = 0; i < PHI.rows(); i++) {
		if (NodeCoordinates[i].x >= 0 && NodeCoordinates[i].x <= 1638.4) {
			fout_PHI << PHI(i) << endl;
			fout_U << U(i) << endl;
			fout_X << NodeCoordinates[i].x << endl;
			fout_Y << NodeCoordinates[i].y << endl;
			fout_plot << NodeCoordinates[i].x << " " << NodeCoordinates[i].y << " " << PHI(i) << " " << U(i) << " " << Theta(i) << endl;
		}
	}

	// Output element boundary for plotting
	for (unsigned lv = 0; lv < LevelElementList.size(); lv++) {
		for (unsigned e = 0; e < LevelElementList[lv].size(); e++) {
			if (LevelElementList[lv][e]->apChildren[0] == 0) {
				for (int i = 0; i < 4; i++) {
					if (LevelElementList[lv][e]->acNodalCoordinates[i].x >= 0 && LevelElementList[lv][e]->acNodalCoordinates[i].x <= 1638.4) {
						foutX << LevelElementList[lv][e]->acNodalCoordinates[i].x << endl;
						foutY << LevelElementList[lv][e]->acNodalCoordinates[i].y << endl;
					}
				}
			}
		}
	}
	
	fout_PHI.close();
	fout_U.close();
	fout_X.close();
	fout_Y.close();
	fout_plot.close();
	foutX.close();
	foutY.close();
}
