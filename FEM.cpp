#include "FEM.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <vector>
#include <memory>
#include <fstream>
#include <ctime>
#include "GaussPoints.h"
#include "ShapeFunctions.h"
#include "Quadtree.h"
using namespace std;
using namespace Eigen;

void find_matrixs(VectorXd& Theta, const VectorXd& PHI, const VectorXd& U, const VectorXd& PHIvelocity, const vector<Coord>& vdNodeCoord, const vector<vector<int>>& vvEFT,
	const vector<shared_ptr<Element>>& vpFinalElementList,
    double epsilon, double lambda, double tloop, array<double, 9> model,
    SparseMatrix<double>& mM11, SparseMatrix<double>& mM12, SparseMatrix<double>& mM21, SparseMatrix<double>& mM22, SparseMatrix<double>& mK11, SparseMatrix<double>& mK12,
    SparseMatrix<double>& mK21, SparseMatrix<double>& mK22, VectorXd& vF1) {
    
	// initialization
	const double PI = 4 * atan(1.0);
	double C_inf = model[0];	// (wt%)
	double k = model[1];		// 
	double G = model[2];		// (K/cm)
	double d0 = model[3];		// (1E-6m)
	double alpha = model[4];	// (1E-6m2/s)
	double Vp = model[5];		// (1E-6m/s)
	double T0 = model[6];		// (K)
	double Ts = model[7];		// (K)
	double dT0 = model[8];		// (K)
	double a1 = 0.8839;
	double a2 = 0.6267;
	double W0 = d0 * lambda / a1; // (1E-6m)
	double Tau0 = a2 * lambda * W0 * W0 / alpha; // (s)
	double D = lambda * a2;

	mM11.setZero(); mM12.setZero(); mM21.setZero(); mM22.setZero(); mK11.setZero(); mK12.setZero(); mK21.setZero(); mK22.setZero(); vF1.setZero();
    typedef Triplet<double> T;
    vector<T> tripletList_M11, tripletList_M12, tripletList_M21, tripletList_M22, tripletList_K11, tripletList_K12, tripletList_K21, tripletList_K22;
	int nGp = 2 * 2; // 2 x 2 Gauss point
    MatrixXd LocationsAndWeights = gauss2D(nGp);
	RowVector2d aniso_parameter;
	int m = 6;

	for (unsigned e = 0; e < vvEFT.size(); e++) {
		size_t numNodePerElement = vvEFT[e].size();
		bitset<8> bitElementType = vpFinalElementList[e]->bitElementType;
        // get the coordinates of the nodes in the element
        MatrixXd elementNodesCoord(numNodePerElement,2); // n x 2
        VectorXd phi(numNodePerElement); // n x 1
        VectorXd u(numNodePerElement); // n x 1
		VectorXd v(numNodePerElement); // n x 1

		for (unsigned i = 0; i < numNodePerElement; i++) {
			elementNodesCoord(i, 0) = vdNodeCoord[vvEFT[e][i]].x;
			elementNodesCoord(i, 1) = vdNodeCoord[vvEFT[e][i]].y;
			phi(i) = PHI[vvEFT[e][i]];
			u(i) = U[vvEFT[e][i]];
			v(i) = PHIvelocity[vvEFT[e][i]];
        }
		double RealCoord = W0 * 0.25*(elementNodesCoord(0, 1) + elementNodesCoord(1, 1) + elementNodesCoord(2, 1) + elementNodesCoord(3, 1)) * 1E-6; // n x 2 (m)
		double RealTime = Tau0 * tloop; // (s)

		MatrixXd Ce = MatrixXd::Zero(numNodePerElement, numNodePerElement);
		MatrixXd Ae = MatrixXd::Zero(numNodePerElement, numNodePerElement);
		MatrixXd Ee = MatrixXd::Zero(numNodePerElement, numNodePerElement);
		VectorXd Fe = VectorXd::Zero(numNodePerElement); // n x 1

		RowVectorXd N0 = ShapeFunction(0, 0, bitElementType);
		MatrixXd dN0 = NaturalDerivatives(0, 0, bitElementType); // 2 x n
		MatrixXd B0 = XYDerivatives(elementNodesCoord, dN0); // 2 x n
		MatrixXd cotangent = get_cotangent(phi, B0); // 2 x 1
		double DERX = cotangent(0);
		double DERY = cotangent(1);
		double angle = atan2(DERY, DERX);
		double as = 1 + epsilon * cos(m*(angle - PI / 6)); // A(theta)
		double asp = -m * epsilon * sin(m*(angle - PI / 6)); // A'(theta)
		//double as = 1 + epsilon * cos(m*(angle)); // A(theta)
		//double asp = -m * epsilon * sin(m*(angle)); // A'(theta)
		double Temperature = T0 + G * 1E2 * (W0 * N0 * elementNodesCoord.col(1) - Vp*RealTime) * 1E-6; // (K)
		double theta = (Temperature - Ts) / dT0;

		// cycle for Gauss point
        for (int q=0; q<nGp; q++) {
            double xi = LocationsAndWeights(q,0);
            double eta = LocationsAndWeights(q,1);
            double W = LocationsAndWeights(q,2);
			RowVectorXd N = ShapeFunction(xi, eta, bitElementType); // 1 x n
			MatrixXd dN = NaturalDerivatives(xi, eta, bitElementType); // 2 x n
			MatrixXd B = XYDerivatives(elementNodesCoord, dN); // 2 x n
			double J = detJacobian(elementNodesCoord, dN); // 1 x 1

            // matrixs of a element
            Ce     += N.transpose() * N * W * J; // n x n
            Ae     -= B.transpose() * B * W * J; // n x n
			Ee	   -= (B.row(1).transpose()*B.row(0) - B.row(0).transpose()*B.row(1)) * W * J; // n x n
			Fe	   += N.transpose() * f(N*phi, N*u, theta, lambda) * W * J; // n x 1
        }
        // cycle for element matrixs
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        for (unsigned i=0; i<numNodePerElement; i++) {
            for (unsigned j=0; j<numNodePerElement; j++) {
				if (Ce(i, j) > 1.0E-12 || Ce(i, j) < -1.0E-12) {
					tripletList_M22.push_back(T(vvEFT[e][i], vvEFT[e][j], Ce(i, j)));
					tripletList_M21.push_back(T(vvEFT[e][i], vvEFT[e][j], -0.5*Ce(i, j)));
					tripletList_M11.push_back(T(vvEFT[e][i], vvEFT[e][j], as * as * Ce(i, j)));
				}
				if (Ae(i, j) > 1.0E-12 || Ae(i, j) < -1.0E-12) {
					tripletList_K22.push_back(T(vvEFT[e][i], vvEFT[e][j], -D * q(N0 * phi, 0.7) * Ae(i, j)));
					tripletList_K11.push_back(T(vvEFT[e][i], vvEFT[e][j], -as * as * Ae(i, j)));
				}
				if (Ee(i, j) > 1.0E-12 || Ee(i, j) < -1.0E-12)
					tripletList_K11.push_back(T(vvEFT[e][i], vvEFT[e][j], -as * asp * Ee(i, j)));
            }
			if (Fe(i) > 1.0E-12 || Fe(i) < -1.0E-12)
				vF1(vvEFT[e][i]) += Fe(i);
        }
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		/*for (unsigned i = 0; i<numNodePerElement; i++) {
			for (unsigned j = 0; j<numNodePerElement; j++) {
				if (Ce(i, j) > 1.0E-12 || Ce(i, j) < -1.0E-12) {
					tripletList_C_u1.push_back(T(vvEFT[e][i], vvEFT[e][j], 0.5 * (1 + k) * Ce(i, j)));
					tripletList_C_u2.push_back(T(vvEFT[e][i], vvEFT[e][j], 0.5 * (1 + (1 - k) * N0 * u) * Ce(i, j)));
					tripletList_C_u3.push_back(T(vvEFT[e][i], vvEFT[e][j], 0.5 * (0 + (1 - k) * N0 * phi) * Ce(i, j)));
					tripletList_C_phi.push_back(T(vvEFT[e][i], vvEFT[e][j], (1 - (1 - k) * theta)* as * as * Ce(i, j)));

				}
				if (Ae(i, j) > 1.0E-12 || Ae(i, j) < -1.0E-12) {
					tripletList_A_u1.push_back(T(vvEFT[e][i], vvEFT[e][j], D * q(N0 * phi, 0.7) * Ae(i, j)));
					if (DERX*DERX + DERY*DERY >= 1.0E-12 && N0 * v >= 1.0E-12) tripletList_A_u2.push_back(T(vvEFT[e][i], vvEFT[e][j], (1 + (1 - k) * N0 * u) * (N0 * v + 0) / sqrt(DERX*DERX + DERY*DERY) / 2 / sqrt(2.0) * Ae(i, j)));
					tripletList_A_phi.push_back(T(vvEFT[e][i], vvEFT[e][j], as * as * Ae(i, j)));
				}
				if (Ee(i, j) > 1.0E-12 || Ee(i, j) < -1.0E-12)
					tripletList_E_phi.push_back(T(vvEFT[e][i], vvEFT[e][j], as * asp * Ee(i, j)));
			}
			if (Fe(i) > 1.0E-12 || Fe(i) < -1.0E-12)
				vF_phi(vvEFT[e][i]) += Fe(i);
			Theta(vvEFT[e][i]) = theta;
		}*/
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    }
	mM11.setFromTriplets(tripletList_M11.begin(), tripletList_M11.end());
	mM12.setFromTriplets(tripletList_M12.begin(), tripletList_M12.end());
	mM21.setFromTriplets(tripletList_M21.begin(), tripletList_M21.end());
	mM22.setFromTriplets(tripletList_M22.begin(), tripletList_M22.end());
	mK11.setFromTriplets(tripletList_K11.begin(), tripletList_K11.end());
	mK12.setFromTriplets(tripletList_K12.begin(), tripletList_K12.end());
	mK21.setFromTriplets(tripletList_K21.begin(), tripletList_K21.end());
	mK22.setFromTriplets(tripletList_K22.begin(), tripletList_K22.end());
}

void time_discretization(ofstream& fout_time,
	const unsigned tloop, VectorXd& Theta, VectorXd& PHI, VectorXd& U, VectorXd& PHIvelocity, VectorXd& Uvelocity, double dt, double D,
	SparseMatrix<double>& mM11, SparseMatrix<double>& mM12, SparseMatrix<double>& mM21, SparseMatrix<double>& mM22, SparseMatrix<double>& mK11, SparseMatrix<double>& mK12,
	SparseMatrix<double>& mK21, SparseMatrix<double>& mK22, VectorXd& vF1,
	const vector<Coord>& vdNodeCoord, const vector<vector<int>>& vvEFT,
	const vector<shared_ptr<Element>>& vpFinalElementList,
	double ephilon, double lambda) {

	clock_t t;
	clock_t solver_time = 0;
	clock_t matrix_time = 0; 
	clock_t scheme_time = 0;

	t = clock(); //-> solver
	BiCGSTAB<SparseMatrix<double> > solver;
	solver_time += clock() - t; //<- solver
	//VectorXd dPHI = solver.compute(mC_phi).solve(dt*(mA_phi + mE_phi)*PHI + dt*vF_phi);
	//PHI += dPHI;
	//U += solver.compute( mC_u1 ).solve( dt*mA_u1*U + mC_u2*dPHI );

	///////////////////////////////////////////////////////////////////////////////////////////////////
	t = clock(); //-> scheme
	double rho = 0;
	double rhos = 0;
	double W1L4 = 1 / (1 + rho);
	double W2L5 = 1 / ((1 + rho) * (1 + rhos));
	double W1L6 = (3 + rho + rhos - rho*rhos) / (2 * (1 + rho) * (1 + rhos));
	double lambda4 = 1;
	double lambda5 = 1 / (1 + rhos);
	unsigned nNode = mM11.rows();

	//////////////////////////////////////////////////////////////////////////////////////////
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
	array<double, 9> model = { C_inf, k, G, d0, alpha, Vp, T0, Ts, dT0 };					//
	//////////////////////////////////////////////////////////////////////////////////////////

	typedef Triplet<double> T;
	vector<T> tripletList_q;
	vector<T> tripletList_Up, tripletList_Down, tripletList_Left, tripletList_Right;
	for (unsigned i = 0; i < nNode; i++) {
		tripletList_Up.push_back(T(i, i, 1));
		tripletList_Down.push_back(T(i + nNode, i, 1));
		tripletList_Left.push_back(T(i, i, 1));
		tripletList_Right.push_back(T(i, i + nNode, 1));
	}

	SparseMatrix<double> Up(nNode * 2, nNode); 
	Up.setFromTriplets(tripletList_Up.begin(), tripletList_Up.end());
	SparseMatrix<double> Down(nNode * 2, nNode);
	Down.setFromTriplets(tripletList_Down.begin(), tripletList_Down.end());
	SparseMatrix<double> Left(nNode, nNode * 2);
	Left.setFromTriplets(tripletList_Left.begin(), tripletList_Left.end());
	SparseMatrix<double> Right(nNode, nNode * 2);
	Right.setFromTriplets(tripletList_Right.begin(), tripletList_Right.end());


	VectorXd d1 = Up * PHI + Down * U;
	VectorXd v1;
	if (tloop == 0) {
		PHIvelocity *= 0;
		find_matrixs(Theta, PHI, U, PHIvelocity, vdNodeCoord, vvEFT, vpFinalElementList, ephilon, lambda, tloop, model, mM11, mM12, mM21, mM22, mK11, mK12, mK21, mK22, vF1);
		SparseMatrix<double> M = Up*(mM11)*Left + Down*(mM21)*Left + Down*(mM22)*Right;
		SparseMatrix<double> K = Up*(mK11)*Left + Down*(mK21)*Left + Down*(mK22)*Right;
		VectorXd F = Up * vF1;
		v1 = solver.compute(M).solve(F - K*d1);
	} else {
		v1 = Up * PHIvelocity + Down * Uvelocity;
	}

	VectorXd d_telda = d1 + W1L4 * v1 * dt;
	PHI = d_telda.topRows(nNode);
	U = d_telda.bottomRows(nNode);
	scheme_time += clock() - t; //<- scheme
	
	t = clock(); //-> matrix
	find_matrixs(Theta, PHI, U, PHIvelocity, vdNodeCoord, vvEFT, vpFinalElementList, ephilon, lambda, tloop, model, mM11, mM12, mM21, mM22, mK11, mK12, mK21, mK22, vF1);
	matrix_time += clock() - t;	 //<- matrix

	t = clock(); //-> scheme
	SparseMatrix<double> M = Up*(mM11)*Left + Down*(mM21)*Left + Down*(mM22)*Right;
	SparseMatrix<double> K = Up*(mK11)*Left + Down*(mK21)*Left + Down*(mK22)*Right;
	VectorXd F = Up * vF1;
	scheme_time += clock() - t; //<- scheme

	t = clock(); //-> solver
	VectorXd v_telda = solver.compute( M ).solve( F - K*d_telda );
	solver_time += clock() - t; //<- solver

	t = clock(); //-> scheme
	VectorXd dv = (-v1 + v_telda) / W1L6;
	VectorXd d2 = d1 + lambda4 * v1 * dt + lambda5 * dv * dt;
	VectorXd v2 = v1 + dv;

	PHI = d2.topRows(nNode);
	U = d2.bottomRows(nNode);
	PHIvelocity = v2.topRows(nNode);
	Uvelocity = v2.bottomRows(nNode);
	scheme_time += clock() - t; //<- scheme

	//fout_time << "\tmatrix: " << 1.*matrix_time/CLOCKS_PER_SEC << " sec" << endl;
	//fout_time << "\tsolver: " << 1.*solver_time/CLOCKS_PER_SEC << " sec" << endl;
	//fout_time << "\tscheme: " << 1.*scheme_time/CLOCKS_PER_SEC << " sec" << endl;
	//cout << "\tmatrix: " << 1.*matrix_time/CLOCKS_PER_SEC << " sec" << endl;
	//cout << "\tsolver: " << 1.*solver_time/CLOCKS_PER_SEC << " sec" << endl;
	//cout << "\tscheme: " << 1.*scheme_time/CLOCKS_PER_SEC << " sec" << endl;
	
}

MatrixXd get_cotangent(const VectorXd& phi, const MatrixXd& B) {
////////////////////////////////////////////////////////////////////////
// phi = / phi1 \     B =  / N1,x N2,x N3,x N4,x \     cot = / DERX \ //
//       | phi2 |          \ N1,y N2,y N3,y N4,y /           \ DERY / //
//       | phi3 |                                                     //
//       \ phi4 /                                                     //
////////////////////////////////////////////////////////////////////////
    return B * phi; // 2x1
}

// g'(phi) - lambda*U*P'(phi)
double f(double phi, double u, double theta, double lambda) {
	return phi * (1 - phi*phi) - lambda * u * pow(1 - phi*phi, 2.0);
	//return phi * (1 - phi*phi) - lambda * pow(1 - phi*phi, 2.0) * (u + 0.9 * phi * (1 - phi*phi) * ((double(rand()) / RAND_MAX) - 0.5));
	//return phi * (1 - phi*phi) - lambda * pow((1 - phi*phi), 2.0) * (u + theta);
	//return phi * (1 - phi*phi) - lambda * pow((1 - phi*phi), 2.0) * (u + theta + 0.3 * phi * (1 - phi*phi) * ((double(rand()) / RAND_MAX) - 0.5));
}

double q(double phi, double k) {
	//return (phi >= 1) ? 0 : (1 - phi) / (1 + k - (1 - k) * phi);
	return (phi >= 1) ? 0 : (1 - phi) / (1 + k - (1 - k) * phi) + (1 + phi) * 0.2 / 2;
	//return (phi >= 1) ? 0 : (1 - phi) / 2;
	//return (phi >= 1) ? 0 : (1 - phi) / 2 + (1 + phi) * 0.2 / 2;
}

void MeshRefinement(unsigned maxLv, double gamma, ofstream& fout_time,
	VectorXd& Theta, VectorXd& PHI, VectorXd& U, VectorXd& PHIvelocity, VectorXd& Uvelocity,
	map<Coord, double>& PhiCoordinateList, map<Coord, double>& UCoordinateList,
	map<Coord, double>&PhiVelocityCoordinateList, map<Coord, double>&UVelocityCoordinateList,
	vector<Coord>& NodeCoordinates, vector<vector<int>>& EFT, vector<vector<shared_ptr<Element>>>& LevelElementList,
	map<Coord, unsigned>& NodeCoordinateList, vector<shared_ptr<Element>>& FinalElementList) {

	PhiCoordinateList.clear();
	UCoordinateList.clear();
	PhiVelocityCoordinateList.clear();
	UVelocityCoordinateList.clear();
	for (int i = 0; i < PHI.rows(); i++) {
		PhiCoordinateList[NodeCoordinates[i]] = PHI(i);
		UCoordinateList[NodeCoordinates[i]] = U(i);
		PhiVelocityCoordinateList[NodeCoordinates[i]] = PHIvelocity(i);
		UVelocityCoordinateList[NodeCoordinates[i]] = Uvelocity(i);
	}

	Quadtree_MeshGenerate(maxLv, gamma, LevelElementList, 10, PhiCoordinateList, UCoordinateList, PhiVelocityCoordinateList, UVelocityCoordinateList); // case = 10
	Quadtree_AddNodes(LevelElementList, NodeCoordinateList);
	ReportElement(LevelElementList, FinalElementList, NodeCoordinateList, EFT, NodeCoordinates);

	PHI.resize(NodeCoordinates.size());
	U.resize(NodeCoordinates.size());
	Theta.resize(NodeCoordinates.size());
	PHIvelocity.resize(NodeCoordinates.size());
	Uvelocity.resize(NodeCoordinates.size());
	for (unsigned i = 0; i < NodeCoordinates.size(); i++) {
		PHI(i) = PhiCoordinateList[NodeCoordinates[i]];
		U(i) = UCoordinateList[NodeCoordinates[i]];
		Theta(i) = 0;
		PHIvelocity(i) = PhiVelocityCoordinateList[NodeCoordinates[i]];
		Uvelocity(i) = UVelocityCoordinateList[NodeCoordinates[i]];
	}
	//fout_time << endl;
	//fout_time << EFT.size() << "\tElements" << endl;
	//fout_time << NodeCoordinateList.size() << "\tNodes" << endl;
	//fout_time << endl;
}
