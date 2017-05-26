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
using namespace AMR;

void find_matrixs(VectorXd& Theta, const VectorXd& PHI, const VectorXd& U, const VectorXd& PHIvelocity, const vector<Coord>& vdNodeCoord, const vector<vector<int>>& vvEFT,
	const vector<shared_ptr<Element>>& vpFinalElementList,
    double tloop, double dt,
    SparseMatrix<double>& mM11, SparseMatrix<double>& mM12, SparseMatrix<double>& mM21, SparseMatrix<double>& mM22, SparseMatrix<double>& mK11, SparseMatrix<double>& mK12,
    SparseMatrix<double>& mK21, SparseMatrix<double>& mK22, VectorXd& vF1) {
    
	// initialization

	double epsilon_4 = 0.05;
	double fold = 4;
	double alpha = 0; // degree
	double beta = 0; //degree

	double Cinf = 0.25;
	double k = 0.1;
	double Cl = 0.74;
	double a1 = 5.0 * sqrt(2.0) / 8.0;
	double a2 = 47.0 / 75.0;
	double epsilon = 100; // W/d0
	double Vp = 5.0E-6;
	double d0 = 1.3E-8;
	double D = 1.0E-9;
	double mCinf = 2;
	double m = mCinf / Cinf;
	double G = 100.0E2;
	double lT = m*(1 - k)*Cl / G;
	double theta_inf = 1.0;
	
	double Dtelda = a1 * a2 * epsilon;
	double Vptelda = a1 * a2 * Vp * d0 / D * pow(epsilon, 2.0);
	double lTtelda = lT / d0 / epsilon;
	double lambda = a1 * epsilon;
	double tau = a1 * a2 * pow(epsilon, 3.0) * pow(d0, 2.0);
	
	//cout << "Dtelda  = " << Dtelda << endl;
	//cout << "Vptelda = " << Vptelda << endl;
	//cout << "lTtelda = " << lTtelda << endl;
	//cout << "lambda  = " << lambda << endl;
	//cout << "tau     = " << tau << endl;
	//cout << endl;
	

	double T0 = -mCinf / k;
	double Tinf = T0 + G * Vp * theta_inf * lTtelda / Vptelda;
	if (int(tloop) % 100 == 0 && tloop >= 100)
		cout << "\ttemperature: " << Tinf << " --> " << T0 << endl;

	mM11.setZero(); mM12.setZero(); mM21.setZero(); mM22.setZero(); mK11.setZero(); mK12.setZero(); mK21.setZero(); mK22.setZero(); vF1.setZero();
    typedef Triplet<double> T;
    vector<T> tripletList_M11, tripletList_M12, tripletList_M21, tripletList_M22, tripletList_K11, tripletList_K12, tripletList_K21, tripletList_K22;
	int nGp = 3 * 3; // 3 x 3 Gauss point
    MatrixXd LocationsAndWeights = gauss2D(nGp);

	size_t numNodePerElement;
	bitset<8> bitElementType;
	MatrixXd elementNodesCoord;
	VectorXd phi, u, v, Q;
	MatrixXd M11e, M12e, M21e, M22e, K11e, K12e, K21e, K22e;
	VectorXd F1e;
	RowVectorXd N;
	MatrixXd dN, B, cotangent;
	double xi, eta, W, J, DERX, DERY, angle, as, asp, PHIe, Ue, Ve, Qe;

	for (unsigned e = 0; e < vvEFT.size(); e++) {
		numNodePerElement = vvEFT[e].size();
		bitElementType = vpFinalElementList[e]->bitElementType;
        // get the coordinates of the nodes in the element
        elementNodesCoord.resize(numNodePerElement,2); // n x 2
        phi.resize(numNodePerElement); // n x 1
        u.resize(numNodePerElement); // n x 1
		v.resize(numNodePerElement); // n x 1
		Q.resize(numNodePerElement); // n x 1

		for (unsigned i = 0; i < numNodePerElement; i++) {
			elementNodesCoord(i, 0) = vdNodeCoord[vvEFT[e][i]].x;
			elementNodesCoord(i, 1) = vdNodeCoord[vvEFT[e][i]].y;
			phi(i) = PHI[vvEFT[e][i]];
			u(i) = U[vvEFT[e][i]];
			v(i) = PHIvelocity[vvEFT[e][i]];
			//if (v(i) != 0)
			//	cout << v(i) << endl;
			if ((elementNodesCoord(i, 1)*cos(beta * M_PI / 180) + elementNodesCoord(i, 0)*sin(beta * M_PI / 180) - Vptelda * dt * tloop) / lTtelda /*+ theta_inf*/> theta_inf)
				Q(i) = theta_inf;
			else if ((elementNodesCoord(i, 1)*cos(beta * M_PI / 180) + elementNodesCoord(i, 0)*sin(beta * M_PI / 180) - Vptelda * dt * tloop) / lTtelda /*+ theta_inf*/ < 0)
				Q(i) = 0;
			else
				Q(i) = (elementNodesCoord(i, 1)*cos(beta * M_PI / 180) + elementNodesCoord(i, 0)*sin(beta * M_PI / 180) - Vptelda * dt * tloop) / lTtelda /*+ theta_inf*/;
			//Q(i) = (elementNodesCoord(i, 1)*cos(beta * M_PI / 180) + elementNodesCoord(i, 0)*sin(beta * M_PI / 180) - Vptelda * dt * tloop) / lTtelda;
			//Q(i) = 0;
        }
		
		M11e = M12e = M21e = M22e = K11e = K12e = K21e = K22e = MatrixXd::Zero(numNodePerElement, numNodePerElement);
		F1e = VectorXd::Zero(numNodePerElement);

		// cycle for Gauss point
        for (int gp=0; gp<nGp; gp++) {
            xi = LocationsAndWeights(gp,0);
            eta = LocationsAndWeights(gp,1);
            W = LocationsAndWeights(gp,2);
			N = ShapeFunction(xi, eta, bitElementType); // 1 x n
			dN = NaturalDerivatives(xi, eta, bitElementType); // 2 x n
			B = XYDerivatives(elementNodesCoord, dN); // 2 x n
			J = detJacobian(elementNodesCoord, dN); // 1 x 1
			
			cotangent = get_cotangent(phi, B); // 2 x 1
			DERX = cotangent(0);
			DERY = cotangent(1);
			angle = atan2(DERY, DERX);
			//as = 1 + epsilon * cos(fold*(angle - M_PI / 6)); // A(theta)
			//asp = -fold * epsilon * sin(fold*(angle - M_PI / 6)); // A'(theta)
			as = 1 + epsilon_4 * cos(fold * (angle - alpha * M_PI / 180)); // A(theta)
			asp = -fold * epsilon_4 * sin(fold * (angle - alpha * M_PI / 180)); // A'(theta)
			
			PHIe = N * phi;
			Ue = N * u;
			Ve = N * v;
			Qe = N * Q;

            // matrixs of a element
			M11e += (1 - (1 - k) * Qe)* as * as * N.transpose() * N * W * J; // n x n
			M21e += -(1 + (1 - k) * Ue) * 0.5 * N.transpose() * N * W * J; // n x n
			M22e += ((1 + k) * 0.5 - (1 - k) * 0.5 * PHIe) * N.transpose() * N * W * J; // n x n
			K11e += as * as * B.transpose() * B * W * J; // n x n
			K11e += as * asp * (B.row(1).transpose()*B.row(0) - B.row(0).transpose()*B.row(1)) * W * J; // n x n
			if (DERX*DERX + DERY*DERY > 1.0E-12)
				K21e += (1 + (1 - k) * Ue) * Ve / sqrt(8.0) / sqrt(DERX*DERX + DERY*DERY) * B.transpose() * B * W * J; // n x n
			K22e += Dtelda * q(PHIe, k) * B.transpose() * B * W * J; // n x n
			F1e += N.transpose() * f(PHIe, Ue, Qe, lambda) * W * J;
			
        }
        // cycle for element matrixs
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
        /*for (unsigned i=0; i<numNodePerElement; i++) {
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
        }*/
		////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		
		for (unsigned i = 0; i<numNodePerElement; i++) {
			for (unsigned j = 0; j<numNodePerElement; j++) {
				if (abs(M11e(i, j)) > 1.0E-10)
					tripletList_M11.push_back(T(vvEFT[e][i], vvEFT[e][j], M11e(i, j)));
				if (abs(M12e(i, j)) > 1.0E-10)
					tripletList_M12.push_back(T(vvEFT[e][i], vvEFT[e][j], M12e(i, j)));
				if (abs(M21e(i, j)) > 1.0E-10)
					tripletList_M21.push_back(T(vvEFT[e][i], vvEFT[e][j], M21e(i, j)));
				if (abs(M22e(i, j)) > 1.0E-10)
					tripletList_M22.push_back(T(vvEFT[e][i], vvEFT[e][j], M22e(i, j)));
				if (abs(K11e(i, j)) > 1.0E-10)
					tripletList_K11.push_back(T(vvEFT[e][i], vvEFT[e][j], K11e(i, j)));
				if (abs(K12e(i, j)) > 1.0E-10)
					tripletList_K12.push_back(T(vvEFT[e][i], vvEFT[e][j], K12e(i, j)));
				if (abs(K21e(i, j)) > 1.0E-10)
					tripletList_K21.push_back(T(vvEFT[e][i], vvEFT[e][j], K21e(i, j)));
				if (abs(K22e(i, j)) > 1.0E-10)
					tripletList_K22.push_back(T(vvEFT[e][i], vvEFT[e][j], K22e(i, j)));
			}
			if (abs(F1e(i)) > 1.0E-10)
				vF1(vvEFT[e][i]) += F1e(i);
			Theta(vvEFT[e][i]) = Q(i);
		}
		
		
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
	const unsigned tloop, VectorXd& Theta, VectorXd& PHI, VectorXd& U, VectorXd& PHIvelocity, VectorXd& Uvelocity, double dt,
	SparseMatrix<double>& mM11, SparseMatrix<double>& mM12, SparseMatrix<double>& mM21, SparseMatrix<double>& mM22, SparseMatrix<double>& mK11, SparseMatrix<double>& mK12,
	SparseMatrix<double>& mK21, SparseMatrix<double>& mK22, VectorXd& vF1,
	const vector<Coord>& vdNodeCoord, const vector<vector<int>>& vvEFT,
	const vector<shared_ptr<Element>>& vpFinalElementList) {

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
		find_matrixs(Theta, PHI, U, PHIvelocity, vdNodeCoord, vvEFT, vpFinalElementList, tloop, dt, mM11, mM12, mM21, mM22, mK11, mK12, mK21, mK22, vF1);
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
	find_matrixs(Theta, PHI, U, PHIvelocity, vdNodeCoord, vvEFT, vpFinalElementList, tloop, dt, mM11, mM12, mM21, mM22, mK11, mK12, mK21, mK22, vF1);
	matrix_time += clock() - t;	 //<- matrix

	t = clock(); //-> scheme
	SparseMatrix<double> M = Up*(mM11)*Left + Down*(mM21)*Left + Down*(mM22)*Right;
	SparseMatrix<double> K = Up*(mK11)*Left + Down*(mK21)*Left + Down*(mK22)*Right;
	VectorXd F = Up * vF1;
	scheme_time += clock() - t; //<- scheme

	t = clock(); //-> solver
	VectorXd v_telda = solver.compute( M ).solve( F - K*d_telda );
	//std::cout << "#iterations:     " << solver.iterations() << std::endl;
	//std::cout << "estimated error: " << solver.error() << std::endl;
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
	//return phi * (1 - phi*phi) - lambda * u * pow(1 - phi*phi, 2.0);
	//return phi * (1 - phi*phi) - lambda * pow(1 - phi*phi, 2.0) * (u + 0.9 * phi * (1 - phi*phi) * ((double(rand()) / RAND_MAX) - 0.5));
	return phi * (1 - phi*phi) - lambda * pow((1 - phi*phi), 2.0) * (u + theta + 0.03 * phi * (1 - phi*phi) * ((double(rand()) / RAND_MAX) - 0.5));
	//return phi * (1 - phi*phi) - lambda * pow((1 - phi*phi), 2.0) * (u + theta);
	//return phi * (1 - phi*phi) - lambda * pow((1 - phi*phi), 2.0) * (u + theta + 0.3 * phi * (1 - phi*phi) * ((double(rand()) / RAND_MAX) - 0.5));
	//return phi * sqrt(2.0) - lambda * (1 - phi*phi) * sqrt(2.0) * (u + theta);
}

double q(double phi, double k) {
	return (phi >= 1) ? 0 : (1.0 - phi) / (1.0 + k - (1.0 - k) * phi);
	//return (phi >= 1) ? 0 : (1 - phi) / (1 + k - (1 - k) * phi) + (1 + phi) * 0.2 / 2;
	//return (phi >= 1) ? 0 : (1 - phi) / 2;
	//return (phi >= 1) ? 0 : (1 - phi) / 2 + (1 + phi) * 0.2 / 2;
	//return 1;
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
