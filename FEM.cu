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

// Our Kernel
// __global__ void cu_element(double* mM11,
// 						   double* mM21,
// 						   double* mM22,
// 						   double* mK11,
// 						   double* mK21,
// 						   double* mK22,
// 						   double* vF1,
// 						   double* aPHI,
// 						   double* aU,
// 						   int** aaEFT,
// 						   int*  numNodePerElement,
// 						   uint8_t* elementType,
// 						   Coord* nodeCoord,
// 						   int numElement,
// 						   double tloop,
// 						   double epsilon,
// 						   double lambda){
// 	const double PI = 3.14159265358979323846;
// 	const double C_inf = 3;
// 	const double k = 0.14;
// 	const double G = 140 * 40;
// 	const double Ts = 273;
// 	const double dT0 = 2.6 * C_inf * (1 - k) / k; // m = 2.6
// 	const double T0 = Ts - dT0 / 10;
// 	const double Vp = 3000;
// 	const double dTn = 8;
// 	const double d0 = 5E-3;
// 	const double alpha = 3000;
// 	const double a1 = 0.8839;
// 	const double a2 = 0.6267;
// 	const double W0 = d0 * lambda / a1;
// 	const double Tau0 = a2 * lambda * W0 * W0 / alpha;
// 	const double D = lambda * a2;
// 	const int m = 6;

// 	// gaussian weight are all ones, ignore
// 	const double XIs[] = {-0.577350269189626, 0.577350269189626, 0.577350269189626, -0.577350269189626};
               
//     const double ETAs[] = {-0.577350269189626, -0.577350269189626, 0.577350269189626, 0.577350269189626};

//     int n = 4;
//     // get the coordinates of the nodes in the element
//     MatrixXd elementNodesCoord(n,2); // n x 2
//     VectorXd phi(n); // n x 1
//     VectorXd u(n); // n x 1

// }

// #define PI 3.14159265358979323846
void cu_find_matrixs(double lambda,
					 double tloop,
					 double epsilon,
					 int mSize,
					 const vector<vector<int>>& EFT,
					 const vector<shared_ptr<Element>>& FinalElementList,
					 const vector<Coord>& NodeCoordinates,
					 const VectorXd& PHI,
					 const VectorXd& U){
	double* mM11;
	cudaMalloc((void**)&mM11, sizeof(double)*mSize * mSize);

	// double* mM21;
	// cudaMallocManaged((void**)&mM21, sizeof(double)*mSize * mSize);

	// double* mM22;
	// cudaMallocManaged((void**)&mM22, sizeof(double)*mSize * mSize);

	// double* mK11;
	// cudaMallocManaged((void**)&mK11, sizeof(double)*mSize * mSize);

	// double* mK21;
	// cudaMallocManaged((void**)&mK21, sizeof(double)*mSize * mSize);

	// double* mK22;
	// cudaMallocManaged((void**)&mK22, sizeof(double)*mSize * mSize);

	// double* vF1;
	// cudaMallocManaged((void**)&vF1, sizeof(double)*NodeCoordinates.size());
	

	// size_t numElement = EFT.size();
 //    int** aaEFT;
 //    cudaMallocManaged((void**)&aaEFT, sizeof(int*)*numElement);

 //    int*  numNodePerElement;
 //    cudaMallocManaged((void**)&numNodePerElement, sizeof(int)*numElement);

 //    uint8_t* elementType;
 //    cudaMallocManaged((void**)&elementType, sizeof(uint8_t)*numElement);

 //    for(unsigned e = 0; e < numElement; ++e){
 //    	int _n = EFT[e].size();
 //    	numNodePerElement[e] = _n;
 //    	cudaMallocManaged((void**)&aaEFT[e], sizeof(int)*_n);
 //    	copy(EFT[e].begin(), EFT[e].end(), aaEFT[e]);

 //    	elementType[e] = (uint8_t)FinalElementList[e]->bitElementType.to_ulong();
 //    }

 //    Coord* nodeCoord;
 //    cudaMallocManaged((void**)&nodeCoord, sizeof(Coord)*NodeCoordinates.size());
 //    copy(NodeCoordinates.begin(), NodeCoordinates.end(), nodeCoord);

 //    double* aPHI;
 //    cudaMallocManaged((void**)&aPHI, sizeof(double)*PHI.size());
 //    Map<VectorXd>(aPHI, PHI.size()) = PHI;

 //    double* aU;
 //    cudaMallocManaged((void**)&aU, sizeof(double)*U.size());
 //    Map<VectorXd>(aU, U.size()) = U;

    // const int BLOCKSIZE = 512;
    // const int numBLOCK = ((numElement-1)/BLOCKSIZE + 1);
    // cu_element<<<numBLOCK, BLOCKSIZE>>>(mM11,
				// 						mM21,
				// 						mM22,
				// 						mK11,
				// 						mK21,
				// 						mK22,
				// 						vF1,
				// 						aPHI,
				// 						aU,
				// 						aaEFT,
				// 						numNodePerElement,
				// 						elementType,
				// 						nodeCoord,
				// 						numElement,
				// 						tloop,
				// 						epsilon,
				// 						lambda);

    
    cudaDeviceReset();
    
}



void FEM::MeshRefinement() {
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

	int ncSize = NodeCoordinates.size();

	mM11 = SparseMatrix<double>(ncSize, ncSize);
	mM21 = SparseMatrix<double>(ncSize, ncSize);
	mM22 = SparseMatrix<double>(ncSize, ncSize);
	mK11 = SparseMatrix<double>(ncSize, ncSize);
	mK21 = SparseMatrix<double>(ncSize, ncSize);
	mK22 = SparseMatrix<double>(ncSize, ncSize);
	vF1 = VectorXd(ncSize);
}




// Remind :
// 	Theta not used
//	
void FEM::find_matrixs(double lambda, double epsilon, unsigned tloop, double dt) {
	// initialization
	const double PI = 3.14159265358979323846;
	double C_inf = 3;	// (wt%)
	double k = 0.14;		// 
	double G = 140 * 40;		// (K/cm)
	double d0 = 5E-3;		// (1E-6m)
	double alpha = 3000;	// (1E-6m2/s)
	double Vp = 3000;		// (1E-6m/s)
	double Ts = 273;		// (K)
	double dT0 = 2.6 * C_inf * (1 - k) / k;		// (K)
	double T0 = Ts - dT0 / 10;		// (K)
	double a1 = 0.8839;
	double a2 = 0.6267;
	double W0 = d0 * lambda / a1; // (1E-6m)
	double Tau0 = a2 * lambda * W0 * W0 / alpha; // (s)
	double D = lambda * a2;

	mM11.setZero(); mM21.setZero(); mM22.setZero(); mK11.setZero(); mK21.setZero(); mK22.setZero(); vF1.setZero();
    typedef Triplet<double> T;
    vector<T> tripletList_M11, tripletList_M12, tripletList_M21, tripletList_M22, tripletList_K11, tripletList_K12, tripletList_K21, tripletList_K22;
	int nGp = 2 * 2; // 2 x 2 Gauss point
    MatrixXd LocationsAndWeights = gauss2D(nGp);
	int m = 6;


	double RealTime = Tau0 * tloop; // (s)

	for (unsigned e = 0; e < EFT.size(); e++) {
		size_t numNodePerElement = EFT[e].size();
		bitset<8> bitElementType = FinalElementList[e]->bitElementType;
        // get the coordinates of the nodes in the element
        MatrixXd elementNodesCoord(numNodePerElement,2); // n x 2
        VectorXd phi(numNodePerElement); // n x 1
        VectorXd u(numNodePerElement); // n x 1

        // element info
		for (unsigned i = 0; i < numNodePerElement; i++) {
			int nodeSerial = EFT[e][i];

			elementNodesCoord(i, 0) = NodeCoordinates[nodeSerial].x;
			elementNodesCoord(i, 1) = NodeCoordinates[nodeSerial].y;
			phi(i) = PHI[nodeSerial];
			u(i) = U[nodeSerial];
        }
		double RealCoord = W0 * 0.25*(elementNodesCoord(0, 1) + elementNodesCoord(1, 1) + elementNodesCoord(2, 1) + elementNodesCoord(3, 1)) * 1E-6; // n x 2 (m)
		
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
					tripletList_M22.push_back(T(EFT[e][i], EFT[e][j], Ce(i, j)));
					tripletList_M21.push_back(T(EFT[e][i], EFT[e][j], -0.5*Ce(i, j)));
					tripletList_M11.push_back(T(EFT[e][i], EFT[e][j], as * as * Ce(i, j)));
				}
				if (Ae(i, j) > 1.0E-12 || Ae(i, j) < -1.0E-12) {
					tripletList_K22.push_back(T(EFT[e][i], EFT[e][j], -D * q(N0 * phi, 0.7) * Ae(i, j)));
					tripletList_K11.push_back(T(EFT[e][i], EFT[e][j], -as * as * Ae(i, j)));
				}
				if (Ee(i, j) > 1.0E-12 || Ee(i, j) < -1.0E-12)
					tripletList_K11.push_back(T(EFT[e][i], EFT[e][j], -as * asp * Ee(i, j)));
            }
			if (Fe(i) > 1.0E-12 || Fe(i) < -1.0E-12)
				vF1(EFT[e][i]) += Fe(i);
        }
    }
	mM11.setFromTriplets(tripletList_M11.begin(), tripletList_M11.end());
	mM21.setFromTriplets(tripletList_M21.begin(), tripletList_M21.end());
	mM22.setFromTriplets(tripletList_M22.begin(), tripletList_M22.end());
	mK11.setFromTriplets(tripletList_K11.begin(), tripletList_K11.end());
	mK21.setFromTriplets(tripletList_K21.begin(), tripletList_K21.end());
	mK22.setFromTriplets(tripletList_K22.begin(), tripletList_K22.end());
}

void FEM::time_discretization(
	double lambda,
	double epsilon,
	unsigned tloop,
	double dt) {
	clock_t t;
	clock_t solver_time = 0;
	clock_t matrix_time = 0;
	clock_t scheme_time = 0;

	t = clock(); //-> solver
	BiCGSTAB<SparseMatrix<double> > solver;
	solver_time += clock() - t; //<- solver

	///////////////////////////////////////////////////////////////////////////////////////////////////
	t = clock(); //-> scheme
	double rho = 0;
	double rhos = 0;
	double W1L4 = 1 / (1 + rho);
	// double W2L5 = 1 / ((1 + rho) * (1 + rhos));
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
		find_matrixs(lambda, epsilon, tloop, dt);
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

	// Call Cuda here!!!!!!!
	// cu_find_matrixs(lambda,
	// 				tloop,
	// 				epsilon,
	// 				nNode,
	// 				EFT,
	// 				FinalElementList,
	// 				NodeCoordinates,
	// 				PHI,
	// 				U);

	find_matrixs(lambda, epsilon, tloop, dt);
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
