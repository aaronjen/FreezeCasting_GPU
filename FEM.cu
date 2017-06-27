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

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

#define BLOCK_SIZE 256
#define BLOCK_WARP 8




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

	ncSize = NodeCoordinates.size();
	elemSize = EFT.size();


	PHI.setZero(ncSize);
	U.setZero(ncSize);
	Theta.setZero(ncSize);
	PHIvelocity.setZero(ncSize);
	Uvelocity.setZero(ncSize);
	for (unsigned i = 0; i < ncSize; i++) {
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

	cudaFree(aPHI);
	cudaFree(aU);
	cudaFree(aEFT);
	cudaFree(aNodeNum);
	cudaFree(elementType);
	cudaFree(aCoordX);
	cudaFree(aCoordY);

	cudaMalloc(&aPHI, sizeof(double)*ncSize);
	cudaMalloc(&aU, sizeof(double)*ncSize);
	cudaMallocManaged(&aEFT, sizeof(int)*elemSize*8); //Element at max 8 nodes
	cudaMallocManaged(&aNodeNum, sizeof(int)*elemSize);
	cudaMallocManaged(&elementType, sizeof(unsigned char)*elemSize);
	cudaMallocManaged(&aCoordX, sizeof(double)*ncSize);
	cudaMallocManaged(&aCoordY, sizeof(double)*ncSize);

	// copy EFT to array
	for(int i = 0; i < elemSize; ++i){
		aNodeNum[i] = EFT[i].size();
		// cudaMemset(aNodeNum+i, EFT[i].size(), sizeof(int));

		for(int j = 0; j < EFT[i].size(); ++j){
			aEFT[i * 8+j] = EFT[i][j];
			// cudaMemset(aEFT+i*8, EFT[i][j], sizeof(int));
		}
		elementType[i] = (unsigned char)FinalElementList[i]->bitElementType.to_ulong();
		// unsigned char c = (unsigned char)FinalElementList[i]->bitElementType.to_ulong();
		// cudaMemset(elementType+i, c,  sizeof(unsigned char));
	}

	//copy elementType to array
	for(int i = 0; i < ncSize; ++i){
		aCoordX[i] = NodeCoordinates[i].x;
		// cudaMemset(aCoordX+i, NodeCoordinates[i].x, sizeof(int));
		aCoordY[i] = NodeCoordinates[i].y;
		// cudaMemset(aCoordY+i, NodeCoordinates[i].y, sizeof(int));
	}


	// device pointer
	cudaFree(adM11);
	cudaFree(adM21);
	cudaFree(adM22);
	cudaFree(adK11);
	cudaFree(adK21);
	cudaFree(adK22);
	cudaFree(adF1);

	cudaMalloc(&adM11, sizeof(double)*ncSize*ncSize);
	cudaMalloc(&adM21, sizeof(double)*ncSize*ncSize);
	cudaMalloc(&adM22, sizeof(double)*ncSize*ncSize);
	cudaMalloc(&adK11, sizeof(double)*ncSize*ncSize);
	cudaMalloc(&adK21, sizeof(double)*ncSize*ncSize);
	cudaMalloc(&adK22, sizeof(double)*ncSize*ncSize);
	cudaMalloc(&adF1, sizeof(double)*ncSize);

	// host pointer
	free(aM11);
	free(aM21);
	free(aM22);
	free(aK11);
	free(aK21);
	free(aK22);
	free(aF1);

	aM11 = (double*)malloc(sizeof(double)*ncSize*ncSize);
	aM21 = (double*)malloc(sizeof(double)*ncSize*ncSize);
	aM22 = (double*)malloc(sizeof(double)*ncSize*ncSize);
	aK11 = (double*)malloc(sizeof(double)*ncSize*ncSize);
	aK21 = (double*)malloc(sizeof(double)*ncSize*ncSize);
	aK22 = (double*)malloc(sizeof(double)*ncSize*ncSize);
	aF1 = (double*)malloc(sizeof(double)*ncSize);
}

__device__ __host__ RowVectorXd cuShapeFunction(double xi, double eta, unsigned char bitElementType) {
	double Ni[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	Ni[7] = (1 - eta*eta) * (1 - xi) / 2; // N8
	Ni[6] = (1 - xi*xi) * (1 + eta) / 2;  // N7
	Ni[5] = (1 - eta*eta) * (1 + xi) / 2; // N6
	Ni[4] = (1 - xi*xi) * (1 - eta) / 2;  // N5
	int numberOfNodes = 0;
	for (int i=0; i<8; ++i){
		if (bitElementType & (1 << i)) ++numberOfNodes;
		else (Ni[i]=0);
	}
	Ni[3] = (1 - xi) * (1 + eta) / 4 - (Ni[6] + Ni[7]) / 2; // N4 - (N7 + N8) / 2
	Ni[2] = (1 + xi) * (1 + eta) / 4 - (Ni[5] + Ni[6]) / 2; // N3 - (N6 + N7) / 2
	Ni[1] = (1 + xi) * (1 - eta) / 4 - (Ni[4] + Ni[5]) / 2; // N2 - (N5 + N6) / 2
	Ni[0] = (1 - xi) * (1 - eta) / 4 - (Ni[7] + Ni[4]) / 2; // N1 - (N8 + N5) / 2
	RowVectorXd shape(numberOfNodes); // 1 x n
	int cnt=0;
	for (int i=0; i<8; ++i)
		if(bitElementType & (1 << i)) shape(cnt++)=Ni[i];
	return shape;
}

__device__ __host__ MatrixXd cuNaturalDerivatives(double xi, double eta, unsigned char bitElementType) {
	double Ni_xi[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	double Ni_eta[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
	Ni_xi[4]  = - xi * (1 - eta);
	Ni_eta[4] = -(1 - xi*xi) / 2;
	Ni_xi[5]  =  (1 - eta*eta) / 2;
	Ni_eta[5] = - eta * (1 + xi);
	Ni_xi[6]  = - xi * (1 + eta);
	Ni_eta[6] =  (1 - xi*xi) / 2;
	Ni_xi[7]  = -(1 - eta*eta) / 2;
	Ni_eta[7] = - eta * (1 - xi);
	int numberOfNodes = 0;
	for (int i=0; i<8; ++i){
		if (bitElementType & (1 << i)) ++numberOfNodes;
		else (Ni_xi[i]=Ni_eta[i]=0);
	}
	Ni_xi[0]  = -(1 - eta) / 4 - (Ni_xi[7]  + Ni_xi[4])  / 2;
	Ni_eta[0] = -(1 - xi)  / 4 - (Ni_eta[7] + Ni_eta[4]) / 2;
	Ni_xi[1]  =  (1 - eta) / 4 - (Ni_xi[4]  + Ni_xi[5])  / 2;
	Ni_eta[1] = -(1 + xi)  / 4 - (Ni_eta[4] + Ni_eta[5]) / 2;
	Ni_xi[2]  =  (1 + eta) / 4 - (Ni_xi[5]  + Ni_xi[6])  / 2;
	Ni_eta[2] =  (1 + xi)  / 4 - (Ni_eta[5] + Ni_eta[6]) / 2;
	Ni_xi[3]  = -(1 + eta) / 4 - (Ni_xi[6]  + Ni_xi[7])  / 2;
	Ni_eta[3] =  (1 - xi)  / 4 - (Ni_eta[6] + Ni_eta[7]) / 2;
	MatrixXd naturalDerivatives(2,numberOfNodes);
	int cnt=0;
	for (int i=0; i<8; ++i) {
		if(bitElementType & (1 << i)) {
			naturalDerivatives(0,cnt) = Ni_xi[i];
			naturalDerivatives(1,cnt) = Ni_eta[i];
			++cnt;
		}
	}
	return naturalDerivatives;
}

__device__ __host__ Matrix2d cuinvJacobian(const MatrixXd& nodeCoord, const MatrixXd& naturalDerivatives) {
    return Jacobian(nodeCoord,naturalDerivatives).inverse();
}

__device__ __host__ MatrixXd cuXYDerivatives(const MatrixXd& nodeCoord, const MatrixXd& naturalDerivatives) {
    return cuinvJacobian(nodeCoord,naturalDerivatives) * naturalDerivatives;
}

__device__ __host__ double cudetJacobian(const MatrixXd& nodeCoord, const MatrixXd& naturalDerivatives) {
    return Jacobian(nodeCoord,naturalDerivatives).determinant();
}

__global__ void cu_element(
	double lambda,
	double tloop,
	double epsilon,
	const double* aPHI,
	const double* aU,
	const int* aEFT,
	const int* aNodeNum,
	const unsigned char* elementType,
	const double* aCoordX,
	const double* aCoordY,
	double* aM11,
	double* aM21,
	double* aM22,
	double* aK11,
	double* aK21,
	double* aK22,
	double* aF1,
	int ncSize,
	int elemSize
){
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	if (e < elemSize){
		Map<MatrixXd> mM11(aM11, ncSize, ncSize);
		Map<MatrixXd> mM21(aM21, ncSize, ncSize);
		Map<MatrixXd> mM22(aM22, ncSize, ncSize);
		Map<MatrixXd> mK11(aK11, ncSize, ncSize);
		Map<MatrixXd> mK21(aK21, ncSize, ncSize);
		Map<MatrixXd> mK22(aK22, ncSize, ncSize);
		Map<VectorXd> vF1(aF1, ncSize);

		mM11 = MatrixXd::Zero(ncSize, ncSize);
		mM21 = MatrixXd::Zero(ncSize, ncSize);
		mM22 = MatrixXd::Zero(ncSize, ncSize);
		mK11 = MatrixXd::Zero(ncSize, ncSize);
		mK21 = MatrixXd::Zero(ncSize, ncSize);
		mK22 = MatrixXd::Zero(ncSize, ncSize);
		vF1  = VectorXd::Zero(ncSize);

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

		// Gaussian point
		const double xis[4] = {-0.577350269189626, 0.577350269189626, 0.577350269189626, -0.577350269189626};
		const double etas[4] = {-0.577350269189626, -0.577350269189626, 0.577350269189626, 0.577350269189626};

		int m = 6;
        double RealTime = Tau0 * tloop; // (s)

        int numNodePerElement = aNodeNum[e];
        unsigned char bitElementType = elementType[e];
        MatrixXd elementNodesCoord(numNodePerElement,2);
        VectorXd phi(numNodePerElement);
        VectorXd u(numNodePerElement);

        for (unsigned i = 0; i < numNodePerElement; i++) {
        	int nodeSerial = aEFT[e*8 + i];
        	elementNodesCoord(i, 0) = aCoordX[nodeSerial];
        	elementNodesCoord(i, 1) = aCoordY[nodeSerial];
        	phi(i) = aPHI[nodeSerial];
        	u(i) = aU[nodeSerial];
        }

        double RealCoord = W0 * 0.25*(elementNodesCoord(0, 1) + elementNodesCoord(1, 1) + elementNodesCoord(2, 1) + elementNodesCoord(3, 1)) * 1E-6;

        MatrixXd Ce = MatrixXd::Zero(numNodePerElement, numNodePerElement);
        MatrixXd Ae = MatrixXd::Zero(numNodePerElement, numNodePerElement);
        MatrixXd Ee = MatrixXd::Zero(numNodePerElement, numNodePerElement);
		VectorXd Fe = VectorXd::Zero(numNodePerElement); // n x 1

		RowVectorXd N0 = cuShapeFunction(0, 0, bitElementType);
		MatrixXd dN0 = cuNaturalDerivatives(0, 0, bitElementType); // 2 x n
		MatrixXd B0 = cuXYDerivatives(elementNodesCoord, dN0); // 2 x n
		MatrixXd cotangent = get_cotangent(phi, B0); // 2 x 1
		double DERX = cotangent(0);
		double DERY = cotangent(1);
		double angle = atan2(DERY, DERX);
		double as = 1 + epsilon * cos(m*(angle - PI / 6)); // A(theta)
		double asp = -m * epsilon * sin(m*(angle - PI / 6)); // A'(theta)
		double col1 = 0;
		for(int i = 0; i < N0.size(); ++i){
			col1 += N0(i) * elementNodesCoord(i, 1);
		}
		double Temperature = T0 + G * 1E2 * (W0 * col1 - Vp*RealTime) * 1E-6; // (K)
		double theta = (Temperature - Ts) / dT0;

		int nGp = 4;
		for (int q=0; q<nGp; q++) {
			double xi = xis[q];
			double eta = etas[q];
			double W = 1;
			RowVectorXd N = cuShapeFunction(xi, eta, bitElementType); // 1 x n
			MatrixXd dN = cuNaturalDerivatives(xi, eta, bitElementType); // 2 x n
			MatrixXd B = cuXYDerivatives(elementNodesCoord, dN); // 2 x n
			double J = cudetJacobian(elementNodesCoord, dN); // 1 x 1
	                // matrixs of a element
			Ce += N.transpose() * N * W * J; // n x n
			Ae -= B.transpose() * B * W * J; // n x n
			Ee -= (B.row(1).transpose()*B.row(0) - B.row(0).transpose()*B.row(1)) * W * J; // n x n

			double Nphi = 0;
			double Nu = 0;
			for(int i = 0; i < N.size(); ++i){
				Nphi += N(i) * phi(i);
				Nu += N(i) * u(i);
			}
			Fe += N.transpose() * f(Nphi, Nu, theta, lambda) * W * J; // n x 1
		}
		
		for (unsigned i=0; i<numNodePerElement; i++) {
			int x = aEFT[e * 8 + i];
			for (unsigned j=0; j<numNodePerElement; j++) {
				int y = aEFT[e * 8 + j];
				int idx = y * ncSize + x;
				if (Ce(i, j) > 1.0E-12 || Ce(i, j) < -1.0E-12) {
					atomicAdd(&aM22[idx], Ce(i, j));
					atomicAdd(&aM21[idx], -0.5*Ce(i, j));
					atomicAdd(&aM11[idx], as * as * Ce(i, j));

					// mM22(x, y) += Ce(i, j);
					// mM21(x, y) += -0.5*Ce(i, j);
					// mM11(x, y) += as * as * Ce(i, j);
				}
				if (Ae(i, j) > 1.0E-12 || Ae(i, j) < -1.0E-12) {
					double N0phi = 0;
					for(int i = 0; i < N0.size(); ++i){
						N0phi += N0(i) * phi(i);
					}
					atomicAdd(&aK22[idx], -D * q(N0phi, 0.7) * Ae(i, j));
					atomicAdd(&aK11[idx], -as * as * Ae(i, j));

					// mK22(x, y) += -D * q(N0phi, 0.7) * Ae(i, j);
					// mK11(x, y) += -as * as * Ae(i, j);
				}
				if (Ee(i, j) > 1.0E-12 || Ee(i, j) < -1.0E-12)
					atomicAdd(&aK11[idx], -as * asp * Ee(i, j));

					// mK11(x, y) += -as * asp * Ee(i, j);
			}
			if (Fe(i) > 1.0E-12 || Fe(i) < -1.0E-12)
				atomicAdd(&aF1[x], Fe(i));

				// vF1(x) += Fe(i);
		}
	}
}

void FEM::cu_find_matrixs(double lambda, double epsilon, unsigned tloop, double dt){

	cudaMemcpy(aPHI, PHI.data(), sizeof(double)*ncSize, cudaMemcpyHostToDevice);
	cudaMemcpy(aU, U.data(), sizeof(double)*ncSize, cudaMemcpyHostToDevice);

	cudaMemcpy(adM11, aM11, sizeof(double)*ncSize*ncSize, cudaMemcpyHostToDevice);
	cudaMemcpy(adM21, aM21, sizeof(double)*ncSize*ncSize, cudaMemcpyHostToDevice);
	cudaMemcpy(adM22, aM22, sizeof(double)*ncSize*ncSize, cudaMemcpyHostToDevice);
	cudaMemcpy(adK11, aK11, sizeof(double)*ncSize*ncSize, cudaMemcpyHostToDevice);
	cudaMemcpy(adK21, aK21, sizeof(double)*ncSize*ncSize, cudaMemcpyHostToDevice);
	cudaMemcpy(adK22, aK22, sizeof(double)*ncSize*ncSize, cudaMemcpyHostToDevice);
	cudaMemcpy(adF1,  aF1, sizeof(double)*ncSize,         cudaMemcpyHostToDevice);

	for(int i = 0; i < ncSize ; ++i){
		if(PHI[i] != PHI[i]){
			cout << tloop << endl;
			exit(1);
		}
	}

	cudaDeviceSynchronize();


	cu_element<<<1024,1024>>>(lambda, tloop, epsilon, aPHI, aU, aEFT, aNodeNum, elementType, aCoordX, aCoordY, adM11, adM21, adM22, adK11, adK21, adK22, adF1, ncSize, elemSize);

	cudaDeviceSynchronize();
	
	cudaMemcpy(aM11, adM11, sizeof(double)*ncSize*ncSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(aM21, adM21, sizeof(double)*ncSize*ncSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(aM22, adM22, sizeof(double)*ncSize*ncSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(aK11, adK11, sizeof(double)*ncSize*ncSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(aK21, adK21, sizeof(double)*ncSize*ncSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(aK22, adK22, sizeof(double)*ncSize*ncSize, cudaMemcpyDeviceToHost);
	cudaMemcpy(aF1, adF1,  sizeof(double)*ncSize,         cudaMemcpyDeviceToHost);
}

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

	const double xis[4] = {-0.577350269189626, 0.577350269189626, 0.577350269189626, -0.577350269189626};
	const double etas[4] = {-0.577350269189626, -0.577350269189626, 0.577350269189626, 0.577350269189626};

	
	Map<MatrixXd> mM11(aM11, ncSize, ncSize);
	Map<MatrixXd> mM21(aM21, ncSize, ncSize);
	Map<MatrixXd> mM22(aM22, ncSize, ncSize);
	Map<MatrixXd> mK11(aK11, ncSize, ncSize);
	Map<MatrixXd> mK21(aK21, ncSize, ncSize);
	Map<MatrixXd> mK22(aK22, ncSize, ncSize);
	Map<VectorXd> vF1(aF1, ncSize);

	mM11 = MatrixXd::Zero(ncSize, ncSize);
	mM21 = MatrixXd::Zero(ncSize, ncSize);
	mM22 = MatrixXd::Zero(ncSize, ncSize);
	mK11 = MatrixXd::Zero(ncSize, ncSize);
	mK21 = MatrixXd::Zero(ncSize, ncSize);
	mK22 = MatrixXd::Zero(ncSize, ncSize);
	vF1  = VectorXd::Zero(ncSize);

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
		int nGp = 2 * 2;
		for (int q=0; q<nGp; q++) {
			double xi = xis[q];
			double eta = etas[q];
			double W = 1;
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

		for (unsigned i=0; i<numNodePerElement; i++) {
			int x = EFT[e][i];
			for (unsigned j=0; j<numNodePerElement; j++) {
				int y = EFT[e][j];
				if (Ce(i, j) > 1.0E-12 || Ce(i, j) < -1.0E-12) {
					mM22(x, y) += Ce(i, j);
					mM21(x, y) += -0.5*Ce(i, j);
					mM11(x, y) += as * as * Ce(i, j);
				}
				if (Ae(i, j) > 1.0E-12 || Ae(i, j) < -1.0E-12) {
					mK22(x, y) += -D * q(N0 * phi, 0.7) * Ae(i, j);
					mK11(x, y) += -as * as * Ae(i, j);
				}
				if (Ee(i, j) > 1.0E-12 || Ee(i, j) < -1.0E-12)
					mK11(x, y) += -as * asp * Ee(i, j);
			}
			if (Fe(i) > 1.0E-12 || Fe(i) < -1.0E-12)
				vF1(x) += Fe(i);
        	}
	}
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
	double W1L6 = (3 + rho + rhos - rho*rhos) / (2 * (1 + rho) * (1 + rhos));
	double lambda4 = 1;
	double lambda5 = 1 / (1 + rhos);
	unsigned nNode = ncSize;

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

		SparseMatrix<double> mM11 = Map<MatrixXd>(aM11, ncSize, ncSize).sparseView();
		SparseMatrix<double> mM21 = Map<MatrixXd>(aM21, ncSize, ncSize).sparseView();
		SparseMatrix<double> mM22 = Map<MatrixXd>(aM22, ncSize, ncSize).sparseView();
		SparseMatrix<double> mK11 = Map<MatrixXd>(aK11, ncSize, ncSize).sparseView();
		SparseMatrix<double> mK21 = Map<MatrixXd>(aK21, ncSize, ncSize).sparseView();
		SparseMatrix<double> mK22 = Map<MatrixXd>(aK22, ncSize, ncSize).sparseView();
		Map<VectorXd> vF1(aF1, ncSize);

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

	cu_find_matrixs(lambda, epsilon, tloop, dt);

	// find_matrixs(lambda, epsilon, tloop, dt);
	SparseMatrix<double> mM11 = Map<MatrixXd>(aM11, ncSize, ncSize).sparseView();
	SparseMatrix<double> mM21 = Map<MatrixXd>(aM21, ncSize, ncSize).sparseView();
	SparseMatrix<double> mM22 = Map<MatrixXd>(aM22, ncSize, ncSize).sparseView();
	SparseMatrix<double> mK11 = Map<MatrixXd>(aK11, ncSize, ncSize).sparseView();
	SparseMatrix<double> mK21 = Map<MatrixXd>(aK21, ncSize, ncSize).sparseView();
	SparseMatrix<double> mK22 = Map<MatrixXd>(aK22, ncSize, ncSize).sparseView();
	Map<VectorXd> vF1(aF1, ncSize);

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

	// cout << PHI << endl;

	// cout << endl;
	// cout << "next" << endl;
	// cout << endl;

	PHI = d2.topRows(nNode);
	
	// if(tloop == 50){
	// 	cout << PHI << endl;
	// 	exit(1);
	// }



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

__device__ __host__ MatrixXd get_cotangent(const VectorXd& phi, const MatrixXd& B) {
////////////////////////////////////////////////////////////////////////
// phi = / phi1 \     B =  / N1,x N2,x N3,x N4,x \     cot = / DERX \ //
//       | phi2 |          \ N1,y N2,y N3,y N4,y /           \ DERY / //
//       | phi3 |                                                     //
//       \ phi4 /                                                     //
////////////////////////////////////////////////////////////////////////
    return B * phi; // 2x1
}

// g'(phi) - lambda*U*P'(phi)
__device__ __host__ double f(double phi, double u, double theta, double lambda) {
	return phi * (1 - phi*phi) - lambda * u * pow(1 - phi*phi, 2.0);
	//return phi * (1 - phi*phi) - lambda * pow(1 - phi*phi, 2.0) * (u + 0.9 * phi * (1 - phi*phi) * ((double(rand()) / RAND_MAX) - 0.5));
	//return phi * (1 - phi*phi) - lambda * pow((1 - phi*phi), 2.0) * (u + theta);
	//return phi * (1 - phi*phi) - lambda * pow((1 - phi*phi), 2.0) * (u + theta + 0.3 * phi * (1 - phi*phi) * ((double(rand()) / RAND_MAX) - 0.5));
}

__device__ __host__ double q(double phi, double k) {
	//return (phi >= 1) ? 0 : (1 - phi) / (1 + k - (1 - k) * phi);
	return (phi >= 1) ? 0 : (1 - phi) / (1 + k - (1 - k) * phi) + (1 + phi) * 0.2 / 2;
	//return (phi >= 1) ? 0 : (1 - phi) / 2;
	//return (phi >= 1) ? 0 : (1 - phi) / 2 + (1 + phi) * 0.2 / 2;
}
