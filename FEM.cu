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

		for(int j = 0; j < EFT[i].size(); ++j){
			aEFT[i * 8+j] = EFT[i][j];
		}
		elementType[i] = (unsigned char)FinalElementList[i]->bitElementType.to_ulong();
	}

	//copy elementType to array
	for(int i = 0; i < ncSize; ++i){
		aCoordX[i] = NodeCoordinates[i].x;
		aCoordY[i] = NodeCoordinates[i].y;
	}


	// device pointer
	cudaFree(adM11);
	cudaFree(adM21);
	cudaFree(adM22);
	cudaFree(adK11);
	cudaFree(adK21);
	cudaFree(adK22);
    cudaFree(adF1);


	cudaMallocManaged((void **)&adM11, sizeof(float)*ncSize*ncSize*4);
	cudaMallocManaged((void **)&adM21, sizeof(float)*ncSize*ncSize*4);
	cudaMallocManaged((void **)&adM22, sizeof(float)*ncSize*ncSize*4);
	cudaMallocManaged((void **)&adK11, sizeof(float)*ncSize*ncSize*4);
	cudaMallocManaged((void **)&adK21, sizeof(float)*ncSize*ncSize*4);
	cudaMallocManaged((void **)&adK22, sizeof(float)*ncSize*ncSize*4);
	cudaMallocManaged((void **)&adF1,  sizeof(float)*ncSize*4);
}

__device__ __host__ RowVectorXf cuShapeFunction(float xi, float eta, unsigned char bitElementType) {
    //RowVectorXf shape(numberOfNodes);
    float Ni [8] = {(1 - xi) * (1 - eta) / 4,
        (1 + xi) * (1 - eta) / 4,
        (1 + xi) * (1 + eta) / 4,
        (1 - xi) * (1 + eta) / 4,
        (1 - xi*xi) * (1 - eta) / 2,
        (1 - eta*eta) * (1 + xi) / 2,
        (1 - xi*xi) * (1 + eta) / 2,
        (1 - eta*eta) * (1 - xi) / 2};
    RowVectorXf shape(8);
    switch((int)bitElementType){
        //Q8
        case(255): //11111111
            break;
        //Q7
        case(127): //01111111
            Ni[7] = 0;
            shape.resize(7);
            break;
        case(191): //10111111
            Ni[6] = 0;
            shape.resize(7);
            break;
        case(223): //11011111
            Ni[5] = 0;
            shape.resize(7);
            break;
        case(239): //11101111
            Ni[4] = 0;      
            shape.resize(7);
            break;
        //Q6
        case(63):  //00111111
            Ni[7] = Ni[6] = 0;
            shape.resize(6);
            break;
        case(159): //10011111
            Ni[6] = Ni[5] = 0;
            shape.resize(6);
            break;
        case(207): //11001111       
            Ni[5] = Ni[4] = 0;
            shape.resize(6);
            break;
        case(111): //01101111
            Ni[4] = Ni[7] = 0;
            shape.resize(6);
            break;
        case(175): //10101111
            Ni[6] = Ni[4] = 0;
            shape.resize(6);
            break;
        case(95): //01011111
            Ni[7] = Ni[5] = 0;
            shape.resize(6);
            break;
        //Q5
        case(31): //00011111
            Ni[7] = Ni[6] = Ni[5] = 0;
            shape.resize(5);
            break;
        case(143): //10001111
            Ni[6] = Ni[5] = Ni[4] = 0;
            shape.resize(5);
            break;
        case(79): //01001111
            Ni[7] = Ni[5] = Ni[4] = 0;
            shape.resize(5);
            break;
        case(47): //00101111
            Ni[7] = Ni[6] = Ni[4] = 0;
            shape.resize(5);
            break;
        //Q4
        case(15): //00001111
            Ni[7] = Ni[6] = Ni[5] = Ni[4] = 0;
            shape.resize(4);
            break;
    }
    int cnt=4;
        for (int i = 4;i<8;++i)
            if (bitElementType&(1<<i)) shape(cnt++)=Ni[i];
    shape(3) = Ni[3] - (Ni[6] + Ni[7]) / 2;
    shape(2) = Ni[2] - (Ni[5] + Ni[6]) / 2;
    shape(1) = Ni[1] - (Ni[4] + Ni[5]) / 2;
    shape(0) = Ni[0] - (Ni[7] + Ni[4]) / 2;
    return shape;
}

__device__ __host__ MatrixXf cuNaturalDerivatives(float xi, float eta, unsigned char bitElementType) {
    MatrixXf naturalDerivatives(2,8);
    float Ni_xi[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    float Ni_eta[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
    Ni_xi[4]  = - xi * (1 - eta);
    Ni_eta[4] = -(1 - xi*xi) / 2;
    Ni_xi[5]  =  (1 - eta*eta) / 2;
    Ni_eta[5] = - eta * (1 + xi);
    Ni_xi[6]  = - xi * (1 + eta);
    Ni_eta[6] =  (1 - xi*xi) / 2;
    Ni_xi[7]  = -(1 - eta*eta) / 2;
    Ni_eta[7] = - eta * (1 - xi);
    switch((int)bitElementType){
        //Q8
        case(255): //11111111
            break;
        //Q7
        case(127): //01111111
            Ni_xi[7] = 0;
            Ni_eta[7] = 0;
            naturalDerivatives.resize(2,7);
            break;
        case(191): //10111111
            Ni_xi[6] = 0;
            Ni_eta[6] = 0;
            naturalDerivatives.resize(2,7);
            break;
        case(223): //11011111
            Ni_xi[5] = 0;
            Ni_eta[5] = 0;
            naturalDerivatives.resize(2,7);
            break;
        case(239): //11101111
            Ni_xi[4] = 0;
            Ni_eta[4] = 0;
            naturalDerivatives.resize(2,7);
            break;
        //Q6
        case(63):  //00111111
            Ni_xi[7] = Ni_xi[6] = 0;
            Ni_eta[7] = Ni_eta[6] = 0;
            naturalDerivatives.resize(2,6);
            break;
        case(159): //10011111
            Ni_xi[6] = Ni_xi[5] = 0;
            Ni_eta[6] = Ni_eta[5] = 0;
            naturalDerivatives.resize(2,6);
            break;
        case(207): //11001111          
            Ni_xi[5] = Ni_xi[4] = 0;
            Ni_eta[5] = Ni_eta[4] = 0;
            naturalDerivatives.resize(2,6);
            break;
        case(111): //01101111
            Ni_xi[4] = Ni_xi[7] = 0;
            Ni_eta[4] = Ni_eta[7] = 0;
            naturalDerivatives.resize(2,6);
            break;
        case(175): //10101111
            Ni_xi[6] = Ni_xi[4] = 0;
            Ni_eta[6] = Ni_eta[4] = 0;
            naturalDerivatives.resize(2,6);
            break;
        case(95): //01011111
            Ni_xi[7] = Ni_xi[5] = 0;
            Ni_eta[7] = Ni_eta[5] = 0;
            naturalDerivatives.resize(2,6);
            break;
        //Q5
        case(31): //00011111
            Ni_xi[7] = Ni_xi[6] = Ni_xi[5] = 0;
            Ni_eta[7] = Ni_eta[6] = Ni_eta[5] = 0;
            naturalDerivatives.resize(2,5);
            break;
        case(143): //10001111
            Ni_xi[6] = Ni_xi[5] = Ni_xi[4] = 0;
            Ni_eta[6] = Ni_eta[5] = Ni_eta[4] = 0;
            naturalDerivatives.resize(2,5);
            break;
        case(79): //01001111
            Ni_xi[7] = Ni_xi[5] = Ni_xi[4] = 0;
            Ni_eta[7] = Ni_eta[5] = Ni_eta[4] = 0;
            naturalDerivatives.resize(2,5);
            break;
        case(47): //00101111
            Ni_xi[7] = Ni_xi[6] = Ni_xi[4] = 0;
            Ni_eta[7] = Ni_eta[6] = Ni_eta[4] = 0;
            naturalDerivatives.resize(2,5);
            break;
        //Q4
        case(15): //00001111
            Ni_xi[7] = Ni_xi[6] = Ni_xi[5] = Ni_xi[4] = 0;
            Ni_eta[7] = Ni_eta[6] = Ni_eta[5] = Ni_eta[4] = 0;
            naturalDerivatives.resize(2,4);
            break;
    }
    int cnt = 4;
    for (int i=4; i<8; ++i) {
        if (bitElementType & ( 1 << i ) ) {
            naturalDerivatives(0,cnt) = Ni_xi[i];
            naturalDerivatives(1,cnt) = Ni_eta[i];
            ++cnt;
        }
    }
    naturalDerivatives(0,0)  = -(1 - eta) / 4 - (Ni_xi[7]  + Ni_xi[4])  / 2;
    naturalDerivatives(1,0) = -(1 - xi)  / 4 - (Ni_eta[7] + Ni_eta[4]) / 2;
    naturalDerivatives(0,1)  =  (1 - eta) / 4 - (Ni_xi[4]  + Ni_xi[5])  / 2;
    naturalDerivatives(1,1) = -(1 + xi)  / 4 - (Ni_eta[4] + Ni_eta[5]) / 2;
    naturalDerivatives(0,2)  =  (1 + eta) / 4 - (Ni_xi[5]  + Ni_xi[6])  / 2;
    naturalDerivatives(1,2) =  (1 + xi)  / 4 - (Ni_eta[5] + Ni_eta[6]) / 2;
    naturalDerivatives(0,3)  = -(1 + eta) / 4 - (Ni_xi[6]  + Ni_xi[7])  / 2;
    naturalDerivatives(1,3) =  (1 - xi)  / 4 - (Ni_eta[6] + Ni_eta[7]) / 2;
    return naturalDerivatives;
}

__device__ float determinant(const Matrix2f& target){
	return target(0,0)*target(1,1)-target(0,1)*target(1,0);
}


__device__ Matrix2f inverse(const Matrix2f& target){
	float det=determinant(target);
	Matrix2f inv;
	inv(0,0)=target(1,1)/det;
	inv(1,1)=target(0,0)/det;
	inv(0,1)=-target(0,1)/det;
	inv(1,0)=-target(1,0)/det;
	return inv;
}

__device__ Matrix2f cuJacobian(const MatrixXf& nodeCoord, const MatrixXf& naturalDerivatives) {
    return naturalDerivatives * nodeCoord;
}

__device__ Matrix2f cuinvJacobian(const MatrixXf& nodeCoord, const MatrixXf& naturalDerivatives) {
    return inverse(cuJacobian(nodeCoord,naturalDerivatives));
}

__device__ MatrixXf cuXYDerivatives(const MatrixXf& nodeCoord, const MatrixXf& naturalDerivatives) {
    return cuinvJacobian(nodeCoord,naturalDerivatives) * naturalDerivatives;
}

__device__ float cudetJacobian(const MatrixXf& nodeCoord, const MatrixXf& naturalDerivatives) {
    return determinant(cuJacobian(nodeCoord,naturalDerivatives));
}

__device__ __host__ MatrixXf cu_get_cotangent(const VectorXf& phi, const MatrixXf& B) {
////////////////////////////////////////////////////////////////////////
// phi = / phi1 \     B =  / N1,x N2,x N3,x N4,x \     cot = / DERX \ //
//       | phi2 |          \ N1,y N2,y N3,y N4,y /           \ DERY / //
//       | phi3 |                                                     //
//       \ phi4 /                                                     //
////////////////////////////////////////////////////////////////////////
    return B * phi; // 2x1
}

// g'(phi) - lambda*U*P'(phi)
__device__ __host__ float cuF(float phi, float u, float theta, float lambda) {
	return phi * (1 - phi*phi) - lambda * u * pow(1 - phi*phi, 2.0);
	//return phi * (1 - phi*phi) - lambda * pow(1 - phi*phi, 2.0) * (u + 0.9 * phi * (1 - phi*phi) * ((double(rand()) / RAND_MAX) - 0.5));
	//return phi * (1 - phi*phi) - lambda * pow((1 - phi*phi), 2.0) * (u + theta);
	//return phi * (1 - phi*phi) - lambda * pow((1 - phi*phi), 2.0) * (u + theta + 0.3 * phi * (1 - phi*phi) * ((double(rand()) / RAND_MAX) - 0.5));
}

__device__ __host__ float cuQ(float phi, float k) {
	//return (phi >= 1) ? 0 : (1 - phi) / (1 + k - (1 - k) * phi);
	return (phi >= 1) ? 0 : (1 - phi) / (1 + k - (1 - k) * phi) + (1 + phi) * 0.2 / 2;
	//return (phi >= 1) ? 0 : (1 - phi) / 2;
	//return (phi >= 1) ? 0 : (1 - phi) / 2 + (1 + phi) * 0.2 / 2;
}

__global__ void cu_element(
	float lambda,
	float tloop,
	float epsilon,
	const double* aPHI,
	const double* aU,
	const int* aEFT,
	const int* aNodeNum,
	const unsigned char* elementType,
	const double* aCoordX,
	const double* aCoordY,
	float* aM11,
	float* aM21,
	float* aM22,
	float* aK11,
	float* aK21,
	float* aK22,
	float* aF1,
	int ncSize,
	int elemSize
){
	int e = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (e >= elemSize) return;

	const float PI = 3.14159265358979323846;
	float C_inf = 3;	// (wt%)
	float k = 0.14;		// 
	float G = 140 * 40;		// (K/cm)
	float d0 = 5E-3;		// (1E-6m)
	float alpha = 3000;	// (1E-6m2/s)
	float Vp = 3000;		// (1E-6m/s)
	float Ts = 273;		// (K)
	float dT0 = 2.6 * C_inf * (1 - k) / k;		// (K)
	float T0 = Ts - dT0 / 10;		// (K)
	float a1 = 0.8839;
	float a2 = 0.6267;
	float W0 = d0 * lambda / a1; // (1E-6m)
	float Tau0 = a2 * lambda * W0 * W0 / alpha; // (s)
	float D = lambda * a2;

	// Gaussian point
	const float xis[4] = {-0.577350269189626, 0.577350269189626, 0.577350269189626, -0.577350269189626};
	const float etas[4] = {-0.577350269189626, -0.577350269189626, 0.577350269189626, 0.577350269189626};

	int m = 6;
	float RealTime = Tau0 * tloop; // (s)
	int numNodePerElement = aNodeNum[e];
	unsigned char bitElementType = elementType[e];
	MatrixXf elementNodesCoord(numNodePerElement,2);
	VectorXf phi(numNodePerElement);
	VectorXf u(numNodePerElement);

	for (unsigned i = 0; i < numNodePerElement; i++) {
		int nodeSerial = aEFT[e*8 + i];
		elementNodesCoord(i, 0) = aCoordX[nodeSerial];
		elementNodesCoord(i, 1) = aCoordY[nodeSerial];
		phi(i) = aPHI[nodeSerial];
		u(i) = aU[nodeSerial];
	}

	MatrixXf Ce = MatrixXf::Zero(numNodePerElement, numNodePerElement);
	MatrixXf Ae = MatrixXf::Zero(numNodePerElement, numNodePerElement);
	MatrixXf Ee = MatrixXf::Zero(numNodePerElement, numNodePerElement);
	VectorXf Fe = VectorXf::Zero(numNodePerElement); // n x 1

	RowVectorXf N0 = cuShapeFunction(0, 0, bitElementType);
	MatrixXf dN0 = cuNaturalDerivatives(0, 0, bitElementType); // 2 x n
	MatrixXf B0 = cuXYDerivatives(elementNodesCoord, dN0); // 2 x n
	MatrixXf cotangent = cu_get_cotangent(phi, B0); // 2 x 1
	float DERX = cotangent(0);
	float DERY = cotangent(1);
	float angle = atan2(DERY, DERX);
	float as = 1 + epsilon * cos(m*(angle - PI / 6)); // A(theta)
	float asp = -m * epsilon * sin(m*(angle - PI / 6)); // A'(theta)
	float col1 = 0;
	for(int i = 0; i < N0.size(); ++i){
		col1 += N0(i) * elementNodesCoord(i, 1);
	}
	float Temperature = T0 + G * 1E2 * (W0 * col1 - Vp*RealTime) * 1E-6; // (K)
	float theta = (Temperature - Ts) / dT0;

	int nGp = 4;
	for (int q=0; q<nGp; q++) {
		float xi = xis[q];
		float eta = etas[q];
		float W = 1;
		RowVectorXf N = cuShapeFunction(xi, eta, bitElementType); // 1 x n
		MatrixXf dN = cuNaturalDerivatives(xi, eta, bitElementType); // 2 x n
		MatrixXf B = cuXYDerivatives(elementNodesCoord, dN); // 2 x n
		float J = cudetJacobian(elementNodesCoord, dN); // 1 x 1

                // matrixs of a element
		Ce += N.transpose() * N * W * J; // n x n
		Ae -= B.transpose() * B * W * J; // n x n
		Ee -= (B.row(1).transpose()*B.row(0) - B.row(0).transpose()*B.row(1)) * W * J; // n x n

		float Nphi = 0;
		float Nu = 0;
		for(int i = 0; i < N.size(); ++i){
			Nphi += N(i) * phi(i);
			Nu += N(i) * u(i);
		}
		Fe += N.transpose() * cuF(Nphi, Nu, theta, lambda) * W * J; // n x 1
	}
	int site[8] = {0};
	// int ElementTypeToSite[8] = {2, 3, 0, 1, 2, 3, 0, 1};
    int ElementTypeToSite[8] = {0, 1, 2, 3, 0, 1, 2, 3};
	int cnt = 0;
	for(int i = 0; i < 8; ++i){
		if(bitElementType & (1 << i)){
			site[cnt] = ElementTypeToSite[i];
			++cnt;
		}
	}
	
	for (unsigned i=0; i<numNodePerElement; i++) {
		int x = aEFT[e * 8 + i];
		for (unsigned j=0; j<numNodePerElement; j++) {
			int y = aEFT[e * 8 + j];
			int idx = y * ncSize + x + ncSize*ncSize*site[i];
			if (Ce(i, j) > 1.0E-12 || Ce(i, j) < -1.0E-12) {
				aM22[idx] = Ce(i, j);
				aM21[idx] = -0.5*Ce(i, j);
				aM11[idx] = as * as * Ce(i, j);

			}
			if (Ae(i, j) > 1.0E-12 || Ae(i, j) < -1.0E-12) {
				float N0phi = 0;
				for(int i = 0; i < N0.size(); ++i){
					N0phi += N0(i) * phi(i);
				}
				aK22[idx] = -D * cuQ(N0phi, 0.7) * Ae(i, j);
				aK11[idx] = -as * as * Ae(i, j);
			}
			if (Ee(i, j) > 1.0E-12 || Ee(i, j) < -1.0E-12)
				aK11[idx] = -as * asp * Ee(i, j);

		}
		if (Fe(i) > 1.0E-12 || Fe(i) < -1.0E-12)
			aF1[x+ncSize*site[i]] = Fe(i);
	}
}

__global__ void cu_sum(float* adM11,
					   float* adM21,
					   float* adM22,
					   float* adK11,
					   float* adK21,
					   float* adK22,
					   float* adF1,
					   size_t ncSize){
	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if(idx >= ncSize*ncSize) return;
	size_t s = ncSize*ncSize;
	for(int i = 1; i < 4; ++i){
		adM11[idx] += adM11[idx + s*i];
		adM21[idx] += adM21[idx + s*i];
		adM22[idx] += adM22[idx + s*i];
		adK11[idx] += adK11[idx + s*i];
		adK21[idx] += adK21[idx + s*i];
		adK22[idx] += adK22[idx + s*i];
	}

	if(idx >= ncSize) return;
	s = ncSize;
	for(int i = 1; i < 4; ++i){
		adF1[idx] += adF1[idx + s*i];
	}
}


void FEM::cu_find_matrixs(float lambda, float epsilon, unsigned tloop, float dt){

	cudaMemcpy(aPHI, PHI.data(), sizeof(double)*ncSize, cudaMemcpyHostToDevice);
	cudaMemcpy(aU, U.data(), sizeof(double)*ncSize, cudaMemcpyHostToDevice);

	cudaMemset(adM11, 0, sizeof(float)*ncSize*ncSize*4);
	cudaMemset(adM21, 0, sizeof(float)*ncSize*ncSize*4);
	cudaMemset(adM22, 0, sizeof(float)*ncSize*ncSize*4);
	cudaMemset(adK11, 0, sizeof(float)*ncSize*ncSize*4);
	cudaMemset(adK21, 0, sizeof(float)*ncSize*ncSize*4);
	cudaMemset(adK22, 0, sizeof(float)*ncSize*ncSize*4);
	cudaMemset(adF1,  0, sizeof(float)*ncSize*4);

	cu_element<<<CeilDiv(elemSize, 256), 256>>>(lambda, tloop, epsilon, aPHI, aU, aEFT, aNodeNum, elementType, aCoordX, aCoordY, adM11, adM21, adM22, adK11, adK21, adK22, adF1, ncSize, elemSize);
	cudaDeviceSynchronize();


	cu_sum<<<CeilDiv(ncSize*ncSize, 256), 256>>>(adM11, adM21, adM22, adK11, adK21, adK22, adF1, ncSize);

	cudaDeviceSynchronize();
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
		cu_find_matrixs(lambda, epsilon, tloop, dt);

		SparseMatrix<double> mM11 = Map<MatrixXf>(adM11, ncSize, ncSize).cast<double>().sparseView();
		SparseMatrix<double> mM21 = Map<MatrixXf>(adM21, ncSize, ncSize).cast<double>().sparseView();
		SparseMatrix<double> mM22 = Map<MatrixXf>(adM22, ncSize, ncSize).cast<double>().sparseView();
		SparseMatrix<double> mK11 = Map<MatrixXf>(adK11, ncSize, ncSize).cast<double>().sparseView();
		SparseMatrix<double> mK21 = Map<MatrixXf>(adK21, ncSize, ncSize).cast<double>().sparseView();
		SparseMatrix<double> mK22 = Map<MatrixXf>(adK22, ncSize, ncSize).cast<double>().sparseView();
		VectorXd vF1 = Map<VectorXf>(adF1, ncSize).cast<double>();

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
	

	
	SparseMatrix<double> mM11 = Map<MatrixXf>(adM11, ncSize, ncSize).cast<double>().sparseView();
	SparseMatrix<double> mM21 = Map<MatrixXf>(adM21, ncSize, ncSize).cast<double>().sparseView();
	SparseMatrix<double> mM22 = Map<MatrixXf>(adM22, ncSize, ncSize).cast<double>().sparseView();
	SparseMatrix<double> mK11 = Map<MatrixXf>(adK11, ncSize, ncSize).cast<double>().sparseView();
	SparseMatrix<double> mK21 = Map<MatrixXf>(adK21, ncSize, ncSize).cast<double>().sparseView();
	SparseMatrix<double> mK22 = Map<MatrixXf>(adK22, ncSize, ncSize).cast<double>().sparseView();
	VectorXd vF1 = Map<VectorXf>(adF1, ncSize).cast<double>();

    // cout << vF1 << endl;
    // exit(1);

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
