#include "ShapeFunctions.h"
#include <Eigen/Dense>
#include <bitset>
#include <iostream>
#include <array>
using namespace std;
using namespace Eigen;

RowVectorXd ShapeFunction(double xi, double eta, unsigned char bitElementType) {
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

MatrixXd NaturalDerivatives(double xi, double eta, unsigned char bitElementType) {
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
	for (int i=0; i<8; ++i) if(bitElementType & (1 << i)) naturalDerivatives(0,cnt++) = Ni_xi[i];
	cnt=0;
	for (int i=0; i<8; ++i) if(bitElementType & (1 << i)) naturalDerivatives(1,cnt++) = Ni_eta[i];
	return naturalDerivatives;
}

RowVectorXd ShapeFunction(double xi, double eta, bitset<8> bitElementType) {
	RowVectorXd shape(bitElementType.count()); // 1 x n
	array<double, 8> Ni = { 0, 0, 0, 0, 0, 0, 0, 0 };
	size_t cnt = bitElementType.count() - 1; 
	for (int i = 7; i >= 0; i--) {
		if (bitElementType.test(i)) {
			switch (i)
			{
			case 0:
				Ni[0] = (1 - xi) * (1 - eta) / 4 - (Ni[7] + Ni[4]) / 2; // N1 - (N8 + N5) / 2
				break;
			case 1:
				Ni[1] = (1 + xi) * (1 - eta) / 4 - (Ni[4] + Ni[5]) / 2; // N2 - (N5 + N6) / 2
				break;
			case 2:
				Ni[2] = (1 + xi) * (1 + eta) / 4 - (Ni[5] + Ni[6]) / 2; // N3 - (N6 + N7) / 2
				break;
			case 3:
				Ni[3] = (1 - xi) * (1 + eta) / 4 - (Ni[6] + Ni[7]) / 2; // N4 - (N7 + N8) / 2
				break;
			case 4:
				Ni[4] = (1 - xi*xi) * (1 - eta) / 2; // N5
				break;
			case 5:
				Ni[5] = (1 - eta*eta) * (1 + xi) / 2; // N6
				break;
			case 6:
				Ni[6] = (1 - xi*xi) * (1 + eta) / 2; // N7
				break;
			case 7:
				Ni[7] = (1 - eta*eta) * (1 - xi) / 2; // N8
				break;
			default:
				cerr << "No such node number!" << endl;
				break;
			}
			shape(cnt--) = Ni[i];
		}
	}

	return shape;
}

MatrixXd NaturalDerivatives(double xi, double eta, bitset<8> bitElementType) {
	MatrixXd naturalDerivatives(2, bitElementType.count()); // 2 x n
	array<double, 8> Ni_xi = { 0, 0, 0, 0, 0, 0, 0, 0 };
	array<double, 8> Ni_eta = { 0, 0, 0, 0, 0, 0, 0, 0 };
	size_t cnt = bitElementType.count() - 1;
	for (int i = 7; i >= 0; i--) {
		if (bitElementType.test(i)) {
			switch (i)
			{
			case 0: // dN1 - (dN8 + dN5) / 2
				Ni_xi[0]	= -(1 - eta) / 4	- (Ni_xi[7]  + Ni_xi[4])  / 2; 
				Ni_eta[0]	= -(1 - xi)	 / 4	- (Ni_eta[7] + Ni_eta[4]) / 2;
				break;
			case 1: // dN2 - (dN5 + dN6) / 2
				Ni_xi[1]	=  (1 - eta) / 4	- (Ni_xi[4]  + Ni_xi[5])  / 2; 
				Ni_eta[1]	= -(1 + xi)  / 4	- (Ni_eta[4] + Ni_eta[5]) / 2;
				break;
			case 2: // dN3 - (dN6 + dN7) / 2
				Ni_xi[2]	=  (1 + eta) / 4	- (Ni_xi[5]  + Ni_xi[6])  / 2;
				Ni_eta[2]	=  (1 + xi)  / 4	- (Ni_eta[5] + Ni_eta[6]) / 2; 
				break;
			case 3: // dN4 - (dN7 + dN8) / 2
				Ni_xi[3]	= -(1 + eta) / 4	- (Ni_xi[6]  + Ni_xi[7])  / 2;
				Ni_eta[3]	=  (1 - xi)	 / 4	- (Ni_eta[6] + Ni_eta[7]) / 2;
				break;
			case 4: // dN5
				Ni_xi[4]	= - xi * (1 - eta);
				Ni_eta[4]	= -(1 - xi*xi) / 2;
				break;
			case 5: // dN6
				Ni_xi[5]	=  (1 - eta*eta) / 2;
				Ni_eta[5]	= - eta * (1 + xi);
				break;
			case 6: // dN7
				Ni_xi[6]	= - xi * (1 + eta);
				Ni_eta[6]	=  (1 - xi*xi) / 2;
				break;
			case 7: // dN8
				Ni_xi[7]	= -(1 - eta*eta) / 2;
				Ni_eta[7]	= - eta * (1 - xi);
				break;
			default:
				cerr << "No such node number!" << endl;
				break;
			}
			naturalDerivatives(0, cnt) = Ni_xi[i];
			naturalDerivatives(1, cnt--) = Ni_eta[i];
		}
	}

	return naturalDerivatives;
}

Matrix2d Jacobian(const MatrixXd& nodeCoord, const MatrixXd& naturalDerivatives) {
//////////////////////////////////////////////////////////////////
// nodeCoord             naturalDerivatives                     //
//     =  / x1 y1 \          =  / N1,xi  N2,xi  N3,xi  N4,xi  \ //
//        | x2 y2 |             \ N1,eta N2,eta N3,eta N4,eta / //
//        | x3 y3 |                                             //
//        \ x4 y4 /                                             //
//////////////////////////////////////////////////////////////////
    return naturalDerivatives * nodeCoord;
}

Matrix2d invJacobian(const MatrixXd& nodeCoord, const MatrixXd& naturalDerivatives) {
    return Jacobian(nodeCoord,naturalDerivatives).inverse();
}

MatrixXd XYDerivatives(const MatrixXd& nodeCoord, const MatrixXd& naturalDerivatives) {
//////////////////////////////////////
// output = / N1,x N2,x N3,x N4,x \ //
//          \ N1,y N2,y N3,y N4,y / //
//////////////////////////////////////
    return invJacobian(nodeCoord,naturalDerivatives) * naturalDerivatives;
}

double detJacobian(const MatrixXd& nodeCoord, const MatrixXd& naturalDerivatives) {
    return Jacobian(nodeCoord,naturalDerivatives).determinant();
}
