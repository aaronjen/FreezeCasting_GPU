#include "Quadtree.h"
#include "ShapeFunctions.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <array>
#include <vector>
#include <memory>
#include <Eigen/Dense>
using namespace std;
using namespace Eigen;

	///////////////////
	// (3)--(6)--(2) //
	//  |         |  //
	// (7)       (5) //
	//  |         |  //
	// (0)--(4)--(1) //
	///////////////////

void Quadtree_Initialization(double Nx, double Ny, unsigned maxLv, double gamma,
							vector<vector<shared_ptr<Element>>>& vvpLevelElementList, 
							map<Coord, unsigned>& mNodeCoordinateList,
							vector<vector<int>>& vvEFT,
							int option, map<Coord, double>& mPhiCoordinateList, map<Coord, double>& mUCoordinateList,
							map<Coord, double>& mPhiVelocityCoordinateList, map<Coord, double>& mUVelocityCoordinateList) {

	// Offer containers for elements from level 0 to maxLv
	vvpLevelElementList.resize(maxLv + 1);
	// Level 0
	Quadtree_ConstructRoot(Nx, Ny, maxLv, vvpLevelElementList);

	// Subdivide Level 1 to maxLv
	Quadtree_Subdivide(maxLv, gamma, vvpLevelElementList, option, mPhiCoordinateList, mUCoordinateList, mPhiVelocityCoordinateList, mUVelocityCoordinateList);
}


void Quadtree_MeshGenerate(unsigned maxLv, double gamma, vector<vector<shared_ptr<Element>>>& vvpLevelElementList, int option, map<Coord, double>& mPhiCoordinateList, map<Coord, double>& mUCoordinateList,
						  map<Coord, double>& mPhiVelocityCoordinateList, map<Coord, double>& mUVelocityCoordinateList) {

	// Fuse Level maxLv-1 to 0
	//cout << endl << "Fuse:" << endl;
	Quadtree_Fuse(maxLv, gamma, vvpLevelElementList, option, mPhiCoordinateList, mUCoordinateList, mPhiVelocityCoordinateList, mUVelocityCoordinateList);

	// Subdivide Level 1 to maxLv
	//cout << endl << "Subdivide:" << endl;
	Quadtree_Subdivide(maxLv, gamma, vvpLevelElementList, option, mPhiCoordinateList, mUCoordinateList, mPhiVelocityCoordinateList, mUVelocityCoordinateList);
}


void Quadtree_ConstructRoot(double Nx, double Ny, unsigned maxLv,
						   vector<vector<shared_ptr<Element>>>& vvpLevelElementList) {

	Coord SW(0, 0);
	Coord SE(Nx, 0);
	Coord NE(Nx, Ny);
	Coord NW(0, Nx);
	array<Coord, 4> RootCoordinates = { SW, SE, NE, NW };

	shared_ptr<Element> Root = CreateElement(RootCoordinates, 0, 0, maxLv, vvpLevelElementList);
	//cout << "\tLevel 0 \tDone!" << endl;
}


void Quadtree_Subdivide(unsigned maxLv, double gamma, vector<vector<shared_ptr<Element>>>& vvpLevelElementList, int option, map<Coord, double>& mPhiCoordinateList, map<Coord, double>& mUCoordinateList, 
					   map<Coord, double>& mPhiVelocityCoordinateList, map<Coord, double>& mUVelocityCoordinateList) {

	for (unsigned lv = 0; lv < maxLv; lv++) {
		for (unsigned i = 0; i < vvpLevelElementList[lv].size(); i++) {
			if (vvpLevelElementList[lv][i]->CheckError(option, mPhiCoordinateList, mUCoordinateList, gamma)) {
				vvpLevelElementList[lv][i]->Subdivide(maxLv, vvpLevelElementList, mPhiCoordinateList, mUCoordinateList, mPhiVelocityCoordinateList, mUVelocityCoordinateList);
			}
		}
		//cout << "\tLevel " << lv + 1 << " \tDone!" << endl;
	}
}


void Quadtree_Fuse(unsigned maxLv, double gamma, vector<vector<shared_ptr<Element>>>& vvpLevelElementList, int option, map<Coord, double>& mPhiCoordinateList, map<Coord, double>& mUCoordinateList,
				  map<Coord, double>& mPhiVelocityCoordinateList, map<Coord, double>& mUVelocityCoordinateList) {

	for (unsigned lv = 0; lv < maxLv; lv++) {
		unsigned reverseLv = maxLv - 1 - lv;
		if (vvpLevelElementList[reverseLv].size() > 0) {
			for (unsigned i = 0; i < vvpLevelElementList[reverseLv].size(); i++) {
				vvpLevelElementList[reverseLv][i]->Fuse(vvpLevelElementList, option, mPhiCoordinateList, mUCoordinateList, mPhiVelocityCoordinateList, mUVelocityCoordinateList, gamma);
			}
		}
		//cout << "\tLevel " << reverseLv << " \tDone!" << endl;
	}

	//cout << endl << "Deleting unused elements..." << endl;
	DeleteElement(vvpLevelElementList);
}

void Quadtree_AddNodes(vector<vector<shared_ptr<Element>>>& vvpLevelElementList, map<Coord, unsigned>& mNodeCoordinateList) {

	// Initialization
	mNodeCoordinateList.clear();
	
	// Add nodes for leaf elements
	for (unsigned lv = 0; lv < vvpLevelElementList.size(); lv++)
		for (unsigned e = 0; e < vvpLevelElementList[lv].size(); e++)
			if (!vvpLevelElementList[lv][e]->apChildren[0])
				vvpLevelElementList[lv][e]->AddNodes(mNodeCoordinateList);
}

void Element::Subdivide(unsigned maxLv, vector<vector<shared_ptr<Element>>>& vvpLevelElementList,
					   map<Coord, double>& mPhiCoordinateList, map<Coord, double>& mUCoordinateList,
					   map<Coord, double>& mPhiVelocityCoordinateList, map<Coord, double>& mUVelocityCoordinateList) {

	if (!apChildren[0] && CheckNeighbors(maxLv, vvpLevelElementList, mPhiCoordinateList, mUCoordinateList, mPhiVelocityCoordinateList, mUVelocityCoordinateList)) { // If no children

		array<Coord, 4> BoundingBox;
		Coord SW, SE, NE, NW;
		array<shared_ptr<Element>, 8> ChildNeighbors;
		unsigned uChildLevel = uLevel + 1;
		
		// Create sub-elements ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// apChildren[0]
		SW.set(acNodalCoordinates[0].x, acNodalCoordinates[0].y);
		SE.set((acNodalCoordinates[0].x + acNodalCoordinates[1].x) / 2, acNodalCoordinates[0].y);
		NE.set((acNodalCoordinates[0].x + acNodalCoordinates[1].x) / 2, (acNodalCoordinates[1].y + acNodalCoordinates[2].y) / 2);
		NW.set(acNodalCoordinates[0].x, (acNodalCoordinates[1].y + acNodalCoordinates[2].y) / 2);
		BoundingBox = { SW, SE, NE, NW };
		apChildren[0] = CreateElement(BoundingBox, pMe, uChildLevel, maxLv, vvpLevelElementList);

		if (!mPhiCoordinateList.count(SE)) {
			mPhiCoordinateList[SE] = 0.5 *  (mPhiCoordinateList[acNodalCoordinates[0]] + mPhiCoordinateList[acNodalCoordinates[1]]);
			mUCoordinateList[SE] = 0.5 *  (mUCoordinateList[acNodalCoordinates[0]] + mUCoordinateList[acNodalCoordinates[1]]);
			mPhiVelocityCoordinateList[SE] = 0.5 *  (mPhiVelocityCoordinateList[acNodalCoordinates[0]] + mPhiVelocityCoordinateList[acNodalCoordinates[1]]);
			mUVelocityCoordinateList[SE] = 0.5 *  (mUVelocityCoordinateList[acNodalCoordinates[0]] + mUVelocityCoordinateList[acNodalCoordinates[1]]);

		}
		if (!mPhiCoordinateList.count(NE)) {
			mPhiCoordinateList[NE] = 0.25 * (mPhiCoordinateList[acNodalCoordinates[0]] + mPhiCoordinateList[acNodalCoordinates[1]] +
				mPhiCoordinateList[acNodalCoordinates[2]] + mPhiCoordinateList[acNodalCoordinates[3]]);
			mUCoordinateList[NE] = 0.25 * (mUCoordinateList[acNodalCoordinates[0]] + mUCoordinateList[acNodalCoordinates[1]] +
				mUCoordinateList[acNodalCoordinates[2]] + mUCoordinateList[acNodalCoordinates[3]]);
			mPhiVelocityCoordinateList[NE] = 0.25 * (mPhiVelocityCoordinateList[acNodalCoordinates[0]] + mPhiVelocityCoordinateList[acNodalCoordinates[1]] +
				mPhiVelocityCoordinateList[acNodalCoordinates[2]] + mPhiVelocityCoordinateList[acNodalCoordinates[3]]);
			mUVelocityCoordinateList[NE] = 0.25 * (mUVelocityCoordinateList[acNodalCoordinates[0]] + mUVelocityCoordinateList[acNodalCoordinates[1]] +
				mUVelocityCoordinateList[acNodalCoordinates[2]] + mUVelocityCoordinateList[acNodalCoordinates[3]]);
		}
		if (!mPhiCoordinateList.count(NW)) {
			mPhiCoordinateList[NW] = 0.5 *  (mPhiCoordinateList[acNodalCoordinates[0]] + mPhiCoordinateList[acNodalCoordinates[3]]);
			mUCoordinateList[NW] = 0.5 *  (mUCoordinateList[acNodalCoordinates[0]] + mUCoordinateList[acNodalCoordinates[3]]);
			mPhiVelocityCoordinateList[NW] = 0.5 *  (mPhiVelocityCoordinateList[acNodalCoordinates[0]] + mPhiVelocityCoordinateList[acNodalCoordinates[3]]);
			mUVelocityCoordinateList[NW] = 0.5 *  (mUVelocityCoordinateList[acNodalCoordinates[0]] + mUVelocityCoordinateList[acNodalCoordinates[3]]);
		}
		
		// apChildren[1]
		SW.set((acNodalCoordinates[0].x + acNodalCoordinates[1].x) / 2, acNodalCoordinates[0].y);
		SE.set(acNodalCoordinates[1].x, acNodalCoordinates[0].y);
		NE.set(acNodalCoordinates[1].x, (acNodalCoordinates[1].y + acNodalCoordinates[2].y) / 2);
		NW.set((acNodalCoordinates[0].x + acNodalCoordinates[1].x) / 2, (acNodalCoordinates[1].y + acNodalCoordinates[2].y) / 2);
		BoundingBox = { SW, SE, NE, NW };
		apChildren[1] = CreateElement(BoundingBox, pMe, uChildLevel, maxLv, vvpLevelElementList);

		if (!mPhiCoordinateList.count(NE)) {
			mPhiCoordinateList[NE] = 0.5 *  (mPhiCoordinateList[acNodalCoordinates[1]] + mPhiCoordinateList[acNodalCoordinates[2]]);
			mUCoordinateList[NE] = 0.5 *  (mUCoordinateList[acNodalCoordinates[1]] + mUCoordinateList[acNodalCoordinates[2]]);
			mPhiVelocityCoordinateList[NE] = 0.5 *  (mPhiVelocityCoordinateList[acNodalCoordinates[1]] + mPhiVelocityCoordinateList[acNodalCoordinates[2]]);
			mUVelocityCoordinateList[NE] = 0.5 *  (mUVelocityCoordinateList[acNodalCoordinates[1]] + mUVelocityCoordinateList[acNodalCoordinates[2]]);
		}

		// apChildren[2]
		SW.set((acNodalCoordinates[0].x + acNodalCoordinates[1].x) / 2, (acNodalCoordinates[1].y + acNodalCoordinates[2].y) / 2);
		SE.set(acNodalCoordinates[1].x, (acNodalCoordinates[1].y + acNodalCoordinates[2].y) / 2);
		NE.set(acNodalCoordinates[1].x, acNodalCoordinates[2].y);
		NW.set((acNodalCoordinates[0].x + acNodalCoordinates[1].x) / 2, acNodalCoordinates[2].y);
		BoundingBox = { SW, SE, NE, NW };
		apChildren[2] = CreateElement(BoundingBox, pMe, uChildLevel, maxLv, vvpLevelElementList);

		if (!mPhiCoordinateList.count(NW)) {
			mPhiCoordinateList[NW] = 0.5 *  (mPhiCoordinateList[acNodalCoordinates[2]] + mPhiCoordinateList[acNodalCoordinates[3]]);
			mUCoordinateList[NW] = 0.5 *  (mUCoordinateList[acNodalCoordinates[2]] + mUCoordinateList[acNodalCoordinates[3]]);
			mPhiVelocityCoordinateList[NW] = 0.5 *  (mPhiVelocityCoordinateList[acNodalCoordinates[2]] + mPhiVelocityCoordinateList[acNodalCoordinates[3]]);
			mUVelocityCoordinateList[NW] = 0.5 *  (mUVelocityCoordinateList[acNodalCoordinates[2]] + mUVelocityCoordinateList[acNodalCoordinates[3]]);
		}

		// apChildren[3]
		SW.set(acNodalCoordinates[0].x, (acNodalCoordinates[1].y + acNodalCoordinates[2].y) / 2);
		SE.set((acNodalCoordinates[0].x + acNodalCoordinates[1].x) / 2, (acNodalCoordinates[1].y + acNodalCoordinates[2].y) / 2);
		NE.set((acNodalCoordinates[0].x + acNodalCoordinates[1].x) / 2, acNodalCoordinates[2].y);
		NW.set(acNodalCoordinates[0].x, acNodalCoordinates[2].y);
		BoundingBox = { SW, SE, NE, NW };
		apChildren[3] = CreateElement(BoundingBox, pMe, uChildLevel, maxLv, vvpLevelElementList);

		// Find neighbors /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// apChildren[0]
		ChildNeighbors[0] = apNeighbors[0]; // P0
		ChildNeighbors[1] = apNeighbors[0]; // P0
		ChildNeighbors[2] = apChildren[1];  // C1
		ChildNeighbors[3] = apChildren[1];  // C1
		ChildNeighbors[4] = apChildren[3];  // C3
		ChildNeighbors[5] = apChildren[3];  // C3
		ChildNeighbors[6] = apNeighbors[7]; // P7
		ChildNeighbors[7] = apNeighbors[7]; // P7
		apChildren[0]->apNeighbors = ChildNeighbors;

		// apChildren[1]
		ChildNeighbors[0] = apNeighbors[1]; // P1
		ChildNeighbors[1] = apNeighbors[1]; // P1
		ChildNeighbors[2] = apNeighbors[2]; // P2
		ChildNeighbors[3] = apNeighbors[2]; // P2
		ChildNeighbors[4] = apChildren[2];  // C2
		ChildNeighbors[5] = apChildren[2];  // C2
		ChildNeighbors[6] = apChildren[0];  // C0
		ChildNeighbors[7] = apChildren[0];  // C0
		apChildren[1]->apNeighbors = ChildNeighbors;

		// apChildren[2]
		ChildNeighbors[0] = apChildren[1];  // C1
		ChildNeighbors[1] = apChildren[1];  // C1
		ChildNeighbors[2] = apNeighbors[3]; // P3
		ChildNeighbors[3] = apNeighbors[3]; // P3
		ChildNeighbors[4] = apNeighbors[4]; // P4
		ChildNeighbors[5] = apNeighbors[4]; // P4
		ChildNeighbors[6] = apChildren[3];  // C3
		ChildNeighbors[7] = apChildren[3];  // C3
		apChildren[2]->apNeighbors = ChildNeighbors;

		// apChildren[3]
		ChildNeighbors[0] = apChildren[0];  // C0
		ChildNeighbors[1] = apChildren[0];  // C0
		ChildNeighbors[2] = apChildren[2];  // C2
		ChildNeighbors[3] = apChildren[2];  // C2
		ChildNeighbors[4] = apNeighbors[5]; // P5
		ChildNeighbors[5] = apNeighbors[5]; // P5
		ChildNeighbors[6] = apNeighbors[6]; // P6
		ChildNeighbors[7] = apNeighbors[6]; // P6
		apChildren[3]->apNeighbors = ChildNeighbors;

		// Greet neighbors ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// apNeighbors[0] & apNeighbors[1]
		if (apNeighbors[0] != apNeighbors[1] && apNeighbors[0] && apNeighbors[1]) {
			apNeighbors[0]->apNeighbors[4] = apChildren[0]; // C0
			apNeighbors[0]->apNeighbors[5] = apChildren[0]; // C0
			apNeighbors[1]->apNeighbors[4] = apChildren[1]; // C1
			apNeighbors[1]->apNeighbors[5] = apChildren[1]; // C1
		}
		else if (apNeighbors[0] == apNeighbors[1] && apNeighbors[0]) {
			apNeighbors[0]->apNeighbors[4] = apChildren[1]; // C1
			apNeighbors[0]->apNeighbors[5] = apChildren[0]; // C0
		}

		// apNeighbors[2] & apNeighbors[3]
		if (apNeighbors[2] != apNeighbors[3] && apNeighbors[2] && apNeighbors[3]) {
			apNeighbors[2]->apNeighbors[6] = apChildren[1]; // C1
			apNeighbors[2]->apNeighbors[7] = apChildren[1]; // C1
			apNeighbors[3]->apNeighbors[6] = apChildren[2]; // C2
			apNeighbors[3]->apNeighbors[7] = apChildren[2]; // C2
		}
		else if (apNeighbors[2] == apNeighbors[3] && apNeighbors[2]) {
			apNeighbors[2]->apNeighbors[6] = apChildren[2]; // C2
			apNeighbors[2]->apNeighbors[7] = apChildren[1]; // C1
		}

		// apNeighbors[4] & apNeighbors[5]
		if (apNeighbors[4] != apNeighbors[5] && apNeighbors[4] && apNeighbors[5]) {
			apNeighbors[4]->apNeighbors[0] = apChildren[2]; // C2
			apNeighbors[4]->apNeighbors[1] = apChildren[2]; // C2
			apNeighbors[5]->apNeighbors[0] = apChildren[3]; // C3
			apNeighbors[5]->apNeighbors[1] = apChildren[3]; // C3
		}
		else if (apNeighbors[4] == apNeighbors[5] && apNeighbors[4]) {
			apNeighbors[4]->apNeighbors[0] = apChildren[3]; // C3
			apNeighbors[4]->apNeighbors[1] = apChildren[2]; // C2
		}

		// apNeighbors[6] & apNeighbors[7]
		if (apNeighbors[6] != apNeighbors[7] && apNeighbors[6] && apNeighbors[7]) {
			apNeighbors[6]->apNeighbors[2] = apChildren[3]; // C3
			apNeighbors[6]->apNeighbors[3] = apChildren[3]; // C3
			apNeighbors[7]->apNeighbors[2] = apChildren[0]; // C0
			apNeighbors[7]->apNeighbors[3] = apChildren[0]; // C0
		}
		else if (apNeighbors[6] == apNeighbors[7] && apNeighbors[6]) {
			apNeighbors[6]->apNeighbors[2] = apChildren[0]; // C0
			apNeighbors[6]->apNeighbors[3] = apChildren[3]; // C3
		}
	}
}


void Element::Fuse(vector<vector<shared_ptr<Element>>>& vvpLevelElementList, int option, map<Coord, double>& mPhiCoordinateList, map<Coord, double>& mUCoordinateList, 
				  map<Coord, double>& mPhiVelocityCoordinateList, map<Coord, double>& mUVelocityCoordinateList, double gamma) {
	if (apChildren[0] && // If the element has children
		apChildren[0]->apChildren[0] == 0 &&
		apChildren[1]->apChildren[1] == 0 &&
		apChildren[2]->apChildren[2] == 0 &&
		apChildren[3]->apChildren[3] == 0 &&
		!apChildren[0]->CheckError(option, mPhiCoordinateList, mUCoordinateList, gamma) &&
		!apChildren[1]->CheckError(option, mPhiCoordinateList, mUCoordinateList, gamma) &&
		!apChildren[2]->CheckError(option, mPhiCoordinateList, mUCoordinateList, gamma) &&
		!apChildren[3]->CheckError(option, mPhiCoordinateList, mUCoordinateList, gamma) &&
		!CheckError(option, mPhiCoordinateList, mUCoordinateList, gamma) &&
		apChildren[0]->apNeighbors[6] == apChildren[0]->apNeighbors[7] &&
		apChildren[0]->apNeighbors[0] == apChildren[0]->apNeighbors[1] &&
		apChildren[1]->apNeighbors[0] == apChildren[1]->apNeighbors[1] &&
		apChildren[1]->apNeighbors[2] == apChildren[1]->apNeighbors[3] &&
		apChildren[2]->apNeighbors[2] == apChildren[2]->apNeighbors[3] &&
		apChildren[2]->apNeighbors[4] == apChildren[2]->apNeighbors[5] &&
		apChildren[3]->apNeighbors[4] == apChildren[3]->apNeighbors[5] &&
		apChildren[3]->apNeighbors[6] == apChildren[3]->apNeighbors[7]   ) {

		// Reset neithbors
		apNeighbors[0] = apChildren[0]->apNeighbors[0];
		apNeighbors[1] = apChildren[1]->apNeighbors[0];
		apNeighbors[2] = apChildren[1]->apNeighbors[2];
		apNeighbors[3] = apChildren[2]->apNeighbors[2];
		apNeighbors[4] = apChildren[2]->apNeighbors[4];
		apNeighbors[5] = apChildren[3]->apNeighbors[4];
		apNeighbors[6] = apChildren[3]->apNeighbors[6];
		apNeighbors[7] = apChildren[0]->apNeighbors[6];
		// Children are marked to be deleted
		for (unsigned i = 0; i < 4; i++) {
			apChildren[i]->Clean();
			apChildren[i].reset();
		}
		
		// Greet neighbors ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// apNeighbors[0] & apNeighbors[1]
		if (apNeighbors[0] != apNeighbors[1] && apNeighbors[0] && apNeighbors[1]) {
			apNeighbors[0]->apNeighbors[4] = pMe; // P
			apNeighbors[0]->apNeighbors[5] = pMe; // P
			apNeighbors[1]->apNeighbors[4] = pMe; // P
			apNeighbors[1]->apNeighbors[5] = pMe; // P
		}
		else if (apNeighbors[0] == apNeighbors[1] && apNeighbors[0]) {
			apNeighbors[0]->apNeighbors[4] = pMe; // P
			apNeighbors[0]->apNeighbors[5] = pMe; // P
		}

		// apNeighbors[2] & apNeighbors[3]
		if (apNeighbors[2] != apNeighbors[3] && apNeighbors[2] && apNeighbors[3]) {
			apNeighbors[2]->apNeighbors[6] = pMe; // P
			apNeighbors[2]->apNeighbors[7] = pMe; // P
			apNeighbors[3]->apNeighbors[6] = pMe; // P
			apNeighbors[3]->apNeighbors[7] = pMe; // P
		}
		else if (apNeighbors[2] == apNeighbors[3] && apNeighbors[2]) {
			apNeighbors[2]->apNeighbors[6] = pMe; // P
			apNeighbors[2]->apNeighbors[7] = pMe; // P
		}

		// apNeighbors[4] & apNeighbors[5]
		if (apNeighbors[4] != apNeighbors[5] && apNeighbors[4] && apNeighbors[5]) {
			apNeighbors[4]->apNeighbors[0] = pMe; // P
			apNeighbors[4]->apNeighbors[1] = pMe; // P
			apNeighbors[5]->apNeighbors[0] = pMe; // P
			apNeighbors[5]->apNeighbors[1] = pMe; // P
		}
		else if (apNeighbors[4] == apNeighbors[5] && apNeighbors[4]) {
			apNeighbors[4]->apNeighbors[0] = pMe; // P
			apNeighbors[4]->apNeighbors[1] = pMe; // P
		}

		// apNeighbors[6] & apNeighbors[7]
		if (apNeighbors[6] != apNeighbors[7] && apNeighbors[6] && apNeighbors[7]) {
			apNeighbors[6]->apNeighbors[2] = pMe; // P
			apNeighbors[6]->apNeighbors[3] = pMe; // P
			apNeighbors[7]->apNeighbors[2] = pMe; // P
			apNeighbors[7]->apNeighbors[3] = pMe; // P
		}
		else if (apNeighbors[6] == apNeighbors[7] && apNeighbors[6]) {
			apNeighbors[6]->apNeighbors[2] = pMe; // P
			apNeighbors[6]->apNeighbors[3] = pMe; // P
		}
	}
}


void Element::AddNodes(map<Coord, unsigned>& mNodeCoordinateList) {

	// Initialize bitElementType
	bitElementType.set();
	for (unsigned i = 0; i < 4; i++)
		if (apNeighbors[2 * i] == apNeighbors[2 * i + 1])
			bitElementType[i + 4] = false;

	// Assign nodal coordinates
	for (unsigned i = 0; i < 8; i++) {
		if (bitElementType.test(i)) {
			switch (i)
			{
			case 4:
				acNodalCoordinates[4].set((acNodalCoordinates[0].x + acNodalCoordinates[1].x) / 2, acNodalCoordinates[0].y);
				break;
			case 5:
				acNodalCoordinates[5].set(acNodalCoordinates[1].x, (acNodalCoordinates[1].y + acNodalCoordinates[2].y) / 2);
				break;
			case 6:
				acNodalCoordinates[6].set((acNodalCoordinates[2].x + acNodalCoordinates[3].x) / 2, acNodalCoordinates[2].y);
				break;
			case 7:
				acNodalCoordinates[7].set(acNodalCoordinates[0].x, (acNodalCoordinates[0].y + acNodalCoordinates[3].y) / 2);
				break;
			default: // i = 0 or 1 ro 2 or 3
				break;
			}
			if (mNodeCoordinateList.count(acNodalCoordinates[i]) == 0) {
				size_t uNode = mNodeCoordinateList.size();
				mNodeCoordinateList[acNodalCoordinates[i]] = uNode;
			}
		}
	}
}


bool Element::CheckNeighbors(unsigned maxLv, vector<vector<shared_ptr<Element>>>& vvpLevelElementList,
							 map<Coord, double>& mPhiCoordinateList, map<Coord, double>& mUCoordinateList,
							 map<Coord, double>& mPhiVelocityCoordinateList, map<Coord, double>& mUVelocityCoordinateList) {
	for (unsigned i = 0; i < apNeighbors.size(); i++)
		if (apNeighbors[i])
			if (uLevel > apNeighbors[i]->uLevel)
				apNeighbors[i]->Subdivide(maxLv, vvpLevelElementList, mPhiCoordinateList, mUCoordinateList, mPhiVelocityCoordinateList, mUVelocityCoordinateList);
	return true;
}

bool Element::CheckError(int option, map<Coord, double>& mPhiCoordinateList, map<Coord, double>& mUCoordinateList, double gamma) {

	double x_mid = mPhiCoordinateList.crbegin()->first.x / 2.0;
	double y_mid = mPhiCoordinateList.crbegin()->first.y / 2.0;
	MatrixXd elementNodesCoord(4, 2);
	double f[4];
	VectorXd psi(4);
	MatrixXd dN;
	MatrixXd B;
	MatrixXd cotangent;
	double DERX;
	double DERY;

	switch (option) // For phase field
	{
	case 10: // Mesh refinement
		for (unsigned i = 0; i < 4; i++) {
			elementNodesCoord(i, 0) = acNodalCoordinates[i].x;
			elementNodesCoord(i, 1) = acNodalCoordinates[i].y;
			if (mPhiCoordinateList.count(acNodalCoordinates[i]))
				psi[i] = mPhiCoordinateList[acNodalCoordinates[i]] + gamma * mUCoordinateList[acNodalCoordinates[i]];
			else
				cerr << "NO!!!!!!!!!!!!!!!!!!!!!" << endl;
		}
		dN = NaturalDerivatives(0, 0, bitset<8>("00001111")); // at mass center
		B = XYDerivatives(elementNodesCoord, dN);
		cotangent = B * psi; // 2x1
		DERX = cotangent(0);
		DERY = cotangent(1);
		return (sqrt(DERX*DERX + DERY*DERY) > 0.3 && acNodalCoordinates[0].x >= 0 && acNodalCoordinates[1].x <= 1638.4 || uLevel <= 1);
		break;
	case 11: // Initialization
		for (unsigned i = 0; i < 4; i++) {
			elementNodesCoord(i, 0) = acNodalCoordinates[i].x;
			elementNodesCoord(i, 1) = acNodalCoordinates[i].y;
			double rad = x_mid / 256;
			double seeds = 4;
			double dist = sqrt(pow(elementNodesCoord(i,0)-x_mid,2.0) + pow(elementNodesCoord(i,1)-y_mid,2.0)) - rad;
			psi[i] = -tanh(dist / sqrt(2));
		}
		dN = NaturalDerivatives(0, 0, bitset<8>("00001111")); // at mass center
		B = XYDerivatives(elementNodesCoord, dN);
		cotangent = B * psi; // 2x1
		DERX = cotangent(0);
		DERY = cotangent(1);
		return (sqrt(DERX*DERX + DERY*DERY) > 0.01 || uLevel <= 4);
	case 12: // Regular mesh
		return true;
	default:
		break;
	}

	for (unsigned i = 0; i < 4; i++) {
		double x, y;
		x = acNodalCoordinates[i].x;
		y = acNodalCoordinates[i].y;
		switch (option)
		{
		case 1: // 10 x 10
			f[i] = pow((x - 5), 4.0) + pow((y - 7.5), 4.0) + 100 / (pow((x - 5), 4.0) + pow((2 * (y - 7.5) - 2), 4.0) + pow((2 * (y - 7.5) - 1), 2.0)) + 100 / (pow((x - 5), 4.0) + pow((2 * (y - 7.5) + 2), 4.0) + pow((2 * (y - 7.5) + 1), 2.0)) - 1 / (pow(((y - 7.5) + 3), 4.0) + pow(((x - 5) / 15), 4.0)) - 1 / (pow(((y - 7.5) + 4), 4.0) + pow(((x - 5) / 15), 4.0)) - 1 / (pow(((y - 7.5) + 5), 4.0) + pow(((x - 5) / 15), 4.0)) - 1 / (4 * pow(((x - 5) + (y - 7.5) + 4), 4.0) + pow((((x - 5) - (y - 7.5) + 1) / 5), 4.0)) - pow(100, 16.0) / pow((pow((pow(((x - 5) - 4), 2.0) + pow(((y - 7.5) + 5), 2.0) - 13), 2.0) + pow((pow(((x - 5) - 19), 2.0) + pow(((y - 7.5) + 12), 2.0) - 400), 2.0)), 16.0) - 25;
			break;
		case 2: // 4 x 4
			f[i] = sin(x) - y + 2;
			break;
		case 3:
			f[i] = 1;
			break;
		case 4: // 4 x 4
			f[i] = pow((pow(x - 2, 2.0) + pow(y - 2, 2.0) - 1), 3.0) - pow(x - 2, 2.0)*pow(y - 2, 3.0);
			break;
		default:
			cerr << "ERROR!";
			break;
		}
	}

	if (f[0] * f[1] <= 0 || f[1] * f[2] <= 0 || f[2] * f[3] <= 0 || f[3] * f[0] <= 0 || uLevel <= 0)
		return true;

	return false;
}


void Element::Clean() {
	pMe.reset();
	pParent.reset();
	for (unsigned i = 0; i < apChildren.size(); i++)
		apChildren[i].reset();
	for (unsigned i = 0; i < apNeighbors.size(); i++)
		apNeighbors[i].reset();
}


shared_ptr<Element> CreateElement(array<Coord, 4> Bounding, shared_ptr<Element> Parent, unsigned uLevel, unsigned maxLv,
								  vector<vector<shared_ptr<Element>>>& vvpLevelElementList) {

	shared_ptr<Element>  pElement(new Element(Parent, uLevel, Bounding)); // create new element
	vvpLevelElementList[uLevel].push_back(pElement);
	pElement->pMe = pElement;

	return pElement;
}


void DeleteElement(vector<vector<shared_ptr<Element>>>& vvpLevelElementList) {
	
	vector<vector<shared_ptr<Element>>> NEWvvpLevelElementList(vvpLevelElementList.size());

	for (unsigned lv = 0; lv < vvpLevelElementList.size(); lv++) {
		for (unsigned e = 0; e < vvpLevelElementList[lv].size(); e++) {
			if (vvpLevelElementList[lv][e]->pMe != 0 && vvpLevelElementList[lv][e]->uLevel < vvpLevelElementList.size()) // Delete Elements with pME = 0
					NEWvvpLevelElementList[lv].push_back(vvpLevelElementList[lv][e]);
			else 
				vvpLevelElementList[lv][e].reset();
		}
	}
	vvpLevelElementList = NEWvvpLevelElementList;
}


void ReportElement(const vector<vector<shared_ptr<Element>>>& vvpLevelElementList, 
				   vector<shared_ptr<Element>>& vpFinalElementList,
				   map<Coord, unsigned>& mNodeCoordinateList,
				   vector<vector<int>>& vvEFT,
				   vector<Coord>& vcNodeCoordinates) {

	// Initialization
	vpFinalElementList.clear();
	vvEFT.clear();
	vcNodeCoordinates.clear();

	// Output element boundary for plotting
	for (unsigned lv = 0; lv < vvpLevelElementList.size(); lv++) {
		for (unsigned e = 0; e < vvpLevelElementList[lv].size(); e++) {
			if (vvpLevelElementList[lv][e]->apChildren[0] == 0 && vvpLevelElementList[lv][e]->acNodalCoordinates[0].x >= 0 && vvpLevelElementList[lv][e]->acNodalCoordinates[1].x <= 1638.4) {
				vpFinalElementList.push_back(vvpLevelElementList[lv][e]);
			}
		}
	}

	// Output the global node coordinate list
	map<unsigned, Coord> mNode2Coord;
	for (const auto cnNodeCoordinate : mNodeCoordinateList)
		mNode2Coord[cnNodeCoordinate.second] = cnNodeCoordinate.first; // reverse key and value
	for (const auto ncNodeCoordinate : mNode2Coord) {
		vcNodeCoordinates.push_back(Coord(ncNodeCoordinate.second.x, ncNodeCoordinate.second.y));
	}

	// Output the element freedom table
	for (const auto pFinalElement : vpFinalElementList) {
		vector<int> viElementNode;
		for (unsigned i = 0; i < 8; i++) {
			if (pFinalElement->bitElementType.test(i)) {
				viElementNode.push_back(mNodeCoordinateList[pFinalElement->acNodalCoordinates[i]]);
			}
		}
		vvEFT.push_back(viElementNode);
	}
}
