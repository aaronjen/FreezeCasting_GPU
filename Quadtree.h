#ifndef QUADTREE_H
#define QUADTREE_H

#include <iostream>
#include <string>
#include <array>
#include <vector>
#include <map>
#include <memory>
#include <bitset>

class Coord {
public:
	Coord() = default;
	Coord(double xCoord, double yCoord) : x(xCoord), y(yCoord) {};
	void set(double xCoord, double yCoord) { x = xCoord; y = yCoord; }
	bool operator< (const Coord c) const {
		if (int(x*1e6) != int(c.x*1e6))
			return x < c.x;
		else if (int(y*1e6) != int(c.y*1e6))
			return y < c.y;
		else 
			return false;
/*
 		if (abs(x-c.x) > 1e-16)
 			return x < c.x;
 		else if (abs(y-c.y) > 1e-16)
 			return y < c.y;
 		else
 			return false;
*/
	}
	bool operator== (const Coord c) const {
		return (x == c.x && y == c.y);
	}
	double x = -1;
	double y = -1;
};

class Element {
public:
	Element() = default;
	Element(std::shared_ptr<Element> Parent, unsigned Level, std::array<Coord, 4> Bounding) :
		pParent(Parent), uLevel(Level) {for (unsigned i = 0; i < 4; i++) acNodalCoordinates[i] = Bounding[i];};

	unsigned								uLevel;
	std::array<Coord, 8>					acNodalCoordinates;
	std::bitset<8>							bitElementType; 
	std::shared_ptr<Element>				pMe;
	std::shared_ptr<Element>				pParent;
	std::array<std::shared_ptr<Element>, 4>	apChildren;
	std::array<std::shared_ptr<Element>, 8>	apNeighbors;
	double									tolerance;

	void		Subdivide(unsigned maxLv, std::vector<std::vector<std::shared_ptr<Element>>>& vvpLevelElementList,
					  std::map<Coord, double>& mPhiCoordinateList, std::map<Coord, double>& mUCoordinateList,
					  std::map<Coord, double>& mPhiVelocityCoordinateList, std::map<Coord, double>& mUVelocityCoordinateList);
	void		Fuse(std::vector<std::vector<std::shared_ptr<Element>>>& vvpLevelElementList, int option, std::map<Coord, double>& mPhiCoordinateList, std::map<Coord, double>& mUCoordinateList, 
				 std::map<Coord, double>& mPhiVelocityCoordinateList, std::map<Coord, double>& mUVelocityCoordinateList, double gamma);
	void		AddNodes(std::map<Coord, unsigned>& mNodeCoordinateList);
	bool	CheckNeighbors(unsigned maxLv, std::vector<std::vector<std::shared_ptr<Element>>>& vvpLevelElementList,
						   std::map<Coord, double>& mPhiCoordinateList, std::map<Coord, double>& mUCoordinateList,
						   std::map<Coord, double>& mPhiVelocityCoordinateList, std::map<Coord, double>& mUVelocityCoordinateList);
	bool	CheckError(int option, std::map<Coord, double>& mPhiCoordinateList, std::map<Coord, double>& mUCoordinateList, double gamma);
	void		Clean();

};

void Quadtree_Initialization(double Nx, double Ny, unsigned maxLv, double gamma,
							std::vector<std::vector<std::shared_ptr<Element>>>& vvpLevelElementList, 
							std::map<Coord, unsigned>& mNodeCoordinateList,
							std::vector<std::vector<int>>& vvEFT,
							int option, std::map<Coord, double>& mPhiCoordinateList, std::map<Coord, double>& mUCoordinateList,
							std::map<Coord, double>& mPhiVelocityCoordinateList, std::map<Coord, double>& mUVelocityCoordinateList);

void Quadtree_MeshGenerate(unsigned maxLv, double gamma, std::vector<std::vector<std::shared_ptr<Element>>>& vvpLevelElementList, int option, std::map<Coord, double>& mPhiCoordinateList, std::map<Coord, double>& mUCoordinateList,
						  std::map<Coord, double>& mPhiVelocityCoordinateList, std::map<Coord, double>& mUVelocityCoordinateList);

void Quadtree_ConstructRoot(double Nx, double Ny, unsigned maxLv,
						   std::vector<std::vector<std::shared_ptr<Element>>>& vvpLevelElementList);

void Quadtree_Subdivide(unsigned maxLv, double gamma, std::vector<std::vector<std::shared_ptr<Element>>>& vvpLevelElementList, int option, std::map<Coord, double>& mPhiCoordinateList, std::map<Coord, double>& mUCoordinateList,
					   std::map<Coord, double>& mPhiVelocityCoordinateList, std::map<Coord, double>& mUVelocityCoordinateList);

void Quadtree_Fuse(unsigned maxLv, double gamma, std::vector<std::vector<std::shared_ptr<Element>>>& vvpLevelElementList, int option, std::map<Coord, double>& mPhiCoordinateList, std::map<Coord, double>& mUCoordinateList, 
				  std::map<Coord, double>& mPhiVelocityCoordinateList, std::map<Coord, double>& mUVelocityCoordinateList);

void Quadtree_AddNodes(std::vector<std::vector<std::shared_ptr<Element>>>& vvpLevelElementList, std::map<Coord, unsigned>& mNodeCoordinateList);

std::shared_ptr<Element> CreateElement(std::array<Coord, 4> Bounding, std::shared_ptr<Element> Parent, unsigned uLevel, unsigned maxLv,
									   std::vector<std::vector<std::shared_ptr<Element>>>& vvpLevelElementList);

void DeleteElement(std::vector<std::vector<std::shared_ptr<Element>>>& vvpLevelElementList);

void ReportElement(const std::vector<std::vector<std::shared_ptr<Element>>>& vvpLevelElementList, 
				   std::vector<std::shared_ptr<Element>>& vpFinalElementList,
				   std::map<Coord, unsigned>& mNodeCoordinateList,
				   std::vector<std::vector<int>>& vvEFT,
				   std::vector<Coord>& vcNodeCoordinates);

#endif // QUADTREE_H
