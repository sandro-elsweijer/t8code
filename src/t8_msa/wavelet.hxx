/**
 * @file   wavelet.hpp
 * @author Nils Gerhard <gerhard@igpm.rwth-aachen.de>
 * @date   Tue Jun 4 2013
 *
 * @brief L2-Projection and Multiscale transformation on Triangles
 *
 *
 */

#ifndef __WAVELET_INCLUDE__
#define __WAVELET_INCLUDE__

// siehe http://www.cplusplus.com/reference/map/map/
#include <map>
#include <t8.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#define _USE_MATH_DEFINES
#include <cmath>
#include <gsl/gsl_math.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include "vecmat.hxx"
#include "dunavant.hxx"
//T8_EXTERN_C_BEGIN ();
using namespace std;

typedef double (*func) (double, double);
typedef double (*spline) (const gsl_spline2d *spline, const double x, const double y, gsl_interp_accel *xacc,
                          gsl_interp_accel *yacc);

// punkt in R^2
struct node
{
  double x;
  double y;
};

// key in der hashmap, Nummerierung der Zellen
typedef long int Key;

// Container für benötigte Daten auf einer Zelle
struct CellData
{
  // zum testen und überprüfen
  int level;

  // Eckpunkte des Dreiecks:
  vector<node> nodes;

  // informationen über Vaterzelle
  bool hasFather;
  Key Fkey;

  // informationen über Kinder
  bool hasChilds;
  vector<Key> childs;

  // Single- und Multiskalenkoeffizienten
  vector<double> u_coeff;
  vector<double> d_coeff;

  double flaeche;
  // LR-Zerlegung fuer Transformation von kartesischen zu baryzentrischen Koordinaten
  mat A;
  vector<int> r;
  bool adaptiert;
};

typedef map<Key, CellData> GridMap;
typedef vector<GridMap> GlobalGrid;

ostream &
operator<< (ostream &lhs, node &OUt)
{
  lhs << "(" << OUt.x << "," << OUt.y << ")";
  return lhs;
}

ostream &
operator<< (ostream &lhs, CellData &OUt)
{
  lhs << endl;

  cout << "Data:" << endl;
  cout << "level: " << OUt.level << endl << endl;
  ;
  for (unsigned int i = 0; i < OUt.nodes.size (); i++)
    cout << "nodes[" << i << "] = " << OUt.nodes[i] << endl;
  cout << endl << "hasFather: " << OUt.hasFather << endl;
  cout << "Fkey: " << OUt.Fkey << endl;
  cout << "hasChilds: " << OUt.hasChilds << endl << endl;
  ;
  for (unsigned int i = 0; i < OUt.childs.size (); i++)
    cout << "childs[" << i << "] = " << OUt.childs[i] << endl;
  cout << "Coefficients: " << endl;
  for (unsigned int i = 0; i < OUt.u_coeff.size (); i++)
    cout << "u_coeff[" << i << "]= " << OUt.u_coeff[i] << endl;
  for (unsigned int i = 0; i < OUt.d_coeff.size (); i++)
    cout << "u_coeff[" << i << "]= " << OUt.d_coeff[i] << endl;

  return lhs;
}

ostream &
operator<< (ostream &lhs, GlobalGrid &rhs)
{
  lhs << endl;
  for (unsigned int i = 0; i < rhs.size (); i++) {
    cout << "Cells on Level " << i << ":\n";
    for (GridMap::iterator it = (rhs[i]).begin (); it != (rhs[i]).end (); ++it) {
      cout << "Index: " << it->first << endl;
      // cout << it->second << endl;
      cout << "\t points: \n";
      cout << "\t \t no 1: (" << it->second.nodes[0].x << " , " << it->second.nodes[0].y << ")" << endl;
      cout << "\t \t no 2: (" << it->second.nodes[1].x << " , " << it->second.nodes[1].y << ")" << endl;
      cout << "\t \t no 3: (" << it->second.nodes[2].x << " , " << it->second.nodes[2].y << ")" << endl;
    }
  }

  return lhs;
}

//T8_EXTERN_C_END ();
#endif
