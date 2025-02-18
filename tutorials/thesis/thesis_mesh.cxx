#include <t8.h>                          /* General t8code header, always include this. */
#include <t8_cmesh.hxx>                  /* cmesh definition and basic interface. */
#include <t8_forest/t8_forest_general.h> /* forest definition and basic interface. */
#include <t8_vtk/t8_vtk_writer.h>
#include <t8_geometry/t8_geometry_implementations/t8_geometry_linear.hxx> /* linear geometry of the cmesh */
#include <t8_forest/t8_forest_io.h>                                       /* forest io interface. */
#include <t8_schemes/t8_default/t8_default.hxx>                           /* default refinement scheme. */
#include <t8_forest/t8_forest_geometrical.h>                              /* geometrical information */
#include <t8_msa/vecmat.hxx>
#include <t8_msa/wavelet.hxx>
#include <t8_msa/basis_functions.hxx>
#include <t8_msa/mask_coefficients.hxx>
#include <cmath>
#include <vector>
#include <t8_vec.h> /* Basic operations on 3D vectors. */
#include <sc_statistics.h>
#include <t8_refcount.h>
#include <t8_forest/t8_forest_general.h>
#include <t8_forest/t8_forest_profiling.h>
#include <t8_forest/t8_forest_private.h>
#include <t8_forest/t8_forest_types.h>
#include <t8_forest/t8_forest_partition.h>
#include <t8_forest/t8_forest_ghost.h>
#include <t8_forest/t8_forest_adapt.h>
#include <t8_forest/t8_forest_balance.h>
#include <t8_cmesh/t8_cmesh_offset.h>
#include <t8_cmesh/t8_cmesh_trees.h>
//#include <t8_element_c_interface.h>
#include <iostream>
#include <t8_eclass.h>
#include <t8_cmesh/t8_cmesh_examples.h>
#include <t8_forest/t8_forest_iterate.h>
#include <t8_schemes/t8_default/t8_default_tri/t8_dtri.h>
#include <t8_schemes/t8_default/t8_default_tri/t8_dtri_bits.h>
#include <t8_schemes/t8_default/t8_default_tri/t8_dtri_connectivity.h>
#include <t8_schemes/t8_default/t8_default_tri/t8_default_tri.hxx>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_spline2d.h>
#include <fstream>
#include <time.h>

// // Declare variables to store the start and end times
//     clock_t start, end;
//     double cpu_time_used;
//
//     // Measure the time for the first computation
//     start = clock(); // Start the clock
//     // Example computation 1: Sum of numbers from 1 to 1 million
//     long sum = 0;
//     for (long i = 1; i <= 1000000; i++) {
//         sum += i;
//     }
//     end = clock(); // End the clock
//     cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC; // Compute the time in seconds
//     printf("Time for first computation: %f seconds\n", cpu_time_used);
//
//     // Measure the time for the second computation
//     start = clock(); // Start the clock again
//     // Example computation 2: Another computation (e.g., product of numbers)
//     long product = 1;
//     for (long i = 1; i <= 100000; i++) {
//         product *= i;
//     }
//     end = clock(); // End the clock
//     cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC; // Compute the time in seconds
//     printf("Time for second computation: %f seconds\n", cpu_time_used);

/* TO DO:
 * - Check if all headers still necessary after changes.
 * - change lmi s.t. it works with binary algorithm(faster runtime, level-independent)
 * - redundant data, functions, variables need to be deleted
 * - vgl Neuberechnung ordnung Punkte vs Speichern Laufzeit und Speicher
 * - test if MultiScaleOperator and InverseMultiScaleOperator work: done
 * - plot L2 error
 * - prediction function schreiben
 * - Speicherfehler am Ende loswerden
 * - Matrixklasse für schnellere Tauschen wenn das irgendwann relevant wird
 */
using namespace std;
typedef int8_t t8_dtri_cube_id_t;

/* defines the maximum refinement level (should currently be less than approx. 10 depending on how many basecells there are) */
const int max_level = 4;

/* defines the polynomial degree, for a given p, we use polynomials up to degree p-1, maximal value is 4*/
const int p = 4;

/* M defines the local degrees of freedom of the single-scale spaces, which is just the dim(space of Polynomials up to degree p-1)
 * For d=2 here and given p, this is equal to (p+d-1) choose (d) = (p+1) choose (2) =(p+1)!/((2!)*(p-1)!)=(p+1)p/2 */
const int M = (p * (p + 1)) / 2;

/* declaring the matrices */
mat M0, M1, M2, M3, N0, N1, N2, N3;

/* Arrays only needed for non-binary implementation of level-multiindex lmi */
/* storing the powers of 10 and 4 so we have a lookup table, if more than 13 levels to be used, we need to add more powers */

// Precompute powers of 4 up to level 29(T8_DTRI_MAXLEVEL)
#define MAX_LEVEL T8_DTRI_MAXLEVEL
//BIT Length for bitwise level multi index
const int PATH_BITS = 2;       // Each path segment is 2 bits
const int LEVEL_BITS = 5;      // Level is encoded in 5 bits
const int BASECELL_BITS = 21;  // Basecell is encoded in 21 bits
uint64_t pow4[MAX_LEVEL + 1];

// Function to initialize the powers of 4
void
initialize_pow4 ()
{
  pow4[0] = 1;
  for (int i = 1; i <= MAX_LEVEL; ++i) {
    pow4[i] = pow4[i - 1] * 4;
  }
}

// static long long int pow10[14] = {
//       1, 10, 100, 1000, 10000,
//       100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000,1000000000000,10000000000000
//   };

// static long long int pow4[14] = {
//         1, 4, 16, 64, 256,
//         1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864
//     };

/* To store the data for the 3d case (e.g. in AuswertungSinglescale) */
struct double_3d_array
{
  double dim_val[3];
};

/* This struct stores the level multi index:
 * Each lmi is of the form: (path, level, basecell)
 * path: consists of the path how to get to an element, say: start with basecell a, go to its third child, and from there to its first child
 * this corresponds to  02a (from right to left, and the children are indexed from 0 to 3)
 * level = level of the given elements
 * basecell = index of the cmesh element that contains the current element
 */
// struct lmi
// {
// 	long long int lmi_arr[3];
// };

/*
 * Stores the t8_locidx_t of the children of an element
 */
struct children
{
  t8_locidx_t child_arr[4];
};

/* We can drop the volume of the element and the level, hasFather, haschilds, childs ids, father ids*/
/* This struct stores the element data */
struct t8_data_per_element
{
  double u_coeff[M];      //single-scale coefficients for all dof/ basis polynomials
  double d_coeff[3 * M];  //difference coefficients
  bool adaptiert;         //whether an element is adapted or not
  bool signifikant;       // whether an element is significant or not
  t8_locidx_t Father_id;  //what is the t8_locidx_t of the Father then?
  uint64_t lmi;
  struct children child_ids;  //what are the t8_locidx_t of the children
  double
    u_val;  // This is for plotting purposes: the 0th u_coeff corresponds to the average over a cell because it is constant
  unsigned int first : 2;
  unsigned int second : 2;
  unsigned int third : 2;
};

/* This struct stores the element data */
struct t8_data_per_element_3d
{
  double u_coeff_d1[M];  //single-scale coefficients for all dof/ basis polynomials
  double u_coeff_d2[M];
  double u_coeff_d3[M];
  double d_coeff_d1[3 * M];  //difference coefficients
  double d_coeff_d2[3 * M];
  double d_coeff_d3[3 * M];
  bool adaptiert;         //whether an element is adapted or not
  bool signifikant;       // whether an element is significant or not
  t8_locidx_t Father_id;  //what is the t8_locidx_t of the Father then?
  uint64_t lmi;
  struct children child_ids;  //what are the t8_locidx_t of the children
  double
    u_val_d1;  // This is for plotting purposes: the 0th u_coeff corresponds to the average over a cell because it is constant
  double u_val_d2;
  double u_val_d3;
  unsigned int first : 2;
  unsigned int second : 2;
  unsigned int third : 2;
};

/* This struct stores the element data */
struct t8_data_per_element_waveletfree
{
  double u_coeff[M];  //single-scale coefficients for all dof/ basis polynomials
  bool adaptiert;     //whether an element is adapted or not
  bool signifikant;   // whether an element is significant or not
  double d_coeff_wavelet_free
    [M][4];  //we need to store in case of the wavelet free approach more coefficients for the difference information
  t8_locidx_t Father_id;  //what is the t8_locidx_t of the Father then?
  uint64_t lmi;
  struct children child_ids;  //what are the t8_locidx_t of the children
  double
    u_val;  // This is for plotting purposes: the 0th u_coeff corresponds to the average over a cell because it is constant
  unsigned int first : 2;
  unsigned int second : 2;
  unsigned int third : 2;
};

/* This struct stores the element data */
struct t8_data_per_element_waveletfree_3d
{
  double u_coeff_d1[M];  //single-scale coefficients for all dof/ basis polynomials
  double u_coeff_d2[M];
  double u_coeff_d3[M];
  bool adaptiert;    //whether an element is adapted or not
  bool signifikant;  // whether an element is significant or not
  double d_coeff_wavelet_free_d1
    [M][4];  //we need to store in case of the wavelet free approach more coefficients for the difference information
  double d_coeff_wavelet_free_d2[M][4];
  double d_coeff_wavelet_free_d3[M][4];
  t8_locidx_t Father_id;  //what is the t8_locidx_t of the Father then?
  uint64_t lmi;
  struct children child_ids;  //what are the t8_locidx_t of the children
  double
    u_val_d1;  // This is for plotting purposes: the 0th u_coeff corresponds to the average over a cell because it is constant
  double u_val_d2;
  double u_val_d3;
  unsigned int first : 2;
  unsigned int second : 2;
  unsigned int third : 2;
};

/* for the adaptation process we need to compare the elements in our forest with the elements in the grid hierarchy, thus we need to store the level multi indices */
struct t8_data_per_element_adapt
{
  //struct lmi lmi;
  uint64_t lmi;
};

/* for each level of our grid hierarchy, we store the forest and the element data */
struct grid_level
{
  t8_forest_t forest_arr;
  struct t8_data_per_element *data_arr;
};

/* for each level of our grid hierarchy, we store the forest and the element data */
struct grid_level_3d
{
  t8_forest_t forest_arr;
  struct t8_data_per_element_3d *data_arr;
};

/* for each level of our grid hierarchy, we store the forest and the element data */
struct grid_level_waveletfree
{
  t8_forest_t forest_arr;
  struct t8_data_per_element_waveletfree *data_arr;
};

/* for each level of our grid hierarchy, we store the forest and the element data */
struct grid_level_waveletfree_3d
{
  t8_forest_t forest_arr;
  struct t8_data_per_element_waveletfree_3d *data_arr;
};

/* then we store the grid levels in a grid hierarchy */
struct grid_hierarchy
{
  struct grid_level lev_arr[max_level + 1];
};

/* then we store the grid levels in a grid hierarchy */
struct grid_hierarchy_3d
{
  struct grid_level_3d lev_arr[max_level + 1];
};

/* then we store the grid levels in a grid hierarchy */
struct grid_hierarchy_waveletfree
{
  struct grid_level_waveletfree lev_arr[max_level + 1];
};

/* then we store the grid levels in a grid hierarchy */
struct grid_hierarchy_waveletfree_3d
{
  struct grid_level_waveletfree_3d lev_arr[max_level + 1];
};

/* we need to hand over this data to the grid adaptation call, so that we have both the data of the ref grid and the grid to be adapted */
struct lmi_adapt
{
  struct t8_data_per_element *ref_grid_data;
  struct t8_data_per_element_adapt *adapt_lmi_data;
};

/* we need to hand over this data to the grid adaptation call, so that we have both the data of the ref grid and the grid to be adapted */
struct lmi_adapt_3d
{
  struct t8_data_per_element_3d *ref_grid_data;
  struct t8_data_per_element_adapt *adapt_lmi_data;
};

/* we need to hand over this data to the grid adaptation call, so that we have both the data of the ref grid and the grid to be adapted */
struct lmi_adapt_waveletfree
{
  struct t8_data_per_element_waveletfree *ref_grid_data;
  struct t8_data_per_element_adapt *adapt_lmi_data;
};

/* we need to hand over this data to the grid adaptation call, so that we have both the data of the ref grid and the grid to be adapted */
struct lmi_adapt_waveletfree_3d
{
  struct t8_data_per_element_waveletfree_3d *ref_grid_data;
  struct t8_data_per_element_adapt *adapt_lmi_data;
};

/* We build 4 kinds of cmeshes in the following by hand. */

/* This cmesh is [0,1]² with 8 triangles. */
t8_cmesh_t
t8_cmesh_new_basic (sc_MPI_Comm comm)
{

  /* 1. Defining an array with all vertices */
  /* Just all vertices of all trees. partly duplicated */
  double vertices[72] = {
    0,   0,   0, 0.5, 0,   0, 0.5, 0.5, 0,  //triangle 2
    0,   0,   0, 0,   0.5, 0, 0.5, 0.5, 0,  //triangle 1
    0.5, 0,   0, 1,   0,   0, 1,   0.5, 0,  //triangle 3
    0.5, 0,   0, 0.5, 0.5, 0, 1,   0.5, 0,  //triangle 4
    0,   0.5, 0, 0.5, 0.5, 0, 0.5, 1,   0,  //triangle 5
    0,   0.5, 0, 0,   1,   0, 0.5, 1,   0,  //triangle 6
    0.5, 0.5, 0, 1,   0.5, 0, 1,   1,   0,  //triangle 7
    0.5, 0.5, 0, 0.5, 1,   0, 1,   1,   0,  //triangle 8
  };

  /* 2. Initialization of the mesh */
  t8_cmesh_t cmesh;
  t8_cmesh_init (&cmesh);

  /* 3. Definition of the geometry */
  t8_cmesh_register_geometry<t8_geometry_linear> (cmesh);
  ; /* Use linear geometry */

  /* 4. Definition of the classes of the different trees */
  t8_cmesh_set_tree_class (cmesh, 0, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 1, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 2, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 3, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 4, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 5, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 6, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 7, T8_ECLASS_TRIANGLE);

  /* 5. Classification of the vertices for each tree */
  t8_cmesh_set_tree_vertices (cmesh, 0, vertices, 3);
  t8_cmesh_set_tree_vertices (cmesh, 1, vertices + 9, 3);
  t8_cmesh_set_tree_vertices (cmesh, 2, vertices + 18, 3);
  t8_cmesh_set_tree_vertices (cmesh, 3, vertices + 27, 3);
  t8_cmesh_set_tree_vertices (cmesh, 4, vertices + 36, 3);
  t8_cmesh_set_tree_vertices (cmesh, 5, vertices + 45, 3);
  t8_cmesh_set_tree_vertices (cmesh, 6, vertices + 54, 3);
  t8_cmesh_set_tree_vertices (cmesh, 7, vertices + 63, 3);

  /* 6. Definition of the face neighbors between the different trees */
  t8_cmesh_set_join (cmesh, 0, 1, 1, 1, 0);
  t8_cmesh_set_join (cmesh, 0, 3, 0, 2, 0);
  t8_cmesh_set_join (cmesh, 1, 4, 0, 2, 0);
  t8_cmesh_set_join (cmesh, 2, 3, 1, 1, 0);
  t8_cmesh_set_join (cmesh, 3, 6, 0, 2, 0);
  t8_cmesh_set_join (cmesh, 4, 5, 1, 1, 0);
  t8_cmesh_set_join (cmesh, 4, 7, 0, 2, 0);
  t8_cmesh_set_join (cmesh, 6, 7, 1, 1, 0);

  /* 7. Commit the mesh */
  t8_cmesh_commit (cmesh, comm);

  return cmesh;
}

/* This cmesh is [0,1]² with 2 triangles. for debugging purposes to compare to results from Florian Sieglar */
t8_cmesh_t
t8_cmesh_new_debugging (sc_MPI_Comm comm)
{

  /* 1. Defining an array with all vertices */
  /* Just all vertices of all trees. partly duplicated */
  double vertices[18] = {
    0, 0, 0, 0, 1, 0, 1, 0, 0,  //triangle 1
    1, 1, 0, 1, 0, 0, 0, 1, 0,  //triangle 2
  };

  /* 2. Initialization of the mesh */
  t8_cmesh_t cmesh;
  t8_cmesh_init (&cmesh);

  /* 3. Definition of the geometry */
  t8_cmesh_register_geometry<t8_geometry_linear> (cmesh);
  ; /* Use linear geometry */

  /* 4. Definition of the classes of the different trees */
  t8_cmesh_set_tree_class (cmesh, 0, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 1, T8_ECLASS_TRIANGLE);

  /* 5. Classification of the vertices for each tree */
  t8_cmesh_set_tree_vertices (cmesh, 0, vertices, 3);
  t8_cmesh_set_tree_vertices (cmesh, 1, vertices + 9, 3);

  /* 6. Definition of the face neighbors between the different trees */
  t8_cmesh_set_join (cmesh, 0, 1, 0, 0, 0);

  /* 7. Commit the mesh */
  t8_cmesh_commit (cmesh, comm);

  return cmesh;
}

/* This cmesh is an octagon with 14 elements */
t8_cmesh_t
t8_cmesh_new_octagon (sc_MPI_Comm comm)
{
  double a = 2.4142135623731;  //1+sqrt(2)
  /* 1. Defining an array with all vertices */
  /* Just all vertices of all trees. partly duplicated */
  double vertices[126] = {
    -1, -a, 0, -1, -1, 0, -a, -1, 0,  //triangle 1
    -1, -a, 0, 1,  -a, 0, 1,  -1, 0,  //triangle 2
    -1, -a, 0, -1, -1, 0, 1,  -1, 0,  //triangle 3
    1,  -a, 0, 1,  -1, 0, a,  -1, 0,  //triangle 4
    -a, -1, 0, -1, -1, 0, -1, 1,  0,  //triangle 5
    -a, -1, 0, -a, 1,  0, -1, 1,  0,  //triangle 6
    -1, -1, 0, 1,  -1, 0, 1,  1,  0,  //triangle 7
    -1, -1, 0, -1, 1,  0, 1,  1,  0,  //triangle 8
    1,  -1, 0, a,  -1, 0, 1,  1,  0,  //triangle 9
    a,  -1, 0, 1,  1,  0, a,  1,  0,  //triangle 10
    -a, 1,  0, -1, 1,  0, -1, a,  0,  //triangle 11
    -1, 1,  0, 1,  a,  0, 1,  1,  0,  //triangle 12
    -1, 1,  0, -1, a,  0, 1,  a,  0,  //triangle 13
    1,  a,  0, 1,  1,  0, a,  1,  0,  //triangle 14
  };

  /* 2. Initialization of the mesh */
  t8_cmesh_t cmesh;
  t8_cmesh_init (&cmesh);

  /* 3. Definition of the geometry */
  t8_cmesh_register_geometry<t8_geometry_linear> (cmesh);
  ; /* Use linear geometry */

  /* 4. Definition of the classes of the different trees */
  t8_cmesh_set_tree_class (cmesh, 0, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 1, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 2, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 3, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 4, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 5, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 6, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 7, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 8, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 9, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 10, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 11, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 12, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 13, T8_ECLASS_TRIANGLE);

  /* 5. Classification of the vertices for each tree */
  t8_cmesh_set_tree_vertices (cmesh, 0, vertices, 3);
  t8_cmesh_set_tree_vertices (cmesh, 1, vertices + 9, 3);
  t8_cmesh_set_tree_vertices (cmesh, 2, vertices + 18, 3);
  t8_cmesh_set_tree_vertices (cmesh, 3, vertices + 27, 3);
  t8_cmesh_set_tree_vertices (cmesh, 4, vertices + 36, 3);
  t8_cmesh_set_tree_vertices (cmesh, 5, vertices + 45, 3);
  t8_cmesh_set_tree_vertices (cmesh, 6, vertices + 54, 3);
  t8_cmesh_set_tree_vertices (cmesh, 7, vertices + 63, 3);
  t8_cmesh_set_tree_vertices (cmesh, 8, vertices + 72, 3);
  t8_cmesh_set_tree_vertices (cmesh, 9, vertices + 81, 3);
  t8_cmesh_set_tree_vertices (cmesh, 10, vertices + 90, 3);
  t8_cmesh_set_tree_vertices (cmesh, 11, vertices + 99, 3);
  t8_cmesh_set_tree_vertices (cmesh, 12, vertices + 108, 3);
  t8_cmesh_set_tree_vertices (cmesh, 13, vertices + 117, 3);

  /* 6. Definition of the face neighbors between the different trees */
  t8_cmesh_set_join (cmesh, 0, 2, 2, 2, 0);
  t8_cmesh_set_join (cmesh, 0, 4, 0, 2, 1);
  t8_cmesh_set_join (cmesh, 1, 2, 1, 1, 0);
  t8_cmesh_set_join (cmesh, 1, 3, 0, 2, 0);
  t8_cmesh_set_join (cmesh, 2, 6, 0, 2, 0);
  t8_cmesh_set_join (cmesh, 3, 8, 0, 2, 0);
  t8_cmesh_set_join (cmesh, 4, 5, 1, 1, 0);
  t8_cmesh_set_join (cmesh, 4, 7, 0, 2, 0);
  t8_cmesh_set_join (cmesh, 5, 10, 0, 2, 0);
  t8_cmesh_set_join (cmesh, 6, 7, 1, 1, 0);
  t8_cmesh_set_join (cmesh, 6, 8, 0, 1, 1);
  t8_cmesh_set_join (cmesh, 7, 11, 0, 1, 1);
  t8_cmesh_set_join (cmesh, 8, 9, 0, 2, 0);
  t8_cmesh_set_join (cmesh, 9, 13, 0, 0, 0);
  t8_cmesh_set_join (cmesh, 10, 12, 0, 2, 0);
  t8_cmesh_set_join (cmesh, 11, 12, 2, 1, 1);
  t8_cmesh_set_join (cmesh, 11, 13, 0, 2, 0);

  /* 7. Commit the mesh */
  t8_cmesh_commit (cmesh, comm);

  return cmesh;
}

/* This cmesh is a complex polygonal shape with 8 triangles. */
t8_cmesh_t
t8_cmesh_new_complex_polygonal_shape (sc_MPI_Comm comm)
{

  /* 1. Defining an array with all vertices */
  /* Just all vertices of all trees. partly duplicated */
  double vertices[72] = {
    0.4, 0.5, 0, 1,   0,   0, 1,   0.6, 0,  //triangle 1
    0.4, 0.2, 0, 0.4, 0.5, 0, 1,   0,   0,  //triangle 2
    0,   0,   0, 0.4, 0.2, 0, 0.4, 0.5, 0,  //triangle 3
    0,   0,   0, 0.2, 0.4, 0, 0.4, 0.5, 0,  //triangle 4
    0.4, 0.8, 0, 0.2, 0.4, 0, 0.4, 0.5, 0,  //triangle 5
    0.4, 0.8, 0, 0.2, 0.4, 0, 0,   0.6, 0,  //triangle 6
    0.4, 0.8, 0, 0.4, 1,   0, 0,   0.6, 0,  //triangle 7
    0.4, 0.8, 0, 0.4, 1,   0, 1,   0.8, 0,  //triangle 8
  };

  /* 2. Initialization of the mesh */
  t8_cmesh_t cmesh;
  t8_cmesh_init (&cmesh);

  /* 3. Definition of the geometry */
  t8_cmesh_register_geometry<t8_geometry_linear> (cmesh);
  ; /* Use linear geometry */

  /* 4. Definition of the classes of the different trees */
  t8_cmesh_set_tree_class (cmesh, 0, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 1, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 2, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 3, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 4, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 5, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 6, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 7, T8_ECLASS_TRIANGLE);

  /* 5. Classification of the vertices for each tree */
  t8_cmesh_set_tree_vertices (cmesh, 0, vertices, 3);
  t8_cmesh_set_tree_vertices (cmesh, 1, vertices + 9, 3);
  t8_cmesh_set_tree_vertices (cmesh, 2, vertices + 18, 3);
  t8_cmesh_set_tree_vertices (cmesh, 3, vertices + 27, 3);
  t8_cmesh_set_tree_vertices (cmesh, 4, vertices + 36, 3);
  t8_cmesh_set_tree_vertices (cmesh, 5, vertices + 45, 3);
  t8_cmesh_set_tree_vertices (cmesh, 6, vertices + 54, 3);
  t8_cmesh_set_tree_vertices (cmesh, 7, vertices + 63, 3);

  /* 6. Definition of the face neighbors between the different trees */
  t8_cmesh_set_join (cmesh, 0, 1, 2, 0, 0);
  t8_cmesh_set_join (cmesh, 1, 2, 2, 0, 0);
  t8_cmesh_set_join (cmesh, 2, 3, 1, 1, 0);
  t8_cmesh_set_join (cmesh, 3, 4, 0, 0, 0);
  t8_cmesh_set_join (cmesh, 4, 5, 2, 2, 0);
  t8_cmesh_set_join (cmesh, 5, 6, 1, 1, 0);
  t8_cmesh_set_join (cmesh, 6, 7, 2, 2, 0);

  /* 7. Commit the mesh */
  t8_cmesh_commit (cmesh, comm);

  return cmesh;
}

/* This cmesh is an L-shape with 4 triangles. */
t8_cmesh_t
t8_cmesh_new_l_shape (sc_MPI_Comm comm)
{

  /* 1. Defining an array with all vertices */
  /* Just all vertices of all trees. partly duplicated */
  double vertices[36] = {
    0.5, 0.5, 0, 1, 0, 0, 1,   0.5, 0,  //triangle 1
    0.5, 0.5, 0, 1, 0, 0, 0,   0,   0,  //triangle 2
    0.5, 0.5, 0, 0, 1, 0, 0,   0,   0,  //triangle 3
    0.5, 0.5, 0, 0, 1, 0, 0.5, 1,   0,  //triangle 4
  };

  /* 2. Initialization of the mesh */
  t8_cmesh_t cmesh;
  t8_cmesh_init (&cmesh);

  /* 3. Definition of the geometry */
  t8_cmesh_register_geometry<t8_geometry_linear> (cmesh);
  /* Use linear geometry */

  /* 4. Definition of the classes of the different trees */
  t8_cmesh_set_tree_class (cmesh, 0, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 1, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 2, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 3, T8_ECLASS_TRIANGLE);

  /* 5. Classification of the vertices for each tree */
  t8_cmesh_set_tree_vertices (cmesh, 0, vertices, 3);
  t8_cmesh_set_tree_vertices (cmesh, 1, vertices + 9, 3);
  t8_cmesh_set_tree_vertices (cmesh, 2, vertices + 18, 3);
  t8_cmesh_set_tree_vertices (cmesh, 3, vertices + 27, 3);

  /* 6. Definition of the face neighbors between the different trees */
  t8_cmesh_set_join (cmesh, 0, 1, 2, 2, 0);
  t8_cmesh_set_join (cmesh, 1, 2, 1, 1, 0);
  t8_cmesh_set_join (cmesh, 2, 3, 2, 2, 0);

  /* 7. Commit the mesh */
  t8_cmesh_commit (cmesh, comm);

  return cmesh;
}

/* This cmesh is (x-axis)[0,360]x(y-axis)[-90,90] with 4 triangles (specifically for the era5 data). */
t8_cmesh_t
t8_cmesh_new_earth (sc_MPI_Comm comm)
{

  double vertices[36] = {
    0,   -90, 0, 180, -90, 0, 180, 90,  0,  //triangle 1
    360, 90,  0, 180, -90, 0, 360, -90, 0,  //triangle 2
    0,   -90, 0, 0,   90,  0, 180, 90,  0,  //triangle 3
    360, 90,  0, 180, -90, 0, 180, 90,  0   //triangle 4
  };

  t8_cmesh_t cmesh;
  t8_cmesh_init (&cmesh);

  t8_cmesh_register_geometry<t8_geometry_linear> (cmesh);

  t8_cmesh_set_tree_class (cmesh, 0, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 1, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 2, T8_ECLASS_TRIANGLE);
  t8_cmesh_set_tree_class (cmesh, 3, T8_ECLASS_TRIANGLE);

  t8_cmesh_set_tree_vertices (cmesh, 0, vertices, 3);
  t8_cmesh_set_tree_vertices (cmesh, 1, vertices + 9, 3);
  t8_cmesh_set_tree_vertices (cmesh, 2, vertices + 18, 3);
  t8_cmesh_set_tree_vertices (cmesh, 3, vertices + 27, 3);

  t8_cmesh_set_join (cmesh, 0, 2, 1, 1, 0);
  t8_cmesh_set_join (cmesh, 0, 3, 0, 0, 0);
  t8_cmesh_set_join (cmesh, 1, 3, 2, 2, 0);

  t8_cmesh_commit (cmesh, comm);

  return cmesh;
}

/* Compute the cube-id of t's ancestor of level "level" in constant time.
 * If "level" is greater then t->level then the cube-id 0 is returned. */
static t8_dtri_cube_id_t
compute_cubeid (const t8_dtri_t *t, int level)
{
  t8_dtri_cube_id_t id = 0;
  t8_dtri_coord_t h;

  /* TODO: assert that 0 < level? This may simplify code elsewhere */

  T8_ASSERT (0 <= level && level <= T8_DTRI_MAXLEVEL);
  h = T8_DTRI_LEN (level);
  if (level == 0) {
    return 0;
  }

  id |= ((t->x & h) ? 0x01 : 0);
  id |= ((t->y & h) ? 0x02 : 0);
#ifdef T8_DTRI_TO_DTET
  id |= ((t->z & h) ? 0x04 : 0);
#endif

  return id;
}

/* A routine to compute the type of t's ancestor of level "level", if its type at an intermediate level is already
 * known. If "level" equals t's level then t's type is returned. It is not allowed to call this function with "level"
 * greater than t->level. This method runs in O(t->level - level).
 */
static t8_dtri_type_t
compute_type_ext (const t8_dtri_t *t, int level, t8_dtri_type_t known_type, int known_level)
{
  int8_t type = known_type;
  t8_dtri_cube_id_t cid;
  int i;

  T8_ASSERT (0 <= level && level <= known_level);
  T8_ASSERT (known_level <= t->level);
  if (level == known_level) {
    return known_type;
  }
  if (level == 0) {
    /* TODO: the type of the root tet is hardcoded to 0
     *       maybe once we want to allow the root tet to have different types */
    return 0;
  }
  for (i = known_level; i > level; i--) {
    cid = compute_cubeid (t, i);
    /* compute type as the type of T^{i+1}, that is T's ancestor of level i+1 */
    type = t8_dtri_cid_type_to_parenttype[cid][type];
  }
  return type;
}

/* A routine to compute the type of t's ancestor of level "level". If "level" equals t's level then t's type is
 * returned. It is not allowed to call this function with "level" greater than t->level. This method runs in
 * O(t->level - level).
 */
static t8_dtri_type_t
compute_type (const t8_dtri_t *t, int level)
{
  return compute_type_ext (t, level, t->type, t->level);
}

static void
get_point_order (int *first, int *second, int *third, t8_dtri_cube_id_t cube_id)
{
  if (*first == 0 && *second == 1 && *third == 2) {
    if (cube_id == 0) {
      *first = 0;
      *second = 1;
      *third = 2;
    }
    else if (cube_id == 1) {
      *first = 2;   //1
      *second = 0;  //2
      *third = 1;   //0
    }
    else if (cube_id == 2) {
      *first = 1;   //2
      *second = 2;  //0
      *third = 0;   //1
    }
    else if (cube_id == 3) {
      *first = 0;
      *second = 2;
      *third = 1;
    }
  }
  else if (*first == 2 && *second == 0 && *third == 1) {
    if (cube_id == 0) {
      *first = 0;
      *second = 1;
      *third = 2;
    }
    else if (cube_id == 1) {
      *first = 2;
      *second = 0;
      *third = 1;
    }
    else if (cube_id == 2) {
      *first = 1;
      *second = 2;
      *third = 0;
    }
    else if (cube_id == 3) {
      *first = 2;   //1
      *second = 1;  //0
      *third = 0;   //2
    }
  }
  else if (*first == 1 && *second == 2 && *third == 0) {
    if (cube_id == 0) {
      *first = 0;
      *second = 1;
      *third = 2;
    }
    else if (cube_id == 1) {
      *first = 2;
      *second = 0;
      *third = 1;
    }
    else if (cube_id == 2) {
      *first = 1;
      *second = 2;
      *third = 0;
    }
    else if (cube_id == 3) {
      *first = 1;   //2
      *second = 0;  //1
      *third = 2;   //0
    }
  }
  else if (*first == 0 && *second == 2 && *third == 1) {
    if (cube_id == 0) {
      *first = 0;
      *second = 2;
      *third = 1;
    }
    else if (cube_id == 1) {
      *first = 1;
      *second = 0;
      *third = 2;
    }
    else if (cube_id == 2) {
      *first = 2;
      *second = 1;
      *third = 0;
    }
    else if (cube_id == 3) {
      *first = 2;
      *second = 0;
      *third = 1;
    }
  }
  else if (*first == 1 && *second == 0 && *third == 2) {
    if (cube_id == 0) {
      *first = 0;
      *second = 2;
      *third = 1;
    }
    else if (cube_id == 1) {
      *first = 1;
      *second = 0;
      *third = 2;
    }
    else if (cube_id == 2) {
      *first = 2;
      *second = 1;
      *third = 0;
    }
    else if (cube_id == 3) {
      *first = 0;
      *second = 1;
      *third = 2;
    }
  }
  else if (*first == 2 && *second == 1 && *third == 0) {
    if (cube_id == 0) {
      *first = 0;
      *second = 2;
      *third = 1;
    }
    else if (cube_id == 1) {
      *first = 1;
      *second = 0;
      *third = 2;
    }
    else if (cube_id == 2) {
      *first = 2;
      *second = 1;
      *third = 0;
    }
    else if (cube_id == 3) {
      *first = 1;
      *second = 2;
      *third = 0;
    }
  }
}

// int get_correct_order_children(t8_dtri_cube_id_t cube_id,int first, int second, int third){
//   if(first==0&&second==1&&third==2){
//     if(cube_id==0){
//       return 1;//3;
//     }
//     else if(cube_id==1){
//       return 2;//0;
//     }
//     else if(cube_id==2){
//       return 3;//1;
//     }
//     else if(cube_id==3){
//       return 0;//2;
//     }
//   }
//   else if(first==2&&second==0&&third==1){
//     if(cube_id==0){
//       return 3;//2;
//     }
//     else if(cube_id==1){
//       return 2;//3;
//     }
//     else if(cube_id==2){
//       return 0;//1;
//     }
//     else if(cube_id==3){
//       return 1;//0;
//     }
//   }
//   else if(first==1&&second==2&&third==0){
//     if(cube_id==0){
//       return 3;//3;
//     }
//     else if(cube_id==1){
//       return 1;//1;
//     }
//     else if(cube_id==2){
//       return 2;//2;
//     }
//     else if(cube_id==3){
//       return 0;//0;
//     }
//   }
//   else if(first==0&&second==2&&third==1){
//     if(cube_id==0){
//       return 3;//1;
//     }
//     else if(cube_id==1){
//       return 0;//3;
//     }
//     else if(cube_id==2){
//       return 2;//2;
//     }
//     else if(cube_id==3){
//       return 1;//0;
//     }
//   }
//   else if(first==1&&second==0&&third==2){
//     if(cube_id==0){
//       return 3;//2;
//     }
//     else if(cube_id==1){
//       return 1;//1;
//     }
//     else if(cube_id==2){
//       return 0;//3;
//     }
//     else if(cube_id==3){
//       return 2;//0;
//     }
//   }
//   else if(first==2&&second==1&&third==0){
//     if(cube_id==0){
//       return 3;//3;
//     }
//     else if(cube_id==1){
//       return 2;//2;
//     }
//     else if(cube_id==2){
//       return 1;//2;
//     }
//     else if(cube_id==3){
//       return 0;//0;
//     }
//   }
// }

static void
invert_order (int *first, int *second, int *third)
{
  int first_new = *first;
  int second_new = *second;
  int third_new = *third;
  if (first_new == 0) {
    *first = 0;
  }
  else if (second_new == 0) {
    *first = 1;
  }
  else if (third_new == 0) {
    *first = 2;
  }
  if (first_new == 1) {
    *second = 0;
  }
  else if (second_new == 1) {
    *second = 1;
  }
  else if (third_new == 1) {
    *second = 2;
  }
  if (first_new == 2) {
    *third = 0;
  }
  else if (second_new == 2) {
    *third = 1;
  }
  else if (third_new == 2) {
    *third = 2;
  }
}

int
get_correct_order_children (int type, int child_id, int first, int second, int third)
{
  if (type == 1) {
    if (first == 0 && second == 1 && third == 2) {
      if (child_id == 0) {
        return 1;  //3;
      }
      else if (child_id == 1) {
        return 0;  //0;
      }
      else if (child_id == 2) {
        return 2;  //1;
      }
      else if (child_id == 3) {
        return 3;  //2;
      }
    }
    else if (first == 2 && second == 0 && third == 1) {
      if (child_id == 0) {
        return 1;  //2;
      }
      else if (child_id == 1) {
        return 3;  //3;
      }
      else if (child_id == 2) {
        return 0;  //1;
      }
      else if (child_id == 3) {
        return 2;  //0;
      }
    }
    else if (first == 1 && second == 2 && third == 0) {
      if (child_id == 0) {
        return 1;  //3;
      }
      else if (child_id == 1) {
        return 2;  //1;
      }
      else if (child_id == 2) {
        return 3;  //2;
      }
      else if (child_id == 3) {
        return 0;  //0;
      }
    }
    else if (first == 0 && second == 2 && third == 1) {
      if (child_id == 0) {
        return 1;  //1;
      }
      else if (child_id == 1) {
        return 0;  //3;
      }
      else if (child_id == 2) {
        return 3;  //2;
      }
      else if (child_id == 3) {
        return 2;  //0;
      }
    }
    else if (first == 1 && second == 0 && third == 2) {
      if (child_id == 0) {
        return 1;  //2;
      }
      else if (child_id == 1) {
        return 2;  //1;
      }
      else if (child_id == 2) {
        return 0;  //3;
      }
      else if (child_id == 3) {
        return 3;  //0;
      }
    }
    else if (first == 2 && second == 1 && third == 0) {
      if (child_id == 0) {
        return 1;  //3;
      }
      else if (child_id == 1) {
        return 3;  //2;
      }
      else if (child_id == 2) {
        return 2;  //2;
      }
      else if (child_id == 3) {
        return 0;  //0;
      }
    }
  }
  else {
    if (first == 0 && second == 1 && third == 2) {
      if (child_id == 0) {
        return 2;  //3;
      }
      else if (child_id == 1) {
        return 0;  //0;
      }
      else if (child_id == 2) {
        return 1;  //1;
      }
      else if (child_id == 3) {
        return 3;  //2;
      }
    }
    else if (first == 2 && second == 0 && third == 1) {
      if (child_id == 0) {
        return 2;  //2;
      }
      else if (child_id == 1) {
        return 3;  //3;
      }
      else if (child_id == 2) {
        return 0;  //1;
      }
      else if (child_id == 3) {
        return 1;  //0;
      }
    }
    else if (first == 1 && second == 2 && third == 0) {
      if (child_id == 0) {
        return 2;  //3;
      }
      else if (child_id == 1) {
        return 1;  //1;
      }
      else if (child_id == 2) {
        return 3;  //2;
      }
      else if (child_id == 3) {
        return 0;  //0;
      }
    }
    else if (first == 0 && second == 2 && third == 1) {
      if (child_id == 0) {
        return 2;  //1;
      }
      else if (child_id == 1) {
        return 0;  //3;
      }
      else if (child_id == 2) {
        return 3;  //2;
      }
      else if (child_id == 3) {
        return 1;  //0;
      }
    }
    else if (first == 1 && second == 0 && third == 2) {
      if (child_id == 0) {
        return 2;  //2;
      }
      else if (child_id == 1) {
        return 1;  //1;
      }
      else if (child_id == 2) {
        return 0;  //3;
      }
      else if (child_id == 3) {
        return 3;  //0;
      }
    }
    else if (first == 2 && second == 1 && third == 0) {
      if (child_id == 0) {
        return 2;  //3;
      }
      else if (child_id == 1) {
        return 3;  //2;
      }
      else if (child_id == 2) {
        return 1;  //2;
      }
      else if (child_id == 3) {
        return 0;  //0;
      }
    }
  }
}

inline bool
isZero (double x)
{
  const double epsilon = 1e-15;
  return std::abs (x) <= epsilon;
}

/* brauchen wir in binary lmi nicht mehr */
/* Counts the digits of an integer */
// int countDigit(long long int n)
// {
//     if (n == 0)
//         return 1;
//     int count = 0;
//     while (n != 0) {
//         n = n / 10;
//         ++count;
//     }
//     return count;
// }

uint64_t
t8_lmi_to_elem_id_binary (uint64_t lmi)
{
  uint64_t elem_id = 0;

  uint64_t basecell = lmi & ((1ULL << BASECELL_BITS) - 1);  // Extract the basecell (21 bits)
  lmi >>= BASECELL_BITS;                                    // Shift right to remove the basecell part

  uint64_t level = lmi & ((1ULL << LEVEL_BITS) - 1);  // Extract the 5 bits for level
  lmi >>= LEVEL_BITS;                                 // Shift right to remove the level part

  elem_id = basecell * pow4[level];  // Initialize elem_id with basecell * pow4[level]

  // Extract and process the path bits (2 bits per level)
  for (int lev_ind = 0; lev_ind < level; ++lev_ind) {
    uint64_t path_segment = lmi & 3;          // Extract the 2 lowest bits (path segment)
    lmi >>= 2;                                // Shift to the right to remove the path segment
    elem_id += pow4[lev_ind] * path_segment;  // Update elem_id using path segment
  }

  return elem_id;
}

// Function to clear path and basecell bits, keeping only the level
uint64_t
get_level_only_lmi (uint64_t lmi)
{
  // Shift lmi right to remove the basecell part
  lmi >>= BASECELL_BITS;

  // Extract the level (5 bits) by masking with (1ULL << LEVEL_BITS) - 1
  uint64_t level = lmi & ((1ULL << LEVEL_BITS) - 1);

  // We don't need to extract the path because we are going to clear it

  // Reconstruct the LMI with only the level and set path and basecell bits to zero
  // Clear the path bits and the basecell bits, leaving only the level
  uint64_t new_lmi = (level << BASECELL_BITS);  // Shift the level into place
  return new_lmi;
}

// Function to encode LMI
uint64_t
t8_elem_id_to_lmi_binary (t8_locidx_t elem_id, int level, const t8_locidx_t basecell)
{
  uint64_t lmi = 0;

  // Path encoding (2 bits per level)
  for (int lev_ind = 0; lev_ind < level; ++lev_ind) {
    uint64_t j = (elem_id / pow4[level - lev_ind - 1]) % 4;  // Get the path segment
    //printf("path ist %llu\n", j);
    lmi = (lmi << PATH_BITS) | j;  // Shift left by 2 bits and add path segment
  }

  // Level encoding (5 bits)
  lmi = (lmi << LEVEL_BITS) | level;

  // Basecell encoding (21 bits)
  lmi = (lmi << BASECELL_BITS) | basecell;

  return lmi;
}

// Function to decode LMI and print path, basecell, and level
void
decode_lmi (uint64_t lmi)
{
  // Extract the basecell (21 bits)
  uint64_t basecell = lmi & ((1ULL << BASECELL_BITS) - 1);  // Extract the lowest 21 bits (basecell)
  lmi >>= BASECELL_BITS;                                    // Shift to remove the basecell part

  // Extract the level (5 bits)
  uint64_t level = lmi & ((1ULL << LEVEL_BITS) - 1);  // Extract the lowest 5 bits (level)
  lmi >>= LEVEL_BITS;                                 // Shift to remove the level part

  // Extract the path (38 bits, 2 bits per path segment)
  uint64_t path = lmi & ((1ULL << (PATH_BITS * level)) - 1);  // Extract the path bits
  lmi >>= (PATH_BITS * level);                                // Shift to remove the path part

  // Print the decoded information
  printf ("Decoded LMI:\n");
  printf ("Level: %i\n", (int) level);        // Cast to int to match expected format
  printf ("Basecell: %i\n", (int) basecell);  // Cast to int to match expected format
  printf ("Path (in 2-bit segments): ");

  // Print each path segment as a number between 0 and 3
  for (int i = 0; i < level; ++i) {
    uint64_t segment = path & 3;  // Extract the lowest 2 bits for each segment
    printf ("%llu ", segment);    // Print the current segment
    path >>= 2;                   // Shift right by 2 to process the next segment
  }
  printf ("\n");
}

// Function to decrease the level and reset corresponding path bits to zero, returning the new LMI
uint64_t
get_parents_lmi_binary (uint64_t lmi)
{
  // Extract the basecell (21 bits) - the lowest bits
  uint64_t basecell = lmi & ((1ULL << BASECELL_BITS) - 1);
  lmi >>= BASECELL_BITS;  // Shift to remove the basecell part

  // Extract the level (5 bits)
  uint64_t level = lmi & ((1ULL << LEVEL_BITS) - 1);
  lmi >>= LEVEL_BITS;  // Shift to remove the level part

  // Extract the path (38 bits)
  uint64_t path = lmi;  // Remaining part is the path

  // Decrease the level by 1 if it is greater than 0
  if (level > 0) {
    level--;  // Decrease the level

    // Reset the path bits for the decreased level (for this we just shift out the last 2 bits for the current level)
    path >>= PATH_BITS;  // Shift to remove the last path bits

    // Re-encode the LMI with the decreased level, reset path, and basecell
    // Reinsert the level (shifted to its position)
    lmi = (path << (LEVEL_BITS + BASECELL_BITS)) | (level << BASECELL_BITS) | basecell;  // Reassemble the LMI
  }

  return lmi;
}

// Function to create an LMI from a given level with path and basecell bits set to zero
uint64_t
create_lmi_from_level (int level)
{
  // Ensure level is valid (between 0 and 31 for 5 bits)
  if (level < 0 || level >= (1 << LEVEL_BITS)) {
    printf ("Invalid level: %i. Level must be between 0 and %i.\n", level, (1 << LEVEL_BITS));  //-1
    return 0;  // Return 0 for invalid level
  }

  // Create an LMI with the path and basecell bits set to zero, and only the level populated
  uint64_t lmi = 0;

  // Step 1: Shift to make space for the path (38 bits) and basecell (21 bits)
  // Path will be zero, so we just need to shift space for path and basecell.
  lmi = (lmi << BASECELL_BITS);       // Shift to make room for the basecell (21 bits)
  lmi = (lmi << LEVEL_BITS) | level;  // Shift to make room for the level (5 bits) and set the level

  // The final LMI will have only the level in place, path and basecell will be zero.

  return lmi;
}

// Function to increase the level by 1 and insert a new path segment (j) at the corresponding level, returning the new LMI
uint64_t
get_jth_child_lmi_binary (uint64_t lmi, uint64_t j)
{
  // Extract the basecell (21 bits) from the lowest bits
  uint64_t basecell = lmi & ((1ULL << BASECELL_BITS) - 1);
  lmi >>= BASECELL_BITS;  // Remove the basecell part

  // Extract the level (5 bits)
  uint64_t level = lmi & ((1ULL << LEVEL_BITS) - 1);
  lmi >>= LEVEL_BITS;  // Remove the level part

  // Extract the path (38 bits) as the remaining part
  uint64_t path = lmi;  // Remaining part after removing level and basecell

  // Increase the level by 1
  level++;  // Increment the level

  // Insert the new path segment (j) at the corresponding position (2 bits)
  path
    = (path << 2) | (j & 3);  // Add the new path segment by shifting and OR-ing with j (ensuring j is between 0 and 3)

  // Re-encode the LMI with the increased level, new path segment, and basecell in the correct order:
  // 1. Shift the path into its position
  // 2. Insert the level (shifted into the correct position)
  // 3. Insert the basecell (shifted into the correct position)
  lmi = (path << (LEVEL_BITS + BASECELL_BITS)) | (level << BASECELL_BITS) | basecell;

  return lmi;
}

// Function to calculate the LMI from elem_id, then increase the level and return the corresponding elem_id from the new LMI
t8_locidx_t
get_jth_child_t8code_id_binary (t8_locidx_t elem_id, int level, t8_locidx_t basecell, uint64_t j)
{
  uint64_t lmi = t8_elem_id_to_lmi_binary (elem_id, level, basecell);

  uint64_t new_lmi = get_jth_child_lmi_binary (lmi, j);

  uint64_t new_elem_id = t8_lmi_to_elem_id_binary (new_lmi);

  return new_elem_id;
}

// Function to calculate the LMI from elem_id, then decrease the level and return the corresponding elem_id from the new LMI
t8_locidx_t
get_parents_t8code_id_binary (uint64_t elem_id, int level, t8_locidx_t basecell)
{
  uint64_t lmi = t8_elem_id_to_lmi_binary (elem_id, level, basecell);

  uint64_t new_lmi = get_parents_lmi_binary (lmi);

  uint64_t new_elem_id = t8_lmi_to_elem_id_binary (new_lmi);

  return new_elem_id;
}

//
// /* Transform an t8code element id into the corresponding level multi index. */
// struct lmi t8_elem_id_to_lmi (const t8_element_t *element, const t8_scheme *eclass_scheme, t8_locidx_t elem_id,const long long int basecell_num_digits_offset){
//   struct lmi curr_elem;//declare an lmi struct
//   long long int level = eclass_scheme->t8_element_level (element);//get the level of our element
//   curr_elem.lmi_arr[1]=level;//write level in the correct place in lmi
//   /* NOTE: it is pow4 because each cell is refined into four elements, this has to be changed for different element types */
//   curr_elem.lmi_arr[2]=elem_id/pow4[level];//could replace with which tree
//   long long int path=curr_elem.lmi_arr[2];//start with basecell as path and then add the path through the children on the next levels
//   for (int lev_ind=0;lev_ind<level; lev_ind++ ){
//     path +=pow10[lev_ind+1+basecell_num_digits_offset]*((elem_id/pow4[level-lev_ind-1])%4);
//   }
//   curr_elem.lmi_arr[0]=path;
//   return curr_elem;
// }
//
// /* Transform a level multi index into the corresponding t8code element id. */
// int t8_lmi_to_elem_id (struct lmi curr_elem,const long long int basecell_num_digits_offset){
//   int level=curr_elem.lmi_arr[1];//read the level
//   long long int path=curr_elem.lmi_arr[0];//read the path
//   long long int elem_id=pow4[level]*curr_elem.lmi_arr[2];
//   for (int lev_ind=0;lev_ind<level; lev_ind++ ){
//     elem_id +=pow4[lev_ind]*((path/pow10[level+basecell_num_digits_offset-lev_ind])%10);
//   }
//   return elem_id;
// }
//
// /* Calculates the parent of an element via lmi
//  * Note: basecell_num_digits_offset is needed because in the lmi=(path,level,basecell), if we have more than 10 basecells
//  * then we need to know how many digits the basecell_needs in our path
//  */
// struct lmi get_parents_lmi(struct lmi curr_elem,const long long int basecell_num_digits_offset){
//   struct lmi parent;
//   parent.lmi_arr[1]=curr_elem.lmi_arr[1]-1;//parent is on level-1
//   parent.lmi_arr[2]=curr_elem.lmi_arr[2];//parent has same basecell
//   /* Now just delete the first digit in our path in the lmi */
//   parent.lmi_arr[0]=curr_elem.lmi_arr[0]-((curr_elem.lmi_arr[0]/pow10[curr_elem.lmi_arr[1]+basecell_num_digits_offset+1])%10)*pow10[basecell_num_digits_offset+curr_elem.lmi_arr[1]+1];
//   return parent;
// }
//
// /* Calculate the lmi of the jth child. */
// struct lmi get_jth_child_lmi(const int j, struct lmi curr_elem,const long long int basecell_num_digits_offset){
//   struct lmi child;
//   child.lmi_arr[1]=curr_elem.lmi_arr[1]+1;//children live on level+1
//   child.lmi_arr[2]=curr_elem.lmi_arr[2];//children have the same basecell
//   child.lmi_arr[0]=curr_elem.lmi_arr[0]+j*pow10[curr_elem.lmi_arr[1]+1+basecell_num_digits_offset];//add j at front of path
//   return child;
// }
//
// /* Calculate the t8code id of the jth child. */
// int get_jth_child_t8code_id(const int j,const t8_element_t *element, const t8_scheme *eclass_scheme, t8_locidx_t elem_id,const long long int basecell_num_digits_offset){
//   struct lmi child_lmi;
//   struct lmi elem_lmi=t8_elem_id_to_lmi (element,eclass_scheme,elem_id,basecell_num_digits_offset);//first transform into lmi
//   child_lmi=get_jth_child_lmi(j,elem_lmi,basecell_num_digits_offset);//then calculate the jth child
//   int t8code_id=t8_lmi_to_elem_id (child_lmi,basecell_num_digits_offset);//then transform again back to t8code id
//   return t8code_id;
// }
//
// /* Calculate the t8code id of the parent on one level lower. */
// int get_parents_t8code_id(const t8_element_t *element, const t8_scheme *eclass_scheme, t8_locidx_t elem_id,const long long int basecell_num_digits_offset){
//   struct lmi elem_lmi=t8_elem_id_to_lmi (element,eclass_scheme,elem_id,basecell_num_digits_offset);//first transform into lmi
//   struct lmi parent_lmi=get_parents_lmi(elem_lmi,basecell_num_digits_offset);//then calculate the lmi of the parent one level lower
//   int t8code_id=t8_lmi_to_elem_id (parent_lmi,basecell_num_digits_offset);//then transform again back to t8code id
//   return t8code_id;
// }

/* Alternativer Weg Tresholding durchzuführen: Hier gehen wir von oberen Leveln runter. Ist eine Zelle signifikant,
 * so auch deren Kinder.
 */
void
HierarchischerThresholdOperator (struct grid_hierarchy grid_hierarchy, double c_tresh, double a,
                                 unsigned int &anzahl_gesamt, unsigned int &anzahl_signifikant)
{
  anzahl_gesamt = 0;
  anzahl_signifikant = 0;
  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    const t8_element_t *element;
    for (int l = 1; l < max_level + 1; ++l) {
      c_tresh /= a;  //auskommentieren
      T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
      num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        /* Get the number of elements of this tree. */
        num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          ++anzahl_gesamt;
          grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant = false;
          grid_hierarchy.lev_arr[l].data_arr[current_index].adaptiert = false;
        }
      }
    }

    for (int l = max_level - 1; l >= 0; --l) {
      c_tresh /= a;
      T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
      /* Get the number of ghost elements of forest. */
      num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        /* This loop iterates through all local trees in the forest. */
        /* Get the number of elements of this tree. */
        num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          T8_ASSERT (sizeof (grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff) / sizeof (double) == 3 * M);
          element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
          double volume = t8_forest_element_volume (grid_hierarchy.lev_arr[l].forest_arr, itree, element);
          if (grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant)
            continue;
          for (int i = 0; i < 3 * M; ++i) {
            if (abs (grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[i]) > sqrt (2.0 * volume) * c_tresh) {
              grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[0]]
                .adaptiert
                = true;
              grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[1]]
                .adaptiert
                = true;
              grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[2]]
                .adaptiert
                = true;
              grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[3]]
                .adaptiert
                = true;
              grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant = true;
              ++anzahl_signifikant;
              int level = l;
              t8_locidx_t index = current_index;
              while (level > 0) {
                ++anzahl_signifikant;
                index = grid_hierarchy.lev_arr[level].data_arr[index].Father_id;
                --level;
                grid_hierarchy.lev_arr[level].data_arr[index].signifikant = true;
              }
              break;
            }
          }
          if (!grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant) {
            for (int j = 0; j < 3 * M; ++j) {
              grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[j] = 0.;
            }
          }
        }
      }
    }
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[0].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[0].forest_arr);
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[0].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        if (!(grid_hierarchy.lev_arr[0].data_arr[current_index].signifikant))
          grid_hierarchy.lev_arr[0].data_arr[current_index].adaptiert = true;
      }
    }
  }
  //Hier könnten wir die Anzahl sign. Details printen vs Anzahl gesamt
}

/* Alternativer Weg Tresholding durchzuführen: Hier gehen wir von oberen Leveln runter. Ist eine Zelle signifikant,
 * so auch deren Kinder. In der ThresholdOperator Funktion machen wir es genau andersrum.
 */
void
HierarchischerThresholdOperator_3d (struct grid_hierarchy_3d grid_hierarchy, double c_tresh, double a,
                                    unsigned int &anzahl_gesamt, unsigned int &anzahl_signifikant)
{
  anzahl_gesamt = 0;
  anzahl_signifikant = 0;
  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    const t8_element_t *element;
    double avg_per_dim_arr[3] = { 0, 0, 0 }; /* We need this for the thresholding */
    double area;                             /*volume/area of the whole domain */

    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[0].forest_arr);
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      /* Get the number of elements of this tree. */
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[0].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[0].forest_arr, itree, ielement);
        double A = t8_forest_element_volume (grid_hierarchy.lev_arr[0].forest_arr, itree, element);
        avg_per_dim_arr[0] += A * grid_hierarchy.lev_arr[0].data_arr[current_index].u_coeff_d1[0];
        avg_per_dim_arr[1] += A * grid_hierarchy.lev_arr[0].data_arr[current_index].u_coeff_d2[0];
        avg_per_dim_arr[2] += A * grid_hierarchy.lev_arr[0].data_arr[current_index].u_coeff_d3[0];
        area += A;
      }
    }
    avg_per_dim_arr[0] /= area;
    avg_per_dim_arr[1] /= area;
    avg_per_dim_arr[2] /= area;

    avg_per_dim_arr[0] = max (avg_per_dim_arr[0], 1.0);
    avg_per_dim_arr[1] = max (avg_per_dim_arr[1], 1.0);
    avg_per_dim_arr[2] = max (avg_per_dim_arr[2], 1.0);

    for (int l = 1; l < max_level + 1; ++l) {
      c_tresh /= a;  //auskommentieren und testen, entspricht nicht BA Theorie Florian Sieglar so wie hier geschrieben
      T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
      num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        /* Get the number of elements of this tree. */
        num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          ++anzahl_gesamt;
          grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant = false;
          grid_hierarchy.lev_arr[l].data_arr[current_index].adaptiert = false;
        }
      }
    }

    for (int l = max_level - 1; l >= 0; --l) {
      c_tresh /= a;
      T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
      num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        /* This loop iterates through all local trees in the forest. */
        /* Get the number of elements of this tree. */
        num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          T8_ASSERT (sizeof (grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d1) / sizeof (double) == 3 * M);
          element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
          double volume = t8_forest_element_volume (grid_hierarchy.lev_arr[l].forest_arr, itree, element);
          if (grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant)
            continue;
          for (int i = 0; i < 3 * M; ++i) {
            if (max ({ abs (grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d1[i]) / avg_per_dim_arr[0],
                       abs (grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d2[i]) / avg_per_dim_arr[1],
                       abs (grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d3[i]) / avg_per_dim_arr[2] })
                > sqrt (2.0 * volume) * c_tresh) {
              grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[0]]
                .adaptiert
                = true;
              grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[1]]
                .adaptiert
                = true;
              grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[2]]
                .adaptiert
                = true;
              grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[3]]
                .adaptiert
                = true;
              grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant = true;
              ++anzahl_signifikant;
              int level = l;
              t8_locidx_t index = current_index;
              while (level > 0) {
                ++anzahl_signifikant;
                index = grid_hierarchy.lev_arr[level].data_arr[index].Father_id;
                --level;
                grid_hierarchy.lev_arr[level].data_arr[index].signifikant = true;
              }
              break;
            }
          }
          if (!grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant) {
            for (int j = 0; j < 3 * M; ++j) {
              grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d1[j] = 0.;
              grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d2[j] = 0.;
              grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d3[j] = 0.;
            }
          }
        }
      }
    }
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[0].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[0].forest_arr);
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[0].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        if (!(grid_hierarchy.lev_arr[0].data_arr[current_index].signifikant))
          grid_hierarchy.lev_arr[0].data_arr[current_index].adaptiert = true;
      }
    }
  }
  //Hier könnten wir die Anzahl sign. Details printen vs Anzahl gesamt
}

/* Waveletfree version of the above function. */
void
HierarchischerThresholdOperatorwaveletfree (struct grid_hierarchy_waveletfree grid_hierarchy, double c_tresh, double a,
                                            unsigned int &anzahl_gesamt, unsigned int &anzahl_signifikant)
{
  anzahl_gesamt = 0;
  anzahl_signifikant = 0;
  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    const t8_element_t *element;
    for (int l = 1; l < max_level + 1; ++l) {
      c_tresh /= a;  //auskommentieren
      T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
      num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        /* Get the number of elements of this tree. */
        num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          ++anzahl_gesamt;
          grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant = false;
          grid_hierarchy.lev_arr[l].data_arr[current_index].adaptiert = false;
        }
      }
    }
    for (int l = max_level - 1; l >= 0; --l) {  // vorher -2 int l = levels-2; l>=0; --l
      c_tresh /= a;
      T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
      num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        /* This loop iterates through all local trees in the forest. */
        /* Get the number of elements of this tree. */
        num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          T8_ASSERT (sizeof (grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free) / sizeof (double)
                     == 3 * M);
          element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
          if (grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant)
            continue;
          //calculate the 2 norm of the vector
          double sumSq = 0;
          for (int i = 0; i < M; ++i) {
            for (int j = 0; j < 4; ++j) {
              sumSq += grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][j]
                       * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][j];
            }
          }

          sumSq = sqrt (sumSq);
          if (sumSq > sqrt (2.0 * (t8_forest_element_volume (grid_hierarchy.lev_arr[l].forest_arr, itree, element)))
                        * c_tresh) {
            grid_hierarchy.lev_arr[l + 1]
              .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[0]]
              .adaptiert
              = true;
            grid_hierarchy.lev_arr[l + 1]
              .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[1]]
              .adaptiert
              = true;
            grid_hierarchy.lev_arr[l + 1]
              .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[2]]
              .adaptiert
              = true;
            grid_hierarchy.lev_arr[l + 1]
              .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[3]]
              .adaptiert
              = true;
            grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant = true;
            int level = l;
            t8_locidx_t index = current_index;
            while (level > 0) {
              ++anzahl_signifikant;
              index = grid_hierarchy.lev_arr[level].data_arr[index].Father_id;
              --level;
              grid_hierarchy.lev_arr[level].data_arr[index].signifikant = true;
            }
          }

          if (!grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant) {
            for (int i = 0; i < M; ++i) {
              for (int j = 0; j < 4; ++j) {
                grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][j] = 0.;
              }
            }
          }
        }
      }
    }
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[0].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[0].forest_arr);
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[0].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        if (!(grid_hierarchy.lev_arr[0].data_arr[current_index].signifikant))
          grid_hierarchy.lev_arr[0].data_arr[current_index].adaptiert = true;
      }
    }
  }
}

/* Waveletfree version of the above function. */
void
HierarchischerThresholdOperatorwaveletfree_3d (struct grid_hierarchy_waveletfree_3d grid_hierarchy, double c_tresh,
                                               double a, unsigned int &anzahl_gesamt, unsigned int &anzahl_signifikant)
{
  anzahl_gesamt = 0;
  anzahl_signifikant = 0;
  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    const t8_element_t *element;
    double avg_per_dim_arr[3] = { 0, 0, 0 }; /* We need this for the thresholding */
    double area;                             /*volume/area of the whole domain */

    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[0].forest_arr);
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      /* Get the number of elements of this tree. */
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[0].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[0].forest_arr, itree, ielement);
        double A = t8_forest_element_volume (grid_hierarchy.lev_arr[0].forest_arr, itree, element);
        avg_per_dim_arr[0] += A * grid_hierarchy.lev_arr[0].data_arr[current_index].u_coeff_d1[0];
        avg_per_dim_arr[1] += A * grid_hierarchy.lev_arr[0].data_arr[current_index].u_coeff_d2[0];
        avg_per_dim_arr[2] += A * grid_hierarchy.lev_arr[0].data_arr[current_index].u_coeff_d3[0];
        area += A;
      }
    }
    avg_per_dim_arr[0] /= area;
    avg_per_dim_arr[1] /= area;
    avg_per_dim_arr[2] /= area;

    avg_per_dim_arr[0] = max (avg_per_dim_arr[0], 1.0);
    avg_per_dim_arr[1] = max (avg_per_dim_arr[1], 1.0);
    avg_per_dim_arr[2] = max (avg_per_dim_arr[2], 1.0);

    for (int l = 1; l < max_level + 1; ++l) {
      c_tresh /= a;  //auskommentieren
      T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
      num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        /* Get the number of elements of this tree. */
        num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          ++anzahl_gesamt;
          grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant = false;
          grid_hierarchy.lev_arr[l].data_arr[current_index].adaptiert = false;
        }
      }
    }
    for (int l = max_level - 1; l >= 0; --l) {  // vorher -2 int l = levels-2; l>=0; --l
      c_tresh /= a;
      T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
      num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        /* This loop iterates through all local trees in the forest. */
        /* Get the number of elements of this tree. */
        num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          T8_ASSERT (sizeof (grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d1)
                       / sizeof (double)
                     == 3 * M);
          element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
          if (grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant)
            continue;
          //calculate the 2 norm of the vector for each dimension
          double sumSq[3] = { 0, 0, 0 };
          for (int i = 0; i < M; ++i) {
            for (int j = 0; j < 4; ++j) {
              sumSq[0] += grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d1[i][j]
                          * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d1[i][j];
              sumSq[1] += grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d2[i][j]
                          * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d2[i][j];
              sumSq[2] += grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d3[i][j]
                          * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d3[i][j];
            }
          }

          sumSq[0] = sqrt (sumSq[0]);
          sumSq[1] = sqrt (sumSq[1]);
          sumSq[2] = sqrt (sumSq[2]);
          if (max ({ sumSq[0] / avg_per_dim_arr[0], sumSq[2] / avg_per_dim_arr[1], sumSq[2] / avg_per_dim_arr[2] })
              > sqrt (2.0 * (t8_forest_element_volume (grid_hierarchy.lev_arr[l].forest_arr, itree, element)))
                  * c_tresh) {
            grid_hierarchy.lev_arr[l + 1]
              .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[0]]
              .adaptiert
              = true;
            grid_hierarchy.lev_arr[l + 1]
              .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[1]]
              .adaptiert
              = true;
            grid_hierarchy.lev_arr[l + 1]
              .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[2]]
              .adaptiert
              = true;
            grid_hierarchy.lev_arr[l + 1]
              .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[3]]
              .adaptiert
              = true;
            grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant = true;
            int level = l;
            t8_locidx_t index = current_index;
            while (level > 0) {
              ++anzahl_signifikant;
              index = grid_hierarchy.lev_arr[level].data_arr[index].Father_id;
              --level;
              grid_hierarchy.lev_arr[level].data_arr[index].signifikant = true;
            }
          }

          if (!grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant) {
            for (int i = 0; i < M; ++i) {
              for (int j = 0; j < 4; ++j) {
                grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d1[i][j] = 0.;
                grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d2[i][j] = 0.;
                grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d3[i][j] = 0.;
              }
            }
          }
        }
      }
    }
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[0].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[0].forest_arr);
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[0].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        if (!(grid_hierarchy.lev_arr[0].data_arr[current_index].signifikant))
          grid_hierarchy.lev_arr[0].data_arr[current_index].adaptiert = true;
      }
    }
  }
}

/* Verwenden nur Hierarchisches Thresholding, da wir Baumstruktur wollen
void ThresholdOperator(struct grid_hierarchy grid_hierarchy, double c_tresh, double a, unsigned int & anzahl_gesamt, unsigned int & anzahl_klein) {
  anzahl_gesamt = 0; anzahl_klein = 0;
  {
    t8_locidx_t num_local_elements;
    t8_locidx_t num_ghost_elements;
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    t8_eclass_t tree_class;
    const t8_scheme *eclass_scheme;
    const t8_element_t *element;
    for (int l = max_level-1; l>=0; --l) {//-2
      c_tresh /= a;
      T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
      num_local_elements = t8_forest_get_local_num_elements (grid_hierarchy.lev_arr[l].forest_arr);
      num_ghost_elements = t8_forest_get_num_ghosts (grid_hierarchy.lev_arr[l].forest_arr);
      num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);

      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        tree_class = t8_forest_get_tree_class (grid_hierarchy.lev_arr[l].forest_arr, itree);
        eclass_scheme = t8_forest_get_scheme(grid_hierarchy.lev_arr[l].forest_arr, tree_class);
        num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
          double volume = t8_forest_element_volume (grid_hierarchy.lev_arr[l].forest_arr, itree, element);
          T8_ASSERT(sizeof(grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff) / sizeof(double) == 3*M);
          for (int i = 0; i < 3*M; ++i) {
            ++anzahl_gesamt;
            if (abs(grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[i]) <= sqrt(2.0*volume)*c_tresh){
              grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[i] = 0.;
              ++anzahl_klein;
            }
          }
        }
      }
    }

  }
}

void ThresholdOperatorwaveletfree(struct grid_hierarchy grid_hierarchy, double c_tresh, double a, unsigned int & anzahl_gesamt, unsigned int & anzahl_klein) {
  anzahl_gesamt = 0; anzahl_klein = 0;
  {
    t8_locidx_t num_local_elements;
    t8_locidx_t num_ghost_elements;
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    t8_eclass_t tree_class;
    const t8_scheme *eclass_scheme;
    const t8_element_t *element;
    for (int l = max_level-1; l>=0; --l) {//-2
      c_tresh /= a;
      T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));

      num_local_elements = t8_forest_get_local_num_elements (grid_hierarchy.lev_arr[l].forest_arr);

      num_ghost_elements = t8_forest_get_num_ghosts (grid_hierarchy.lev_arr[l].forest_arr);
      num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);

      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        tree_class = t8_forest_get_tree_class (grid_hierarchy.lev_arr[l].forest_arr, itree);
        eclass_scheme = t8_forest_get_scheme(grid_hierarchy.lev_arr[l].forest_arr, tree_class);
        num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
          double volume = t8_forest_element_volume (grid_hierarchy.lev_arr[l].forest_arr, itree, element);
          T8_ASSERT(sizeof(grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff) / sizeof(double) == 3*M);
          for (int i = 0; i < M; ++i) {
            for (int j = 0; j < 4; ++j) {
              ++anzahl_gesamt;
              if (abs(grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][j]) <= sqrt(2.0*volume)*c_tresh){
                grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][j] = 0.;
                ++anzahl_klein;
              }
            }
          }
        }
      }
    }
  }
}
*/

// // folgende Funktionen vgl mit Paper
// double err_thr(struct grid_hierarchy grid_hierarchy, double c_tresh, double a) {
//   double sum = 0;
//   {
//     t8_locidx_t itree, num_local_trees;
//     t8_locidx_t current_index;
//     t8_locidx_t ielement, num_elements_in_tree;
//     const t8_element_t *element;
//     for (int l = max_level-1; l>=0; --l) {//-2
//       c_tresh /= a;
//       T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
//       num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
//
//       for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
//         num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
//         for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
//           element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
//           double volume = t8_forest_element_volume (grid_hierarchy.lev_arr[l].forest_arr, itree, element);
//           T8_ASSERT(sizeof(grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff) / sizeof(double) == 3*M);
//           for (int i = 0; i < 3*M; ++i) {
//             if (abs(grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[i]) > sqrt(2.0*volume)*c_tresh){
//               sum +=(grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[i])*(grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[i]);
//             }
//           }
//         }
//       }
//     }
//   }
//   return sqrt(sum);
// }
//
// double err_thr_waveletfree(struct grid_hierarchy grid_hierarchy, double c_tresh, double a) {
//   double sum = 0;
//   {
//     t8_locidx_t itree, num_local_trees;
//     t8_locidx_t current_index;
//     t8_locidx_t ielement, num_elements_in_tree;
//     const t8_element_t *element;
//     for (int l = max_level-1; l>=0; --l) {//-2
//       c_tresh /= a;
//       T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
//       num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
//
//       for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
//         num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
//         for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
//           element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
//           double volume = t8_forest_element_volume (grid_hierarchy.lev_arr[l].forest_arr, itree, element);
//           T8_ASSERT(sizeof(grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff) / sizeof(double) == 3*M);
//           for (int i = 0; i < 3*M; ++i) {
//             if (abs(grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[i]) > sqrt(2.0*volume)*c_tresh){
//               sum +=(grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[i])*(grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[i]);
//             }
//           }
//         }
//       }
//     }
//   }
//   return sqrt(sum);
// }
//
// // Kleinstes c_tresh, sodass kein Koeffizient mehr signifikant,
// // also größte sinnvolle Wahl für c_tresh
// double max_c_tresh(struct grid_hierarchy grid_hierarchy, double a) {
//   double max = 0.; double b = 1.;
//   {
//     t8_locidx_t itree, num_local_trees;
//     t8_locidx_t current_index;
//     t8_locidx_t ielement, num_elements_in_tree;
//     const t8_element_t *element;
//     for (int l = max_level-1; l>=0; --l) {//-2
//       b /= a;
//       T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
//       num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
//
//       for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
//         num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
//         for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
//           element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
//           double volume = t8_forest_element_volume (grid_hierarchy.lev_arr[l].forest_arr, itree, element);
//           T8_ASSERT(sizeof(grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff) / sizeof(double) == 3*M);
//           for (int i = 0; i < 3*M; ++i) {
//             double value = abs(grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[i])/(sqrt(2.0*volume)*b);
//             if  (value > max) max = value;
//           }
//         }
//       }
//     }
//   }
//   return max;
// }
//
// // Kleinstes c_tresh, sodass kein Koeffizient mehr signifikant,
// // also größte sinnvolle Wahl für c_tresh
// double max_c_tresh_waveletfree(struct grid_hierarchy grid_hierarchy, double a) {
//   double max = 0.; double b = 1.;
//   {
//     t8_locidx_t itree, num_local_trees;
//     t8_locidx_t current_index;
//     t8_locidx_t ielement, num_elements_in_tree;
//     const t8_element_t *element;
//     for (int l = max_level-1; l>=0; --l) {//-2
//       b /= a;
//       T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
//       num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
//
//       for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
//         num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
//         for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
//           element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
//           double volume = t8_forest_element_volume (grid_hierarchy.lev_arr[l].forest_arr, itree, element);
//           T8_ASSERT(sizeof(grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff) / sizeof(double) == 3*M);
//           for (int i = 0; i < 3*M; ++i) {
//             double value = abs(grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[i])/(sqrt(2.0*volume)*b);
//             if  (value > max) max = value;
//           }
//         }
//       }
//     }
//   }
//   return max;
// }
//
//
//
// double c_tresh_opt(struct grid_hierarchy grid_hierarchy, double err, double a) {
//   double delta = pow(10.,floor(log10(err))-1);
//   double c_tresh_l = 0.;
//   double c_tresh_r = max_c_tresh(grid_hierarchy, a);
//   double err_r = err_thr(grid_hierarchy, c_tresh_r, a);
//   if (err_r < err) return c_tresh_r;
//   while ((c_tresh_r - c_tresh_l) >= delta) {
//     double c_tresh_m = (c_tresh_l+c_tresh_r)/2.;
//     double err_m = err_thr(grid_hierarchy, c_tresh_m, a);
//     if (err_m == err) return c_tresh_m;
//     if (err_m < err) { c_tresh_l = c_tresh_m;}
//     else { c_tresh_r = c_tresh_m; err_r = err_m; }
//   }
//   return c_tresh_r;
// }

void
MultiScaleOperator (struct grid_hierarchy grid_hierarchy)
{
  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  for (int l = max_level - 1; l >= 0; --l) {  //-2
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);

    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        const t8_element_t *element;
        element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
        int first = grid_hierarchy.lev_arr[l].data_arr[current_index].first;
        int second = grid_hierarchy.lev_arr[l].data_arr[current_index].second;
        int third = grid_hierarchy.lev_arr[l].data_arr[current_index].third;
        invert_order (&first, &second, &third);
        for (int i = 0; i < M; ++i) {
          double u_sum = 0., d_sum = 0.;
          for (int j = 0; j < M; ++j) {
            double v0;
            double v1;
            double v2;
            double v3;

            v0 = grid_hierarchy.lev_arr[l + 1]
                   .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr
                               [get_correct_order_children ((((t8_dtri_t *) element)->type), 0, first, second, third)]]
                   .u_coeff[j];
            v1 = grid_hierarchy.lev_arr[l + 1]
                   .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr
                               [get_correct_order_children ((((t8_dtri_t *) element)->type), 1, first, second, third)]]
                   .u_coeff[j];
            v2 = grid_hierarchy.lev_arr[l + 1]
                   .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr
                               [get_correct_order_children ((((t8_dtri_t *) element)->type), 2, first, second, third)]]
                   .u_coeff[j];
            v3 = grid_hierarchy.lev_arr[l + 1]
                   .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr
                               [get_correct_order_children ((((t8_dtri_t *) element)->type), 3, first, second, third)]]
                   .u_coeff[j];

            u_sum += M0 (i, j) * v0;
            u_sum += M1 (i, j) * v1;
            u_sum += M2 (i, j) * v2;
            u_sum += M3 (i, j) * v3;

            d_sum += N0 (i, j) * v0;
            d_sum += N1 (i, j) * v1;
            d_sum += N2 (i, j) * v2;
            d_sum += N3 (i, j) * v3;
          }
          grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff[i] = u_sum;
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[i] = d_sum;
        }
        for (int i = M; i < 3 * M; ++i) {
          double sum = 0.;
          for (int j = 0; j < M; ++j) {
            sum += N0 (i, j)
                   * grid_hierarchy.lev_arr[l + 1]
                       .data_arr[grid_hierarchy.lev_arr[l]
                                   .data_arr[current_index]
                                   .child_ids.child_arr[get_correct_order_children ((((t8_dtri_t *) element)->type), 0,
                                                                                    first, second, third)]]
                       .u_coeff[j];
            sum += N1 (i, j)
                   * grid_hierarchy.lev_arr[l + 1]
                       .data_arr[grid_hierarchy.lev_arr[l]
                                   .data_arr[current_index]
                                   .child_ids.child_arr[get_correct_order_children ((((t8_dtri_t *) element)->type), 1,
                                                                                    first, second, third)]]
                       .u_coeff[j];
            sum += N2 (i, j)
                   * grid_hierarchy.lev_arr[l + 1]
                       .data_arr[grid_hierarchy.lev_arr[l]
                                   .data_arr[current_index]
                                   .child_ids.child_arr[get_correct_order_children ((((t8_dtri_t *) element)->type), 2,
                                                                                    first, second, third)]]
                       .u_coeff[j];
            sum += N3 (i, j)
                   * grid_hierarchy.lev_arr[l + 1]
                       .data_arr[grid_hierarchy.lev_arr[l]
                                   .data_arr[current_index]
                                   .child_ids.child_arr[get_correct_order_children ((((t8_dtri_t *) element)->type), 3,
                                                                                    first, second, third)]]
                       .u_coeff[j];
          }
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[i] = sum;
        }
      }
    }
  }
}

void
MultiScaleOperator_3d (struct grid_hierarchy_3d grid_hierarchy)
{
  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  for (int l = max_level - 1; l >= 0; --l) {
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);

    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        const t8_element_t *element;
        element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
        int first = grid_hierarchy.lev_arr[l].data_arr[current_index].first;
        int second = grid_hierarchy.lev_arr[l].data_arr[current_index].second;
        int third = grid_hierarchy.lev_arr[l].data_arr[current_index].third;
        invert_order (&first, &second, &third);
        int first_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 0, first, second, third);
        int second_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 1, first, second, third);
        int third_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 2, first, second, third);
        int fourth_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 3, first, second, third);
        for (int i = 0; i < M; ++i) {
          double u_sum[3] = { 0, 0, 0 };
          double d_sum[3] = { 0, 0, 0 };
          for (int j = 0; j < M; ++j) {
            double v0[3]
              = { grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
                    .u_coeff_d1[j],
                  grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
                    .u_coeff_d2[j],
                  grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
                    .u_coeff_d3[j] };
            double v1[3]
              = { grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
                    .u_coeff_d1[j],
                  grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
                    .u_coeff_d2[j],
                  grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
                    .u_coeff_d3[j] };
            double v2[3]
              = { grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
                    .u_coeff_d1[j],
                  grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
                    .u_coeff_d2[j],
                  grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
                    .u_coeff_d3[j] };
            double v3[3]
              = { grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
                    .u_coeff_d1[j],
                  grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
                    .u_coeff_d2[j],
                  grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
                    .u_coeff_d3[j] };

            u_sum[0] += M0 (i, j) * v0[0];
            u_sum[0] += M1 (i, j) * v1[0];
            u_sum[0] += M2 (i, j) * v2[0];
            u_sum[0] += M3 (i, j) * v3[0];

            u_sum[1] += M0 (i, j) * v0[1];
            u_sum[1] += M1 (i, j) * v1[1];
            u_sum[1] += M2 (i, j) * v2[1];
            u_sum[1] += M3 (i, j) * v3[1];

            u_sum[2] += M0 (i, j) * v0[2];
            u_sum[2] += M1 (i, j) * v1[2];
            u_sum[2] += M2 (i, j) * v2[2];
            u_sum[2] += M3 (i, j) * v3[2];

            d_sum[0] += N0 (i, j) * v0[0];
            d_sum[0] += N1 (i, j) * v1[0];
            d_sum[0] += N2 (i, j) * v2[0];
            d_sum[0] += N3 (i, j) * v3[0];

            d_sum[1] += N0 (i, j) * v0[1];
            d_sum[1] += N1 (i, j) * v1[1];
            d_sum[1] += N2 (i, j) * v2[1];
            d_sum[1] += N3 (i, j) * v3[1];

            d_sum[2] += N0 (i, j) * v0[2];
            d_sum[2] += N1 (i, j) * v1[2];
            d_sum[2] += N2 (i, j) * v2[2];
            d_sum[2] += N3 (i, j) * v3[2];
          }
          grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d1[i] = u_sum[0];
          grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d2[i] = u_sum[1];
          grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d3[i] = u_sum[2];

          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d1[i] = d_sum[0];
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d2[i] = d_sum[1];
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d3[i] = d_sum[2];
        }
        for (int i = M; i < 3 * M; ++i) {
          double sum[3] = { 0, 0, 0 };
          for (int j = 0; j < M; ++j) {
            sum[0] += N0 (i, j)
                      * grid_hierarchy.lev_arr[l + 1]
                          .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
                          .u_coeff_d1[j];
            sum[0] += N1 (i, j)
                      * grid_hierarchy.lev_arr[l + 1]
                          .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
                          .u_coeff_d1[j];
            sum[0] += N2 (i, j)
                      * grid_hierarchy.lev_arr[l + 1]
                          .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
                          .u_coeff_d1[j];
            sum[0] += N3 (i, j)
                      * grid_hierarchy.lev_arr[l + 1]
                          .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
                          .u_coeff_d1[j];

            sum[1] += N0 (i, j)
                      * grid_hierarchy.lev_arr[l + 1]
                          .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
                          .u_coeff_d2[j];
            sum[1] += N1 (i, j)
                      * grid_hierarchy.lev_arr[l + 1]
                          .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
                          .u_coeff_d2[j];
            sum[1] += N2 (i, j)
                      * grid_hierarchy.lev_arr[l + 1]
                          .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
                          .u_coeff_d2[j];
            sum[1] += N3 (i, j)
                      * grid_hierarchy.lev_arr[l + 1]
                          .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
                          .u_coeff_d2[j];

            sum[2] += N0 (i, j)
                      * grid_hierarchy.lev_arr[l + 1]
                          .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
                          .u_coeff_d3[j];
            sum[2] += N1 (i, j)
                      * grid_hierarchy.lev_arr[l + 1]
                          .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
                          .u_coeff_d3[j];
            sum[2] += N2 (i, j)
                      * grid_hierarchy.lev_arr[l + 1]
                          .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
                          .u_coeff_d3[j];
            sum[2] += N3 (i, j)
                      * grid_hierarchy.lev_arr[l + 1]
                          .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
                          .u_coeff_d3[j];
          }
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d1[i] = sum[0];
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d2[i] = sum[1];
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d3[i] = sum[2];
        }
      }
    }
  }
}

void
MultiScaleOperatorWaveletFree (struct grid_hierarchy_waveletfree grid_hierarchy)
{
  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  for (int l = max_level - 1; l >= 0; --l) {  //-2
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);

    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        const t8_element_t *element;
        element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
        int first = grid_hierarchy.lev_arr[l].data_arr[current_index].first;
        int second = grid_hierarchy.lev_arr[l].data_arr[current_index].second;
        int third = grid_hierarchy.lev_arr[l].data_arr[current_index].third;
        invert_order (&first, &second, &third);
        int first_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 0, first, second, third);
        int second_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 1, first, second, third);
        int third_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 2, first, second, third);
        int fourth_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 3, first, second, third);
        for (int i = 0; i < M; ++i) {
          double u_sum = 0.;
          for (int j = 0; j < M; ++j) {
            double v0 = grid_hierarchy.lev_arr[l + 1]
                          .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
                          .u_coeff[j];
            double v1 = grid_hierarchy.lev_arr[l + 1]
                          .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
                          .u_coeff[j];
            double v2 = grid_hierarchy.lev_arr[l + 1]
                          .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
                          .u_coeff[j];
            double v3 = grid_hierarchy.lev_arr[l + 1]
                          .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
                          .u_coeff[j];

            u_sum += M0 (i, j) * v0;
            u_sum += M1 (i, j) * v1;
            u_sum += M2 (i, j) * v2;
            u_sum += M3 (i, j) * v3;
          }
          grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff[i] = u_sum;
        }

        for (int i = 0; i < M; ++i) {
          double sum0 = 0., sum1 = 0., sum2 = 0., sum3 = 0.;
          for (int j = 0; j < M; ++j) {
            sum0 += M0 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff[j];
            sum1 += M1 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff[j];
            sum2 += M2 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff[j];
            sum3 += M3 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff[j];
          }
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][0]
            = grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
                .u_coeff[i]
              - sum0;
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][1]
            = grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
                .u_coeff[i]
              - sum1;
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][2]
            = grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
                .u_coeff[i]
              - sum2;
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][3]
            = grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
                .u_coeff[i]
              - sum3;
        }
      }
    }
  }
}

void
MultiScaleOperatorWaveletFree_3d (struct grid_hierarchy_waveletfree_3d grid_hierarchy)
{
  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  for (int l = max_level - 1; l >= 0; --l) {  //-2
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);

    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        const t8_element_t *element;
        element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
        int first = grid_hierarchy.lev_arr[l].data_arr[current_index].first;
        int second = grid_hierarchy.lev_arr[l].data_arr[current_index].second;
        int third = grid_hierarchy.lev_arr[l].data_arr[current_index].third;
        invert_order (&first, &second, &third);
        int first_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 0, first, second, third);
        int second_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 1, first, second, third);
        int third_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 2, first, second, third);
        int fourth_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 3, first, second, third);
        for (int i = 0; i < M; ++i) {
          double u_sum[3] = { 0, 0, 0 };
          for (int j = 0; j < M; ++j) {
            double v0[3]
              = { grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
                    .u_coeff_d1[j],
                  grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
                    .u_coeff_d2[j],
                  grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
                    .u_coeff_d3[j] };
            double v1[3]
              = { grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
                    .u_coeff_d1[j],
                  grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
                    .u_coeff_d2[j],
                  grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
                    .u_coeff_d3[j] };
            double v2[3]
              = { grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
                    .u_coeff_d1[j],
                  grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
                    .u_coeff_d2[j],
                  grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
                    .u_coeff_d3[j] };
            double v3[3]
              = { grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
                    .u_coeff_d1[j],
                  grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
                    .u_coeff_d2[j],
                  grid_hierarchy.lev_arr[l + 1]
                    .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
                    .u_coeff_d3[j] };

            u_sum[0] += M0 (i, j) * v0[0];
            u_sum[0] += M1 (i, j) * v1[0];
            u_sum[0] += M2 (i, j) * v2[0];
            u_sum[0] += M3 (i, j) * v3[0];

            u_sum[1] += M0 (i, j) * v0[1];
            u_sum[1] += M1 (i, j) * v1[1];
            u_sum[1] += M2 (i, j) * v2[1];
            u_sum[1] += M3 (i, j) * v3[1];

            u_sum[2] += M0 (i, j) * v0[2];
            u_sum[2] += M1 (i, j) * v1[2];
            u_sum[2] += M2 (i, j) * v2[2];
            u_sum[2] += M3 (i, j) * v3[2];
          }
          grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d1[i] = u_sum[0];
          grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d2[i] = u_sum[1];
          grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d3[i] = u_sum[2];
        }

        for (int i = 0; i < M; ++i) {
          double sum0[3] = { 0, 0, 0 };
          double sum1[3] = { 0, 0, 0 };
          double sum2[3] = { 0, 0, 0 };
          double sum3[3] = { 0, 0, 0 };
          for (int j = 0; j < M; ++j) {
            sum0[0] += M0 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d1[j];
            sum1[0] += M1 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d1[j];
            sum2[0] += M2 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d1[j];
            sum3[0] += M3 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d1[j];

            sum0[1] += M0 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d2[j];
            sum1[1] += M1 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d2[j];
            sum2[1] += M2 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d2[j];
            sum3[1] += M3 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d2[j];

            sum0[2] += M0 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d3[j];
            sum1[2] += M1 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d3[j];
            sum2[2] += M2 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d3[j];
            sum3[2] += M3 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d3[j];
          }
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d1[i][0]
            = grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
                .u_coeff_d1[i]
              - sum0[0];
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d1[i][1]
            = grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
                .u_coeff_d1[i]
              - sum1[0];
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d1[i][2]
            = grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
                .u_coeff_d1[i]
              - sum2[0];
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d1[i][3]
            = grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
                .u_coeff_d1[i]
              - sum3[0];

          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d2[i][0]
            = grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
                .u_coeff_d2[i]
              - sum0[1];
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d2[i][1]
            = grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
                .u_coeff_d2[i]
              - sum1[1];
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d2[i][2]
            = grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
                .u_coeff_d2[i]
              - sum2[1];
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d2[i][3]
            = grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
                .u_coeff_d2[i]
              - sum3[1];

          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d3[i][0]
            = grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
                .u_coeff_d3[i]
              - sum0[2];
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d3[i][1]
            = grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
                .u_coeff_d3[i]
              - sum1[2];
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d3[i][2]
            = grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
                .u_coeff_d3[i]
              - sum2[2];
          grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d3[i][3]
            = grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
                .u_coeff_d3[i]
              - sum3[2];
        }
      }
    }
  }
}

void
InverseMultiScaleOperator (struct grid_hierarchy grid_hierarchy)
{
  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;

  for (int l = 0; l < max_level; ++l) {
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);

    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        const t8_element_t *element;
        element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
        int first = grid_hierarchy.lev_arr[l].data_arr[current_index].first;
        int second = grid_hierarchy.lev_arr[l].data_arr[current_index].second;
        int third = grid_hierarchy.lev_arr[l].data_arr[current_index].third;
        invert_order (&first, &second, &third);
        for (int i = 0; i < M; ++i) {
          double sum0 = 0., sum1 = 0., sum2 = 0., sum3 = 0.;
          for (int j = 0; j < M; ++j) {
            sum0 += M0 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff[j];
            sum1 += M1 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff[j];
            sum2 += M2 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff[j];
            sum3 += M3 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff[j];
          }
          for (int j = 0; j < 3 * M; ++j) {
            sum0 += N0 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[j];
            sum1 += N1 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[j];
            sum2 += N2 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[j];
            sum3 += N3 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[j];
          }
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[get_correct_order_children (
              (((t8_dtri_t *) element)->type), 0, first, second, third)]]
            .u_coeff[i]
            = sum0; /*0*/
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[get_correct_order_children (
              (((t8_dtri_t *) element)->type), 1, first, second, third)]]
            .u_coeff[i]
            = sum1;
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[get_correct_order_children (
              (((t8_dtri_t *) element)->type), 2, first, second, third)]]
            .u_coeff[i]
            = sum2;
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[get_correct_order_children (
              (((t8_dtri_t *) element)->type), 3, first, second, third)]]
            .u_coeff[i]
            = sum3;
        }
      }
    }
  }
}

void
InverseMultiScaleOperator_3d (struct grid_hierarchy_3d grid_hierarchy)
{
  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  for (int l = 0; l < max_level; ++l) {
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);

    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        const t8_element_t *element;
        element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
        int first = grid_hierarchy.lev_arr[l].data_arr[current_index].first;
        int second = grid_hierarchy.lev_arr[l].data_arr[current_index].second;
        int third = grid_hierarchy.lev_arr[l].data_arr[current_index].third;
        invert_order (&first, &second, &third);
        int first_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 0, first, second, third);
        int second_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 1, first, second, third);
        int third_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 2, first, second, third);
        int fourth_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 3, first, second, third);
        for (int i = 0; i < M; ++i) {
          double sum0[3] = { 0, 0, 0 };
          double sum1[3] = { 0, 0, 0 };
          double sum2[3] = { 0, 0, 0 };
          double sum3[3] = { 0, 0, 0 };
          for (int j = 0; j < M; ++j) {
            sum0[0] += M0 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d1[j];
            sum1[0] += M1 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d1[j];
            sum2[0] += M2 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d1[j];
            sum3[0] += M3 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d1[j];

            sum0[1] += M0 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d2[j];
            sum1[1] += M1 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d2[j];
            sum2[1] += M2 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d2[j];
            sum3[1] += M3 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d2[j];

            sum0[2] += M0 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d3[j];
            sum1[2] += M1 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d3[j];
            sum2[2] += M2 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d3[j];
            sum3[2] += M3 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d3[j];
          }
          for (int j = 0; j < 3 * M; ++j) {
            sum0[0] += N0 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d1[j];
            sum1[0] += N1 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d1[j];
            sum2[0] += N2 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d1[j];
            sum3[0] += N3 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d1[j];

            sum0[1] += N0 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d2[j];
            sum1[1] += N1 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d2[j];
            sum2[1] += N2 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d2[j];
            sum3[1] += N3 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d2[j];

            sum0[2] += N0 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d3[j];
            sum1[2] += N1 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d3[j];
            sum2[2] += N2 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d3[j];
            sum3[2] += N3 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d3[j];
          }
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
            .u_coeff_d1[i]
            = sum0[0];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
            .u_coeff_d1[i]
            = sum1[0];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
            .u_coeff_d1[i]
            = sum2[0];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
            .u_coeff_d1[i]
            = sum3[0];

          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
            .u_coeff_d2[i]
            = sum0[1];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
            .u_coeff_d2[i]
            = sum1[1];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
            .u_coeff_d2[i]
            = sum2[1];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
            .u_coeff_d2[i]
            = sum3[1];

          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
            .u_coeff_d3[i]
            = sum0[2];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
            .u_coeff_d3[i]
            = sum1[2];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
            .u_coeff_d3[i]
            = sum2[2];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
            .u_coeff_d3[i]
            = sum3[2];
        }
      }
    }
  }
}

void
InverseMultiScaleOperatorwaveletfree (struct grid_hierarchy_waveletfree grid_hierarchy)
{
  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  for (int l = 0; l < max_level; ++l) {
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        const t8_element_t *element;
        element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
        int first = grid_hierarchy.lev_arr[l].data_arr[current_index].first;
        int second = grid_hierarchy.lev_arr[l].data_arr[current_index].second;
        int third = grid_hierarchy.lev_arr[l].data_arr[current_index].third;
        invert_order (&first, &second, &third);
        for (int i = 0; i < M; ++i) {
          double sum0 = 0., sum1 = 0., sum2 = 0., sum3 = 0.;
          for (int j = 0; j < M; ++j) {
            sum0 += M0 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff[j];
            sum1 += M1 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff[j];
            sum2 += M2 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff[j];
            sum3 += M3 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff[j];
          }
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[get_correct_order_children (
              (((t8_dtri_t *) element)->type), 0, first, second, third)]]
            .u_coeff[i]
            = sum0 + grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][0];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[get_correct_order_children (
              (((t8_dtri_t *) element)->type), 1, first, second, third)]]
            .u_coeff[i]
            = sum1 + grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][1];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[get_correct_order_children (
              (((t8_dtri_t *) element)->type), 2, first, second, third)]]
            .u_coeff[i]
            = sum2 + grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][2];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[get_correct_order_children (
              (((t8_dtri_t *) element)->type), 3, first, second, third)]]
            .u_coeff[i]
            = sum3 + grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][3];
          // grid_hierarchy.lev_arr[l+1].data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[get_correct_order_children((((t8_dtri_t *)element)->type),0,first,second, third)]].u_coeff[i]=sum0+grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][first_child];
          // grid_hierarchy.lev_arr[l+1].data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[get_correct_order_children((((t8_dtri_t *)element)->type),1,first,second, third)]].u_coeff[i]=sum1+grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][second_child];
          // grid_hierarchy.lev_arr[l+1].data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[get_correct_order_children((((t8_dtri_t *)element)->type),2,first,second, third)]].u_coeff[i]=sum2+grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][third_child];
          // grid_hierarchy.lev_arr[l+1].data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[get_correct_order_children((((t8_dtri_t *)element)->type),3,first,second, third)]].u_coeff[i]=sum3+grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][fourth_child];
        }
      }
    }
  }
}

void
InverseMultiScaleOperatorwaveletfree_3d (struct grid_hierarchy_waveletfree_3d grid_hierarchy)
{
  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  for (int l = 0; l < max_level; ++l) {
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);

    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        const t8_element_t *element;
        element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
        int first = grid_hierarchy.lev_arr[l].data_arr[current_index].first;
        int second = grid_hierarchy.lev_arr[l].data_arr[current_index].second;
        int third = grid_hierarchy.lev_arr[l].data_arr[current_index].third;
        invert_order (&first, &second, &third);
        int first_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 0, first, second, third);
        int second_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 1, first, second, third);
        int third_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 2, first, second, third);
        int fourth_child = get_correct_order_children ((((t8_dtri_t *) element)->type), 3, first, second, third);
        for (int i = 0; i < M; ++i) {
          double sum0[3] = { 0, 0, 0 };
          double sum1[3] = { 0, 0, 0 };
          double sum2[3] = { 0, 0, 0 };
          double sum3[3] = { 0, 0, 0 };
          for (int j = 0; j < M; ++j) {
            sum0[0] += M0 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d1[j];
            sum1[0] += M1 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d1[j];
            sum2[0] += M2 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d1[j];
            sum3[0] += M3 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d1[j];

            sum0[1] += M0 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d2[j];
            sum1[1] += M1 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d2[j];
            sum2[1] += M2 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d2[j];
            sum3[1] += M3 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d2[j];

            sum0[2] += M0 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d3[j];
            sum1[2] += M1 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d3[j];
            sum2[2] += M2 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d3[j];
            sum3[2] += M3 (j, i) * grid_hierarchy.lev_arr[l].data_arr[current_index].u_coeff_d3[j];
          }
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
            .u_coeff_d1[i]
            = sum0[0] + grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d1[i][0];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
            .u_coeff_d1[i]
            = sum1[0] + grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d1[i][1];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
            .u_coeff_d1[i]
            = sum2[0] + grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d1[i][2];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
            .u_coeff_d1[i]
            = sum3[0] + grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d1[i][3];

          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
            .u_coeff_d2[i]
            = sum0[1] + grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d2[i][0];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
            .u_coeff_d2[i]
            = sum1[1] + grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d2[i][1];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
            .u_coeff_d2[i]
            = sum2[1] + grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d2[i][2];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
            .u_coeff_d2[i]
            = sum3[1] + grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d2[i][3];

          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[first_child]]
            .u_coeff_d3[i]
            = sum0[2] + grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d3[i][0];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[second_child]]
            .u_coeff_d3[i]
            = sum1[2] + grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d3[i][1];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[third_child]]
            .u_coeff_d3[i]
            = sum2[2] + grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d3[i][2];
          grid_hierarchy.lev_arr[l + 1]
            .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[fourth_child]]
            .u_coeff_d3[i]
            = sum3[2] + grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d3[i][3];
        }
      }
    }
  }
}

void
GridAdaptation (struct grid_hierarchy grid_hierarchy, double c_tresh, double a)
{
  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;
  for (int l = 1; l < max_level + 1; ++l) {
    c_tresh /= a;
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        grid_hierarchy.lev_arr[l].data_arr[current_index].adaptiert = false;
      }
    }
  }
  T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[0].forest_arr));
  num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[0].forest_arr);
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[0].forest_arr, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      grid_hierarchy.lev_arr[0].data_arr[current_index].adaptiert = true;
    }
  }
  for (int l = 0; l < max_level; ++l) {
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
        double volume = t8_forest_element_volume (grid_hierarchy.lev_arr[l].forest_arr, itree, element);
        if (grid_hierarchy.lev_arr[l].data_arr[current_index].adaptiert) {
          T8_ASSERT (sizeof (grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff) / sizeof (double) == 3 * M);
          for (int i = 0; i < 3 * M; ++i) {
            if (abs (grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[i]) > sqrt (2.0 * volume) * c_tresh) {
              //grid_hierarchy.lev_arr[l].data_arr[current_index].adaptiert=false;
              grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[0]]
                .adaptiert
                = true;
              grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[1]]
                .adaptiert
                = true;
              grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[2]]
                .adaptiert
                = true;
              grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[3]]
                .adaptiert
                = true;
              break;
            }
          }
        }
      }
    }
    c_tresh *= a;
  }
}

void
GridAdaptation_3d (struct grid_hierarchy_3d grid_hierarchy, double c_tresh, double a)
{
  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;
  double avg_per_dim_arr[3] = { 0, 0, 0 }; /* We need this for the thresholding */
  double area;                             /*volume/area of the whole domain */
  for (int l = 1; l < max_level + 1; ++l) {
    c_tresh /= a;
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        grid_hierarchy.lev_arr[l].data_arr[current_index].adaptiert = false;
      }
    }
  }
  T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[0].forest_arr));
  num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[0].forest_arr);
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[0].forest_arr, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      grid_hierarchy.lev_arr[0].data_arr[current_index].adaptiert = true;
      element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[0].forest_arr, itree, ielement);
      double A = t8_forest_element_volume (grid_hierarchy.lev_arr[0].forest_arr, itree, element);
      avg_per_dim_arr[0] += A * grid_hierarchy.lev_arr[0].data_arr[current_index].u_coeff_d1[0];
      avg_per_dim_arr[1] += A * grid_hierarchy.lev_arr[0].data_arr[current_index].u_coeff_d2[0];
      avg_per_dim_arr[2] += A * grid_hierarchy.lev_arr[0].data_arr[current_index].u_coeff_d3[0];
      area += A;
    }
  }
  avg_per_dim_arr[0] /= area;
  avg_per_dim_arr[1] /= area;
  avg_per_dim_arr[2] /= area;

  avg_per_dim_arr[0] = max (avg_per_dim_arr[0], 1.0);
  avg_per_dim_arr[1] = max (avg_per_dim_arr[1], 1.0);
  avg_per_dim_arr[2] = max (avg_per_dim_arr[2], 1.0);

  for (int l = 0; l < max_level; ++l) {
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
        double volume = t8_forest_element_volume (grid_hierarchy.lev_arr[l].forest_arr, itree, element);
        if (grid_hierarchy.lev_arr[l].data_arr[current_index].adaptiert) {
          T8_ASSERT (sizeof (grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d1) / sizeof (double) == 3 * M);
          for (int i = 0; i < 3 * M; ++i) {
            if (max ({ abs (grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d1[i]) / avg_per_dim_arr[0],
                       abs (grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d2[i]) / avg_per_dim_arr[1],
                       abs (grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d3[i]) / avg_per_dim_arr[2] })
                > sqrt (2.0 * volume) * c_tresh) {
              grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[0]]
                .adaptiert
                = true;
              grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[1]]
                .adaptiert
                = true;
              grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[2]]
                .adaptiert
                = true;
              grid_hierarchy.lev_arr[l + 1]
                .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[3]]
                .adaptiert
                = true;
              break;
            }
          }
        }
      }
    }
    c_tresh *= a;
  }
}

void
GridAdaptationwaveletfree (struct grid_hierarchy_waveletfree grid_hierarchy, double c_tresh, double a)
{
  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;

  for (int l = 1; l < max_level + 1; ++l) {
    c_tresh /= a;
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        grid_hierarchy.lev_arr[l].data_arr[current_index].adaptiert = false;
      }
    }
  }
  T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[0].forest_arr));
  num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[0].forest_arr);
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[0].forest_arr, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      grid_hierarchy.lev_arr[0].data_arr[current_index].adaptiert = true;
    }
  }

  for (int l = 0; l < max_level; ++l) {
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
        if (grid_hierarchy.lev_arr[l].data_arr[current_index].adaptiert) {
          double sumSq = 0;
          for (int i = 0; i < M; ++i) {
            for (int j = 0; j < 4; ++j) {
              sumSq += grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][j]
                       * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][j];
            }
          }

          sumSq = sqrt (sumSq);
          if (sumSq > sqrt (2.0 * (t8_forest_element_volume (grid_hierarchy.lev_arr[l].forest_arr, itree, element)))
                        * c_tresh) {
            grid_hierarchy.lev_arr[l + 1]
              .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[0]]
              .adaptiert
              = true;
            grid_hierarchy.lev_arr[l + 1]
              .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[1]]
              .adaptiert
              = true;
            grid_hierarchy.lev_arr[l + 1]
              .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[2]]
              .adaptiert
              = true;
            grid_hierarchy.lev_arr[l + 1]
              .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[3]]
              .adaptiert
              = true;
          }
        }
      }
    }
    c_tresh *= a;
  }
}

void
GridAdaptationwaveletfree_3d (struct grid_hierarchy_waveletfree_3d grid_hierarchy, double c_tresh, double a)
{
  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;
  double avg_per_dim_arr[3] = { 0, 0, 0 }; /* We need this for the thresholding */
  double area;                             /*volume/area of the whole domain */

  for (int l = 1; l < max_level + 1; ++l) {
    c_tresh /= a;  //check
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        grid_hierarchy.lev_arr[l].data_arr[current_index].adaptiert = false;
      }
    }
  }
  T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[0].forest_arr));
  num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[0].forest_arr);
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[0].forest_arr, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      grid_hierarchy.lev_arr[0].data_arr[current_index].adaptiert = true;
      element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[0].forest_arr, itree, ielement);
      double A = t8_forest_element_volume (grid_hierarchy.lev_arr[0].forest_arr, itree, element);
      avg_per_dim_arr[0] += A * grid_hierarchy.lev_arr[0].data_arr[current_index].u_coeff_d1[0];
      avg_per_dim_arr[1] += A * grid_hierarchy.lev_arr[0].data_arr[current_index].u_coeff_d2[0];
      avg_per_dim_arr[2] += A * grid_hierarchy.lev_arr[0].data_arr[current_index].u_coeff_d3[0];
      area += A;
    }
  }

  avg_per_dim_arr[0] /= area;
  avg_per_dim_arr[1] /= area;
  avg_per_dim_arr[2] /= area;

  avg_per_dim_arr[0] = max (avg_per_dim_arr[0], 1.0);
  avg_per_dim_arr[1] = max (avg_per_dim_arr[1], 1.0);
  avg_per_dim_arr[2] = max (avg_per_dim_arr[2], 1.0);

  for (int l = 0; l < max_level; ++l) {
    T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
    num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
        if (grid_hierarchy.lev_arr[l].data_arr[current_index].adaptiert) {
          //calculate the 2 norm of the vector for each dimension
          double sumSq[3] = { 0, 0, 0 };
          for (int i = 0; i < M; ++i) {
            for (int j = 0; j < 4; ++j) {
              sumSq[0] += grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d1[i][j]
                          * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d1[i][j];
              sumSq[1] += grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d2[i][j]
                          * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d2[i][j];
              sumSq[2] += grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d3[i][j]
                          * grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d3[i][j];
            }
          }
          sumSq[0] = sqrt (sumSq[0]);
          sumSq[1] = sqrt (sumSq[1]);
          sumSq[2] = sqrt (sumSq[2]);

          if (max ({ sumSq[0] / avg_per_dim_arr[0], sumSq[2] / avg_per_dim_arr[1], sumSq[2] / avg_per_dim_arr[2] })
              > sqrt (2.0 * (t8_forest_element_volume (grid_hierarchy.lev_arr[l].forest_arr, itree, element)))
                  * c_tresh) {
            grid_hierarchy.lev_arr[l + 1]
              .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[0]]
              .adaptiert
              = true;
            grid_hierarchy.lev_arr[l + 1]
              .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[1]]
              .adaptiert
              = true;
            grid_hierarchy.lev_arr[l + 1]
              .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[2]]
              .adaptiert
              = true;
            grid_hierarchy.lev_arr[l + 1]
              .data_arr[grid_hierarchy.lev_arr[l].data_arr[current_index].child_ids.child_arr[3]]
              .adaptiert
              = true;
          }
        }
      }
    }
    c_tresh *= a;
  }
}

/* Grading makes sure that a element has only neighbors of level differing at most by +-1
 * We do this by checking the face neighbors.
 */
void
grading_grid (struct grid_hierarchy grid_hierarchy)
{
  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    t8_eclass_t tree_class;
    const t8_scheme *eclass_scheme;
    const t8_element_t *element;
    for (int l = max_level; l >= 1; --l) {  //vorher 2
      T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
      num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        /* Each tree may have a different element class (quad/tri/hex/tet etc.) and therefore
         * also a different way to interpret its elements. In order to be able to handle elements
         * of a tree, we need to get its eclass_scheme, and in order to so we first get its eclass. */
        eclass_scheme = t8_forest_get_scheme (grid_hierarchy.lev_arr[l].forest_arr);
        tree_class = t8_forest_get_tree_class (grid_hierarchy.lev_arr[l].forest_arr, itree);
        /* This loop iterates through all local trees in the forest. */
        num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
          T8_ASSERT (sizeof (grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff) / sizeof (double) == 3 * M);
          if (grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant) {
            /* Loop over all faces of an element. */
            int num_faces = eclass_scheme->element_get_num_faces (tree_class, element);
            for (int iface = 0; iface < num_faces; iface++) {
              int num_neighbors;        /**< Number of neighbors for each face */
              int *dual_faces;          /**< The face indices of the neighbor elements */
              t8_locidx_t *neighids;    /**< Indices of the neighbor elements */
              t8_element_t **neighbors; /*< Neighboring elements. */
              t8_eclass_t neigh_class;  /*< Neighboring elements tree class. */
              /* Collect all neighbors at the current face. */
              t8_forest_leaf_face_neighbors (grid_hierarchy.lev_arr[l].forest_arr, itree, element, &neighbors, iface,
                                             &dual_faces, &num_neighbors, &neighids, &neigh_class, 1);

              if (num_neighbors > 0) {
                for (int ineigh = 0; ineigh < num_neighbors; ineigh++) {
                  if (!grid_hierarchy.lev_arr[l - 1]
                         .data_arr[grid_hierarchy.lev_arr[l].data_arr[neighids[ineigh]].Father_id]
                         .signifikant) {
                    int level = l;
                    t8_locidx_t index = neighids[ineigh];
                    while (level > 0) {
                      index = grid_hierarchy.lev_arr[level].data_arr[index].Father_id;
                      --level;
                      grid_hierarchy.lev_arr[level].data_arr[index].signifikant = true;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/* Grading makes sure that a element has only neighbors of level differing at most by +-1
 * We do this by checking the face neighbors.
 */
void
grading_grid_3d (struct grid_hierarchy_3d grid_hierarchy)
{
  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    t8_eclass_t tree_class;
    const t8_scheme *eclass_scheme;
    const t8_element_t *element;
    for (int l = max_level; l >= 2; --l) {
      T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
      num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        /* Each tree may have a different element class (quad/tri/hex/tet etc.) and therefore
         * also a different way to interpret its elements. In order to be able to handle elements
         * of a tree, we need to get its eclass_scheme, and in order to so we first get its eclass. */
        tree_class = t8_forest_get_tree_class (grid_hierarchy.lev_arr[l].forest_arr, itree);
        eclass_scheme = t8_forest_get_scheme (grid_hierarchy.lev_arr[l].forest_arr);
        /* This loop iterates through all local trees in the forest. */
        num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
          T8_ASSERT (sizeof (grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d1) / sizeof (double) == 3 * M);
          if (grid_hierarchy.lev_arr[l].data_arr[current_index].adaptiert) {
            /* Loop over all faces of an element. */
            int num_faces = eclass_scheme->element_get_num_faces (tree_class, element);
            for (int iface = 0; iface < num_faces; iface++) {
              int num_neighbors;        /**< Number of neighbors for each face */
              int *dual_faces;          /**< The face indices of the neighbor elements */
              t8_locidx_t *neighids;    /**< Indices of the neighbor elements */
              t8_element_t **neighbors; /*< Neighboring elements. */
              t8_eclass_t neigh_class;  /*< Neighboring elements tree class. */
              /* Collect all neighbors at the current face. */
              t8_forest_leaf_face_neighbors (grid_hierarchy.lev_arr[l].forest_arr, itree, element, &neighbors, iface,
                                             &dual_faces, &num_neighbors, &neighids, &neigh_class, 1);

              if (num_neighbors > 0) {
                for (int ineigh = 0; ineigh < num_neighbors; ineigh++) {
                  if (!grid_hierarchy.lev_arr[l - 1]
                         .data_arr[grid_hierarchy.lev_arr[l].data_arr[neighids[ineigh]].Father_id]
                         .adaptiert) {
                    int level = l;
                    t8_locidx_t index = neighids[ineigh];
                    while (level > 0) {
                      index = grid_hierarchy.lev_arr[level].data_arr[index].Father_id;
                      --level;
                      grid_hierarchy.lev_arr[level].data_arr[index].adaptiert = true;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/* Grading makes sure that a element has only neighbors of level differing at most by +-1
 * We do this by checking the face neighbors.
 */
void
grading_grid_waveletfree (struct grid_hierarchy_waveletfree grid_hierarchy)
{
  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    t8_eclass_t tree_class;
    const t8_scheme *eclass_scheme;
    const t8_element_t *element;
    for (int l = max_level; l >= 2; --l) {
      T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
      num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        /* Each tree may have a different element class (quad/tri/hex/tet etc.) and therefore
         * also a different way to interpret its elements. In order to be able to handle elements
         * of a tree, we need to get its eclass_scheme, and in order to so we first get its eclass. */
        tree_class = t8_forest_get_tree_class (grid_hierarchy.lev_arr[l].forest_arr, itree);
        eclass_scheme = t8_forest_get_scheme (grid_hierarchy.lev_arr[l].forest_arr);
        /* This loop iterates through all local trees in the forest. */
        num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
          T8_ASSERT (sizeof (grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free) / sizeof (double)
                     == 3 * M);
          if (grid_hierarchy.lev_arr[l].data_arr[current_index].adaptiert) {
            /* Loop over all faces of an element. */
            int num_faces = eclass_scheme->element_get_num_faces (tree_class, element);
            for (int iface = 0; iface < num_faces; iface++) {
              int num_neighbors;        /**< Number of neighbors for each face */
              int *dual_faces;          /**< The face indices of the neighbor elements */
              t8_locidx_t *neighids;    /**< Indices of the neighbor elements */
              t8_element_t **neighbors; /*< Neighboring elements. */
              t8_eclass_t neigh_class;  /*< Neighboring elements tree class. */
              /* Collect all neighbors at the current face. */
              t8_forest_leaf_face_neighbors (grid_hierarchy.lev_arr[l].forest_arr, itree, element, &neighbors, iface,
                                             &dual_faces, &num_neighbors, &neighids, &neigh_class, 1);

              if (num_neighbors > 0) {
                for (int ineigh = 0; ineigh < num_neighbors; ineigh++) {
                  //height = height + element_data[neighids[ineigh]].height;
                  if (!grid_hierarchy.lev_arr[l - 1]
                         .data_arr[grid_hierarchy.lev_arr[l].data_arr[neighids[ineigh]].Father_id]
                         .adaptiert) {
                    int level = l;
                    t8_locidx_t index = neighids[ineigh];
                    while (level > 0) {
                      index = grid_hierarchy.lev_arr[level].data_arr[index].Father_id;
                      --level;
                      grid_hierarchy.lev_arr[level].data_arr[index].adaptiert = true;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/* Grading makes sure that a element has only neighbors of level differing at most by +-1
 * We do this by checking the face neighbors.
 */
void
grading_grid_waveletfree_3d (struct grid_hierarchy_waveletfree_3d grid_hierarchy)
{
  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    t8_eclass_t tree_class;
    const t8_scheme *eclass_scheme;
    const t8_element_t *element;
    for (int l = max_level; l >= 2; --l) {
      T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
      num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        /* This loop iterates through all local trees in the forest. */
        /* Each tree may have a different element class (quad/tri/hex/tet etc.) and therefore
         * also a different way to interpret its elements. In order to be able to handle elements
         * of a tree, we need to get its eclass_scheme, and in order to so we first get its eclass. */
        tree_class = t8_forest_get_tree_class (grid_hierarchy.lev_arr[l].forest_arr, itree);
        eclass_scheme = t8_forest_get_scheme (grid_hierarchy.lev_arr[l].forest_arr);
        /* Get the number of elements of this tree. */
        num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          element = t8_forest_get_element_in_tree (grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
          T8_ASSERT (sizeof (grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d1)
                       / sizeof (double)
                     == 3 * M);
          if (grid_hierarchy.lev_arr[l].data_arr[current_index].adaptiert) {
            /* Loop over all faces of an element. */
            //int num_faces = eclass_scheme->t8_element_num_faces (element);
            int num_faces = eclass_scheme->element_get_num_faces (tree_class, element);
            for (int iface = 0; iface < num_faces; iface++) {
              int num_neighbors;        /**< Number of neighbors for each face */
              int *dual_faces;          /**< The face indices of the neighbor elements */
              t8_locidx_t *neighids;    /**< Indices of the neighbor elements */
              t8_element_t **neighbors; /*< Neighboring elements. */
              t8_eclass_t neigh_class;  /*< Neighboring elements tree class. */
              /* Collect all neighbors at the current face. */
              t8_forest_leaf_face_neighbors (grid_hierarchy.lev_arr[l].forest_arr, itree, element, &neighbors, iface,
                                             &dual_faces, &num_neighbors, &neighids, &neigh_class, 1);

              /* Retrieve the `height` of the face neighbor. Account for two neighbors in case
                 of a non-conforming interface by computing the average. */
              if (num_neighbors > 0) {
                for (int ineigh = 0; ineigh < num_neighbors; ineigh++) {
                  //height = height + element_data[neighids[ineigh]].height;
                  if (!grid_hierarchy.lev_arr[l - 1]
                         .data_arr[grid_hierarchy.lev_arr[l].data_arr[neighids[ineigh]].Father_id]
                         .adaptiert) {
                    int level = l;
                    t8_locidx_t index = neighids[ineigh];
                    while (level > 0) {
                      index = grid_hierarchy.lev_arr[level].data_arr[index].Father_id;
                      --level;
                      grid_hierarchy.lev_arr[level].data_arr[index].adaptiert = true;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/* Grading makes sure that a element has only neighbors of level differing at most by +-1
 * We do this by checking the face neighbors.
 */
void
prediction_step_grid (struct grid_hierarchy grid_hierarchy)
{
  {
    struct grid_hierarchy copy_grid_hierarchy = grid_hierarchy;
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    t8_eclass_t tree_class;
    const t8_scheme *eclass_scheme;
    const t8_element_t *element;
    for (int l = max_level; l >= 1; --l) {  //vorher 2
      //T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
      num_local_trees = t8_forest_get_num_local_trees (copy_grid_hierarchy.lev_arr[l].forest_arr);
      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        /* This loop iterates through all local trees in the forest. */
        /* Each tree may have a different element class (quad/tri/hex/tet etc.) and therefore
         * also a different way to interpret its elements. In order to be able to handle elements
         * of a tree, we need to get its eclass_scheme, and in order to so we first get its eclass. */
        tree_class = t8_forest_get_tree_class (copy_grid_hierarchy.lev_arr[l].forest_arr, itree);
        eclass_scheme = t8_forest_get_scheme (copy_grid_hierarchy.lev_arr[l].forest_arr);
        /* Get the number of elements of this tree. */
        num_elements_in_tree = t8_forest_get_tree_num_elements (copy_grid_hierarchy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          element = t8_forest_get_element_in_tree (copy_grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
          T8_ASSERT (sizeof (copy_grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff) / sizeof (double)
                     == 3 * M);
          if (copy_grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant) {
            /* Loop over all faces of an element. */
            //int num_faces = eclass_scheme->t8_element_num_faces (element);
            int num_faces = eclass_scheme->element_get_num_faces (tree_class, element);
            for (int iface = 0; iface < num_faces; iface++) {
              int num_neighbors;        /**< Number of neighbors for each face */
              int *dual_faces;          /**< The face indices of the neighbor elements */
              t8_locidx_t *neighids;    /**< Indices of the neighbor elements */
              t8_element_t **neighbors; /*< Neighboring elements. */
              t8_eclass_t neigh_class;  /*< Neighboring elements tree class. */
              /* Collect all neighbors at the current face. */
              t8_forest_leaf_face_neighbors (grid_hierarchy.lev_arr[l].forest_arr, itree, element, &neighbors, iface,
                                             &dual_faces, &num_neighbors, &neighids, &neigh_class, 1);

              /* Retrieve the `height` of the face neighbor. Account for two neighbors in case
                 of a non-conforming interface by computing the average. */
              if (num_neighbors > 0) {
                for (int ineigh = 0; ineigh < num_neighbors; ineigh++) {
                  //height = height + element_data[neighids[ineigh]].height;
                  if (!grid_hierarchy.lev_arr[l].data_arr[neighids[ineigh]].signifikant) {
                    grid_hierarchy.lev_arr[l].data_arr[neighids[ineigh]].signifikant = true;
                    int level = l;
                    t8_locidx_t index = neighids[ineigh];
                    while (level > 0) {
                      index = grid_hierarchy.lev_arr[level].data_arr[index].Father_id;
                      --level;
                      grid_hierarchy.lev_arr[level].data_arr[index].signifikant = true;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/* Grading makes sure that a element has only neighbors of level differing at most by +-1
 * We do this by checking the face neighbors.
 */
void
prediction_step_grid_3d (struct grid_hierarchy_3d grid_hierarchy)
{
  {
    struct grid_hierarchy_3d copy_grid_hierarchy = grid_hierarchy;
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    t8_eclass_t tree_class;
    const t8_scheme *eclass_scheme;
    const t8_element_t *element;
    for (int l = max_level; l >= 1; --l) {  //vorher 2
      //T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
      num_local_trees = t8_forest_get_num_local_trees (copy_grid_hierarchy.lev_arr[l].forest_arr);
      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        /* This loop iterates through all local trees in the forest. */
        /* Each tree may have a different element class (quad/tri/hex/tet etc.) and therefore
         * also a different way to interpret its elements. In order to be able to handle elements
         * of a tree, we need to get its eclass_scheme, and in order to so we first get its eclass. */
        tree_class = t8_forest_get_tree_class (copy_grid_hierarchy.lev_arr[l].forest_arr, itree);
        eclass_scheme = t8_forest_get_scheme (copy_grid_hierarchy.lev_arr[l].forest_arr);
        /* Get the number of elements of this tree. */
        num_elements_in_tree = t8_forest_get_tree_num_elements (copy_grid_hierarchy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          element = t8_forest_get_element_in_tree (copy_grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
          T8_ASSERT (sizeof (copy_grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_d1) / sizeof (double)
                     == 3 * M);
          if (copy_grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant) {
            /* Loop over all faces of an element. */
            //int num_faces = eclass_scheme->t8_element_num_faces (element);
            int num_faces = eclass_scheme->element_get_num_faces (tree_class, element);
            for (int iface = 0; iface < num_faces; iface++) {
              int num_neighbors;        /**< Number of neighbors for each face */
              int *dual_faces;          /**< The face indices of the neighbor elements */
              t8_locidx_t *neighids;    /**< Indices of the neighbor elements */
              t8_element_t **neighbors; /*< Neighboring elements. */
              t8_eclass_t neigh_class;  /*< Neighboring elements tree class. */
              /* Collect all neighbors at the current face. */
              t8_forest_leaf_face_neighbors (grid_hierarchy.lev_arr[l].forest_arr, itree, element, &neighbors, iface,
                                             &dual_faces, &num_neighbors, &neighids, &neigh_class, 1);

              /* Retrieve the `height` of the face neighbor. Account for two neighbors in case
                 of a non-conforming interface by computing the average. */
              if (num_neighbors > 0) {
                for (int ineigh = 0; ineigh < num_neighbors; ineigh++) {
                  //height = height + element_data[neighids[ineigh]].height;
                  if (!grid_hierarchy.lev_arr[l].data_arr[neighids[ineigh]].signifikant) {
                    grid_hierarchy.lev_arr[l].data_arr[neighids[ineigh]].signifikant = true;
                    int level = l;
                    t8_locidx_t index = neighids[ineigh];
                    while (level > 0) {
                      index = grid_hierarchy.lev_arr[level].data_arr[index].Father_id;
                      --level;
                      grid_hierarchy.lev_arr[level].data_arr[index].signifikant = true;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/* Grading makes sure that a element has only neighbors of level differing at most by +-1
 * We do this by checking the face neighbors.
 */
void
prediction_step_grid_waveletfree (struct grid_hierarchy_waveletfree grid_hierarchy)
{
  {
    struct grid_hierarchy_waveletfree copy_grid_hierarchy = grid_hierarchy;
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    t8_eclass_t tree_class;
    const t8_scheme *eclass_scheme;
    const t8_element_t *element;
    for (int l = max_level; l >= 1; --l) {  //vorher 2
      //T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
      num_local_trees = t8_forest_get_num_local_trees (copy_grid_hierarchy.lev_arr[l].forest_arr);
      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        /* This loop iterates through all local trees in the forest. */
        /* Each tree may have a different element class (quad/tri/hex/tet etc.) and therefore
         * also a different way to interpret its elements. In order to be able to handle elements
         * of a tree, we need to get its eclass_scheme, and in order to so we first get its eclass. */
        tree_class = t8_forest_get_tree_class (copy_grid_hierarchy.lev_arr[l].forest_arr, itree);
        eclass_scheme = t8_forest_get_scheme (copy_grid_hierarchy.lev_arr[l].forest_arr);
        /* Get the number of elements of this tree. */
        num_elements_in_tree = t8_forest_get_tree_num_elements (copy_grid_hierarchy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          element = t8_forest_get_element_in_tree (copy_grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
          T8_ASSERT (sizeof (copy_grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free)
                       / sizeof (double)
                     == 3 * M);
          if (copy_grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant) {
            /* Loop over all faces of an element. */
            //int num_faces = eclass_scheme->t8_element_num_faces (element);
            int num_faces = eclass_scheme->element_get_num_faces (tree_class, element);
            for (int iface = 0; iface < num_faces; iface++) {
              int num_neighbors;        /**< Number of neighbors for each face */
              int *dual_faces;          /**< The face indices of the neighbor elements */
              t8_locidx_t *neighids;    /**< Indices of the neighbor elements */
              t8_element_t **neighbors; /*< Neighboring elements. */
              t8_eclass_t neigh_class;  /*< Neighboring elements tree class. */
              /* Collect all neighbors at the current face. */
              t8_forest_leaf_face_neighbors (grid_hierarchy.lev_arr[l].forest_arr, itree, element, &neighbors, iface,
                                             &dual_faces, &num_neighbors, &neighids, &neigh_class, 1);

              /* Retrieve the `height` of the face neighbor. Account for two neighbors in case
                 of a non-conforming interface by computing the average. */
              if (num_neighbors > 0) {
                for (int ineigh = 0; ineigh < num_neighbors; ineigh++) {
                  //height = height + element_data[neighids[ineigh]].height;
                  if (!grid_hierarchy.lev_arr[l].data_arr[neighids[ineigh]].signifikant) {
                    grid_hierarchy.lev_arr[l].data_arr[neighids[ineigh]].signifikant = true;
                    int level = l;
                    t8_locidx_t index = neighids[ineigh];
                    while (level > 0) {
                      index = grid_hierarchy.lev_arr[level].data_arr[index].Father_id;
                      --level;
                      grid_hierarchy.lev_arr[level].data_arr[index].signifikant = true;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/* Grading makes sure that a element has only neighbors of level differing at most by +-1
 * We do this by checking the face neighbors.
 */
void
prediction_step_grid_waveletfree3d (struct grid_hierarchy_waveletfree_3d grid_hierarchy)
{
  {
    struct grid_hierarchy_waveletfree_3d copy_grid_hierarchy = grid_hierarchy;
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    t8_eclass_t tree_class;
    const t8_scheme *eclass_scheme;
    const t8_element_t *element;
    for (int l = max_level; l >= 1; --l) {  //vorher 2
      //T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
      num_local_trees = t8_forest_get_num_local_trees (copy_grid_hierarchy.lev_arr[l].forest_arr);
      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        /* This loop iterates through all local trees in the forest. */
        /* Each tree may have a different element class (quad/tri/hex/tet etc.) and therefore
         * also a different way to interpret its elements. In order to be able to handle elements
         * of a tree, we need to get its eclass_scheme, and in order to so we first get its eclass. */
        tree_class = t8_forest_get_tree_class (copy_grid_hierarchy.lev_arr[l].forest_arr, itree);
        eclass_scheme = t8_forest_get_scheme (copy_grid_hierarchy.lev_arr[l].forest_arr);
        /* Get the number of elements of this tree. */
        num_elements_in_tree = t8_forest_get_tree_num_elements (copy_grid_hierarchy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          element = t8_forest_get_element_in_tree (copy_grid_hierarchy.lev_arr[l].forest_arr, itree, ielement);
          T8_ASSERT (sizeof (copy_grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free_d1)
                       / sizeof (double)
                     == 3 * M);
          if (copy_grid_hierarchy.lev_arr[l].data_arr[current_index].signifikant) {
            /* Loop over all faces of an element. */
            //int num_faces = eclass_scheme->t8_element_num_faces (element);
            int num_faces = eclass_scheme->element_get_num_faces (tree_class, element);
            for (int iface = 0; iface < num_faces; iface++) {
              int num_neighbors;        /**< Number of neighbors for each face */
              int *dual_faces;          /**< The face indices of the neighbor elements */
              t8_locidx_t *neighids;    /**< Indices of the neighbor elements */
              t8_element_t **neighbors; /*< Neighboring elements. */
              t8_eclass_t neigh_class;  /*< Neighboring elements tree class. */
              /* Collect all neighbors at the current face. */
              t8_forest_leaf_face_neighbors (grid_hierarchy.lev_arr[l].forest_arr, itree, element, &neighbors, iface,
                                             &dual_faces, &num_neighbors, &neighids, &neigh_class, 1);

              /* Retrieve the `height` of the face neighbor. Account for two neighbors in case
                 of a non-conforming interface by computing the average. */
              if (num_neighbors > 0) {
                for (int ineigh = 0; ineigh < num_neighbors; ineigh++) {
                  //height = height + element_data[neighids[ineigh]].height;
                  if (!grid_hierarchy.lev_arr[l].data_arr[neighids[ineigh]].signifikant) {
                    grid_hierarchy.lev_arr[l].data_arr[neighids[ineigh]].signifikant = true;
                    int level = l;
                    t8_locidx_t index = neighids[ineigh];
                    while (level > 0) {
                      index = grid_hierarchy.lev_arr[level].data_arr[index].Father_id;
                      --level;
                      grid_hierarchy.lev_arr[level].data_arr[index].signifikant = true;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/* Adapt a forest according to a callback function.
 * This will create a new forest and return it.
 * Create a new forest that is adapted from \a forest with our adaptation callback.
 * We provide the adapt_data as user data that is stored as the used_data pointer of the
 * new forest (see also t8_forest_set_user_data).
 * The 0, 0 arguments are flags that control
 * \param [in] forest_from      Forest that should be adapted
 * \param [in] adapt_fn         Function that defines how to adapt the forest - Callback function
 * \param [in] do_partition     If non-zero the new_forest should partition the existing forest. As the second parameter
                                is set to NULL, a previously (or later) set forest will be taken
                                (\ref t8_forest_set_adapt, \ref t8_forest_set_balance).
 * \param [in] recursive        If non-zero adaptation is recursive, thus if an element is adapted the children
 *                              or parents are plugged into the callback again recursively until the forest does not
 *                              change any more. If you use this you should ensure that refinement will stop eventually.
 *                              One way is to check the element's level against a given maximum level.
 * \param [in] user_data        User-defined data array to store on the forest
 */
t8_forest_t
t8_adapt_forest (t8_forest_t forest_from, t8_forest_adapt_t adapt_fn, int do_partition, int recursive, void *user_data)
{
  t8_forest_t forest_new;

  t8_forest_init (&forest_new);
  /* Adapt the forest */
  t8_forest_set_adapt (forest_new, forest_from, adapt_fn, recursive);

  /* Set user data for the adapted forest */
  if (user_data != NULL) {
    t8_forest_set_user_data (forest_new, user_data);
  }
  /* Commit the adapted forest */
  t8_forest_commit (forest_new);

  return forest_new;
}

/* Replace callback to decide how to interpolate a refined or coarsened element.
 * If an element is refined, each child gets the value of its parent.
 * If elements are coarsened, the parent gets the average value of the children.
 * Outgoing are the old elements and incoming the new ones
 * \param [in] forest_old        non adapted forest
 * \param [in] forest_new        adapted forest
 * \param [in] which_tree        tree_id of the analyzed element
 * \param [in] ts                eclass scheme
 * \param [in] refine            ==0 - do nothing, == -1 - coarsen, == 1 - refine
 * \param [in] num_outgoing      number of the elements not refined forest
 * \param [in] first_outgoing    index of the old element
 * \param [in] num_incoming      number of the elements corresponding to the element of the not refined forest
 * \param [in] first_incoming    index of the new element
 */
void
t8_forest_replace (t8_forest_t forest_old, t8_forest_t forest_new, t8_locidx_t which_tree, const t8_eclass_t tree_class,
                   const t8_scheme *ts, int refine, int num_outgoing, t8_locidx_t first_outgoing, int num_incoming,
                   t8_locidx_t first_incoming)
{
  struct lmi_adapt *adapt_data_new = (struct lmi_adapt *) t8_forest_get_user_data (forest_new);
  struct lmi_adapt *adapt_data_old = (struct lmi_adapt *) t8_forest_get_user_data (forest_old);

  /* get the index of the data array corresponding to the old and the adapted forest */
  first_incoming += t8_forest_get_tree_element_offset (forest_new, which_tree);
  first_outgoing += t8_forest_get_tree_element_offset (forest_old, which_tree);

  /* Do not adapt or coarsen */
  if (refine == 0) {
    adapt_data_new->adapt_lmi_data[first_incoming].lmi = adapt_data_old->adapt_lmi_data[first_outgoing].lmi;
  }
  /* The old element is refined, we copy the element values */
  else if (refine == 1) {
    for (int i = 0; i < num_incoming; i++) {
      adapt_data_new->adapt_lmi_data[first_incoming + i].lmi
        = get_jth_child_lmi_binary (adapt_data_old->adapt_lmi_data[first_outgoing].lmi, i);
    }
  }
  t8_forest_set_user_data (forest_new, adapt_data_new);
}

/* The data that we want to store for each element.
 * In this example we want to store the element's level and volume. */
static struct t8_data_per_element *
t8_create_element_data (struct grid_hierarchy initial_grid_hierarchy, const int level, func F, const int rule,
                        const int max_lev)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  mat A;
  vector<int> r;
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  t8_locidx_t num_local_elements;
  t8_locidx_t num_ghost_elements;
  struct t8_data_per_element *element_data;
  //initial_grid_hierarchy.lev_arr[max_level].forest_arr

  /* Check that forest is a committed, that is valid and usable, forest. */
  T8_ASSERT (t8_forest_is_committed (initial_grid_hierarchy.lev_arr[level].forest_arr));

  /* Get the number of local elements of forest. */
  num_local_elements = t8_forest_get_local_num_elements (initial_grid_hierarchy.lev_arr[level].forest_arr);
  /* Get the number of ghost elements of forest. */
  num_ghost_elements = t8_forest_get_num_ghosts (initial_grid_hierarchy.lev_arr[level].forest_arr);

  /* Now we need to build an array of our data that is as long as the number
   * of elements plus the number of ghosts. You can use any allocator such as
   * new, malloc or the t8code provide allocation macro T8_ALLOC.
   * Note that in the latter case you need
   * to use T8_FREE in order to free the memory.
   */
  element_data = T8_ALLOC (struct t8_data_per_element, num_local_elements + num_ghost_elements);  //hier
  /* Note: We will later need to associate this data with an sc_array in order to exchange the values for
   *       the ghost elements, which we can do with sc_array_new_data (see t8_step5_exchange_ghost_data).
   *       We could also have directly allocated the data here in an sc_array with
   *       sc_array_new_count (sizeof (struct data_per_element), num_local_elements + num_ghost_elements);
   */

  /* Let us now fill the data with something.
   * For this, we iterate through all trees and for each tree through all its elements, calling
   * t8_forest_get_element_in_tree to get a pointer to the current element.
   * This is the recommended and most performant way.
   * An alternative is to iterate over the number of local elements and use
   * t8_forest_get_element. However, this function needs to perform a binary search
   * for the element and the tree it is in, while t8_forest_get_element_in_tree has a
   * constant look up time. You should only use t8_forest_get_element if you do not know
   * in which tree an element is.
   */
  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    t8_eclass_t tree_class;
    const t8_scheme *eclass_scheme;
    const t8_element_t *element;

    /* Get the number of trees that have elements of this process. */
    num_local_trees = t8_forest_get_num_local_trees (initial_grid_hierarchy.lev_arr[level].forest_arr);
    // long long int basecell_num_digits_offset=countDigit(t8_forest_get_num_global_trees (initial_grid_hierarchy.lev_arr[level].forest_arr)-1)-1;
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      /* This loop iterates through all local trees in the forest. */
      /* Each tree may have a different element class (quad/tri/hex/tet etc.) and therefore
       * also a different way to interpret its elements. In order to be able to handle elements
       * of a tree, we need to get its eclass_scheme, and in order to so we first get its eclass. */
      tree_class = t8_forest_get_tree_class (initial_grid_hierarchy.lev_arr[level].forest_arr, itree);
      eclass_scheme = t8_forest_get_scheme (initial_grid_hierarchy.lev_arr[level].forest_arr);
      /* Get the number of elements of this tree. */
      num_elements_in_tree = t8_forest_get_tree_num_elements (initial_grid_hierarchy.lev_arr[level].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        /* This loop iterates through all the local elements of the forest in the current tree. */
        /* We can now write to the position current_index into our array in order to store
         * data for this element. */
        /* Since in this example we want to compute the data based on the element in question,
         * we need to get a pointer to this element. */

        element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, ielement);

        /* We want to store the elements level and its volume as data. We compute these
         * via the eclass_scheme and the forest_element interface. */
        //element_data[current_index].level = eclass_scheme->t8_element_level (element);
        double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element);
        element_data[current_index].lmi = t8_elem_id_to_lmi_binary (
          current_index, level,
          current_index
            / pow4[level]);  //t8_elem_id_to_lmi (element,eclass_scheme,current_index,basecell_num_digits_offset);
        //t8_global_productionf ("lmi alt: %i \n", element_data[current_index].lmi.lmi_arr[0]);
        printf ("basecell? %i\n", (int) (current_index / pow4[level]));
        uint64_t lmi_bin = t8_elem_id_to_lmi_binary (current_index, level, current_index / pow4[level]);
        decode_lmi (lmi_bin);
        printf ("Elem_id true %i\n", current_index);
        printf ("Elem_id %i\n", (int) t8_lmi_to_elem_id_binary (lmi_bin));
        if (level == 0) {
          element_data[current_index].adaptiert = false;

          element_data[current_index].child_ids.child_arr[0]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 0);
          element_data[current_index].child_ids.child_arr[1]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 1);
          element_data[current_index].child_ids.child_arr[2]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 2);
          element_data[current_index].child_ids.child_arr[3]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 3);
        }
        else {
          element_data[current_index].Father_id = get_parents_t8code_id_binary (
            current_index, level,
            current_index
              / pow4[level]);  //get_parents_t8code_id(element,eclass_scheme,current_index,basecell_num_digits_offset);
          printf ("Parent neu: %i\n", get_parents_t8code_id_binary (current_index, level, current_index / pow4[level]));
          if (eclass_scheme->element_get_level (tree_class, element) == max_lev) {
            element_data[current_index].adaptiert = true;
          }
          else {
            element_data[current_index].adaptiert = false;
            element_data[current_index].child_ids.child_arr[0]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 0);
            element_data[current_index].child_ids.child_arr[1]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 1);
            element_data[current_index].child_ids.child_arr[2]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 2);
            element_data[current_index].child_ids.child_arr[3]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 3);
          }
        }

        double verts[3][3] = { 0 };

        if (level > 0) {
          element_data[current_index].first
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].first;
          element_data[current_index].second
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].second;
          element_data[current_index].third
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].third;

          // Step 1: Create temporary variables to hold the values of the bit-fields
          // Step 1: Create temporary variables as int
          int first_copy = (int) element_data[current_index].first;
          int second_copy = (int) element_data[current_index].second;
          int third_copy = (int) element_data[current_index].third;

          // Step 2: Pass the addresses of the temporary variables to the function
          get_point_order (
            &first_copy, &second_copy, &third_copy,
            t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *) element), level - 1)][current_index % 4]);

          // Step 3: If needed, update the original bit-fields with the modified values
          element_data[current_index].first = (unsigned int) first_copy;
          element_data[current_index].second = (unsigned int) second_copy;
          element_data[current_index].third = (unsigned int) third_copy;

          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                        verts[element_data[current_index].first]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                        verts[element_data[current_index].second]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                        verts[element_data[current_index].third]);
        }
        else {
          element_data[current_index].first = 0;
          element_data[current_index].second = 1;
          element_data[current_index].third = 2;
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                        verts[element_data[current_index].first]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                        verts[element_data[current_index].second]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                        verts[element_data[current_index].third]);
        }
        A.resize (3, 3);
        r.resize (3);
        A (0, 0) = verts[0][0];
        A (0, 1) = verts[1][0];
        A (0, 2) = verts[2][0];
        A (1, 0) = verts[0][1];
        A (1, 1) = verts[1][1];
        A (1, 2) = verts[2][1];
        A (2, 0) = 1;
        A (2, 1) = 1;
        A (2, 2) = 1;
        A.lr_factors (A, r);
        double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
        reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
        for (int i = 0; i < M; ++i) {
          double quad = 0.;
          for (int order = 0; order < order_num; ++order) {
            double x = xytab[order * 2];
            double y = xytab[1 + order * 2];
            vec tau (3);
            tau (0) = x;
            tau (1) = y;
            tau (2) = 1.;
            A.lr_solve (A, r, tau);
            quad += wtab[order] * F (x, y) * sqrt (1. / (2. * volume)) * skalierungsfunktion (i, tau (0), tau (1));
          }
          quad *= volume;
          element_data[current_index].u_coeff[i] = quad;
        }
        element_data[current_index].u_val
          = element_data[current_index].u_coeff[0] * sqrt (1. / (2. * volume)) * skalierungsfunktion (0, 0, 0);
      }
    }
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return element_data;
}

/* The data that we want to store for each element.
 * In this example we want to store the element's level and volume. */
static struct t8_data_per_element_3d *
t8_create_element_data_3d (struct grid_hierarchy_3d initial_grid_hierarchy, int level, func F1, func F2, func F3,
                           int rule, int max_lev)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  mat A;
  vector<int> r;
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  t8_locidx_t num_local_elements;
  t8_locidx_t num_ghost_elements;
  struct t8_data_per_element_3d *element_data;

  /* Check that forest is a committed, that is valid and usable, forest. */
  T8_ASSERT (t8_forest_is_committed (initial_grid_hierarchy.lev_arr[level].forest_arr));

  /* Get the number of local elements of forest. */
  num_local_elements = t8_forest_get_local_num_elements (initial_grid_hierarchy.lev_arr[level].forest_arr);
  /* Get the number of ghost elements of forest. */
  num_ghost_elements = t8_forest_get_num_ghosts (initial_grid_hierarchy.lev_arr[level].forest_arr);

  element_data = T8_ALLOC (struct t8_data_per_element_3d, num_local_elements + num_ghost_elements);  //hier
  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    t8_eclass_t tree_class;
    const t8_scheme *eclass_scheme;
    const t8_element_t *element;
    /* Get the number of trees that have elements of this process. */
    num_local_trees = t8_forest_get_num_local_trees (initial_grid_hierarchy.lev_arr[level].forest_arr);
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      tree_class = t8_forest_get_tree_class (initial_grid_hierarchy.lev_arr[level].forest_arr, itree);
      eclass_scheme = t8_forest_get_scheme (initial_grid_hierarchy.lev_arr[level].forest_arr);
      /* Get the number of elements of this tree. */
      num_elements_in_tree = t8_forest_get_tree_num_elements (initial_grid_hierarchy.lev_arr[level].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {

        element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, ielement);
        double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element);
        element_data[current_index].lmi = t8_elem_id_to_lmi_binary (current_index, level, current_index / pow4[level]);

        if (level == 0) {
          element_data[current_index].adaptiert = false;
          element_data[current_index].child_ids.child_arr[0]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 0);
          element_data[current_index].child_ids.child_arr[1]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 1);
          element_data[current_index].child_ids.child_arr[2]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 2);
          element_data[current_index].child_ids.child_arr[3]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 3);
        }
        else {
          element_data[current_index].Father_id
            = get_parents_t8code_id_binary (current_index, level, current_index / pow4[level]);
          if (level == max_lev) {  //vorher max_lev-1
            element_data[current_index].adaptiert = true;
          }
          else {
            element_data[current_index].adaptiert = false;
            element_data[current_index].child_ids.child_arr[0]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 0);
            element_data[current_index].child_ids.child_arr[1]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 1);
            element_data[current_index].child_ids.child_arr[2]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 2);
            element_data[current_index].child_ids.child_arr[3]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 3);
          }
        }
        double verts[3][3] = { 0 };
        if (level > 0) {
          element_data[current_index].first
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].first;
          element_data[current_index].second
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].second;
          element_data[current_index].third
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].third;

          // Step 1: Create temporary variables to hold the values of the bit-fields
          // Step 1: Create temporary variables as int
          int first_copy = (int) element_data[current_index].first;
          int second_copy = (int) element_data[current_index].second;
          int third_copy = (int) element_data[current_index].third;

          // Step 2: Pass the addresses of the temporary variables to the function
          get_point_order (
            &first_copy, &second_copy, &third_copy,
            t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *) element), level - 1)][current_index % 4]);

          // Step 3: If needed, update the original bit-fields with the modified values
          element_data[current_index].first = (unsigned int) first_copy;
          element_data[current_index].second = (unsigned int) second_copy;
          element_data[current_index].third = (unsigned int) third_copy;

          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                        verts[element_data[current_index].first]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                        verts[element_data[current_index].second]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                        verts[element_data[current_index].third]);
        }
        else {
          element_data[current_index].first = 0;
          element_data[current_index].second = 1;
          element_data[current_index].third = 2;
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                        verts[element_data[current_index].first]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                        verts[element_data[current_index].second]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                        verts[element_data[current_index].third]);
        }
        A.resize (3, 3);
        r.resize (3);
        A (0, 0) = verts[0][0];
        A (0, 1) = verts[1][0];
        A (0, 2) = verts[2][0];
        A (1, 0) = verts[0][1];
        A (1, 1) = verts[1][1];
        A (1, 2) = verts[2][1];
        A (2, 0) = 1;
        A (2, 1) = 1;
        A (2, 2) = 1;
        A.lr_factors (A, r);
        double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
        reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
        for (int i = 0; i < M; ++i) {
          double quad[3] = { 0., 0., 0. };
          for (int order = 0; order < order_num; ++order) {
            double x = xytab[order * 2];
            double y = xytab[1 + order * 2];
            vec tau (3);
            tau (0) = x;
            tau (1) = y;
            tau (2) = 1.;
            A.lr_solve (A, r, tau);
            quad[0] += wtab[order] * F1 (x, y) * sqrt (1. / (2. * volume)) * skalierungsfunktion (i, tau (0), tau (1));
            quad[1] += wtab[order] * F2 (x, y) * sqrt (1. / (2. * volume)) * skalierungsfunktion (i, tau (0), tau (1));
            quad[2] += wtab[order] * F3 (x, y) * sqrt (1. / (2. * volume)) * skalierungsfunktion (i, tau (0), tau (1));
          }
          quad[0] *= volume;
          quad[1] *= volume;
          quad[2] *= volume;
          element_data[current_index].u_coeff_d1[i] = quad[0];
          element_data[current_index].u_coeff_d2[i] = quad[1];
          element_data[current_index].u_coeff_d3[i] = quad[2];
        }
        element_data[current_index].u_val_d1
          = element_data[current_index].u_coeff_d1[0] * sqrt (1. / (2. * volume)) * skalierungsfunktion (0, 0, 0);
        element_data[current_index].u_val_d2
          = element_data[current_index].u_coeff_d2[0] * sqrt (1. / (2. * volume)) * skalierungsfunktion (0, 0, 0);
        element_data[current_index].u_val_d3
          = element_data[current_index].u_coeff_d3[0] * sqrt (1. / (2. * volume)) * skalierungsfunktion (0, 0, 0);
      }
    }
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return element_data;
}

/* The data that we want to store for each element.
 * In this example we want to store the element's level and volume. */
static struct t8_data_per_element_waveletfree *
t8_create_element_data_waveletfree (struct grid_hierarchy_waveletfree initial_grid_hierarchy, int level, func F,
                                    int rule, int max_lev)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  mat A;
  vector<int> r;
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  t8_locidx_t num_local_elements;
  t8_locidx_t num_ghost_elements;
  struct t8_data_per_element_waveletfree *element_data;

  /* Check that forest is a committed, that is valid and usable, forest. */
  T8_ASSERT (t8_forest_is_committed (initial_grid_hierarchy.lev_arr[level].forest_arr));

  /* Get the number of local elements of forest. */
  num_local_elements = t8_forest_get_local_num_elements (initial_grid_hierarchy.lev_arr[level].forest_arr);
  /* Get the number of ghost elements of forest. */
  num_ghost_elements = t8_forest_get_num_ghosts (initial_grid_hierarchy.lev_arr[level].forest_arr);

  element_data = T8_ALLOC (struct t8_data_per_element_waveletfree, num_local_elements + num_ghost_elements);
  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    t8_eclass_t tree_class;
    const t8_scheme *eclass_scheme;
    const t8_element_t *element;
    /* Get the number of trees that have elements of this process. */
    num_local_trees = t8_forest_get_num_local_trees (initial_grid_hierarchy.lev_arr[level].forest_arr);
    // long long int basecell_num_digits_offset=countDigit(t8_forest_get_num_global_trees (forest)-1)-1;
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      tree_class = t8_forest_get_tree_class (initial_grid_hierarchy.lev_arr[level].forest_arr, itree);
      eclass_scheme = t8_forest_get_scheme (initial_grid_hierarchy.lev_arr[level].forest_arr);
      /* Get the number of elements of this tree. */
      num_elements_in_tree = t8_forest_get_tree_num_elements (initial_grid_hierarchy.lev_arr[level].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {

        element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, ielement);
        double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element);
        element_data[current_index].lmi = t8_elem_id_to_lmi_binary (current_index, level, current_index / pow4[level]);
        if (level == 0) {
          element_data[current_index].adaptiert = false;

          element_data[current_index].child_ids.child_arr[0]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 0);
          element_data[current_index].child_ids.child_arr[1]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 1);
          element_data[current_index].child_ids.child_arr[2]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 2);
          element_data[current_index].child_ids.child_arr[3]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 3);
        }
        else {
          element_data[current_index].Father_id = get_parents_t8code_id_binary (
            current_index, level,
            current_index
              / pow4[level]);  //get_parents_t8code_id(element,eclass_scheme,current_index,basecell_num_digits_offset);
          if (level == max_lev) {  //vorher max_lev-1
            element_data[current_index].adaptiert = true;
          }
          else {
            element_data[current_index].adaptiert = false;
            element_data[current_index].child_ids.child_arr[0]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 0);
            element_data[current_index].child_ids.child_arr[1]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 1);
            element_data[current_index].child_ids.child_arr[2]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 2);
            element_data[current_index].child_ids.child_arr[3]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 3);
          }
        }
        double verts[3][3] = { 0 };
        if (level > 0) {
          element_data[current_index].first
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].first;
          element_data[current_index].second
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].second;
          element_data[current_index].third
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].third;
          int first_copy = (int) element_data[current_index].first;
          int second_copy = (int) element_data[current_index].second;
          int third_copy = (int) element_data[current_index].third;

          // Step 2: Pass the addresses of the temporary variables to the function
          get_point_order (
            &first_copy, &second_copy, &third_copy,
            t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *) element), level - 1)][current_index % 4]);

          // Step 3: If needed, update the original bit-fields with the modified values
          element_data[current_index].first = (unsigned int) first_copy;
          element_data[current_index].second = (unsigned int) second_copy;
          element_data[current_index].third = (unsigned int) third_copy;

          //get_point_order(&element_data[current_index].first,&element_data[current_index].second, &element_data[current_index].third, t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *)element), level-1)][current_index%4]);
          //t8_global_productionf ("Cube ID: %i \n", t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *)element), level-1)][current_index%4]);

          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                        verts[element_data[current_index].first]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                        verts[element_data[current_index].second]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                        verts[element_data[current_index].third]);
        }
        else {
          element_data[current_index].first = 0;
          element_data[current_index].second = 1;
          element_data[current_index].third = 2;
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                        verts[element_data[current_index].first]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                        verts[element_data[current_index].second]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                        verts[element_data[current_index].third]);
        }
        A.resize (3, 3);
        r.resize (3);
        A (0, 0) = verts[0][0];
        A (0, 1) = verts[1][0];
        A (0, 2) = verts[2][0];
        A (1, 0) = verts[0][1];
        A (1, 1) = verts[1][1];
        A (1, 2) = verts[2][1];
        A (2, 0) = 1;
        A (2, 1) = 1;
        A (2, 2) = 1;
        A.lr_factors (A, r);
        double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
        reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
        for (int i = 0; i < M; ++i) {
          double quad = 0.;
          for (int order = 0; order < order_num; ++order) {
            double x = xytab[order * 2];
            double y = xytab[1 + order * 2];
            vec tau (3);
            tau (0) = x;
            tau (1) = y;
            tau (2) = 1.;
            A.lr_solve (A, r, tau);
            quad += wtab[order] * F (x, y) * sqrt (1. / (2. * volume)) * skalierungsfunktion (i, tau (0), tau (1));
          }
          quad *= volume;
          element_data[current_index].u_coeff[i] = quad;
        }
        element_data[current_index].u_val
          = element_data[current_index].u_coeff[0] * sqrt (1. / (2. * volume)) * skalierungsfunktion (0, 0, 0);
      }
    }
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return element_data;
}

/* The data that we want to store for each element.
 * In this example we want to store the element's level and volume. */
static struct t8_data_per_element_waveletfree_3d *
t8_create_element_data_waveletfree_3d (struct grid_hierarchy_waveletfree_3d initial_grid_hierarchy, int level, func F1,
                                       func F2, func F3, int rule, int max_lev)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  mat A;
  vector<int> r;
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  t8_locidx_t num_local_elements;
  t8_locidx_t num_ghost_elements;
  struct t8_data_per_element_waveletfree_3d *element_data;

  /* Check that forest is a committed, that is valid and usable, forest. */
  T8_ASSERT (t8_forest_is_committed (initial_grid_hierarchy.lev_arr[level].forest_arr));

  /* Get the number of local elements of forest. */
  num_local_elements = t8_forest_get_local_num_elements (initial_grid_hierarchy.lev_arr[level].forest_arr);
  /* Get the number of ghost elements of forest. */
  num_ghost_elements = t8_forest_get_num_ghosts (initial_grid_hierarchy.lev_arr[level].forest_arr);

  element_data = T8_ALLOC (struct t8_data_per_element_waveletfree_3d, num_local_elements + num_ghost_elements);  //hier

  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    t8_eclass_t tree_class;
    const t8_scheme *eclass_scheme;
    const t8_element_t *element;
    /* Get the number of trees that have elements of this process. */
    num_local_trees = t8_forest_get_num_local_trees (initial_grid_hierarchy.lev_arr[level].forest_arr);
    // long long int basecell_num_digits_offset=countDigit(t8_forest_get_num_global_trees (forest)-1)-1;
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      tree_class = t8_forest_get_tree_class (initial_grid_hierarchy.lev_arr[level].forest_arr, itree);
      eclass_scheme = t8_forest_get_scheme (initial_grid_hierarchy.lev_arr[level].forest_arr);
      /* Get the number of elements of this tree. */
      num_elements_in_tree = t8_forest_get_tree_num_elements (initial_grid_hierarchy.lev_arr[level].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {

        element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, ielement);

        double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element);
        element_data[current_index].lmi = t8_elem_id_to_lmi_binary (
          current_index, level,
          current_index
            / pow4[level]);  //t8_elem_id_to_lmi (element,eclass_scheme,current_index,basecell_num_digits_offset);

        if (level == 0) {
          element_data[current_index].adaptiert = false;

          element_data[current_index].child_ids.child_arr[0]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 0);
          element_data[current_index].child_ids.child_arr[1]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 1);
          element_data[current_index].child_ids.child_arr[2]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 2);
          element_data[current_index].child_ids.child_arr[3]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 3);
        }
        else {
          element_data[current_index].Father_id = get_parents_t8code_id_binary (
            current_index, level,
            current_index
              / pow4[level]);  //get_parents_t8code_id(element,eclass_scheme,current_index,basecell_num_digits_offset);
          if (level == max_lev) {  //vorher max_lev-1
            element_data[current_index].adaptiert = true;
          }
          else {
            element_data[current_index].adaptiert = false;
            element_data[current_index].child_ids.child_arr[0]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 0);
            element_data[current_index].child_ids.child_arr[1]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 1);
            element_data[current_index].child_ids.child_arr[2]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 2);
            element_data[current_index].child_ids.child_arr[3]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 3);
          }
        }
        double verts[3][3] = { 0 };
        if (level > 0) {
          element_data[current_index].first
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].first;
          element_data[current_index].second
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].second;
          element_data[current_index].third
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].third;
          int first_copy = (int) element_data[current_index].first;
          int second_copy = (int) element_data[current_index].second;
          int third_copy = (int) element_data[current_index].third;

          // Step 2: Pass the addresses of the temporary variables to the function
          get_point_order (
            &first_copy, &second_copy, &third_copy,
            t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *) element), level - 1)][current_index % 4]);

          // Step 3: If needed, update the original bit-fields with the modified values
          element_data[current_index].first = (unsigned int) first_copy;
          element_data[current_index].second = (unsigned int) second_copy;
          element_data[current_index].third = (unsigned int) third_copy;

          //get_point_order(&element_data[current_index].first,&element_data[current_index].second, &element_data[current_index].third, t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *)element), level-1)][current_index%4]);
          //t8_global_productionf ("Cube ID: %i \n", t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *)element), level-1)][current_index%4]);

          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                        verts[element_data[current_index].first]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                        verts[element_data[current_index].second]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                        verts[element_data[current_index].third]);
        }
        else {
          element_data[current_index].first = 0;
          element_data[current_index].second = 1;
          element_data[current_index].third = 2;
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                        verts[element_data[current_index].first]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                        verts[element_data[current_index].second]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                        verts[element_data[current_index].third]);
        }
        A.resize (3, 3);
        r.resize (3);
        A (0, 0) = verts[0][0];
        A (0, 1) = verts[1][0];
        A (0, 2) = verts[2][0];
        A (1, 0) = verts[0][1];
        A (1, 1) = verts[1][1];
        A (1, 2) = verts[2][1];
        A (2, 0) = 1;
        A (2, 1) = 1;
        A (2, 2) = 1;
        A.lr_factors (A, r);
        double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
        reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
        for (int i = 0; i < M; ++i) {
          double quad[3] = { 0., 0., 0. };
          for (int order = 0; order < order_num; ++order) {
            double x = xytab[order * 2];
            double y = xytab[1 + order * 2];
            vec tau (3);
            tau (0) = x;
            tau (1) = y;
            tau (2) = 1.;
            A.lr_solve (A, r, tau);
            quad[0] += wtab[order] * F1 (x, y) * sqrt (1. / (2. * volume)) * skalierungsfunktion (i, tau (0), tau (1));
            quad[1] += wtab[order] * F2 (x, y) * sqrt (1. / (2. * volume)) * skalierungsfunktion (i, tau (0), tau (1));
            quad[2] += wtab[order] * F3 (x, y) * sqrt (1. / (2. * volume)) * skalierungsfunktion (i, tau (0), tau (1));
          }
          quad[0] *= volume;
          quad[1] *= volume;
          quad[2] *= volume;
          element_data[current_index].u_coeff_d1[i] = quad[0];
          element_data[current_index].u_coeff_d2[i] = quad[1];
          element_data[current_index].u_coeff_d3[i] = quad[2];
        }
        element_data[current_index].u_val_d1
          = element_data[current_index].u_coeff_d1[0] * sqrt (1. / (2. * volume)) * skalierungsfunktion (0, 0, 0);
        element_data[current_index].u_val_d2
          = element_data[current_index].u_coeff_d2[0] * sqrt (1. / (2. * volume)) * skalierungsfunktion (0, 0, 0);
        element_data[current_index].u_val_d3
          = element_data[current_index].u_coeff_d3[0] * sqrt (1. / (2. * volume)) * skalierungsfunktion (0, 0, 0);
      }
    }
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return element_data;
}

/* Here we evaluate a spline interpolation to calculate the single scale projection. */

static struct t8_data_per_element *
t8_create_element_data_spline (struct grid_hierarchy initial_grid_hierarchy, int level, spline eval_spline,
                               const gsl_spline2d *spline, gsl_interp_accel *xacc, gsl_interp_accel *yacc, int rule,
                               int max_lev)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  mat A;
  vector<int> r;
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  t8_locidx_t num_local_elements;
  t8_locidx_t num_ghost_elements;
  struct t8_data_per_element *element_data;

  T8_ASSERT (t8_forest_is_committed (initial_grid_hierarchy.lev_arr[level].forest_arr));

  num_local_elements = t8_forest_get_local_num_elements (initial_grid_hierarchy.lev_arr[level].forest_arr);
  num_ghost_elements = t8_forest_get_num_ghosts (initial_grid_hierarchy.lev_arr[level].forest_arr);

  element_data = T8_ALLOC (struct t8_data_per_element, num_local_elements + num_ghost_elements);  //hier

  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    t8_eclass_t tree_class;
    const t8_scheme *eclass_scheme;
    const t8_element_t *element;
    num_local_trees = t8_forest_get_num_local_trees (initial_grid_hierarchy.lev_arr[level].forest_arr);
    // long long int basecell_num_digits_offset=countDigit(t8_forest_get_num_global_trees (forest)-1)-1;
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      tree_class = t8_forest_get_tree_class (initial_grid_hierarchy.lev_arr[level].forest_arr, itree);
      eclass_scheme = t8_forest_get_scheme (initial_grid_hierarchy.lev_arr[level].forest_arr);
      num_elements_in_tree = t8_forest_get_tree_num_elements (initial_grid_hierarchy.lev_arr[level].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, ielement);
        double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element);
        element_data[current_index].lmi = t8_elem_id_to_lmi_binary (
          current_index, level,
          current_index
            / pow4[level]);  //t8_elem_id_to_lmi (element,eclass_scheme,current_index,basecell_num_digits_offset);

        if (level == 0) {
          element_data[current_index].adaptiert = false;

          element_data[current_index].child_ids.child_arr[0]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 0);
          element_data[current_index].child_ids.child_arr[1]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 1);
          element_data[current_index].child_ids.child_arr[2]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 2);
          element_data[current_index].child_ids.child_arr[3]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 3);
        }
        else {
          element_data[current_index].Father_id = get_parents_t8code_id_binary (
            current_index, level,
            current_index
              / pow4[level]);  //get_parents_t8code_id(element,eclass_scheme,current_index,basecell_num_digits_offset);
          if (level == max_lev) {  //vorher max_lev-1
            element_data[current_index].adaptiert = true;
          }
          else {
            element_data[current_index].adaptiert = false;
            element_data[current_index].child_ids.child_arr[0]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 0);
            element_data[current_index].child_ids.child_arr[1]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 1);
            element_data[current_index].child_ids.child_arr[2]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 2);
            element_data[current_index].child_ids.child_arr[3]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 3);
          }
        }
        double verts[3][3] = { 0 };
        if (level > 0) {
          element_data[current_index].first
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].first;
          element_data[current_index].second
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].second;
          element_data[current_index].third
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].third;
          int first_copy = (int) element_data[current_index].first;
          int second_copy = (int) element_data[current_index].second;
          int third_copy = (int) element_data[current_index].third;

          // Step 2: Pass the addresses of the temporary variables to the function
          get_point_order (
            &first_copy, &second_copy, &third_copy,
            t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *) element), level - 1)][current_index % 4]);

          // Step 3: If needed, update the original bit-fields with the modified values
          element_data[current_index].first = (unsigned int) first_copy;
          element_data[current_index].second = (unsigned int) second_copy;
          element_data[current_index].third = (unsigned int) third_copy;

          //get_point_order(&element_data[current_index].first,&element_data[current_index].second, &element_data[current_index].third, t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *)element), level-1)][current_index%4]);
          //t8_global_productionf ("Cube ID: %i \n", t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *)element), level-1)][current_index%4]);

          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                        verts[element_data[current_index].first]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                        verts[element_data[current_index].second]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                        verts[element_data[current_index].third]);
        }
        else {
          element_data[current_index].first = 0;
          element_data[current_index].second = 1;
          element_data[current_index].third = 2;
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                        verts[element_data[current_index].first]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                        verts[element_data[current_index].second]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                        verts[element_data[current_index].third]);
        }
        A.resize (3, 3);
        r.resize (3);
        A (0, 0) = verts[0][0];
        A (0, 1) = verts[1][0];
        A (0, 2) = verts[2][0];
        A (1, 0) = verts[0][1];
        A (1, 1) = verts[1][1];
        A (1, 2) = verts[2][1];
        A (2, 0) = 1;
        A (2, 1) = 1;
        A (2, 2) = 1;
        A.lr_factors (A, r);
        double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
        reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
        for (int i = 0; i < M; ++i) {
          double quad = 0.;
          for (int order = 0; order < order_num; ++order) {
            double x = xytab[order * 2];
            double y = xytab[1 + order * 2];
            vec tau (3);
            tau (0) = x;
            tau (1) = y;
            tau (2) = 1.;
            A.lr_solve (A, r, tau);
            quad += wtab[order] * eval_spline (spline, x, y, xacc, yacc) * sqrt (1. / (2. * volume))
                    * skalierungsfunktion (i, tau (0), tau (1));
          }
          quad *= volume;
          element_data[current_index].u_coeff[i] = quad;
        }
        element_data[current_index].u_val
          = element_data[current_index].u_coeff[0] * sqrt (1. / (2. * volume)) * skalierungsfunktion (0, 0, 0);
      }
    }
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return element_data;
}

/* The data that we want to store for each element.
 * In this example we want to store the element's level and volume. */
static struct t8_data_per_element_3d *
t8_create_element_data_3d_spline (struct grid_hierarchy_3d initial_grid_hierarchy, int level, spline eval_spline,
                                  const gsl_spline2d *spline_d1, const gsl_spline2d *spline_d2,
                                  const gsl_spline2d *spline_d3, gsl_interp_accel *xacc_d1, gsl_interp_accel *yacc_d1,
                                  gsl_interp_accel *xacc_d2, gsl_interp_accel *yacc_d2, gsl_interp_accel *xacc_d3,
                                  gsl_interp_accel *yacc_d3, int rule, int max_lev)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  mat A;
  vector<int> r;
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  t8_locidx_t num_local_elements;
  t8_locidx_t num_ghost_elements;
  struct t8_data_per_element_3d *element_data;
  T8_ASSERT (t8_forest_is_committed (initial_grid_hierarchy.lev_arr[level].forest_arr));

  num_local_elements = t8_forest_get_local_num_elements (initial_grid_hierarchy.lev_arr[level].forest_arr);
  num_ghost_elements = t8_forest_get_num_ghosts (initial_grid_hierarchy.lev_arr[level].forest_arr);
  element_data = T8_ALLOC (struct t8_data_per_element_3d, num_local_elements + num_ghost_elements);
  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    t8_eclass_t tree_class;
    const t8_scheme *eclass_scheme;
    const t8_element_t *element;
    /* Get the number of trees that have elements of this process. */
    num_local_trees = t8_forest_get_num_local_trees (initial_grid_hierarchy.lev_arr[level].forest_arr);
    // long long int basecell_num_digits_offset=countDigit(t8_forest_get_num_global_trees (forest)-1)-1;
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      tree_class = t8_forest_get_tree_class (initial_grid_hierarchy.lev_arr[level].forest_arr, itree);
      eclass_scheme = t8_forest_get_scheme (initial_grid_hierarchy.lev_arr[level].forest_arr);
      /* Get the number of elements of this tree. */
      num_elements_in_tree = t8_forest_get_tree_num_elements (initial_grid_hierarchy.lev_arr[level].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, ielement);
        double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element);
        element_data[current_index].lmi = t8_elem_id_to_lmi_binary (
          current_index, level,
          current_index
            / pow4[level]);  //t8_elem_id_to_lmi (element,eclass_scheme,current_index,basecell_num_digits_offset);

        if (level == 0) {
          element_data[current_index].adaptiert = false;

          element_data[current_index].child_ids.child_arr[0]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 0);
          element_data[current_index].child_ids.child_arr[1]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 1);
          element_data[current_index].child_ids.child_arr[2]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 2);
          element_data[current_index].child_ids.child_arr[3]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 3);
        }
        else {
          element_data[current_index].Father_id = get_parents_t8code_id_binary (
            current_index, level,
            current_index
              / pow4[level]);  //get_parents_t8code_id(element,eclass_scheme,current_index,basecell_num_digits_offset);
          if (level == max_lev) {  //vorher max_lev-1
            element_data[current_index].adaptiert = true;
          }
          else {
            element_data[current_index].adaptiert = false;
            element_data[current_index].child_ids.child_arr[0]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 0);
            element_data[current_index].child_ids.child_arr[1]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 1);
            element_data[current_index].child_ids.child_arr[2]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 2);
            element_data[current_index].child_ids.child_arr[3]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 3);
          }
        }
        double verts[3][3] = { 0 };
        if (level > 0) {
          element_data[current_index].first
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].first;
          element_data[current_index].second
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].second;
          element_data[current_index].third
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].third;
          int first_copy = (int) element_data[current_index].first;
          int second_copy = (int) element_data[current_index].second;
          int third_copy = (int) element_data[current_index].third;

          // Step 2: Pass the addresses of the temporary variables to the function
          get_point_order (
            &first_copy, &second_copy, &third_copy,
            t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *) element), level - 1)][current_index % 4]);

          // Step 3: If needed, update the original bit-fields with the modified values
          element_data[current_index].first = (unsigned int) first_copy;
          element_data[current_index].second = (unsigned int) second_copy;
          element_data[current_index].third = (unsigned int) third_copy;

          //get_point_order(&element_data[current_index].first,&element_data[current_index].second, &element_data[current_index].third, t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *)element), level-1)][current_index%4]);
          //t8_global_productionf ("Cube ID: %i \n", t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *)element), level-1)][current_index%4]);

          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                        verts[element_data[current_index].first]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                        verts[element_data[current_index].second]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                        verts[element_data[current_index].third]);
        }
        else {
          element_data[current_index].first = 0;
          element_data[current_index].second = 1;
          element_data[current_index].third = 2;
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                        verts[element_data[current_index].first]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                        verts[element_data[current_index].second]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                        verts[element_data[current_index].third]);
        }
        A.resize (3, 3);
        r.resize (3);
        A (0, 0) = verts[0][0];
        A (0, 1) = verts[1][0];
        A (0, 2) = verts[2][0];
        A (1, 0) = verts[0][1];
        A (1, 1) = verts[1][1];
        A (1, 2) = verts[2][1];
        A (2, 0) = 1;
        A (2, 1) = 1;
        A (2, 2) = 1;
        A.lr_factors (A, r);
        double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
        reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
        for (int i = 0; i < M; ++i) {
          double quad[3] = { 0., 0., 0. };
          for (int order = 0; order < order_num; ++order) {
            double x = xytab[order * 2];
            double y = xytab[1 + order * 2];
            vec tau (3);
            tau (0) = x;
            tau (1) = y;
            tau (2) = 1.;
            A.lr_solve (A, r, tau);
            quad[0] += wtab[order] * eval_spline (spline_d1, x, y, xacc_d1, yacc_d1) * sqrt (1. / (2. * volume))
                       * skalierungsfunktion (i, tau (0), tau (1));
            quad[1] += wtab[order] * eval_spline (spline_d2, x, y, xacc_d2, yacc_d2) * sqrt (1. / (2. * volume))
                       * skalierungsfunktion (i, tau (0), tau (1));
            quad[2] += wtab[order] * eval_spline (spline_d3, x, y, xacc_d3, yacc_d3) * sqrt (1. / (2. * volume))
                       * skalierungsfunktion (i, tau (0), tau (1));
          }
          quad[0] *= volume;
          quad[1] *= volume;
          quad[2] *= volume;
          element_data[current_index].u_coeff_d1[i] = quad[0];
          element_data[current_index].u_coeff_d2[i] = quad[1];
          element_data[current_index].u_coeff_d3[i] = quad[2];
        }
        element_data[current_index].u_val_d1
          = element_data[current_index].u_coeff_d1[0] * sqrt (1. / (2. * volume)) * skalierungsfunktion (0, 0, 0);
        element_data[current_index].u_val_d2
          = element_data[current_index].u_coeff_d2[0] * sqrt (1. / (2. * volume)) * skalierungsfunktion (0, 0, 0);
        element_data[current_index].u_val_d3
          = element_data[current_index].u_coeff_d3[0] * sqrt (1. / (2. * volume)) * skalierungsfunktion (0, 0, 0);
      }
    }
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return element_data;
}

static struct t8_data_per_element_waveletfree *
t8_create_element_data_waveletfree_spline (struct grid_hierarchy_waveletfree initial_grid_hierarchy, int level,
                                           spline eval_spline, const gsl_spline2d *spline, gsl_interp_accel *xacc,
                                           gsl_interp_accel *yacc, int rule, int max_lev)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  mat A;
  vector<int> r;
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  t8_locidx_t num_local_elements;
  t8_locidx_t num_ghost_elements;
  struct t8_data_per_element_waveletfree *element_data;

  /* Check that forest is a committed, that is valid and usable, forest. */
  T8_ASSERT (t8_forest_is_committed (initial_grid_hierarchy.lev_arr[level].forest_arr));

  /* Get the number of local elements of forest. */
  num_local_elements = t8_forest_get_local_num_elements (initial_grid_hierarchy.lev_arr[level].forest_arr);
  /* Get the number of ghost elements of forest. */
  num_ghost_elements = t8_forest_get_num_ghosts (initial_grid_hierarchy.lev_arr[level].forest_arr);

  element_data = T8_ALLOC (struct t8_data_per_element_waveletfree, num_local_elements + num_ghost_elements);
  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    t8_eclass_t tree_class;
    const t8_scheme *eclass_scheme;
    const t8_element_t *element;
    /* Get the number of trees that have elements of this process. */
    num_local_trees = t8_forest_get_num_local_trees (initial_grid_hierarchy.lev_arr[level].forest_arr);
    // long long int basecell_num_digits_offset=countDigit(t8_forest_get_num_global_trees (forest)-1)-1;
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      tree_class = t8_forest_get_tree_class (initial_grid_hierarchy.lev_arr[level].forest_arr, itree);
      eclass_scheme = t8_forest_get_scheme (initial_grid_hierarchy.lev_arr[level].forest_arr);
      /* Get the number of elements of this tree. */
      num_elements_in_tree = t8_forest_get_tree_num_elements (initial_grid_hierarchy.lev_arr[level].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, ielement);

        double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element);
        element_data[current_index].lmi = t8_elem_id_to_lmi_binary (
          current_index, level,
          current_index
            / pow4[level]);  //t8_elem_id_to_lmi (element,eclass_scheme,current_index,basecell_num_digits_offset);

        if (level == 0) {
          element_data[current_index].adaptiert = false;
          element_data[current_index].child_ids.child_arr[0]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 0);
          element_data[current_index].child_ids.child_arr[1]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 1);
          element_data[current_index].child_ids.child_arr[2]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 2);
          element_data[current_index].child_ids.child_arr[3]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 3);
        }
        else {
          element_data[current_index].Father_id = get_parents_t8code_id_binary (
            current_index, level,
            current_index
              / pow4[level]);  //get_parents_t8code_id(element,eclass_scheme,current_index,basecell_num_digits_offset);
          if (level == max_lev) {  //vorher max_lev-1
            element_data[current_index].adaptiert = true;
          }
          else {
            element_data[current_index].adaptiert = false;
            element_data[current_index].child_ids.child_arr[0]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 0);
            element_data[current_index].child_ids.child_arr[1]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 1);
            element_data[current_index].child_ids.child_arr[2]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 2);
            element_data[current_index].child_ids.child_arr[3]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 3);
          }
        }
        double verts[3][3] = { 0 };
        if (level > 0) {
          element_data[current_index].first
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].first;
          element_data[current_index].second
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].second;
          element_data[current_index].third
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].third;
          int first_copy = (int) element_data[current_index].first;
          int second_copy = (int) element_data[current_index].second;
          int third_copy = (int) element_data[current_index].third;

          // Step 2: Pass the addresses of the temporary variables to the function
          get_point_order (
            &first_copy, &second_copy, &third_copy,
            t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *) element), level - 1)][current_index % 4]);

          // Step 3: If needed, update the original bit-fields with the modified values
          element_data[current_index].first = (unsigned int) first_copy;
          element_data[current_index].second = (unsigned int) second_copy;
          element_data[current_index].third = (unsigned int) third_copy;

          //get_point_order(&element_data[current_index].first,&element_data[current_index].second, &element_data[current_index].third, t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *)element), level-1)][current_index%4]);
          //t8_global_productionf ("Cube ID: %i \n", t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *)element), level-1)][current_index%4]);

          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                        verts[element_data[current_index].first]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                        verts[element_data[current_index].second]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                        verts[element_data[current_index].third]);
        }
        else {
          element_data[current_index].first = 0;
          element_data[current_index].second = 1;
          element_data[current_index].third = 2;
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                        verts[element_data[current_index].first]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                        verts[element_data[current_index].second]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                        verts[element_data[current_index].third]);
        }
        A.resize (3, 3);
        r.resize (3);
        A (0, 0) = verts[0][0];
        A (0, 1) = verts[1][0];
        A (0, 2) = verts[2][0];
        A (1, 0) = verts[0][1];
        A (1, 1) = verts[1][1];
        A (1, 2) = verts[2][1];
        A (2, 0) = 1;
        A (2, 1) = 1;
        A (2, 2) = 1;
        A.lr_factors (A, r);
        double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
        reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
        for (int i = 0; i < M; ++i) {
          double quad = 0.;
          for (int order = 0; order < order_num; ++order) {
            double x = xytab[order * 2];
            double y = xytab[1 + order * 2];
            vec tau (3);
            tau (0) = x;
            tau (1) = y;
            tau (2) = 1.;
            A.lr_solve (A, r, tau);
            quad += wtab[order] * eval_spline (spline, x, y, xacc, yacc) * sqrt (1. / (2. * volume))
                    * skalierungsfunktion (i, tau (0), tau (1));
          }
          quad *= volume;
          element_data[current_index].u_coeff[i] = quad;
        }
        element_data[current_index].u_val
          = element_data[current_index].u_coeff[0] * sqrt (1. / (2. * volume)) * skalierungsfunktion (0, 0, 0);
      }
    }
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return element_data;
}

static struct t8_data_per_element_waveletfree_3d *
t8_create_element_data_waveletfree_3d_spline (struct grid_hierarchy_waveletfree_3d initial_grid_hierarchy, int level,
                                              spline eval_spline, const gsl_spline2d *spline_d1,
                                              const gsl_spline2d *spline_d2, const gsl_spline2d *spline_d3,
                                              gsl_interp_accel *xacc_d1, gsl_interp_accel *yacc_d1,
                                              gsl_interp_accel *xacc_d2, gsl_interp_accel *yacc_d2,
                                              gsl_interp_accel *xacc_d3, gsl_interp_accel *yacc_d3, int rule,
                                              int max_lev)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  mat A;
  vector<int> r;
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  t8_locidx_t num_local_elements;
  t8_locidx_t num_ghost_elements;
  struct t8_data_per_element_waveletfree_3d *element_data;

  /* Check that forest is a committed, that is valid and usable, forest. */
  T8_ASSERT (t8_forest_is_committed (initial_grid_hierarchy.lev_arr[level].forest_arr));

  /* Get the number of local elements of forest. */
  num_local_elements = t8_forest_get_local_num_elements (initial_grid_hierarchy.lev_arr[level].forest_arr);
  /* Get the number of ghost elements of forest. */
  num_ghost_elements = t8_forest_get_num_ghosts (initial_grid_hierarchy.lev_arr[level].forest_arr);

  element_data = T8_ALLOC (struct t8_data_per_element_waveletfree_3d, num_local_elements + num_ghost_elements);  //hier

  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    t8_eclass_t tree_class;
    const t8_scheme *eclass_scheme;
    const t8_element_t *element;
    /* Get the number of trees that have elements of this process. */
    num_local_trees = t8_forest_get_num_local_trees (initial_grid_hierarchy.lev_arr[level].forest_arr);
    // long long int basecell_num_digits_offset=countDigit(t8_forest_get_num_global_trees (forest)-1)-1;
    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {

      tree_class = t8_forest_get_tree_class (initial_grid_hierarchy.lev_arr[level].forest_arr, itree);
      eclass_scheme = t8_forest_get_scheme (initial_grid_hierarchy.lev_arr[level].forest_arr);
      /* Get the number of elements of this tree. */
      num_elements_in_tree = t8_forest_get_tree_num_elements (initial_grid_hierarchy.lev_arr[level].forest_arr, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {

        element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, ielement);

        double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element);
        element_data[current_index].lmi = t8_elem_id_to_lmi_binary (
          current_index, level,
          current_index
            / pow4[level]);  //t8_elem_id_to_lmi (element,eclass_scheme,current_index,basecell_num_digits_offset);

        if (level == 0) {
          element_data[current_index].adaptiert = false;

          element_data[current_index].child_ids.child_arr[0]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 0);
          element_data[current_index].child_ids.child_arr[1]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 1);
          element_data[current_index].child_ids.child_arr[2]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 2);
          element_data[current_index].child_ids.child_arr[3]
            = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 3);
        }
        else {
          element_data[current_index].Father_id = get_parents_t8code_id_binary (
            current_index, level,
            current_index
              / pow4[level]);  //get_parents_t8code_id(element,eclass_scheme,current_index,basecell_num_digits_offset);
          if (level == max_lev) {  //vorher max_lev-1
            element_data[current_index].adaptiert = true;
          }
          else {
            element_data[current_index].adaptiert = false;
            element_data[current_index].child_ids.child_arr[0]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 0);
            element_data[current_index].child_ids.child_arr[1]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 1);
            element_data[current_index].child_ids.child_arr[2]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 2);
            element_data[current_index].child_ids.child_arr[3]
              = get_jth_child_t8code_id_binary (current_index, level, current_index / pow4[level], 3);
          }
        }
        double verts[3][3] = { 0 };
        if (level > 0) {
          element_data[current_index].first
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].first;
          element_data[current_index].second
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].second;
          element_data[current_index].third
            = initial_grid_hierarchy.lev_arr[level - 1].data_arr[element_data[current_index].Father_id].third;
          int first_copy = (int) element_data[current_index].first;
          int second_copy = (int) element_data[current_index].second;
          int third_copy = (int) element_data[current_index].third;

          // Step 2: Pass the addresses of the temporary variables to the function
          get_point_order (
            &first_copy, &second_copy, &third_copy,
            t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *) element), level - 1)][current_index % 4]);

          // Step 3: If needed, update the original bit-fields with the modified values
          element_data[current_index].first = (unsigned int) first_copy;
          element_data[current_index].second = (unsigned int) second_copy;
          element_data[current_index].third = (unsigned int) third_copy;

          //get_point_order(&element_data[current_index].first,&element_data[current_index].second, &element_data[current_index].third, t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *)element), level-1)][current_index%4]);
          //t8_global_productionf ("Cube ID: %i \n", t8_dtri_type_cid_to_beyid[compute_type (((t8_dtri_t *)element), level-1)][current_index%4]);

          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                        verts[element_data[current_index].first]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                        verts[element_data[current_index].second]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                        verts[element_data[current_index].third]);
        }
        else {
          element_data[current_index].first = 0;
          element_data[current_index].second = 1;
          element_data[current_index].third = 2;
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                        verts[element_data[current_index].first]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                        verts[element_data[current_index].second]);
          t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                        verts[element_data[current_index].third]);
        }
        A.resize (3, 3);
        r.resize (3);
        A (0, 0) = verts[0][0];
        A (0, 1) = verts[1][0];
        A (0, 2) = verts[2][0];
        A (1, 0) = verts[0][1];
        A (1, 1) = verts[1][1];
        A (1, 2) = verts[2][1];
        A (2, 0) = 1;
        A (2, 1) = 1;
        A (2, 2) = 1;
        A.lr_factors (A, r);
        double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
        reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
        for (int i = 0; i < M; ++i) {
          double quad[3] = { 0., 0., 0. };
          for (int order = 0; order < order_num; ++order) {
            double x = xytab[order * 2];
            double y = xytab[1 + order * 2];
            vec tau (3);
            tau (0) = x;
            tau (1) = y;
            tau (2) = 1.;
            A.lr_solve (A, r, tau);
            quad[0] += wtab[order] * eval_spline (spline_d1, x, y, xacc_d1, yacc_d1) * sqrt (1. / (2. * volume))
                       * skalierungsfunktion (i, tau (0), tau (1));
            quad[1] += wtab[order] * eval_spline (spline_d2, x, y, xacc_d2, yacc_d2) * sqrt (1. / (2. * volume))
                       * skalierungsfunktion (i, tau (0), tau (1));
            quad[2] += wtab[order] * eval_spline (spline_d3, x, y, xacc_d3, yacc_d3) * sqrt (1. / (2. * volume))
                       * skalierungsfunktion (i, tau (0), tau (1));
          }
          quad[0] *= volume;
          quad[1] *= volume;
          quad[2] *= volume;
          element_data[current_index].u_coeff_d1[i] = quad[0];
          element_data[current_index].u_coeff_d2[i] = quad[1];
          element_data[current_index].u_coeff_d3[i] = quad[2];
        }
        element_data[current_index].u_val_d1
          = element_data[current_index].u_coeff_d1[0] * sqrt (1. / (2. * volume)) * skalierungsfunktion (0, 0, 0);
        element_data[current_index].u_val_d2
          = element_data[current_index].u_coeff_d2[0] * sqrt (1. / (2. * volume)) * skalierungsfunktion (0, 0, 0);
        element_data[current_index].u_val_d3
          = element_data[current_index].u_coeff_d3[0] * sqrt (1. / (2. * volume)) * skalierungsfunktion (0, 0, 0);
      }
    }
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return element_data;
}

/* Each process has computed the data entries for its local elements.
 * In order to get the values for the ghost elements, we use t8_forest_ghost_exchange_data.
 * Calling this function will fill all the ghost entries of our element data array with the
 * value on the process that owns the corresponding element. */
// static void
// t8_step5_exchange_ghost_data (t8_forest_t forest, struct t8_data_per_element *data)
// {
//   sc_array *sc_array_wrapper;
//   t8_locidx_t num_elements = t8_forest_get_local_num_elements (forest);
//   t8_locidx_t num_ghosts = t8_forest_get_num_ghosts (forest);
//
//   /* t8_forest_ghost_exchange_data expects an sc_array (of length num_local_elements + num_ghosts).
//    * We wrap our data array to an sc_array. */
//   sc_array_wrapper = sc_array_new_data (data, sizeof (struct t8_data_per_element), num_elements + num_ghosts);
//
//   /* Carry out the data exchange. The entries with indices > num_local_elements will get overwritten.
//    */
//   t8_forest_ghost_exchange_data (forest, sc_array_wrapper);
//
//   /* Destroy the wrapper array. This will not free the data memory since we used sc_array_new_data. */
//   sc_array_destroy (sc_array_wrapper);
// }

/* Write the forest as vtu and also write the element's volumes in the file.
 *
 * t8code supports writing element based data to vtu as long as its stored
 * as doubles. Each of the data fields to write has to be provided in its own
 * array of length num_local_elements.
 * We support two types: T8_VTK_SCALAR - One double per element
 *                  and  T8_VTK_VECTOR - 3 doubles per element
 */
static void
t8_output_data_to_vtu (t8_forest_t forest, struct t8_data_per_element *data, const char *prefix)
{
  t8_locidx_t num_elements = t8_forest_get_local_num_elements (forest);
  t8_locidx_t ielem;
  /* We need to allocate a new array to store the volumes on their own.
   * This array has one entry per local element. */
  double *element_volumes = T8_ALLOC (double, num_elements);
  /* The number of user defined data fields to write. */
  int num_data = 1;
  /* For each user defined data field we need one t8_vtk_data_field_t variable */
  t8_vtk_data_field_t vtk_data;
  /* Set the type of this variable. Since we have one value per element, we pick T8_VTK_SCALAR */
  vtk_data.type = T8_VTK_SCALAR;
  /* The name of the field as should be written to the file. */
  strcpy (vtk_data.description, "Element average u evaluated");
  vtk_data.data = element_volumes;
  /* Copy the element's volumes from our data array to the output array. */
  for (ielem = 0; ielem < num_elements; ++ielem) {
    //t8_global_productionf ("u_val:%f\n",data[ielem].u_val);
    element_volumes[ielem] = data[ielem].u_val;
  }
  {
    /* To write user defined data, we need to extended output function t8_forest_vtk_write_file
     * from t8_forest_vtk.h. Despite writin user data, it also offers more control over which
     * properties of the forest to write. */
    int write_treeid = 1;
    int write_mpirank = 1;
    int write_level = 1;
    int write_element_id = 1;
    int write_ghosts = 0;
    t8_forest_write_vtk_ext (forest, prefix, write_treeid, write_mpirank, write_level, write_element_id, write_ghosts,
                             0, 0, num_data, &vtk_data);
  }
  T8_FREE (element_volumes);
}

double
AuswertungSinglescale (t8_forest_t forest, struct t8_data_per_element *element_data, double x, double y,
                       t8_locidx_t itree, t8_locidx_t ielement, t8_locidx_t current_index)
{
  vec tau (3);
  tau (0) = x;
  tau (1) = y;
  tau (2) = 1.;
  mat A;
  vector<int> r;
  const t8_element_t *element;
  element = t8_forest_get_element_in_tree (forest, itree, ielement);
  double volume = t8_forest_element_volume (forest, itree, element);
  double verts[3][3] = { 0 };
  /*
  t8_forest_element_coordinate (forest, itree,element,  0,
                        verts[0]);
  t8_forest_element_coordinate (forest, itree,element,  1,
                        verts[1]);
  t8_forest_element_coordinate (forest, itree,element,  2,
                        verts[2]);
  */
  t8_forest_element_coordinate (forest, itree, element, 0, verts[element_data[current_index].first]);
  t8_forest_element_coordinate (forest, itree, element, 1, verts[element_data[current_index].second]);
  t8_forest_element_coordinate (forest, itree, element, 2, verts[element_data[current_index].third]);

  A.resize (3, 3);
  r.resize (3);
  A (0, 0) = verts[0][0];
  A (0, 1) = verts[1][0];
  A (0, 2) = verts[2][0];
  A (1, 0) = verts[0][1];
  A (1, 1) = verts[1][1];
  A (1, 2) = verts[2][1];
  A (2, 0) = 1;
  A (2, 1) = 1;
  A (2, 2) = 1;
  A.lr_factors (A, r);
  A.lr_solve (A, r, tau);
  double sum = 0.;
  for (int i = 0; i < M; ++i) {
    sum
      += element_data[current_index].u_coeff[i] * sqrt (1. / (2. * volume)) * skalierungsfunktion (i, tau (0), tau (1));
  }
  return sum;
}

struct double_3d_array
AuswertungSinglescale3d (t8_forest_t forest, struct t8_data_per_element_3d *element_data, double x, double y,
                         t8_locidx_t itree, t8_locidx_t ielement, t8_locidx_t current_index)
{
  vec tau (3);
  tau (0) = x;
  tau (1) = y;
  tau (2) = 1.;
  mat A;
  vector<int> r;
  const t8_element_t *element;
  element = t8_forest_get_element_in_tree (forest, itree, ielement);
  double volume = t8_forest_element_volume (forest, itree, element);
  double verts[3][3] = { 0 };
  t8_forest_element_coordinate (forest, itree, element, 0, verts[element_data[current_index].first]);
  t8_forest_element_coordinate (forest, itree, element, 1, verts[element_data[current_index].second]);
  t8_forest_element_coordinate (forest, itree, element, 2, verts[element_data[current_index].third]);
  A.resize (3, 3);
  r.resize (3);
  A (0, 0) = verts[0][0];
  A (0, 1) = verts[1][0];
  A (0, 2) = verts[2][0];
  A (1, 0) = verts[0][1];
  A (1, 1) = verts[1][1];
  A (1, 2) = verts[2][1];
  A (2, 0) = 1;
  A (2, 1) = 1;
  A (2, 2) = 1;
  A.lr_factors (A, r);
  A.lr_solve (A, r, tau);
  struct double_3d_array sum;
  sum.dim_val[0] = 0.;
  sum.dim_val[1] = 0.;
  sum.dim_val[2] = 0.;
  for (int i = 0; i < M; ++i) {
    sum.dim_val[0] += element_data[current_index].u_coeff_d1[i] * sqrt (1. / (2. * volume))
                      * skalierungsfunktion (i, tau (0), tau (1));
    sum.dim_val[1] += element_data[current_index].u_coeff_d2[i] * sqrt (1. / (2. * volume))
                      * skalierungsfunktion (i, tau (0), tau (1));
    sum.dim_val[2] += element_data[current_index].u_coeff_d3[i] * sqrt (1. / (2. * volume))
                      * skalierungsfunktion (i, tau (0), tau (1));
  }
  return sum;
}

double
AuswertungSinglescaleWaveletfree (t8_forest_t forest, struct t8_data_per_element_waveletfree *element_data, double x,
                                  double y, t8_locidx_t itree, t8_locidx_t ielement, t8_locidx_t current_index)
{
  vec tau (3);
  tau (0) = x;
  tau (1) = y;
  tau (2) = 1.;
  mat A;
  vector<int> r;
  const t8_element_t *element;
  element = t8_forest_get_element_in_tree (forest, itree, ielement);
  double volume = t8_forest_element_volume (forest, itree, element);
  double verts[3][3] = { 0 };
  t8_forest_element_coordinate (forest, itree, element, 0, verts[element_data[current_index].first]);
  t8_forest_element_coordinate (forest, itree, element, 1, verts[element_data[current_index].second]);
  t8_forest_element_coordinate (forest, itree, element, 2, verts[element_data[current_index].third]);
  A.resize (3, 3);
  r.resize (3);
  A (0, 0) = verts[0][0];
  A (0, 1) = verts[1][0];
  A (0, 2) = verts[2][0];
  A (1, 0) = verts[0][1];
  A (1, 1) = verts[1][1];
  A (1, 2) = verts[2][1];
  A (2, 0) = 1;
  A (2, 1) = 1;
  A (2, 2) = 1;
  A.lr_factors (A, r);
  A.lr_solve (A, r, tau);
  double sum = 0.;
  for (int i = 0; i < M; ++i) {
    sum
      += element_data[current_index].u_coeff[i] * sqrt (1. / (2. * volume)) * skalierungsfunktion (i, tau (0), tau (1));
  }
  return sum;
}

struct double_3d_array
AuswertungSinglescaleWaveletfree3d (t8_forest_t forest, struct t8_data_per_element_waveletfree_3d *element_data,
                                    double x, double y, t8_locidx_t itree, t8_locidx_t ielement,
                                    t8_locidx_t current_index)
{
  vec tau (3);
  tau (0) = x;
  tau (1) = y;
  tau (2) = 1.;
  mat A;
  vector<int> r;
  const t8_element_t *element;
  element = t8_forest_get_element_in_tree (forest, itree, ielement);
  double volume = t8_forest_element_volume (forest, itree, element);
  double verts[3][3] = { 0 };
  t8_forest_element_coordinate (forest, itree, element, 0, verts[element_data[current_index].first]);
  t8_forest_element_coordinate (forest, itree, element, 1, verts[element_data[current_index].second]);
  t8_forest_element_coordinate (forest, itree, element, 2, verts[element_data[current_index].third]);
  A.resize (3, 3);
  r.resize (3);
  A (0, 0) = verts[0][0];
  A (0, 1) = verts[1][0];
  A (0, 2) = verts[2][0];
  A (1, 0) = verts[0][1];
  A (1, 1) = verts[1][1];
  A (1, 2) = verts[2][1];
  A (2, 0) = 1;
  A (2, 1) = 1;
  A (2, 2) = 1;
  A.lr_factors (A, r);
  A.lr_solve (A, r, tau);
  struct double_3d_array sum;
  sum.dim_val[0] = 0.;
  sum.dim_val[1] = 0.;
  sum.dim_val[2] = 0.;
  for (int i = 0; i < M; ++i) {
    sum.dim_val[0] += element_data[current_index].u_coeff_d1[i] * sqrt (1. / (2. * volume))
                      * skalierungsfunktion (i, tau (0), tau (1));
    sum.dim_val[1] += element_data[current_index].u_coeff_d2[i] * sqrt (1. / (2. * volume))
                      * skalierungsfunktion (i, tau (0), tau (1));
    sum.dim_val[2] += element_data[current_index].u_coeff_d3[i] * sqrt (1. / (2. * volume))
                      * skalierungsfunktion (i, tau (0), tau (1));
  }
  return sum;
}

double
AuswertungMultiscale (struct grid_hierarchy initial_grid_hierarchy, int max_lev, double x, double y, t8_locidx_t itree,
                      t8_locidx_t ielement, t8_locidx_t current_index)
{
  double sum = 0.;
  vec tau (3);
  tau (0) = x;
  tau (1) = y;
  tau (2) = 1.;
  mat A;
  vector<int> r;
  const t8_element_t *element;
  element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[max_lev].forest_arr, itree, ielement);
  double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[max_lev].forest_arr, itree, element);
  double verts[3][3] = { 0 };
  /*
  if(){
    t8_global_productionf("Gefüllt? %i",initial_grid_hierarchy.lev_arr[max_lev].data_arr[current_index].third);
  }

  t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_lev].forest_arr, itree,element,  0,
                        verts[initial_grid_hierarchy.lev_arr[max_lev].data_arr[current_index].first]);
  t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_lev].forest_arr, itree,element,  1,
                        verts[initial_grid_hierarchy.lev_arr[max_lev].data_arr[current_index].second]);
  t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_lev].forest_arr, itree,element,  2,
                        verts[initial_grid_hierarchy.lev_arr[max_lev].data_arr[current_index].third]);
  */

  t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_lev].forest_arr, itree, element, 0, verts[0]);
  t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_lev].forest_arr, itree, element, 1, verts[1]);
  t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_lev].forest_arr, itree, element, 2, verts[2]);

  A.resize (3, 3);
  r.resize (3);
  A (0, 0) = verts[0][0];
  A (0, 1) = verts[1][0];
  A (0, 2) = verts[2][0];
  A (1, 0) = verts[0][1];
  A (1, 1) = verts[1][1];
  A (1, 2) = verts[2][1];
  A (2, 0) = 1;
  A (2, 1) = 1;
  A (2, 2) = 1;
  A.lr_factors (A, r);
  A.lr_solve (A, r, tau);
  int level = max_lev;
  t8_locidx_t index = current_index;
  //t8_global_productionf ("Neue Zelle \n");
  while (level > 0) {
    index = initial_grid_hierarchy.lev_arr[level].data_arr[index].Father_id;
    --level;
    //t8_global_productionf ("level:%i \n", level);
    //t8_global_productionf ("index:%i \n", index);
    //t8_global_productionf ("itree:%i \n", itree);
    tau (0) = x;
    tau (1) = y;
    tau (2) = 1.;
    t8_locidx_t loc_id
      = index - t8_forest_get_tree_element_offset (initial_grid_hierarchy.lev_arr[level].forest_arr, itree);
    //t8_global_productionf ("loc_id:%i \n", loc_id);
    element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, loc_id);
    volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element);
    //t8_global_productionf ("volume:%f \n", volume);
    double verts_loop[3][3] = { 0 };
    t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                  verts_loop[initial_grid_hierarchy.lev_arr[level].data_arr[index].first]);
    t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                  verts_loop[initial_grid_hierarchy.lev_arr[level].data_arr[index].second]);
    t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                  verts_loop[initial_grid_hierarchy.lev_arr[level].data_arr[index].third]);

    A.resize (3, 3);
    r.resize (3);
    A (0, 0) = verts_loop[0][0];
    A (0, 1) = verts_loop[1][0];
    A (0, 2) = verts_loop[2][0];
    A (1, 0) = verts_loop[0][1];
    A (1, 1) = verts_loop[1][1];
    A (1, 2) = verts_loop[2][1];
    A (2, 0) = 1;
    A (2, 1) = 1;
    A (2, 2) = 1;
    A.lr_factors (A, r);
    A.lr_solve (A, r, tau);
    for (int i = 0; i < 3 * M; ++i) {
      sum += initial_grid_hierarchy.lev_arr[level].data_arr[index].d_coeff[i] * sqrt (1. / (2. * volume))
             * muttermultiwavelet (p, i, tau (0), tau (1));
    }
  }
  for (int j = 0; j < M; ++j) {
    sum += initial_grid_hierarchy.lev_arr[level].data_arr[index].u_coeff[j] * sqrt (1. / (2. * volume))
           * skalierungsfunktion (j, tau (0), tau (1));
  }
  return sum;
}

struct double_3d_array
AuswertungMultiscale3d (struct grid_hierarchy_3d initial_grid_hierarchy, int max_level, double x, double y,
                        t8_locidx_t itree, t8_locidx_t ielement, t8_locidx_t current_index)
{
  vec tau (3);
  tau (0) = x;
  tau (1) = y;
  tau (2) = 1.;
  mat A;
  vector<int> r;
  const t8_element_t *element;
  element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, ielement);
  double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element);
  double verts[3][3] = { 0 };
  /*
  t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree,element,  initial_grid_hierarchy.lev_arr[max_level].data_arr[index].first,
                        verts[0]);
  t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree,element,  initial_grid_hierarchy.lev_arr[max_level].data_arr[index].second,
                        verts[1]);
  t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree,element,  initial_grid_hierarchy.lev_arr[max_level].data_arr[index].third,
                        verts[2]);
  */

  t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 0, verts[0]);
  t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 1, verts[1]);
  t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 2, verts[2]);

  A.resize (3, 3);
  r.resize (3);
  A (0, 0) = verts[0][0];
  A (0, 1) = verts[1][0];
  A (0, 2) = verts[2][0];
  A (1, 0) = verts[0][1];
  A (1, 1) = verts[1][1];
  A (1, 2) = verts[2][1];
  A (2, 0) = 1;
  A (2, 1) = 1;
  A (2, 2) = 1;
  A.lr_factors (A, r);
  A.lr_solve (A, r, tau);
  struct double_3d_array sum;
  sum.dim_val[0] = 0.;
  sum.dim_val[1] = 0.;
  sum.dim_val[2] = 0.;
  int level = max_level;
  t8_locidx_t index = current_index;
  while (level > 0) {
    index = initial_grid_hierarchy.lev_arr[level].data_arr[index].Father_id;
    --level;
    tau (0) = x;
    tau (1) = y;
    tau (2) = 1.;
    element = t8_forest_get_element_in_tree (
      initial_grid_hierarchy.lev_arr[level].forest_arr, itree,
      index - t8_forest_get_tree_element_offset (initial_grid_hierarchy.lev_arr[level].forest_arr, itree));
    volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element);
    double verts_loop[3][3] = { 0 };
    t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                  verts_loop[initial_grid_hierarchy.lev_arr[level].data_arr[index].first]);
    t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                  verts_loop[initial_grid_hierarchy.lev_arr[level].data_arr[index].second]);
    t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                  verts_loop[initial_grid_hierarchy.lev_arr[level].data_arr[index].third]);

    A.resize (3, 3);
    r.resize (3);
    A (0, 0) = verts[0][0];
    A (0, 1) = verts[1][0];
    A (0, 2) = verts[2][0];
    A (1, 0) = verts[0][1];
    A (1, 1) = verts[1][1];
    A (1, 2) = verts[2][1];
    A (2, 0) = 1;
    A (2, 1) = 1;
    A (2, 2) = 1;
    A.lr_factors (A, r);
    A.lr_solve (A, r, tau);

    for (int i = 0; i < 3 * M; ++i) {
      sum.dim_val[0] += initial_grid_hierarchy.lev_arr[level].data_arr[index].d_coeff_d1[i] * sqrt (1. / (2. * volume))
                        * muttermultiwavelet (p, i, tau (0), tau (1));
      sum.dim_val[1] += initial_grid_hierarchy.lev_arr[level].data_arr[index].d_coeff_d2[i] * sqrt (1. / (2. * volume))
                        * muttermultiwavelet (p, i, tau (0), tau (1));
      sum.dim_val[2] += initial_grid_hierarchy.lev_arr[level].data_arr[index].d_coeff_d3[i] * sqrt (1. / (2. * volume))
                        * muttermultiwavelet (p, i, tau (0), tau (1));
    }
  }

  for (int i = 0; i < M; ++i) {
    sum.dim_val[0] += initial_grid_hierarchy.lev_arr[level].data_arr[index].u_coeff_d1[i] * sqrt (1. / (2. * volume))
                      * skalierungsfunktion (i, tau (0), tau (1));
    sum.dim_val[1] += initial_grid_hierarchy.lev_arr[level].data_arr[index].u_coeff_d2[i] * sqrt (1. / (2. * volume))
                      * skalierungsfunktion (i, tau (0), tau (1));
    sum.dim_val[2] += initial_grid_hierarchy.lev_arr[level].data_arr[index].u_coeff_d3[i] * sqrt (1. / (2. * volume))
                      * skalierungsfunktion (i, tau (0), tau (1));
  }
  return sum;
}

double
AuswertungMultiscaleWaveletfree (struct grid_hierarchy_waveletfree initial_grid_hierarchy, int max_level, double x,
                                 double y, t8_locidx_t itree, t8_locidx_t ielement, t8_locidx_t current_index)
{
  vec tau (3);
  tau (0) = x;
  tau (1) = y;
  tau (2) = 1.;
  mat A;
  vector<int> r;
  const t8_element_t *element;
  element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, ielement);
  double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element);
  double verts[3][3] = { 0 };
  t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 0, verts[0]);
  t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 1, verts[1]);
  t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 2, verts[2]);
  A.resize (3, 3);
  r.resize (3);
  A (0, 0) = verts[0][0];
  A (0, 1) = verts[1][0];
  A (0, 2) = verts[2][0];
  A (1, 0) = verts[0][1];
  A (1, 1) = verts[1][1];
  A (1, 2) = verts[2][1];
  A (2, 0) = 1;
  A (2, 1) = 1;
  A (2, 2) = 1;
  A.lr_factors (A, r);
  A.lr_solve (A, r, tau);
  double sum = 0.;
  int level = max_level;
  t8_locidx_t index = current_index;
  while (level > 0) {
    index = initial_grid_hierarchy.lev_arr[level].data_arr[index].Father_id;
    --level;
    tau (0) = x;
    tau (1) = y;
    tau (2) = 1.;
    element = t8_forest_get_element_in_tree (
      initial_grid_hierarchy.lev_arr[level].forest_arr, itree,
      index - t8_forest_get_tree_element_offset (initial_grid_hierarchy.lev_arr[level].forest_arr, itree));
    volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element);
    double verts_loop[3][3] = { 0 };
    t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                  verts_loop[initial_grid_hierarchy.lev_arr[level].data_arr[index].first]);
    t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                  verts_loop[initial_grid_hierarchy.lev_arr[level].data_arr[index].second]);
    t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                  verts_loop[initial_grid_hierarchy.lev_arr[level].data_arr[index].third]);
    A.resize (3, 3);
    r.resize (3);
    A (0, 0) = verts_loop[0][0];
    A (0, 1) = verts_loop[1][0];
    A (0, 2) = verts_loop[2][0];
    A (1, 0) = verts_loop[0][1];
    A (1, 1) = verts_loop[1][1];
    A (1, 2) = verts_loop[2][1];
    A (2, 0) = 1;
    A (2, 1) = 1;
    A (2, 2) = 1;
    A.lr_factors (A, r);
    A.lr_solve (A, r, tau);

    int first = initial_grid_hierarchy.lev_arr[level].data_arr[index].first;
    int second = initial_grid_hierarchy.lev_arr[level].data_arr[index].second;
    int third = initial_grid_hierarchy.lev_arr[level].data_arr[index].third;
    invert_order (&first, &second, &third);

    // for (int i = 0; i < M; ++i) {
    // sum += element_data[current_index].u_coeff[i]* sqrt(1./(2.*volume)) * skalierungsfunktion(i,tau(0),tau(1));
    // sum += initial_grid_hierarchy.lev_arr[level].data_arr[index].u_coeff[j]* sqrt(1./(2.*volume))*skalierungsfunktion(j,tau(0),tau(1));
    for (int j = 0; j < 4; ++j) {
      t8_locidx_t child_index
        = initial_grid_hierarchy.lev_arr[level]
            .data_arr[index]
            .child_ids.child_arr[get_correct_order_children ((((t8_dtri_t *) element)->type), j, first, second, third)];
      vec tau_child (3);
      tau_child (0) = x;
      tau_child (1) = y;
      tau_child (2) = 1.;
      const t8_element_t *element_child;
      element_child = t8_forest_get_element_in_tree (
        initial_grid_hierarchy.lev_arr[level + 1].forest_arr, itree,
        child_index - t8_forest_get_tree_element_offset (initial_grid_hierarchy.lev_arr[level + 1].forest_arr, itree));
      double volume_child
        = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[level + 1].forest_arr, itree, element_child);
      double verts_child[3][3] = { 0 };
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level + 1].forest_arr, itree, element_child, 0,
                                    verts_child[initial_grid_hierarchy.lev_arr[level + 1].data_arr[child_index].first]);
      t8_forest_element_coordinate (
        initial_grid_hierarchy.lev_arr[level + 1].forest_arr, itree, element_child, 1,
        verts_child[initial_grid_hierarchy.lev_arr[level + 1].data_arr[child_index].second]);
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level + 1].forest_arr, itree, element_child, 2,
                                    verts_child[initial_grid_hierarchy.lev_arr[level + 1].data_arr[child_index].third]);
      A.resize (3, 3);
      r.resize (3);
      A (0, 0) = verts_child[0][0];
      A (0, 1) = verts_child[1][0];
      A (0, 2) = verts_child[2][0];
      A (1, 0) = verts_child[0][1];
      A (1, 1) = verts_child[1][1];
      A (1, 2) = verts_child[2][1];
      A (2, 0) = 1;
      A (2, 1) = 1;
      A (2, 2) = 1;
      A.lr_factors (A, r);
      A.lr_solve (A, r, tau_child);
      for (int i = 0; i < M; ++i) {
        sum += initial_grid_hierarchy.lev_arr[level].data_arr[index].d_coeff_wavelet_free[i][j]
               * sqrt (1. / (2. * volume_child)) * skalierungsfunktion_nextlevel (i, tau_child (0), tau_child (1));
      }
    }
  }
  // t8_global_productionf("sum 1:%f \n",sum);
  for (int i = 0; i < M; ++i) {
    sum += initial_grid_hierarchy.lev_arr[level].data_arr[index].u_coeff[i] * sqrt (1. / (2. * volume))
           * skalierungsfunktion (i, tau (0), tau (1));
  }
  // t8_global_productionf("sum 2:%f \n",sum);
  return sum;
}

struct double_3d_array
AuswertungMultiscaleWaveletfree3d (struct grid_hierarchy_waveletfree_3d initial_grid_hierarchy, int max_level, double x,
                                   double y, t8_locidx_t itree, t8_locidx_t ielement, t8_locidx_t current_index)
{
  vec tau (3);
  tau (0) = x;
  tau (1) = y;
  tau (2) = 1.;
  mat A;
  vector<int> r;
  const t8_element_t *element;
  element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, ielement);
  double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element);
  double verts[3][3] = { 0 };
  t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 0, verts[0]);
  t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 1, verts[1]);
  t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 2, verts[2]);
  A.resize (3, 3);
  r.resize (3);
  A (0, 0) = verts[0][0];
  A (0, 1) = verts[1][0];
  A (0, 2) = verts[2][0];
  A (1, 0) = verts[0][1];
  A (1, 1) = verts[1][1];
  A (1, 2) = verts[2][1];
  A (2, 0) = 1;
  A (2, 1) = 1;
  A (2, 2) = 1;
  A.lr_factors (A, r);
  A.lr_solve (A, r, tau);
  struct double_3d_array sum;
  sum.dim_val[0] = 0.;
  sum.dim_val[1] = 0.;
  sum.dim_val[2] = 0.;
  int level = max_level;
  t8_locidx_t index = current_index;
  while (level > 0) {
    index = initial_grid_hierarchy.lev_arr[level].data_arr[index].Father_id;
    --level;
    tau (0) = x;
    tau (1) = y;
    tau (2) = 1.;
    element = t8_forest_get_element_in_tree (
      initial_grid_hierarchy.lev_arr[level].forest_arr, itree,
      index - t8_forest_get_tree_element_offset (initial_grid_hierarchy.lev_arr[level].forest_arr, itree));
    volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element);
    double verts_loop[3][3] = { 0 };
    t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 0,
                                  verts_loop[initial_grid_hierarchy.lev_arr[level].data_arr[index].first]);
    t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 1,
                                  verts_loop[initial_grid_hierarchy.lev_arr[level].data_arr[index].second]);
    t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level].forest_arr, itree, element, 2,
                                  verts_loop[initial_grid_hierarchy.lev_arr[level].data_arr[index].third]);

    A.resize (3, 3);
    r.resize (3);
    A (0, 0) = verts_loop[0][0];
    A (0, 1) = verts_loop[1][0];
    A (0, 2) = verts_loop[2][0];
    A (1, 0) = verts_loop[0][1];
    A (1, 1) = verts_loop[1][1];
    A (1, 2) = verts_loop[2][1];
    A (2, 0) = 1;
    A (2, 1) = 1;
    A (2, 2) = 1;
    A.lr_factors (A, r);
    A.lr_solve (A, r, tau);

    int first = initial_grid_hierarchy.lev_arr[level].data_arr[index].first;
    int second = initial_grid_hierarchy.lev_arr[level].data_arr[index].second;
    int third = initial_grid_hierarchy.lev_arr[level].data_arr[index].third;
    invert_order (&first, &second, &third);

    for (int j = 0; j < 4; ++j) {
      t8_locidx_t child_index
        = initial_grid_hierarchy.lev_arr[level]
            .data_arr[index]
            .child_ids.child_arr[get_correct_order_children ((((t8_dtri_t *) element)->type), j, first, second, third)];
      vec tau_child (3);
      tau_child (0) = x;
      tau_child (1) = y;
      tau_child (2) = 1.;
      const t8_element_t *element_child;
      element_child = t8_forest_get_element_in_tree (
        initial_grid_hierarchy.lev_arr[level + 1].forest_arr, itree,
        child_index - t8_forest_get_tree_element_offset (initial_grid_hierarchy.lev_arr[level + 1].forest_arr, itree));
      double volume_child
        = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[level + 1].forest_arr, itree, element);
      double verts_child[3][3] = { 0 };
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level + 1].forest_arr, itree, element_child, 0,
                                    verts_child[initial_grid_hierarchy.lev_arr[level + 1].data_arr[child_index].first]);
      t8_forest_element_coordinate (
        initial_grid_hierarchy.lev_arr[level + 1].forest_arr, itree, element_child, 1,
        verts_child[initial_grid_hierarchy.lev_arr[level + 1].data_arr[child_index].second]);
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[level + 1].forest_arr, itree, element_child, 2,
                                    verts_child[initial_grid_hierarchy.lev_arr[level + 1].data_arr[child_index].third]);
      A.resize (3, 3);
      r.resize (3);
      A (0, 0) = verts_child[0][0];
      A (0, 1) = verts_child[1][0];
      A (0, 2) = verts_child[2][0];
      A (1, 0) = verts_child[0][1];
      A (1, 1) = verts_child[1][1];
      A (1, 2) = verts_child[2][1];
      A (2, 0) = 1;
      A (2, 1) = 1;
      A (2, 2) = 1;
      A.lr_factors (A, r);
      A.lr_solve (A, r, tau_child);
      for (int i = 0; i < M; ++i) {
        sum.dim_val[0] += initial_grid_hierarchy.lev_arr[level].data_arr[current_index].d_coeff_wavelet_free_d1[i][j]
                          * sqrt (1. / (2. * volume_child))
                          * skalierungsfunktion_nextlevel (i, tau_child (0), tau_child (1));
        sum.dim_val[1] += initial_grid_hierarchy.lev_arr[level].data_arr[current_index].d_coeff_wavelet_free_d2[i][j]
                          * sqrt (1. / (2. * volume_child))
                          * skalierungsfunktion_nextlevel (i, tau_child (0), tau_child (1));
        sum.dim_val[2] += initial_grid_hierarchy.lev_arr[level].data_arr[current_index].d_coeff_wavelet_free_d3[i][j]
                          * sqrt (1. / (2. * volume_child))
                          * skalierungsfunktion_nextlevel (i, tau_child (0), tau_child (1));
      }
    }
  }

  for (int i = 0; i < M; ++i) {
    sum.dim_val[0] += initial_grid_hierarchy.lev_arr[level].data_arr[index].u_coeff_d1[i] * sqrt (1. / (2. * volume))
                      * skalierungsfunktion (i, tau (0), tau (1));
    sum.dim_val[1] += initial_grid_hierarchy.lev_arr[level].data_arr[index].u_coeff_d2[i] * sqrt (1. / (2. * volume))
                      * skalierungsfunktion (i, tau (0), tau (1));
    sum.dim_val[2] += initial_grid_hierarchy.lev_arr[level].data_arr[index].u_coeff_d3[i] * sqrt (1. / (2. * volume))
                      * skalierungsfunktion (i, tau (0), tau (1));
  }
  return sum;
}

double
ErrorSinglescale (t8_forest_t forest, struct t8_data_per_element *element_data, func F, int rule, const char *err_type)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  mat A;
  vector<int> r;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  T8_ASSERT (t8_forest_is_committed (forest));

  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;
  num_local_trees = t8_forest_get_num_local_trees (forest);
  double sum = 0.0;
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    /* This loop iterates through all local trees in the forest. */
    /* Get the number of elements of this tree. */
    num_elements_in_tree = t8_forest_get_tree_num_elements (forest, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      /* This loop iterates through all the local elements of the forest in the current tree. */
      /* We can now write to the position current_index into our array in order to store
       * data for this element. */
      /* Since in this example we want to compute the data based on the element in question,
       * we need to get a pointer to this element. */
      element = t8_forest_get_element_in_tree (forest, itree, ielement);
      double volume = t8_forest_element_volume (forest, itree, element);
      double verts[3][3] = { 0 };

      t8_forest_element_coordinate (forest, itree, element, 0, verts[0]);
      t8_forest_element_coordinate (forest, itree, element, 1, verts[1]);
      t8_forest_element_coordinate (forest, itree, element, 2, verts[2]);

      A.resize (3, 3);
      r.resize (3);
      A (0, 0) = verts[0][0];
      A (0, 1) = verts[1][0];
      A (0, 2) = verts[2][0];
      A (1, 0) = verts[0][1];
      A (1, 1) = verts[1][1];
      A (1, 2) = verts[2][1];
      A (2, 0) = 1;
      A (2, 1) = 1;
      A (2, 2) = 1;
      A.lr_factors (A, r);
      double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
      reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
      double quad = 0.;
      for (int order = 0; order < order_num; ++order) {
        double x = xytab[order * 2];
        double y = xytab[1 + order * 2];
        double value = F (x, y) - AuswertungSinglescale (forest, element_data, x, y, itree, ielement, current_index);
        if (err_type == "L1") {
          quad += wtab[order] * abs (value);
        }
        else if (err_type == "L2") {
          quad += wtab[order] * value * value;
        }
        else if (err_type == "Linf") {
          quad = max (abs (value), quad);
        }
        else {
          printf ("Invalid action: %s\n", err_type);
        }
      }
      if (err_type == "L1" || err_type == "L2") {
        quad *= volume;
        sum += quad;
      }
      else if (err_type == "Linf") {
        sum = max (abs (quad), sum);
      }
    }
  }
  if (err_type == "L2") {
    sum = sqrt (sum);
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return sum;
}

double
ErrorSinglescaleSpline (t8_forest_t forest, struct t8_data_per_element *element_data, spline eval_spline,
                        const gsl_spline2d *spline, gsl_interp_accel *xacc, gsl_interp_accel *yacc, int rule,
                        const char *err_type)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  mat A;
  vector<int> r;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  T8_ASSERT (t8_forest_is_committed (forest));

  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;
  num_local_trees = t8_forest_get_num_local_trees (forest);
  double sum = 0.0;
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    /* This loop iterates through all local trees in the forest. */
    /* Get the number of elements of this tree. */
    num_elements_in_tree = t8_forest_get_tree_num_elements (forest, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      /* This loop iterates through all the local elements of the forest in the current tree. */
      /* We can now write to the position current_index into our array in order to store
       * data for this element. */
      /* Since in this example we want to compute the data based on the element in question,
       * we need to get a pointer to this element. */
      element = t8_forest_get_element_in_tree (forest, itree, ielement);
      double volume = t8_forest_element_volume (forest, itree, element);
      double verts[3][3] = { 0 };
      t8_forest_element_coordinate (forest, itree, element, 0, verts[0]);
      t8_forest_element_coordinate (forest, itree, element, 1, verts[1]);
      t8_forest_element_coordinate (forest, itree, element, 2, verts[2]);
      A.resize (3, 3);
      r.resize (3);
      A (0, 0) = verts[0][0];
      A (0, 1) = verts[1][0];
      A (0, 2) = verts[2][0];
      A (1, 0) = verts[0][1];
      A (1, 1) = verts[1][1];
      A (1, 2) = verts[2][1];
      A (2, 0) = 1;
      A (2, 1) = 1;
      A (2, 2) = 1;
      A.lr_factors (A, r);
      double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
      reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
      double quad = 0.;
      for (int order = 0; order < order_num; ++order) {
        double x = xytab[order * 2];
        double y = xytab[1 + order * 2];
        double value = eval_spline (spline, x, y, xacc, yacc)
                       - AuswertungSinglescale (forest, element_data, x, y, itree, ielement, current_index);
        if (err_type == "L1") {
          quad += wtab[order] * abs (value);
        }
        else if (err_type == "L2") {
          quad += wtab[order] * value * value;
        }
        else if (err_type == "Linf") {
          quad = max (abs (value), quad);
        }
        else {
          printf ("Invalid action: %s\n", err_type);
        }
      }
      if (err_type == "L1" || err_type == "L2") {
        quad *= volume;
        sum += quad;
      }
      else if (err_type == "Linf") {
        sum = max (abs (quad), sum);
      }
    }
  }
  if (err_type == "L2") {
    sum = sqrt (sum);
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return sum;
}

double
ErrorSinglescale3d (t8_forest_t forest, struct t8_data_per_element_3d *element_data, func F1, func F2, func F3,
                    int rule, const char *err_type)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  mat A;
  vector<int> r;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  T8_ASSERT (t8_forest_is_committed (forest));

  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;
  num_local_trees = t8_forest_get_num_local_trees (forest);
  double sum = 0.0;
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    /* This loop iterates through all local trees in the forest. */
    /* Get the number of elements of this tree. */
    num_elements_in_tree = t8_forest_get_tree_num_elements (forest, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      /* This loop iterates through all the local elements of the forest in the current tree. */
      /* We can now write to the position current_index into our array in order to store
       * data for this element. */
      /* Since in this example we want to compute the data based on the element in question,
       * we need to get a pointer to this element. */
      element = t8_forest_get_element_in_tree (forest, itree, ielement);
      double volume = t8_forest_element_volume (forest, itree, element);
      double verts[3][3] = { 0 };
      t8_forest_element_coordinate (forest, itree, element, 0, verts[0]);
      t8_forest_element_coordinate (forest, itree, element, 1, verts[1]);
      t8_forest_element_coordinate (forest, itree, element, 2, verts[2]);
      A.resize (3, 3);
      r.resize (3);
      A (0, 0) = verts[0][0];
      A (0, 1) = verts[1][0];
      A (0, 2) = verts[2][0];
      A (1, 0) = verts[0][1];
      A (1, 1) = verts[1][1];
      A (1, 2) = verts[2][1];
      A (2, 0) = 1;
      A (2, 1) = 1;
      A (2, 2) = 1;
      A.lr_factors (A, r);
      double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
      reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
      double quad = 0.0;
      for (int order = 0; order < order_num; ++order) {
        double x = xytab[order * 2];
        double y = xytab[1 + order * 2];
        struct double_3d_array AuswertungSinglescale_3d;
        AuswertungSinglescale_3d = AuswertungSinglescale3d (forest, element_data, x, y, itree, ielement, current_index);
        double value_dim1 = F1 (x, y) - AuswertungSinglescale_3d.dim_val[0];
        double value_dim2 = F2 (x, y) - AuswertungSinglescale_3d.dim_val[1];
        double value_dim3 = F3 (x, y) - AuswertungSinglescale_3d.dim_val[2];
        if (err_type == "L1") {
          quad += wtab[order] * (abs (value_dim1) + abs (value_dim2) + abs (value_dim3));
        }
        else if (err_type == "L2") {
          quad += wtab[order] * (value_dim1 * value_dim1 + value_dim2 * value_dim2 + value_dim3 * value_dim3);
        }
        else if (err_type == "Linf") {
          quad = max (abs (value_dim1), quad);
          quad = max (abs (value_dim2), quad);
          quad = max (abs (value_dim3), quad);
        }
        else {
          printf ("Invalid action: %s\n", err_type);
        }
      }
      if (err_type == "L1" || err_type == "L2") {
        quad *= volume;
        sum += quad;
      }
      else if (err_type == "Linf") {
        sum = max (abs (quad), sum);
      }
    }
  }
  if (err_type == "L2") {
    sum = sqrt (sum);
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return sum;
}

double
ErrorSinglescale3dSpline (t8_forest_t forest, struct t8_data_per_element_3d *element_data, spline eval_spline,
                          const gsl_spline2d *spline_d1, gsl_interp_accel *xacc_d1, gsl_interp_accel *yacc_d1,
                          const gsl_spline2d *spline_d2, gsl_interp_accel *xacc_d2, gsl_interp_accel *yacc_d2,
                          const gsl_spline2d *spline_d3, gsl_interp_accel *xacc_d3, gsl_interp_accel *yacc_d3, int rule,
                          const char *err_type)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  mat A;
  vector<int> r;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  T8_ASSERT (t8_forest_is_committed (forest));

  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;
  num_local_trees = t8_forest_get_num_local_trees (forest);
  double sum = 0.0;
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    /* This loop iterates through all local trees in the forest. */
    /* Get the number of elements of this tree. */
    num_elements_in_tree = t8_forest_get_tree_num_elements (forest, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      /* This loop iterates through all the local elements of the forest in the current tree. */
      /* We can now write to the position current_index into our array in order to store
       * data for this element. */
      /* Since in this example we want to compute the data based on the element in question,
       * we need to get a pointer to this element. */
      element = t8_forest_get_element_in_tree (forest, itree, ielement);
      double volume = t8_forest_element_volume (forest, itree, element);
      double verts[3][3] = { 0 };
      t8_forest_element_coordinate (forest, itree, element, 0, verts[0]);
      t8_forest_element_coordinate (forest, itree, element, 1, verts[1]);
      t8_forest_element_coordinate (forest, itree, element, 2, verts[2]);
      A.resize (3, 3);
      r.resize (3);
      A (0, 0) = verts[0][0];
      A (0, 1) = verts[1][0];
      A (0, 2) = verts[2][0];
      A (1, 0) = verts[0][1];
      A (1, 1) = verts[1][1];
      A (1, 2) = verts[2][1];
      A (2, 0) = 1;
      A (2, 1) = 1;
      A (2, 2) = 1;
      A.lr_factors (A, r);
      double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
      reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
      double quad = 0.0;
      for (int order = 0; order < order_num; ++order) {
        double x = xytab[order * 2];
        double y = xytab[1 + order * 2];
        struct double_3d_array AuswertungSinglescale_3d;
        AuswertungSinglescale_3d = AuswertungSinglescale3d (forest, element_data, x, y, itree, ielement, current_index);
        double value_dim1 = eval_spline (spline_d1, x, y, xacc_d1, yacc_d1) - AuswertungSinglescale_3d.dim_val[0];
        double value_dim2 = eval_spline (spline_d2, x, y, xacc_d2, yacc_d2) - AuswertungSinglescale_3d.dim_val[1];
        double value_dim3 = eval_spline (spline_d3, x, y, xacc_d3, yacc_d3) - AuswertungSinglescale_3d.dim_val[2];
        if (err_type == "L1") {
          quad += wtab[order] * (abs (value_dim1) + abs (value_dim2) + abs (value_dim3));
        }
        else if (err_type == "L2") {
          quad += wtab[order] * (value_dim1 * value_dim1 + value_dim2 * value_dim2 + value_dim3 * value_dim3);
        }
        else if (err_type == "Linf") {
          quad = max (abs (value_dim1), quad);
          quad = max (abs (value_dim2), quad);
          quad = max (abs (value_dim3), quad);
        }
        else {
          printf ("Invalid action: %s\n", err_type);
        }
      }
      if (err_type == "L1" || err_type == "L2") {
        quad *= volume;
        sum += quad;
      }
      else if (err_type == "Linf") {
        sum = max (abs (quad), sum);
      }
    }
  }
  if (err_type == "L2") {
    sum = sqrt (sum);
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return sum;
}

double
ErrorSinglescaleWaveletfree (t8_forest_t forest, struct t8_data_per_element_waveletfree *element_data, func F, int rule,
                             const char *err_type)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  mat A;
  vector<int> r;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  T8_ASSERT (t8_forest_is_committed (forest));

  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;
  num_local_trees = t8_forest_get_num_local_trees (forest);
  double sum = 0.0;
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    /* This loop iterates through all local trees in the forest. */
    /* Get the number of elements of this tree. */
    num_elements_in_tree = t8_forest_get_tree_num_elements (forest, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      /* This loop iterates through all the local elements of the forest in the current tree. */
      /* We can now write to the position current_index into our array in order to store
       * data for this element. */
      /* Since in this example we want to compute the data based on the element in question,
       * we need to get a pointer to this element. */
      element = t8_forest_get_element_in_tree (forest, itree, ielement);
      double volume = t8_forest_element_volume (forest, itree, element);
      double verts[3][3] = { 0 };
      t8_forest_element_coordinate (forest, itree, element, 0, verts[0]);
      t8_forest_element_coordinate (forest, itree, element, 1, verts[1]);
      t8_forest_element_coordinate (forest, itree, element, 2, verts[2]);
      A.resize (3, 3);
      r.resize (3);
      A (0, 0) = verts[0][0];
      A (0, 1) = verts[1][0];
      A (0, 2) = verts[2][0];
      A (1, 0) = verts[0][1];
      A (1, 1) = verts[1][1];
      A (1, 2) = verts[2][1];
      A (2, 0) = 1;
      A (2, 1) = 1;
      A (2, 2) = 1;
      A.lr_factors (A, r);
      double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
      reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
      double quad = 0.;
      for (int order = 0; order < order_num; ++order) {
        double x = xytab[order * 2];
        double y = xytab[1 + order * 2];
        double value
          = F (x, y) - AuswertungSinglescaleWaveletfree (forest, element_data, x, y, itree, ielement, current_index);
        if (err_type == "L1") {
          quad += wtab[order] * abs (value);
        }
        else if (err_type == "L2") {
          quad += wtab[order] * value * value;
        }
        else if (err_type == "Linf") {
          quad = max (abs (value), quad);
        }
        else {
          printf ("Invalid action: %s\n", err_type);
        }
      }
      if (err_type == "L1" || err_type == "L2") {
        quad *= volume;
        sum += quad;
      }
      else if (err_type == "Linf") {
        sum = max (abs (quad), sum);
      }
    }
  }
  if (err_type == "L2") {
    sum = sqrt (sum);
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return sum;
}

double
ErrorSinglescaleWaveletfree3d (t8_forest_t forest, struct t8_data_per_element_waveletfree_3d *element_data, func F1,
                               func F2, func F3, int rule, const char *err_type)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  mat A;
  vector<int> r;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  T8_ASSERT (t8_forest_is_committed (forest));

  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;
  num_local_trees = t8_forest_get_num_local_trees (forest);
  double sum = 0.0;
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    /* This loop iterates through all local trees in the forest. */
    /* Get the number of elements of this tree. */
    num_elements_in_tree = t8_forest_get_tree_num_elements (forest, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      /* This loop iterates through all the local elements of the forest in the current tree. */
      /* We can now write to the position current_index into our array in order to store
       * data for this element. */
      /* Since in this example we want to compute the data based on the element in question,
       * we need to get a pointer to this element. */
      element = t8_forest_get_element_in_tree (forest, itree, ielement);
      double volume = t8_forest_element_volume (forest, itree, element);
      double verts[3][3] = { 0 };
      t8_forest_element_coordinate (forest, itree, element, 0, verts[0]);
      t8_forest_element_coordinate (forest, itree, element, 1, verts[1]);
      t8_forest_element_coordinate (forest, itree, element, 2, verts[2]);
      A.resize (3, 3);
      r.resize (3);
      A (0, 0) = verts[0][0];
      A (0, 1) = verts[1][0];
      A (0, 2) = verts[2][0];
      A (1, 0) = verts[0][1];
      A (1, 1) = verts[1][1];
      A (1, 2) = verts[2][1];
      A (2, 0) = 1;
      A (2, 1) = 1;
      A (2, 2) = 1;
      A.lr_factors (A, r);
      double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
      reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
      double quad = 0.0;
      for (int order = 0; order < order_num; ++order) {
        double x = xytab[order * 2];
        double y = xytab[1 + order * 2];
        struct double_3d_array AuswertungSinglescale_3d;
        AuswertungSinglescale_3d
          = AuswertungSinglescaleWaveletfree3d (forest, element_data, x, y, itree, ielement, current_index);
        double value_dim1 = F1 (x, y) - AuswertungSinglescale_3d.dim_val[0];
        double value_dim2 = F2 (x, y) - AuswertungSinglescale_3d.dim_val[1];
        double value_dim3 = F3 (x, y) - AuswertungSinglescale_3d.dim_val[2];
        if (err_type == "L1") {
          quad += wtab[order] * (abs (value_dim1) + abs (value_dim2) + abs (value_dim3));
        }
        else if (err_type == "L2") {
          quad += wtab[order] * (value_dim1 * value_dim1 + value_dim2 * value_dim2 + value_dim3 * value_dim3);
        }
        else if (err_type == "Linf") {
          quad = max (abs (value_dim1), quad);
          quad = max (abs (value_dim2), quad);
          quad = max (abs (value_dim3), quad);
        }
        else {
          printf ("Invalid action: %s\n", err_type);
        }
      }
      if (err_type == "L1" || err_type == "L2") {
        quad *= volume;
        sum += quad;
      }
      else if (err_type == "Linf") {
        sum = max (abs (quad), sum);
      }
    }
  }
  if (err_type == "L2") {
    sum = sqrt (sum);
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return sum;
}

double
ErrorSinglescaleWaveletfreeSpline (t8_forest_t forest, struct t8_data_per_element_waveletfree *element_data,
                                   spline eval_spline, const gsl_spline2d *spline, gsl_interp_accel *xacc,
                                   gsl_interp_accel *yacc, int rule, const char *err_type)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  mat A;
  vector<int> r;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  T8_ASSERT (t8_forest_is_committed (forest));

  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;
  num_local_trees = t8_forest_get_num_local_trees (forest);
  double sum = 0.0;
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    /* This loop iterates through all local trees in the forest. */
    /* Get the number of elements of this tree. */
    num_elements_in_tree = t8_forest_get_tree_num_elements (forest, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      /* This loop iterates through all the local elements of the forest in the current tree. */
      /* We can now write to the position current_index into our array in order to store
       * data for this element. */
      /* Since in this example we want to compute the data based on the element in question,
       * we need to get a pointer to this element. */
      element = t8_forest_get_element_in_tree (forest, itree, ielement);
      double volume = t8_forest_element_volume (forest, itree, element);
      double verts[3][3] = { 0 };
      t8_forest_element_coordinate (forest, itree, element, 0, verts[0]);
      t8_forest_element_coordinate (forest, itree, element, 1, verts[1]);
      t8_forest_element_coordinate (forest, itree, element, 2, verts[2]);
      A.resize (3, 3);
      r.resize (3);
      A (0, 0) = verts[0][0];
      A (0, 1) = verts[1][0];
      A (0, 2) = verts[2][0];
      A (1, 0) = verts[0][1];
      A (1, 1) = verts[1][1];
      A (1, 2) = verts[2][1];
      A (2, 0) = 1;
      A (2, 1) = 1;
      A (2, 2) = 1;
      A.lr_factors (A, r);
      double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
      reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
      double quad = 0.;
      for (int order = 0; order < order_num; ++order) {
        double x = xytab[order * 2];
        double y = xytab[1 + order * 2];
        double value = eval_spline (spline, x, y, xacc, yacc)
                       - AuswertungSinglescaleWaveletfree (forest, element_data, x, y, itree, ielement, current_index);
        if (err_type == "L1") {
          quad += wtab[order] * abs (value);
        }
        else if (err_type == "L2") {
          quad += wtab[order] * value * value;
        }
        else if (err_type == "Linf") {
          quad = max (abs (value), quad);
        }
        else {
          printf ("Invalid action: %s\n", err_type);
        }
      }
      if (err_type == "L1" || err_type == "L2") {
        quad *= volume;
        sum += quad;
      }
      else if (err_type == "Linf") {
        sum = max (abs (quad), sum);
      }
    }
  }
  if (err_type == "L2") {
    sum = sqrt (sum);
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return sum;
}

double
ErrorSinglescaleWaveletfree3dSpline (t8_forest_t forest, struct t8_data_per_element_waveletfree_3d *element_data,
                                     spline eval_spline, const gsl_spline2d *spline_d1, gsl_interp_accel *xacc_d1,
                                     gsl_interp_accel *yacc_d1, const gsl_spline2d *spline_d2,
                                     gsl_interp_accel *xacc_d2, gsl_interp_accel *yacc_d2,
                                     const gsl_spline2d *spline_d3, gsl_interp_accel *xacc_d3,
                                     gsl_interp_accel *yacc_d3, int rule, const char *err_type)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  mat A;
  vector<int> r;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  T8_ASSERT (t8_forest_is_committed (forest));

  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;
  num_local_trees = t8_forest_get_num_local_trees (forest);
  double sum = 0.0;
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    /* This loop iterates through all local trees in the forest. */
    /* Get the number of elements of this tree. */
    num_elements_in_tree = t8_forest_get_tree_num_elements (forest, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      /* This loop iterates through all the local elements of the forest in the current tree. */
      /* We can now write to the position current_index into our array in order to store
       * data for this element. */
      /* Since in this example we want to compute the data based on the element in question,
       * we need to get a pointer to this element. */
      element = t8_forest_get_element_in_tree (forest, itree, ielement);
      double volume = t8_forest_element_volume (forest, itree, element);
      double verts[3][3] = { 0 };
      t8_forest_element_coordinate (forest, itree, element, 0, verts[0]);
      t8_forest_element_coordinate (forest, itree, element, 1, verts[1]);
      t8_forest_element_coordinate (forest, itree, element, 2, verts[2]);
      A.resize (3, 3);
      r.resize (3);
      A (0, 0) = verts[0][0];
      A (0, 1) = verts[1][0];
      A (0, 2) = verts[2][0];
      A (1, 0) = verts[0][1];
      A (1, 1) = verts[1][1];
      A (1, 2) = verts[2][1];
      A (2, 0) = 1;
      A (2, 1) = 1;
      A (2, 2) = 1;
      A.lr_factors (A, r);
      double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
      reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
      double quad = 0.0;
      for (int order = 0; order < order_num; ++order) {
        double x = xytab[order * 2];
        double y = xytab[1 + order * 2];
        struct double_3d_array AuswertungSinglescale_3d;
        AuswertungSinglescale_3d
          = AuswertungSinglescaleWaveletfree3d (forest, element_data, x, y, itree, ielement, current_index);
        double value_dim1 = eval_spline (spline_d1, x, y, xacc_d1, yacc_d1) - AuswertungSinglescale_3d.dim_val[0];
        double value_dim2 = eval_spline (spline_d2, x, y, xacc_d2, yacc_d2) - AuswertungSinglescale_3d.dim_val[1];
        double value_dim3 = eval_spline (spline_d3, x, y, xacc_d3, yacc_d3) - AuswertungSinglescale_3d.dim_val[2];
        if (err_type == "L1") {
          quad += wtab[order] * (abs (value_dim1) + abs (value_dim2) + abs (value_dim3));
        }
        else if (err_type == "L2") {
          quad += wtab[order] * (value_dim1 * value_dim1 + value_dim2 * value_dim2 + value_dim3 * value_dim3);
        }
        else if (err_type == "Linf") {
          quad = max (abs (value_dim1), quad);
          quad = max (abs (value_dim2), quad);
          quad = max (abs (value_dim3), quad);
        }
        else {
          printf ("Invalid action: %s\n", err_type);
        }
      }
      if (err_type == "L1" || err_type == "L2") {
        quad *= volume;
        sum += quad;
      }
      else if (err_type == "Linf") {
        sum = max (abs (quad), sum);
      }
    }
  }
  if (err_type == "L2") {
    sum = sqrt (sum);
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return sum;
}

double
ErrorMultiscale (struct grid_hierarchy initial_grid_hierarchy, int max_lev, func F, int rule, const char *err_type)
{
  double sum = 0.0;
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  mat A;
  vector<int> r;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  T8_ASSERT (t8_forest_is_committed (initial_grid_hierarchy.lev_arr[max_lev].forest_arr));
  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;
  num_local_trees = t8_forest_get_num_local_trees (initial_grid_hierarchy.lev_arr[max_lev].forest_arr);
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    /* This loop iterates through all local trees in the forest. */
    /* Get the number of elements of this tree. */
    num_elements_in_tree = t8_forest_get_tree_num_elements (initial_grid_hierarchy.lev_arr[max_lev].forest_arr, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      /* This loop iterates through all the local elements of the forest in the current tree. */
      /* We can now write to the position current_index into our array in order to store
       * data for this element. */
      /* Since in this example we want to compute the data based on the element in question,
       * we need to get a pointer to this element. */
      element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[max_lev].forest_arr, itree, ielement);
      double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[max_lev].forest_arr, itree, element);
      double verts[3][3] = { 0 };
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_lev].forest_arr, itree, element, 0,
                                    verts[initial_grid_hierarchy.lev_arr[max_lev].data_arr[current_index].first]);
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_lev].forest_arr, itree, element, 1,
                                    verts[initial_grid_hierarchy.lev_arr[max_lev].data_arr[current_index].second]);
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_lev].forest_arr, itree, element, 2,
                                    verts[initial_grid_hierarchy.lev_arr[max_lev].data_arr[current_index].third]);
      A.resize (3, 3);
      r.resize (3);
      A (0, 0) = verts[0][0];
      A (0, 1) = verts[1][0];
      A (0, 2) = verts[2][0];
      A (1, 0) = verts[0][1];
      A (1, 1) = verts[1][1];
      A (1, 2) = verts[2][1];
      A (2, 0) = 1;
      A (2, 1) = 1;
      A (2, 2) = 1;
      A.lr_factors (A, r);
      double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
      reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
      double quad = 0.;
      for (int order = 0; order < order_num; ++order) {
        double x = xytab[order * 2];
        double y = xytab[1 + order * 2];
        double value
          = F (x, y) - AuswertungMultiscale (initial_grid_hierarchy, max_lev, x, y, itree, ielement, current_index);
        if (err_type == "L1") {
          quad += wtab[order] * abs (value);
        }
        else if (err_type == "L2") {
          quad += wtab[order] * value * value;
        }
        else if (err_type == "Linf") {
          quad = max (abs (value), quad);
        }
        else {
          printf ("Invalid action: %s\n", err_type);
        }
      }
      if (err_type == "L1" || err_type == "L2") {
        quad *= volume;
        sum += quad;
      }
      else if (err_type == "Linf") {
        sum = max (abs (quad), sum);
      }
    }
  }
  if (err_type == "L2") {
    sum = sqrt (sum);
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return sum;
}

double
ErrorMultiscaleSpline (struct grid_hierarchy initial_grid_hierarchy, int max_level, spline eval_spline,
                       const gsl_spline2d *spline, gsl_interp_accel *xacc, gsl_interp_accel *yacc, int rule,
                       const char *err_type)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  mat A;
  vector<int> r;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  T8_ASSERT (t8_forest_is_committed (initial_grid_hierarchy.lev_arr[max_level].forest_arr));

  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;
  num_local_trees = t8_forest_get_num_local_trees (initial_grid_hierarchy.lev_arr[max_level].forest_arr);
  double sum = 0.0;
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    /* This loop iterates through all local trees in the forest. */
    /* Get the number of elements of this tree. */
    num_elements_in_tree
      = t8_forest_get_tree_num_elements (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      /* This loop iterates through all the local elements of the forest in the current tree. */
      /* We can now write to the position current_index into our array in order to store
       * data for this element. */
      /* Since in this example we want to compute the data based on the element in question,
       * we need to get a pointer to this element. */
      element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, ielement);
      double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element);
      double verts[3][3] = { 0 };
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 0, verts[0]);
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 1, verts[1]);
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 2, verts[2]);
      A.resize (3, 3);
      r.resize (3);
      A (0, 0) = verts[0][0];
      A (0, 1) = verts[1][0];
      A (0, 2) = verts[2][0];
      A (1, 0) = verts[0][1];
      A (1, 1) = verts[1][1];
      A (1, 2) = verts[2][1];
      A (2, 0) = 1;
      A (2, 1) = 1;
      A (2, 2) = 1;
      A.lr_factors (A, r);
      double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
      reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
      double quad = 0.;
      for (int order = 0; order < order_num; ++order) {
        double x = xytab[order * 2];
        double y = xytab[1 + order * 2];
        double value = eval_spline (spline, x, y, xacc, yacc)
                       - AuswertungMultiscale (initial_grid_hierarchy, max_level, x, y, itree, ielement, current_index);
        if (err_type == "L1") {
          quad += wtab[order] * abs (value);
        }
        else if (err_type == "L2") {
          quad += wtab[order] * value * value;
        }
        else if (err_type == "Linf") {
          quad = max (abs (value), quad);
        }
        else {
          printf ("Invalid action: %s\n", err_type);
        }
      }
      if (err_type == "L1" || err_type == "L2") {
        quad *= volume;
        sum += quad;
      }
      else if (err_type == "Linf") {
        sum = max (abs (quad), sum);
      }
    }
  }
  if (err_type == "L2") {
    sum = sqrt (sum);
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return sum;
}

double
ErrorMultiscale3d (struct grid_hierarchy_3d initial_grid_hierarchy, int max_level, func F1, func F2, func F3, int rule,
                   const char *err_type)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  mat A;
  vector<int> r;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  T8_ASSERT (t8_forest_is_committed (initial_grid_hierarchy.lev_arr[max_level].forest_arr));

  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;
  num_local_trees = t8_forest_get_num_local_trees (initial_grid_hierarchy.lev_arr[max_level].forest_arr);
  double sum = 0.0;
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    /* This loop iterates through all local trees in the forest. */
    /* Get the number of elements of this tree. */
    num_elements_in_tree
      = t8_forest_get_tree_num_elements (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      /* This loop iterates through all the local elements of the forest in the current tree. */
      /* We can now write to the position current_index into our array in order to store
       * data for this element. */
      /* Since in this example we want to compute the data based on the element in question,
       * we need to get a pointer to this element. */
      element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, ielement);
      double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element);
      double verts[3][3] = { 0 };
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 0, verts[0]);
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 1, verts[1]);
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 2, verts[2]);
      A.resize (3, 3);
      r.resize (3);
      A (0, 0) = verts[0][0];
      A (0, 1) = verts[1][0];
      A (0, 2) = verts[2][0];
      A (1, 0) = verts[0][1];
      A (1, 1) = verts[1][1];
      A (1, 2) = verts[2][1];
      A (2, 0) = 1;
      A (2, 1) = 1;
      A (2, 2) = 1;
      A.lr_factors (A, r);
      double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
      reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
      double quad = 0.0;
      for (int order = 0; order < order_num; ++order) {
        double x = xytab[order * 2];
        double y = xytab[1 + order * 2];
        struct double_3d_array Auswertung_Multiscale_3d;
        Auswertung_Multiscale_3d
          = AuswertungMultiscale3d (initial_grid_hierarchy, max_level, x, y, itree, ielement, current_index);
        double value_dim1 = F1 (x, y) - Auswertung_Multiscale_3d.dim_val[0];
        double value_dim2 = F2 (x, y) - Auswertung_Multiscale_3d.dim_val[1];
        double value_dim3 = F3 (x, y) - Auswertung_Multiscale_3d.dim_val[2];
        if (err_type == "L1") {
          quad += wtab[order] * (abs (value_dim1) + abs (value_dim2) + abs (value_dim3));
        }
        else if (err_type == "L2") {
          quad += wtab[order] * (value_dim1 * value_dim1 + value_dim2 * value_dim2 + value_dim3 * value_dim3);
        }
        else if (err_type == "Linf") {
          quad = max (abs (value_dim1), quad);
          quad = max (abs (value_dim2), quad);
          quad = max (abs (value_dim3), quad);
        }
        else {
          printf ("Invalid action: %s\n", err_type);
        }
      }
      if (err_type == "L1" || err_type == "L2") {
        quad *= volume;
        sum += quad;
      }
      else if (err_type == "Linf") {
        sum = max (abs (quad), sum);
      }
    }
  }
  if (err_type == "L2") {
    sum = sqrt (sum);
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return sum;
}

double
ErrorMultiscale3dSpline (struct grid_hierarchy_3d initial_grid_hierarchy, int max_level, spline eval_spline,
                         const gsl_spline2d *spline_d1, gsl_interp_accel *xacc_d1, gsl_interp_accel *yacc_d1,
                         const gsl_spline2d *spline_d2, gsl_interp_accel *xacc_d2, gsl_interp_accel *yacc_d2,
                         const gsl_spline2d *spline_d3, gsl_interp_accel *xacc_d3, gsl_interp_accel *yacc_d3, int rule,
                         const char *err_type)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  mat A;
  vector<int> r;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  T8_ASSERT (t8_forest_is_committed (initial_grid_hierarchy.lev_arr[max_level].forest_arr));

  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;
  num_local_trees = t8_forest_get_num_local_trees (initial_grid_hierarchy.lev_arr[max_level].forest_arr);
  double sum = 0.0;
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    /* This loop iterates through all local trees in the forest. */
    /* Get the number of elements of this tree. */
    num_elements_in_tree
      = t8_forest_get_tree_num_elements (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      /* This loop iterates through all the local elements of the forest in the current tree. */
      /* We can now write to the position current_index into our array in order to store
       * data for this element. */
      /* Since in this example we want to compute the data based on the element in question,
       * we need to get a pointer to this element. */
      element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, ielement);
      double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element);
      double verts[3][3] = { 0 };
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 0, verts[0]);
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 1, verts[1]);
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 2, verts[2]);
      A.resize (3, 3);
      r.resize (3);
      A (0, 0) = verts[0][0];
      A (0, 1) = verts[1][0];
      A (0, 2) = verts[2][0];
      A (1, 0) = verts[0][1];
      A (1, 1) = verts[1][1];
      A (1, 2) = verts[2][1];
      A (2, 0) = 1;
      A (2, 1) = 1;
      A (2, 2) = 1;
      A.lr_factors (A, r);
      double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
      reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
      double quad = 0.0;
      for (int order = 0; order < order_num; ++order) {
        double x = xytab[order * 2];
        double y = xytab[1 + order * 2];
        struct double_3d_array Auswertung_Multiscale_3d;
        Auswertung_Multiscale_3d
          = AuswertungMultiscale3d (initial_grid_hierarchy, max_level, x, y, itree, ielement, current_index);
        double value_dim1 = eval_spline (spline_d1, x, y, xacc_d1, yacc_d1) - Auswertung_Multiscale_3d.dim_val[0];
        double value_dim2 = eval_spline (spline_d2, x, y, xacc_d2, yacc_d2) - Auswertung_Multiscale_3d.dim_val[1];
        double value_dim3 = eval_spline (spline_d3, x, y, xacc_d3, yacc_d3) - Auswertung_Multiscale_3d.dim_val[2];
        if (err_type == "L1") {
          quad += wtab[order] * (abs (value_dim1) + abs (value_dim2) + abs (value_dim3));
        }
        else if (err_type == "L2") {
          quad += wtab[order] * (value_dim1 * value_dim1 + value_dim2 * value_dim2 + value_dim3 * value_dim3);
        }
        else if (err_type == "Linf") {
          quad = max (abs (value_dim1), quad);
          quad = max (abs (value_dim2), quad);
          quad = max (abs (value_dim3), quad);
        }
        else {
          printf ("Invalid action: %s\n", err_type);
        }
      }
      if (err_type == "L1" || err_type == "L2") {
        quad *= volume;
        sum += quad;
      }
      else if (err_type == "Linf") {
        sum = max (abs (quad), sum);
      }
    }
  }
  if (err_type == "L2") {
    sum = sqrt (sum);
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return sum;
}

double
ErrorMultiscaleWaveletfree (struct grid_hierarchy_waveletfree initial_grid_hierarchy, int max_level, func F, int rule,
                            const char *err_type)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  mat A;
  vector<int> r;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  T8_ASSERT (t8_forest_is_committed (initial_grid_hierarchy.lev_arr[max_level].forest_arr));

  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;
  num_local_trees = t8_forest_get_num_local_trees (initial_grid_hierarchy.lev_arr[max_level].forest_arr);
  double sum = 0.0;
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    /* This loop iterates through all local trees in the forest. */
    /* Get the number of elements of this tree. */
    num_elements_in_tree
      = t8_forest_get_tree_num_elements (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      /* This loop iterates through all the local elements of the forest in the current tree. */
      /* We can now write to the position current_index into our array in order to store
       * data for this element. */
      /* Since in this example we want to compute the data based on the element in question,
       * we need to get a pointer to this element. */
      element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, ielement);
      double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element);
      double verts[3][3] = { 0 };
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 0, verts[0]);
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 1, verts[1]);
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 2, verts[2]);
      A.resize (3, 3);
      r.resize (3);
      A (0, 0) = verts[0][0];
      A (0, 1) = verts[1][0];
      A (0, 2) = verts[2][0];
      A (1, 0) = verts[0][1];
      A (1, 1) = verts[1][1];
      A (1, 2) = verts[2][1];
      A (2, 0) = 1;
      A (2, 1) = 1;
      A (2, 2) = 1;
      A.lr_factors (A, r);
      double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
      reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
      double quad = 0.;
      for (int order = 0; order < order_num; ++order) {
        double x = xytab[order * 2];
        double y = xytab[1 + order * 2];
        //t8_global_productionf ("AuswertungMS: %f\n",AuswertungMultiscaleWaveletfree(initial_grid_hierarchy,max_level,x,y,itree,ielement,current_index));
        double value
          = F (x, y)
            - AuswertungMultiscaleWaveletfree (initial_grid_hierarchy, max_level, x, y, itree, ielement, current_index);
        if (err_type == "L1") {
          quad += wtab[order] * abs (value);
        }
        else if (err_type == "L2") {
          quad += wtab[order] * value * value;
        }
        else if (err_type == "Linf") {
          quad = max (abs (value), quad);
        }
        else {
          printf ("Invalid action: %s\n", err_type);
        }
      }
      if (err_type == "L1" || err_type == "L2") {
        quad *= volume;
        sum += quad;
      }
      else if (err_type == "Linf") {
        sum = max (abs (quad), sum);
      }
    }
  }
  if (err_type == "L2") {
    sum = sqrt (sum);
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return sum;
}

double
ErrorMultiscaleWaveletfree3d (struct grid_hierarchy_waveletfree_3d initial_grid_hierarchy, int max_level, func F1,
                              func F2, func F3, int rule, const char *err_type)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  mat A;
  vector<int> r;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  T8_ASSERT (t8_forest_is_committed (initial_grid_hierarchy.lev_arr[max_level].forest_arr));

  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;
  num_local_trees = t8_forest_get_num_local_trees (initial_grid_hierarchy.lev_arr[max_level].forest_arr);
  double sum = 0.0;
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    /* This loop iterates through all local trees in the forest. */
    /* Get the number of elements of this tree. */
    num_elements_in_tree
      = t8_forest_get_tree_num_elements (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      /* This loop iterates through all the local elements of the forest in the current tree. */
      /* We can now write to the position current_index into our array in order to store
       * data for this element. */
      /* Since in this example we want to compute the data based on the element in question,
       * we need to get a pointer to this element. */
      element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, ielement);
      double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element);
      double verts[3][3] = { 0 };
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 0, verts[0]);
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 1, verts[1]);
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 2, verts[2]);
      A.resize (3, 3);
      r.resize (3);
      A (0, 0) = verts[0][0];
      A (0, 1) = verts[1][0];
      A (0, 2) = verts[2][0];
      A (1, 0) = verts[0][1];
      A (1, 1) = verts[1][1];
      A (1, 2) = verts[2][1];
      A (2, 0) = 1;
      A (2, 1) = 1;
      A (2, 2) = 1;
      A.lr_factors (A, r);
      double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
      reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
      double quad = 0.0;
      for (int order = 0; order < order_num; ++order) {
        double x = xytab[order * 2];
        double y = xytab[1 + order * 2];
        struct double_3d_array Auswertung_Multiscale_3d;
        Auswertung_Multiscale_3d
          = AuswertungMultiscaleWaveletfree3d (initial_grid_hierarchy, max_level, x, y, itree, ielement, current_index);
        double value_dim1 = F1 (x, y) - Auswertung_Multiscale_3d.dim_val[0];
        double value_dim2 = F2 (x, y) - Auswertung_Multiscale_3d.dim_val[1];
        double value_dim3 = F3 (x, y) - Auswertung_Multiscale_3d.dim_val[2];
        if (err_type == "L1") {
          quad += wtab[order] * (abs (value_dim1) + abs (value_dim2) + abs (value_dim3));
        }
        else if (err_type == "L2") {
          quad += wtab[order] * (value_dim1 * value_dim1 + value_dim2 * value_dim2 + value_dim3 * value_dim3);
        }
        else if (err_type == "Linf") {
          quad = max (abs (value_dim1), quad);
          quad = max (abs (value_dim2), quad);
          quad = max (abs (value_dim3), quad);
        }
        else {
          printf ("Invalid action: %s\n", err_type);
        }
      }
      if (err_type == "L1" || err_type == "L2") {
        quad *= volume;
        sum += quad;
      }
      else if (err_type == "Linf") {
        sum = max (abs (quad), sum);
      }
    }
  }
  if (err_type == "L2") {
    sum = sqrt (sum);
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return sum;
}

double
ErrorMultiscaleWaveletfreeSpline (struct grid_hierarchy_waveletfree initial_grid_hierarchy, int max_level,
                                  spline eval_spline, const gsl_spline2d *spline, gsl_interp_accel *xacc,
                                  gsl_interp_accel *yacc, int rule, const char *err_type)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  mat A;
  vector<int> r;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  T8_ASSERT (t8_forest_is_committed (initial_grid_hierarchy.lev_arr[max_level].forest_arr));

  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;
  num_local_trees = t8_forest_get_num_local_trees (initial_grid_hierarchy.lev_arr[max_level].forest_arr);
  double sum = 0.0;
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    /* This loop iterates through all local trees in the forest. */
    /* Get the number of elements of this tree. */
    num_elements_in_tree
      = t8_forest_get_tree_num_elements (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      /* This loop iterates through all the local elements of the forest in the current tree. */
      /* We can now write to the position current_index into our array in order to store
       * data for this element. */
      /* Since in this example we want to compute the data based on the element in question,
       * we need to get a pointer to this element. */
      element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, ielement);
      double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element);
      double verts[3][3] = { 0 };
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 0, verts[0]);
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 1, verts[1]);
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 2, verts[2]);
      A.resize (3, 3);
      r.resize (3);
      A (0, 0) = verts[0][0];
      A (0, 1) = verts[1][0];
      A (0, 2) = verts[2][0];
      A (1, 0) = verts[0][1];
      A (1, 1) = verts[1][1];
      A (1, 2) = verts[2][1];
      A (2, 0) = 1;
      A (2, 1) = 1;
      A (2, 2) = 1;
      A.lr_factors (A, r);
      double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
      reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
      double quad = 0.;
      for (int order = 0; order < order_num; ++order) {
        double x = xytab[order * 2];
        double y = xytab[1 + order * 2];
        double value
          = eval_spline (spline, x, y, xacc, yacc)
            - AuswertungMultiscaleWaveletfree (initial_grid_hierarchy, max_level, x, y, itree, ielement, current_index);
        if (err_type == "L1") {
          quad += wtab[order] * abs (value);
        }
        else if (err_type == "L2") {
          quad += wtab[order] * value * value;
        }
        else if (err_type == "Linf") {
          quad = max (abs (value), quad);
        }
        else {
          printf ("Invalid action: %s\n", err_type);
        }
      }
      if (err_type == "L1" || err_type == "L2") {
        quad *= volume;
        sum += quad;
      }
      else if (err_type == "Linf") {
        sum = max (abs (quad), sum);
      }
    }
  }
  if (err_type == "L2") {
    sum = sqrt (sum);
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return sum;
}

double
ErrorMultiscaleWaveletfree3dSpline (struct grid_hierarchy_waveletfree_3d initial_grid_hierarchy, int max_level,
                                    spline eval_spline, const gsl_spline2d *spline_d1, gsl_interp_accel *xacc_d1,
                                    gsl_interp_accel *yacc_d1, const gsl_spline2d *spline_d2, gsl_interp_accel *xacc_d2,
                                    gsl_interp_accel *yacc_d2, const gsl_spline2d *spline_d3, gsl_interp_accel *xacc_d3,
                                    gsl_interp_accel *yacc_d3, int rule, const char *err_type)
{
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  mat A;
  vector<int> r;
  order_num = dunavant_order_num (rule);
  wtab = T8_ALLOC (double, order_num);
  xytab = T8_ALLOC (double, 2 * order_num);
  xytab_ref = T8_ALLOC (double, 2 * order_num);
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  T8_ASSERT (t8_forest_is_committed (initial_grid_hierarchy.lev_arr[max_level].forest_arr));

  t8_locidx_t itree, num_local_trees;
  t8_locidx_t current_index;
  t8_locidx_t ielement, num_elements_in_tree;
  const t8_element_t *element;
  num_local_trees = t8_forest_get_num_local_trees (initial_grid_hierarchy.lev_arr[max_level].forest_arr);
  double sum = 0.0;
  for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
    /* This loop iterates through all local trees in the forest. */
    /* Get the number of elements of this tree. */
    num_elements_in_tree
      = t8_forest_get_tree_num_elements (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree);
    for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
      /* This loop iterates through all the local elements of the forest in the current tree. */
      /* We can now write to the position current_index into our array in order to store
       * data for this element. */
      /* Since in this example we want to compute the data based on the element in question,
       * we need to get a pointer to this element. */
      element = t8_forest_get_element_in_tree (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, ielement);
      double volume = t8_forest_element_volume (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element);
      double verts[3][3] = { 0 };
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 0, verts[0]);
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 1, verts[1]);
      t8_forest_element_coordinate (initial_grid_hierarchy.lev_arr[max_level].forest_arr, itree, element, 2, verts[2]);
      A.resize (3, 3);
      r.resize (3);
      A (0, 0) = verts[0][0];
      A (0, 1) = verts[1][0];
      A (0, 2) = verts[2][0];
      A (1, 0) = verts[0][1];
      A (1, 1) = verts[1][1];
      A (1, 2) = verts[2][1];
      A (2, 0) = 1;
      A (2, 1) = 1;
      A (2, 2) = 1;
      A.lr_factors (A, r);
      double eckpunkte[6] = { verts[0][0], verts[0][1], verts[1][0], verts[1][1], verts[2][0], verts[2][1] };
      reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
      double quad = 0.0;
      for (int order = 0; order < order_num; ++order) {
        double x = xytab[order * 2];
        double y = xytab[1 + order * 2];
        struct double_3d_array Auswertung_Multiscale_3d;
        Auswertung_Multiscale_3d
          = AuswertungMultiscaleWaveletfree3d (initial_grid_hierarchy, max_level, x, y, itree, ielement, current_index);
        double value_dim1 = eval_spline (spline_d1, x, y, xacc_d1, yacc_d1) - Auswertung_Multiscale_3d.dim_val[0];
        double value_dim2 = eval_spline (spline_d2, x, y, xacc_d2, yacc_d2) - Auswertung_Multiscale_3d.dim_val[1];
        double value_dim3 = eval_spline (spline_d3, x, y, xacc_d3, yacc_d3) - Auswertung_Multiscale_3d.dim_val[2];
        if (err_type == "L1") {
          quad += wtab[order] * (abs (value_dim1) + abs (value_dim2) + abs (value_dim3));
        }
        else if (err_type == "L2") {
          quad += wtab[order] * (value_dim1 * value_dim1 + value_dim2 * value_dim2 + value_dim3 * value_dim3);
        }
        else if (err_type == "Linf") {
          quad = max (abs (value_dim1), quad);
          quad = max (abs (value_dim2), quad);
          quad = max (abs (value_dim3), quad);
        }
        else {
          printf ("Invalid action: %s\n", err_type);
        }
      }
      if (err_type == "L1" || err_type == "L2") {
        quad *= volume;
        sum += quad;
      }
      else if (err_type == "Linf") {
        sum = max (abs (quad), sum);
      }
    }
  }
  if (err_type == "L2") {
    sum = sqrt (sum);
  }
  T8_FREE (wtab);
  T8_FREE (xytab);
  T8_FREE (xytab_ref);
  return sum;
}

// struct grid_hierarchy initialize_grid_hierarchy(t8_cmesh_t cmesh,t8_scheme * scheme,func F,sc_MPI_Comm comm, const int rule, const int max_level){
//   struct grid_hierarchy initial_grid_hierarchy;
//   for (int level=0; level < max_level+1; ++level) {
//     initial_grid_hierarchy.lev_arr[level].forest_arr=t8_forest_new_uniform (cmesh, scheme, level, 0, comm);
//     initial_grid_hierarchy.lev_arr[level].data_arr= t8_create_element_data (initial_grid_hierarchy.lev_arr[level].forest_arr,F,rule,max_level);
//   }
//   return initial_grid_hierarchy;
// }

struct grid_hierarchy
initialize_grid_hierarchy (t8_cmesh_t cmesh, const t8_scheme *scheme, func F, sc_MPI_Comm comm, const int rule,
                           const int max_level)
{
  struct grid_hierarchy initial_grid_hierarchy;
  for (int level = 0; level < max_level + 1; ++level) {
    initial_grid_hierarchy.lev_arr[level].forest_arr = t8_forest_new_uniform (cmesh, scheme, level, 0, comm);
    initial_grid_hierarchy.lev_arr[level].data_arr
      = t8_create_element_data (initial_grid_hierarchy, level, F, rule, max_level);
  }
  return initial_grid_hierarchy;
}

struct grid_hierarchy_3d
initialize_grid_hierarchy_3d (t8_cmesh_t cmesh, const t8_scheme *scheme, func F1, func F2, func F3, sc_MPI_Comm comm,
                              const int rule, const int max_level)
{
  struct grid_hierarchy_3d initial_grid_hierarchy;
  for (int level = 0; level < max_level + 1; ++level) {
    initial_grid_hierarchy.lev_arr[level].forest_arr = t8_forest_new_uniform (cmesh, scheme, level, 0, comm);
    initial_grid_hierarchy.lev_arr[level].data_arr
      = t8_create_element_data_3d (initial_grid_hierarchy, level, F1, F2, F3, rule, max_level);
  }
  return initial_grid_hierarchy;
}

struct grid_hierarchy_waveletfree
initialize_grid_hierarchy_waveletfree (t8_cmesh_t cmesh, const t8_scheme *scheme, func F, sc_MPI_Comm comm,
                                       const int rule, const int max_level)
{
  struct grid_hierarchy_waveletfree initial_grid_hierarchy;
  for (int level = 0; level < max_level + 1; ++level) {
    initial_grid_hierarchy.lev_arr[level].forest_arr = t8_forest_new_uniform (cmesh, scheme, level, 0, comm);
    initial_grid_hierarchy.lev_arr[level].data_arr
      = t8_create_element_data_waveletfree (initial_grid_hierarchy, level, F, rule, max_level);
  }
  return initial_grid_hierarchy;
}

struct grid_hierarchy_waveletfree_3d
initialize_grid_hierarchy_waveletfree_3d (t8_cmesh_t cmesh, const t8_scheme *scheme, func F1, func F2, func F3,
                                          sc_MPI_Comm comm, const int rule, const int max_level)
{
  struct grid_hierarchy_waveletfree_3d initial_grid_hierarchy;
  for (int level = 0; level < max_level + 1; ++level) {
    initial_grid_hierarchy.lev_arr[level].forest_arr = t8_forest_new_uniform (cmesh, scheme, level, 0, comm);
    initial_grid_hierarchy.lev_arr[level].data_arr
      = t8_create_element_data_waveletfree_3d (initial_grid_hierarchy, level, F1, F2, F3, rule, max_level);
  }
  return initial_grid_hierarchy;
}

struct grid_hierarchy
initialize_grid_hierarchy_spline (t8_cmesh_t cmesh, const t8_scheme *scheme, spline eval_spline,
                                  const gsl_spline2d *spline, gsl_interp_accel *xacc, gsl_interp_accel *yacc,
                                  sc_MPI_Comm comm, const int rule, const int max_level)
{
  struct grid_hierarchy initial_grid_hierarchy;
  for (int level = 0; level < max_level + 1; ++level) {
    initial_grid_hierarchy.lev_arr[level].forest_arr = t8_forest_new_uniform (cmesh, scheme, level, 0, comm);
    initial_grid_hierarchy.lev_arr[level].data_arr
      = t8_create_element_data_spline (initial_grid_hierarchy, level, eval_spline, spline, xacc, yacc, rule, max_level);
  }
  return initial_grid_hierarchy;
}

struct grid_hierarchy_3d
initialize_grid_hierarchy_3d_spline (t8_cmesh_t cmesh, const t8_scheme *scheme, spline eval_spline,
                                     const gsl_spline2d *spline_d1, const gsl_spline2d *spline_d2,
                                     const gsl_spline2d *spline_d3, gsl_interp_accel *xacc_d1,
                                     gsl_interp_accel *yacc_d1, gsl_interp_accel *xacc_d2, gsl_interp_accel *yacc_d2,
                                     gsl_interp_accel *xacc_d3, gsl_interp_accel *yacc_d3, sc_MPI_Comm comm,
                                     const int rule, const int max_level)
{
  struct grid_hierarchy_3d initial_grid_hierarchy;
  for (int level = 0; level < max_level + 1; ++level) {
    initial_grid_hierarchy.lev_arr[level].forest_arr = t8_forest_new_uniform (cmesh, scheme, level, 0, comm);
    initial_grid_hierarchy.lev_arr[level].data_arr
      = t8_create_element_data_3d_spline (initial_grid_hierarchy, level, eval_spline, spline_d1, spline_d2, spline_d3,
                                          xacc_d1, yacc_d1, xacc_d2, yacc_d2, xacc_d3, yacc_d3, rule, max_level);
  }
  return initial_grid_hierarchy;
}

struct grid_hierarchy_waveletfree
initialize_grid_hierarchy_waveletfree_spline (t8_cmesh_t cmesh, const t8_scheme *scheme, spline eval_spline,
                                              const gsl_spline2d *spline, gsl_interp_accel *xacc,
                                              gsl_interp_accel *yacc, sc_MPI_Comm comm, const int rule,
                                              const int max_level)
{
  struct grid_hierarchy_waveletfree initial_grid_hierarchy;
  for (int level = 0; level < max_level + 1; ++level) {
    initial_grid_hierarchy.lev_arr[level].forest_arr = t8_forest_new_uniform (cmesh, scheme, level, 0, comm);
    initial_grid_hierarchy.lev_arr[level].data_arr = t8_create_element_data_waveletfree_spline (
      initial_grid_hierarchy, level, eval_spline, spline, xacc, yacc, rule, max_level);
  }
  return initial_grid_hierarchy;
}

struct grid_hierarchy_waveletfree_3d
initialize_grid_hierarchy_waveletfree_3d_spline (t8_cmesh_t cmesh, const t8_scheme *scheme, spline eval_spline,
                                                 const gsl_spline2d *spline_d1, const gsl_spline2d *spline_d2,
                                                 const gsl_spline2d *spline_d3, gsl_interp_accel *xacc_d1,
                                                 gsl_interp_accel *yacc_d1, gsl_interp_accel *xacc_d2,
                                                 gsl_interp_accel *yacc_d2, gsl_interp_accel *xacc_d3,
                                                 gsl_interp_accel *yacc_d3, sc_MPI_Comm comm, const int rule,
                                                 const int max_level)
{
  struct grid_hierarchy_waveletfree_3d initial_grid_hierarchy;
  for (int level = 0; level < max_level + 1; ++level) {
    initial_grid_hierarchy.lev_arr[level].forest_arr = t8_forest_new_uniform (cmesh, scheme, level, 0, comm);
    initial_grid_hierarchy.lev_arr[level].data_arr = t8_create_element_data_waveletfree_3d_spline (
      initial_grid_hierarchy, level, eval_spline, spline_d1, spline_d2, spline_d3, xacc_d1, yacc_d1, xacc_d2, yacc_d2,
      xacc_d3, yacc_d3, rule, max_level);
  }
  return initial_grid_hierarchy;
}

void
deref_grid_hierarchy (struct grid_hierarchy initial_grid_hierarchy)
{
  for (int l = 0; l < max_level + 1; l++) {
    int refcount = initial_grid_hierarchy.lev_arr[l].forest_arr->rc.refcount;
    initial_grid_hierarchy.lev_arr[l].forest_arr->do_dup = 0;
    while (refcount > 0) {
      if (l <= max_level) {

        if (refcount == 1 && l > 0 && l < max_level - 1) {
          T8_FREE (initial_grid_hierarchy.lev_arr[l].data_arr);
          //t8_scheme_ref (initial_grid_hierarchy.lev_arr[l].forest_arr->scheme);
          t8_cmesh_ref (initial_grid_hierarchy.lev_arr[l].forest_arr->cmesh);
        }
      }
      t8_forest_unref (&initial_grid_hierarchy.lev_arr[l].forest_arr);
      refcount--;
    }
  }
}

void
deref_grid_hierarchy_wf (struct grid_hierarchy_waveletfree initial_grid_hierarchy)
{
  for (int l = 0; l < max_level + 1; l++) {
    int refcount = initial_grid_hierarchy.lev_arr[l].forest_arr->rc.refcount;
    initial_grid_hierarchy.lev_arr[l].forest_arr->do_dup = 0;
    while (refcount > 0) {
      if (l <= max_level) {

        if (refcount == 1 && l > 0 && l < max_level - 1) {
          T8_FREE (initial_grid_hierarchy.lev_arr[l].data_arr);
          //t8_scheme_ref (initial_grid_hierarchy.lev_arr[l].forest_arr->scheme);
          t8_cmesh_ref (initial_grid_hierarchy.lev_arr[l].forest_arr->cmesh);
        }
      }
      t8_forest_unref (&initial_grid_hierarchy.lev_arr[l].forest_arr);
      refcount--;
    }
  }
}

void
deref_grid_hierarchy_3d (struct grid_hierarchy_3d initial_grid_hierarchy)
{
  for (int l = 0; l < max_level + 1; l++) {
    int refcount = initial_grid_hierarchy.lev_arr[l].forest_arr->rc.refcount;
    initial_grid_hierarchy.lev_arr[l].forest_arr->do_dup = 0;
    while (refcount > 0) {
      if (l <= max_level) {

        if (refcount == 1 && l > 0 && l < max_level - 1) {
          T8_FREE (initial_grid_hierarchy.lev_arr[l].data_arr);
          //t8_scheme_ref (initial_grid_hierarchy.lev_arr[l].forest_arr->scheme);
          t8_cmesh_ref (initial_grid_hierarchy.lev_arr[l].forest_arr->cmesh);
        }
      }
      t8_forest_unref (&initial_grid_hierarchy.lev_arr[l].forest_arr);
      refcount--;
    }
  }
}

void
deref_grid_hierarchy_3d_wf (struct grid_hierarchy_waveletfree_3d initial_grid_hierarchy)
{
  for (int l = 0; l < max_level + 1; l++) {
    int refcount = initial_grid_hierarchy.lev_arr[l].forest_arr->rc.refcount;
    initial_grid_hierarchy.lev_arr[l].forest_arr->do_dup = 0;
    while (refcount > 0) {
      if (l <= max_level) {

        if (refcount == 1 && l > 0 && l < max_level - 1) {
          T8_FREE (initial_grid_hierarchy.lev_arr[l].data_arr);
          //t8_scheme_ref (initial_grid_hierarchy.lev_arr[l].forest_arr->scheme);
          t8_cmesh_ref (initial_grid_hierarchy.lev_arr[l].forest_arr->cmesh);
        }
      }
      t8_forest_unref (&initial_grid_hierarchy.lev_arr[l].forest_arr);
      refcount--;
    }
  }
}

int
t8_msa_adapt_callback (t8_forest_t forest, t8_forest_t forest_from, t8_locidx_t which_tree,
                       const t8_eclass_t tree_class, t8_locidx_t lelement_id, const t8_scheme *ts, const int is_family,
                       const int num_elements, t8_element_t *elements[])
{
  /* Our adaptation criterion is to look at whether cells have significant local contributions, then we marked them as adaptiert. */

  /* access user with t8_forest_get_user_data (forest). */
  struct lmi_adapt *adapt_data = (struct lmi_adapt *) t8_forest_get_user_data (forest);

  /* Offset is first element index in tree. */
  t8_locidx_t offset = t8_forest_get_tree_element_offset (forest_from, which_tree);
  /* From this we calculate the local element id. */
  t8_locidx_t elem_id = lelement_id + offset;
  /* You can use T8_ASSERT for assertions that are active in debug mode (when configured with --enable-debug).
   * If the condition is not true, then the code will abort.
   * In this case, we want to make sure that we actually did set a user pointer to forest and thus
   * did not get the NULL pointer from t8_forest_get_user_data.
   */
  //t8_locidx_t elem_id_ref=t8_lmi_to_elem_id (adapt_data->adapt_lmi_data[elem_id].lmi,basecell_num_digits_offset);
  uint64_t elem_id_ref = t8_lmi_to_elem_id_binary (adapt_data->adapt_lmi_data[elem_id].lmi);
  T8_ASSERT (adapt_data != NULL);
  if ((adapt_data->ref_grid_data[elem_id_ref].signifikant)
      && (get_level_only_lmi (adapt_data->adapt_lmi_data[elem_id].lmi)
          == get_level_only_lmi (adapt_data->ref_grid_data[elem_id_ref].lmi))) {
    /* Refine this element. */
    return 1;
  }
  return 0;
}

t8_data_per_element_adapt *
initialize_lmi_adapt_data_new (t8_forest_t forest_adapt)
{
  t8_locidx_t num_local_trees;
  t8_locidx_t num_local_elements;
  t8_locidx_t num_ghost_elements;
  t8_locidx_t num_elements;
  const t8_element_t *element;
  struct t8_data_per_element_adapt *adapt_data;
  T8_ASSERT (t8_forest_is_committed (forest_adapt));
  /* Get the number of local elements of forest. */
  num_local_elements = t8_forest_get_local_num_elements (forest_adapt);
  /* Get the number of ghost elements of forest. */
  num_ghost_elements = t8_forest_get_num_ghosts (forest_adapt);
  num_local_trees = t8_forest_get_num_local_trees (forest_adapt);
  num_elements = num_ghost_elements + num_local_elements;
  adapt_data = T8_ALLOC (struct t8_data_per_element_adapt, num_elements);
  {
    const t8_scheme *eclass_scheme;
    t8_locidx_t itree;
    t8_locidx_t current_index;
    t8_locidx_t ielement;
    t8_eclass_t tree_class;
    t8_locidx_t num_elements_in_tree;

    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      tree_class = t8_forest_get_tree_class (forest_adapt, itree);
      eclass_scheme = t8_forest_get_scheme (forest_adapt);
      tree_class = t8_forest_get_tree_class (forest_adapt, itree);
      /* Get the number of elements of this tree. */
      num_elements_in_tree = t8_forest_get_tree_num_elements (forest_adapt, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        element = t8_forest_get_element_in_tree (forest_adapt, itree, ielement);
        /* We want to store the elements level and its volume as data. We compute these
         * via the eclass_scheme and the forest_element interface. */
        adapt_data[current_index].lmi = create_lmi_from_level (eclass_scheme->element_get_level (tree_class, element));
      }
    }
  }
  return adapt_data;
}

t8_data_per_element_adapt *
initialize_lmi_adapt_data (struct grid_hierarchy initial_grid_hierarchy)
{
  t8_forest_t forest_adapt = initial_grid_hierarchy.lev_arr[0].forest_arr;
  t8_locidx_t num_local_trees;
  t8_locidx_t num_local_elements;
  t8_locidx_t num_ghost_elements;
  struct t8_data_per_element_adapt *adapt_data;
  T8_ASSERT (t8_forest_is_committed (forest_adapt));
  /* Get the number of local elements of forest. */
  num_local_elements = t8_forest_get_local_num_elements (forest_adapt);
  /* Get the number of ghost elements of forest. */
  num_ghost_elements = t8_forest_get_num_ghosts (forest_adapt);
  num_local_trees = t8_forest_get_num_local_trees (forest_adapt);
  adapt_data = T8_ALLOC (struct t8_data_per_element_adapt, num_ghost_elements + num_local_elements);
  {
    t8_locidx_t itree;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;

    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (forest_adapt, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        adapt_data[current_index].lmi = initial_grid_hierarchy.lev_arr[0].data_arr[current_index].lmi;
      }
    }
  }
  return adapt_data;
}

t8_data_per_element_adapt *
initialize_lmi_adapt_data_3d (struct grid_hierarchy_3d initial_grid_hierarchy)
{
  t8_forest_t forest_adapt = initial_grid_hierarchy.lev_arr[0].forest_arr;
  t8_locidx_t num_local_trees;
  t8_locidx_t num_local_elements;
  t8_locidx_t num_ghost_elements;
  struct t8_data_per_element_adapt *adapt_data;
  T8_ASSERT (t8_forest_is_committed (forest_adapt));
  /* Get the number of local elements of forest. */
  num_local_elements = t8_forest_get_local_num_elements (forest_adapt);
  /* Get the number of ghost elements of forest. */
  num_ghost_elements = t8_forest_get_num_ghosts (forest_adapt);
  num_local_trees = t8_forest_get_num_local_trees (forest_adapt);
  adapt_data = T8_ALLOC (struct t8_data_per_element_adapt, num_ghost_elements + num_local_elements);
  {
    t8_locidx_t itree;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;

    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (forest_adapt, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        adapt_data[current_index].lmi = initial_grid_hierarchy.lev_arr[0].data_arr[current_index].lmi;
      }
    }
  }
  return adapt_data;
}

t8_data_per_element_adapt *
initialize_lmi_adapt_data_wf (struct grid_hierarchy_waveletfree initial_grid_hierarchy)
{
  t8_forest_t forest_adapt = initial_grid_hierarchy.lev_arr[0].forest_arr;
  t8_locidx_t num_local_trees;
  t8_locidx_t num_local_elements;
  t8_locidx_t num_ghost_elements;
  struct t8_data_per_element_adapt *adapt_data;
  T8_ASSERT (t8_forest_is_committed (forest_adapt));
  /* Get the number of local elements of forest. */
  num_local_elements = t8_forest_get_local_num_elements (forest_adapt);
  /* Get the number of ghost elements of forest. */
  num_ghost_elements = t8_forest_get_num_ghosts (forest_adapt);
  num_local_trees = t8_forest_get_num_local_trees (forest_adapt);
  adapt_data = T8_ALLOC (struct t8_data_per_element_adapt, num_ghost_elements + num_local_elements);
  {
    t8_locidx_t itree;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;

    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (forest_adapt, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        adapt_data[current_index].lmi = initial_grid_hierarchy.lev_arr[0].data_arr[current_index].lmi;
      }
    }
  }
  return adapt_data;
}

t8_data_per_element_adapt *
initialize_lmi_adapt_data_wf_3d (struct grid_hierarchy_waveletfree_3d initial_grid_hierarchy)
{
  t8_forest_t forest_adapt = initial_grid_hierarchy.lev_arr[0].forest_arr;
  t8_locidx_t num_local_trees;
  t8_locidx_t num_local_elements;
  t8_locidx_t num_ghost_elements;
  struct t8_data_per_element_adapt *adapt_data;
  T8_ASSERT (t8_forest_is_committed (forest_adapt));
  /* Get the number of local elements of forest. */
  num_local_elements = t8_forest_get_local_num_elements (forest_adapt);
  /* Get the number of ghost elements of forest. */
  num_ghost_elements = t8_forest_get_num_ghosts (forest_adapt);
  num_local_trees = t8_forest_get_num_local_trees (forest_adapt);
  adapt_data = T8_ALLOC (struct t8_data_per_element_adapt, num_ghost_elements + num_local_elements);
  {
    t8_locidx_t itree;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;

    for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
      num_elements_in_tree = t8_forest_get_tree_num_elements (forest_adapt, itree);
      for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
        adapt_data[current_index].lmi = initial_grid_hierarchy.lev_arr[0].data_arr[current_index].lmi;
      }
    }
  }
  return adapt_data;
}

/* If someone wants to plot all grid levels in one.  */
static void
t8_output_data_to_vtu_hierarchy (struct grid_hierarchy initial_grid_hierarchy)
{
  char c;
  const char *cstr;
  for (int l = 0; l < max_level + 1; l++) {
    const char *cstr;
    c = l + '0';  //funktioniert nur bis level 9
    string y ("Grid_hierarchy_level_");
    y.push_back (c);
    cstr = y.c_str ();
    t8_output_data_to_vtu (initial_grid_hierarchy.lev_arr[l].forest_arr, initial_grid_hierarchy.lev_arr[l].data_arr,
                           cstr);
  }
}

/*
 * MSA: Multiscaleanalysis: We perform the steps
 * Optionen: Waveletfree oder classical
 */
void
t8_msa (func F, struct grid_hierarchy initial_grid_hierarchy, double c_tresh, int max_lev, const char *err_type)
{
  t8_forest_t forest = initial_grid_hierarchy.lev_arr[0].forest_arr;
  t8_forest_t forest_adapt;
  unsigned int anzahl_gesamt;
  unsigned int anzahl_klein;
  struct lmi_adapt adapt_data;
  struct lmi_adapt adapt_data_new;
  //MultiScaleOperatorWaveletFree(initial_grid_hierarchy);
  //InverseMultiScaleOperatorwaveletfree(initial_grid_hierarchy);

  MultiScaleOperator (initial_grid_hierarchy);
  double error = ErrorSinglescale (initial_grid_hierarchy.lev_arr[max_lev].forest_arr,
                                   initial_grid_hierarchy.lev_arr[max_lev].data_arr, F, 10, err_type);
  t8_global_productionf ("Error SS:%f \n", error);
  InverseMultiScaleOperator (initial_grid_hierarchy);
  error = ErrorSinglescale (initial_grid_hierarchy.lev_arr[max_lev].forest_arr,
                            initial_grid_hierarchy.lev_arr[max_lev].data_arr, F, 10, err_type);
  t8_global_productionf ("Error SS danach:%f \n", error);
  /*
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  */
  //error = ErrorSinglescale(initial_grid_hierarchy.lev_arr[max_lev].forest_arr,initial_grid_hierarchy.lev_arr[max_lev].data_arr,F,10);
  //t8_global_productionf ("Error SS:%f \n",error);
  //MultiScaleOperator(initial_grid_hierarchy);
  //HierarchischerThresholdOperator(initial_grid_hierarchy,c_tresh, 2.0,anzahl_gesamt,anzahl_klein);
  t8_global_productionf ("Vorher \n");
  error = ErrorMultiscale (initial_grid_hierarchy, max_lev, F, 10, err_type);
  t8_global_productionf ("Nachher \n");
  t8_global_productionf ("Error MS:%f \n", error);
  //MultiScaleOperatorWaveletFree(initial_grid_hierarchy);

  // nach trehsholding
  //ThresholdOperatorwaveletfree(initial_grid_hierarchy,c_tresh, 2.0,anzahl_gesamt,anzahl_klein);
  HierarchischerThresholdOperator (initial_grid_hierarchy, c_tresh, 2.0, anzahl_gesamt, anzahl_klein);
  //InverseMultiScaleOperatorwaveletfree(initial_grid_hierarchy);
  error = ErrorSinglescale (initial_grid_hierarchy.lev_arr[max_level].forest_arr,
                            initial_grid_hierarchy.lev_arr[max_level].data_arr, F, 10, err_type);
  //HierarchischerThresholdOperator(initial_grid_hierarchy,c_tresh, 2.0,anzahl_gesamt,anzahl_klein);
  //ThresholdOperator(initial_grid_hierarchy,c_tresh, 2.0,anzahl_gesamt,anzahl_klein);
  //GridAdaptation(initial_grid_hierarchy, c_tresh, 2.0);
  //grading_grid(initial_grid_hierarchy);
  adapt_data.ref_grid_data = initial_grid_hierarchy.lev_arr[0].data_arr;
  adapt_data.adapt_lmi_data = initialize_lmi_adapt_data (initial_grid_hierarchy);
  t8_forest_set_user_data (forest, &adapt_data);
  T8_ASSERT (t8_forest_is_committed (forest));
  t8_forest_ref (forest);
  forest_adapt = t8_adapt_forest (forest, t8_msa_adapt_callback, 0, 0, &adapt_data);

  adapt_data_new.ref_grid_data = initial_grid_hierarchy.lev_arr[0].data_arr;
  adapt_data_new.adapt_lmi_data = initialize_lmi_adapt_data_new (forest_adapt);
  t8_forest_set_user_data (forest_adapt, &adapt_data_new);
  t8_forest_iterate_replace (forest_adapt, forest, t8_forest_replace);
  /* Write the adapted forest to a vtu file */
  adapt_data_new = *(struct lmi_adapt *) t8_forest_get_user_data (forest_adapt);
  //t8_write_vtu (forest_adapt, adapt_data, "t8_step7_adapt_forest");

  /* Free the memory */

  T8_FREE (adapt_data.adapt_lmi_data);
  adapt_data.adapt_lmi_data
    = T8_ALLOC (struct t8_data_per_element_adapt, t8_forest_get_local_num_elements (forest_adapt));

  /* Save the new forest as old forest */
  //t8_scheme_cxx_ref (forest->scheme_cxx);
  //t8_cmesh_ref (forest->cmesh);
  t8_forest_ref (initial_grid_hierarchy.lev_arr[0].forest_arr);
  t8_forest_unref (&forest);
  forest = forest_adapt;
  adapt_data = adapt_data_new;

  for (int lev = 1; lev < max_level; lev++) {
    adapt_data.ref_grid_data = initial_grid_hierarchy.lev_arr[lev].data_arr;
    t8_forest_set_user_data (forest, &adapt_data);
    t8_forest_ref (forest);
    forest_adapt = t8_adapt_forest (forest, t8_msa_adapt_callback, 0, 0, &adapt_data);

    adapt_data_new.ref_grid_data = initial_grid_hierarchy.lev_arr[lev].data_arr;
    adapt_data_new.adapt_lmi_data = initialize_lmi_adapt_data_new (forest_adapt);

    t8_forest_set_user_data (forest_adapt, &adapt_data_new);
    t8_forest_iterate_replace (forest_adapt, forest, t8_forest_replace);

    /* Write the adapted forest to a vtu file */
    struct lmi_adapt *elem_data = (struct lmi_adapt *) t8_forest_get_user_data (forest_adapt);

    //t8_write_vtu (forest_adapt, adapt_data, "t8_step7_adapt_forest");
    /* Free the memory */
    T8_FREE (adapt_data.adapt_lmi_data);
    adapt_data.adapt_lmi_data
      = T8_ALLOC (struct t8_data_per_element_adapt, t8_forest_get_local_num_elements (forest_adapt));

    /* Save the new forest as old forest */
    //t8_forest_ref (initial_grid_hierarchy.lev_arr[0].forest_arr);
    t8_forest_unref (&forest);
    forest = forest_adapt;
    adapt_data = adapt_data_new;
  }
  //t8_forest_unref (&forest_adapt);
  t8_forest_write_vtk (forest, "adapted_forest");
  T8_FREE (adapt_data.adapt_lmi_data);
  T8_FREE (adapt_data.ref_grid_data);
  //T8_FREE (&adapt_data_new);
  //T8_FREE (&adapt_data);
  //t8_forest_unref (&forest_adapt);
}

/*
 * MSA: Multiscaleanalysis: We perform the steps
 * Optionen: Waveletfree oder classical
 */
void
t8_msa_wf (func F, struct grid_hierarchy_waveletfree initial_grid_hierarchy, double c_tresh, int max_lev,
           const char *err_type)
{
  t8_forest_t forest = initial_grid_hierarchy.lev_arr[0].forest_arr;
  t8_forest_t forest_adapt;
  unsigned int anzahl_gesamt;
  unsigned int anzahl_klein;
  struct lmi_adapt_waveletfree adapt_data;
  struct lmi_adapt_waveletfree adapt_data_new;
  //MultiScaleOperatorWaveletFree(initial_grid_hierarchy);
  //InverseMultiScaleOperatorwaveletfree(initial_grid_hierarchy);

  MultiScaleOperatorWaveletFree (initial_grid_hierarchy);
  double error = ErrorSinglescaleWaveletfree (initial_grid_hierarchy.lev_arr[max_lev].forest_arr,
                                              initial_grid_hierarchy.lev_arr[max_lev].data_arr, F, 10, err_type);
  t8_global_productionf ("Error SS:%f \n", error);
  InverseMultiScaleOperatorwaveletfree (initial_grid_hierarchy);
  error = ErrorSinglescaleWaveletfree (initial_grid_hierarchy.lev_arr[max_lev].forest_arr,
                                       initial_grid_hierarchy.lev_arr[max_lev].data_arr, F, 10, err_type);
  t8_global_productionf ("Error SS danach:%f \n", error);
  /*
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  */
  //error = ErrorSinglescale(initial_grid_hierarchy.lev_arr[max_lev].forest_arr,initial_grid_hierarchy.lev_arr[max_lev].data_arr,F,10);
  //t8_global_productionf ("Error SS:%f \n",error);
  //MultiScaleOperator(initial_grid_hierarchy);
  //HierarchischerThresholdOperator(initial_grid_hierarchy,c_tresh, 2.0,anzahl_gesamt,anzahl_klein);
  t8_global_productionf ("Vorher \n");
  error = ErrorMultiscaleWaveletfree (initial_grid_hierarchy, max_lev, F, 10, err_type);
  t8_global_productionf ("Nachher \n");
  t8_global_productionf ("Error MS:%f \n", error);
  //MultiScaleOperatorWaveletFree(initial_grid_hierarchy);

  // nach trehsholding
  //ThresholdOperatorwaveletfree(initial_grid_hierarchy,c_tresh, 2.0,anzahl_gesamt,anzahl_klein);
  HierarchischerThresholdOperatorwaveletfree (initial_grid_hierarchy, c_tresh, 2.0, anzahl_gesamt, anzahl_klein);
  //InverseMultiScaleOperatorwaveletfree(initial_grid_hierarchy);
  error = ErrorSinglescaleWaveletfree (initial_grid_hierarchy.lev_arr[max_level].forest_arr,
                                       initial_grid_hierarchy.lev_arr[max_level].data_arr, F, 10, err_type);
  //HierarchischerThresholdOperator(initial_grid_hierarchy,c_tresh, 2.0,anzahl_gesamt,anzahl_klein);
  //ThresholdOperator(initial_grid_hierarchy,c_tresh, 2.0,anzahl_gesamt,anzahl_klein);
  //GridAdaptation(initial_grid_hierarchy, c_tresh, 2.0);
  //grading_grid(initial_grid_hierarchy);
  adapt_data.ref_grid_data = initial_grid_hierarchy.lev_arr[0].data_arr;
  adapt_data.adapt_lmi_data = initialize_lmi_adapt_data_wf (initial_grid_hierarchy);
  t8_forest_set_user_data (forest, &adapt_data);
  T8_ASSERT (t8_forest_is_committed (forest));
  t8_forest_ref (forest);
  forest_adapt = t8_adapt_forest (forest, t8_msa_adapt_callback, 0, 0, &adapt_data);

  adapt_data_new.ref_grid_data = initial_grid_hierarchy.lev_arr[0].data_arr;
  adapt_data_new.adapt_lmi_data = initialize_lmi_adapt_data_new (forest_adapt);
  t8_forest_set_user_data (forest_adapt, &adapt_data_new);
  t8_forest_iterate_replace (forest_adapt, forest, t8_forest_replace);
  /* Write the adapted forest to a vtu file */
  adapt_data_new = *(struct lmi_adapt_waveletfree *) t8_forest_get_user_data (forest_adapt);
  //t8_write_vtu (forest_adapt, adapt_data, "t8_step7_adapt_forest");

  /* Free the memory */

  T8_FREE (adapt_data.adapt_lmi_data);
  adapt_data.adapt_lmi_data
    = T8_ALLOC (struct t8_data_per_element_adapt, t8_forest_get_local_num_elements (forest_adapt));

  /* Save the new forest as old forest */
  //t8_scheme_cxx_ref (forest->scheme_cxx);
  //t8_cmesh_ref (forest->cmesh);
  t8_forest_ref (initial_grid_hierarchy.lev_arr[0].forest_arr);
  t8_forest_unref (&forest);
  forest = forest_adapt;
  adapt_data = adapt_data_new;

  for (int lev = 1; lev < max_level; lev++) {
    adapt_data.ref_grid_data = initial_grid_hierarchy.lev_arr[lev].data_arr;
    t8_forest_set_user_data (forest, &adapt_data);
    t8_forest_ref (forest);
    forest_adapt = t8_adapt_forest (forest, t8_msa_adapt_callback, 0, 0, &adapt_data);

    adapt_data_new.ref_grid_data = initial_grid_hierarchy.lev_arr[lev].data_arr;
    adapt_data_new.adapt_lmi_data = initialize_lmi_adapt_data_new (forest_adapt);

    t8_forest_set_user_data (forest_adapt, &adapt_data_new);
    t8_forest_iterate_replace (forest_adapt, forest, t8_forest_replace);

    /* Write the adapted forest to a vtu file */
    struct lmi_adapt_waveletfree *elem_data = (struct lmi_adapt_waveletfree *) t8_forest_get_user_data (forest_adapt);

    //t8_write_vtu (forest_adapt, adapt_data, "t8_step7_adapt_forest");
    /* Free the memory */
    T8_FREE (adapt_data.adapt_lmi_data);
    adapt_data.adapt_lmi_data
      = T8_ALLOC (struct t8_data_per_element_adapt, t8_forest_get_local_num_elements (forest_adapt));

    /* Save the new forest as old forest */
    //t8_forest_ref (initial_grid_hierarchy.lev_arr[0].forest_arr);
    t8_forest_unref (&forest);
    forest = forest_adapt;
    adapt_data = adapt_data_new;
  }
  //t8_forest_unref (&forest_adapt);
  t8_forest_write_vtk (forest, "adapted_forest");
  T8_FREE (adapt_data.adapt_lmi_data);
  T8_FREE (adapt_data.ref_grid_data);
  //T8_FREE (&adapt_data_new);
  //T8_FREE (&adapt_data);
  //t8_forest_unref (&forest_adapt);
}

/*
 * MSA: Multiscaleanalysis: We perform the steps
 * Optionen: Waveletfree oder classical
 */
void
t8_msa_3d (func F, struct grid_hierarchy initial_grid_hierarchy, double c_tresh, int max_lev, const char *err_type)
{
  t8_forest_t forest = initial_grid_hierarchy.lev_arr[0].forest_arr;
  t8_forest_t forest_adapt;
  unsigned int anzahl_gesamt;
  unsigned int anzahl_klein;
  struct lmi_adapt adapt_data;
  struct lmi_adapt adapt_data_new;
  //MultiScaleOperatorWaveletFree(initial_grid_hierarchy);
  //InverseMultiScaleOperatorwaveletfree(initial_grid_hierarchy);

  MultiScaleOperator (initial_grid_hierarchy);
  double error = ErrorSinglescale (initial_grid_hierarchy.lev_arr[max_lev].forest_arr,
                                   initial_grid_hierarchy.lev_arr[max_lev].data_arr, F, 10, err_type);
  t8_global_productionf ("Error SS:%f \n", error);
  InverseMultiScaleOperator (initial_grid_hierarchy);
  error = ErrorSinglescale (initial_grid_hierarchy.lev_arr[max_lev].forest_arr,
                            initial_grid_hierarchy.lev_arr[max_lev].data_arr, F, 10, err_type);
  t8_global_productionf ("Error SS danach:%f \n", error);
  /*
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  */
  //error = ErrorSinglescale(initial_grid_hierarchy.lev_arr[max_lev].forest_arr,initial_grid_hierarchy.lev_arr[max_lev].data_arr,F,10);
  //t8_global_productionf ("Error SS:%f \n",error);
  //MultiScaleOperator(initial_grid_hierarchy);
  //HierarchischerThresholdOperator(initial_grid_hierarchy,c_tresh, 2.0,anzahl_gesamt,anzahl_klein);
  t8_global_productionf ("Vorher \n");
  error = ErrorMultiscale (initial_grid_hierarchy, max_lev, F, 10, err_type);
  t8_global_productionf ("Nachher \n");
  t8_global_productionf ("Error MS:%f \n", error);
  //MultiScaleOperatorWaveletFree(initial_grid_hierarchy);

  // nach trehsholding
  //ThresholdOperatorwaveletfree(initial_grid_hierarchy,c_tresh, 2.0,anzahl_gesamt,anzahl_klein);
  HierarchischerThresholdOperator (initial_grid_hierarchy, c_tresh, 2.0, anzahl_gesamt, anzahl_klein);
  //InverseMultiScaleOperatorwaveletfree(initial_grid_hierarchy);
  error = ErrorSinglescale (initial_grid_hierarchy.lev_arr[max_level].forest_arr,
                            initial_grid_hierarchy.lev_arr[max_level].data_arr, F, 10, err_type);
  //HierarchischerThresholdOperator(initial_grid_hierarchy,c_tresh, 2.0,anzahl_gesamt,anzahl_klein);
  //ThresholdOperator(initial_grid_hierarchy,c_tresh, 2.0,anzahl_gesamt,anzahl_klein);
  //GridAdaptation(initial_grid_hierarchy, c_tresh, 2.0);
  //grading_grid(initial_grid_hierarchy);
  adapt_data.ref_grid_data = initial_grid_hierarchy.lev_arr[0].data_arr;
  adapt_data.adapt_lmi_data = initialize_lmi_adapt_data (initial_grid_hierarchy);
  t8_forest_set_user_data (forest, &adapt_data);
  T8_ASSERT (t8_forest_is_committed (forest));
  t8_forest_ref (forest);
  forest_adapt = t8_adapt_forest (forest, t8_msa_adapt_callback, 0, 0, &adapt_data);

  adapt_data_new.ref_grid_data = initial_grid_hierarchy.lev_arr[0].data_arr;
  adapt_data_new.adapt_lmi_data = initialize_lmi_adapt_data_new (forest_adapt);
  t8_forest_set_user_data (forest_adapt, &adapt_data_new);
  t8_forest_iterate_replace (forest_adapt, forest, t8_forest_replace);
  /* Write the adapted forest to a vtu file */
  adapt_data_new = *(struct lmi_adapt *) t8_forest_get_user_data (forest_adapt);
  //t8_write_vtu (forest_adapt, adapt_data, "t8_step7_adapt_forest");

  /* Free the memory */

  T8_FREE (adapt_data.adapt_lmi_data);
  adapt_data.adapt_lmi_data
    = T8_ALLOC (struct t8_data_per_element_adapt, t8_forest_get_local_num_elements (forest_adapt));

  /* Save the new forest as old forest */
  //t8_scheme_cxx_ref (forest->scheme_cxx);
  //t8_cmesh_ref (forest->cmesh);
  t8_forest_ref (initial_grid_hierarchy.lev_arr[0].forest_arr);
  t8_forest_unref (&forest);
  forest = forest_adapt;
  adapt_data = adapt_data_new;

  for (int lev = 1; lev < max_level; lev++) {
    adapt_data.ref_grid_data = initial_grid_hierarchy.lev_arr[lev].data_arr;
    t8_forest_set_user_data (forest, &adapt_data);
    t8_forest_ref (forest);
    forest_adapt = t8_adapt_forest (forest, t8_msa_adapt_callback, 0, 0, &adapt_data);

    adapt_data_new.ref_grid_data = initial_grid_hierarchy.lev_arr[lev].data_arr;
    adapt_data_new.adapt_lmi_data = initialize_lmi_adapt_data_new (forest_adapt);

    t8_forest_set_user_data (forest_adapt, &adapt_data_new);
    t8_forest_iterate_replace (forest_adapt, forest, t8_forest_replace);

    /* Write the adapted forest to a vtu file */
    struct lmi_adapt *elem_data = (struct lmi_adapt *) t8_forest_get_user_data (forest_adapt);

    //t8_write_vtu (forest_adapt, adapt_data, "t8_step7_adapt_forest");
    /* Free the memory */
    T8_FREE (adapt_data.adapt_lmi_data);
    adapt_data.adapt_lmi_data
      = T8_ALLOC (struct t8_data_per_element_adapt, t8_forest_get_local_num_elements (forest_adapt));

    /* Save the new forest as old forest */
    //t8_forest_ref (initial_grid_hierarchy.lev_arr[0].forest_arr);
    t8_forest_unref (&forest);
    forest = forest_adapt;
    adapt_data = adapt_data_new;
  }
  //t8_forest_unref (&forest_adapt);
  t8_forest_write_vtk (forest, "adapted_forest");
  T8_FREE (adapt_data.adapt_lmi_data);
  T8_FREE (adapt_data.ref_grid_data);
  //T8_FREE (&adapt_data_new);
  //T8_FREE (&adapt_data);
  //t8_forest_unref (&forest_adapt);
}

/*
 * MSA: Multiscaleanalysis: We perform the steps
 * Optionen: Waveletfree oder classical
 */
void
t8_msa_3d_wf (func F, struct grid_hierarchy initial_grid_hierarchy, double c_tresh, int max_lev, const char *err_type)
{
  t8_forest_t forest = initial_grid_hierarchy.lev_arr[0].forest_arr;
  t8_forest_t forest_adapt;
  unsigned int anzahl_gesamt;
  unsigned int anzahl_klein;
  struct lmi_adapt adapt_data;
  struct lmi_adapt adapt_data_new;
  //MultiScaleOperatorWaveletFree(initial_grid_hierarchy);
  //InverseMultiScaleOperatorwaveletfree(initial_grid_hierarchy);

  MultiScaleOperator (initial_grid_hierarchy);
  double error = ErrorSinglescale (initial_grid_hierarchy.lev_arr[max_lev].forest_arr,
                                   initial_grid_hierarchy.lev_arr[max_lev].data_arr, F, 10, err_type);
  t8_global_productionf ("Error SS:%f \n", error);
  InverseMultiScaleOperator (initial_grid_hierarchy);
  error = ErrorSinglescale (initial_grid_hierarchy.lev_arr[max_lev].forest_arr,
                            initial_grid_hierarchy.lev_arr[max_lev].data_arr, F, 10, err_type);
  t8_global_productionf ("Error SS danach:%f \n", error);
  /*
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  MultiScaleOperator(initial_grid_hierarchy);
  InverseMultiScaleOperator(initial_grid_hierarchy);
  */
  //error = ErrorSinglescale(initial_grid_hierarchy.lev_arr[max_lev].forest_arr,initial_grid_hierarchy.lev_arr[max_lev].data_arr,F,10);
  //t8_global_productionf ("Error SS:%f \n",error);
  //MultiScaleOperator(initial_grid_hierarchy);
  //HierarchischerThresholdOperator(initial_grid_hierarchy,c_tresh, 2.0,anzahl_gesamt,anzahl_klein);
  t8_global_productionf ("Vorher \n");
  error = ErrorMultiscale (initial_grid_hierarchy, max_lev, F, 10, err_type);
  t8_global_productionf ("Nachher \n");
  t8_global_productionf ("Error MS:%f \n", error);
  //MultiScaleOperatorWaveletFree(initial_grid_hierarchy);

  // nach trehsholding
  //ThresholdOperatorwaveletfree(initial_grid_hierarchy,c_tresh, 2.0,anzahl_gesamt,anzahl_klein);
  HierarchischerThresholdOperator (initial_grid_hierarchy, c_tresh, 2.0, anzahl_gesamt, anzahl_klein);
  //InverseMultiScaleOperatorwaveletfree(initial_grid_hierarchy);
  error = ErrorSinglescale (initial_grid_hierarchy.lev_arr[max_level].forest_arr,
                            initial_grid_hierarchy.lev_arr[max_level].data_arr, F, 10, err_type);
  //HierarchischerThresholdOperator(initial_grid_hierarchy,c_tresh, 2.0,anzahl_gesamt,anzahl_klein);
  //ThresholdOperator(initial_grid_hierarchy,c_tresh, 2.0,anzahl_gesamt,anzahl_klein);
  //GridAdaptation(initial_grid_hierarchy, c_tresh, 2.0);
  //grading_grid(initial_grid_hierarchy);
  adapt_data.ref_grid_data = initial_grid_hierarchy.lev_arr[0].data_arr;
  adapt_data.adapt_lmi_data = initialize_lmi_adapt_data (initial_grid_hierarchy);
  t8_forest_set_user_data (forest, &adapt_data);
  T8_ASSERT (t8_forest_is_committed (forest));
  t8_forest_ref (forest);
  forest_adapt = t8_adapt_forest (forest, t8_msa_adapt_callback, 0, 0, &adapt_data);

  adapt_data_new.ref_grid_data = initial_grid_hierarchy.lev_arr[0].data_arr;
  adapt_data_new.adapt_lmi_data = initialize_lmi_adapt_data_new (forest_adapt);
  t8_forest_set_user_data (forest_adapt, &adapt_data_new);
  t8_forest_iterate_replace (forest_adapt, forest, t8_forest_replace);
  /* Write the adapted forest to a vtu file */
  adapt_data_new = *(struct lmi_adapt *) t8_forest_get_user_data (forest_adapt);
  //t8_write_vtu (forest_adapt, adapt_data, "t8_step7_adapt_forest");

  /* Free the memory */

  T8_FREE (adapt_data.adapt_lmi_data);
  adapt_data.adapt_lmi_data
    = T8_ALLOC (struct t8_data_per_element_adapt, t8_forest_get_local_num_elements (forest_adapt));

  /* Save the new forest as old forest */
  //t8_scheme_cxx_ref (forest->scheme_cxx);
  //t8_cmesh_ref (forest->cmesh);
  t8_forest_ref (initial_grid_hierarchy.lev_arr[0].forest_arr);
  t8_forest_unref (&forest);
  forest = forest_adapt;
  adapt_data = adapt_data_new;

  for (int lev = 1; lev < max_level; lev++) {
    adapt_data.ref_grid_data = initial_grid_hierarchy.lev_arr[lev].data_arr;
    t8_forest_set_user_data (forest, &adapt_data);
    t8_forest_ref (forest);
    forest_adapt = t8_adapt_forest (forest, t8_msa_adapt_callback, 0, 0, &adapt_data);

    adapt_data_new.ref_grid_data = initial_grid_hierarchy.lev_arr[lev].data_arr;
    adapt_data_new.adapt_lmi_data = initialize_lmi_adapt_data_new (forest_adapt);

    t8_forest_set_user_data (forest_adapt, &adapt_data_new);
    t8_forest_iterate_replace (forest_adapt, forest, t8_forest_replace);

    /* Write the adapted forest to a vtu file */
    struct lmi_adapt *elem_data = (struct lmi_adapt *) t8_forest_get_user_data (forest_adapt);

    //t8_write_vtu (forest_adapt, adapt_data, "t8_step7_adapt_forest");
    /* Free the memory */
    T8_FREE (adapt_data.adapt_lmi_data);
    adapt_data.adapt_lmi_data
      = T8_ALLOC (struct t8_data_per_element_adapt, t8_forest_get_local_num_elements (forest_adapt));

    /* Save the new forest as old forest */
    //t8_forest_ref (initial_grid_hierarchy.lev_arr[0].forest_arr);
    t8_forest_unref (&forest);
    forest = forest_adapt;
    adapt_data = adapt_data_new;
  }
  //t8_forest_unref (&forest_adapt);
  t8_forest_write_vtk (forest, "adapted_forest");
  T8_FREE (adapt_data.adapt_lmi_data);
  T8_FREE (adapt_data.ref_grid_data);
  //T8_FREE (&adapt_data_new);
  //T8_FREE (&adapt_data);
  //t8_forest_unref (&forest_adapt);
}

/*
 * MSA: Multiscaleanalysis: We perform the steps
 * Optionen: Waveletfree oder classical
 */
void
t8_msa_spline (spline eval_spline, const gsl_spline2d *spline, gsl_interp_accel *xacc, gsl_interp_accel *yacc,
               struct grid_hierarchy initial_grid_hierarchy, double c_tresh)
{
  t8_forest_t forest = initial_grid_hierarchy.lev_arr[0].forest_arr;
  t8_forest_t forest_adapt;
  unsigned int anzahl_gesamt;
  unsigned int anzahl_klein;
  struct lmi_adapt adapt_data;
  struct lmi_adapt adapt_data_new;

  MultiScaleOperator (initial_grid_hierarchy);
  InverseMultiScaleOperator (initial_grid_hierarchy);
  HierarchischerThresholdOperator (initial_grid_hierarchy, c_tresh, 2.0, anzahl_gesamt, anzahl_klein);

  adapt_data.ref_grid_data = initial_grid_hierarchy.lev_arr[0].data_arr;
  adapt_data.adapt_lmi_data = initialize_lmi_adapt_data (initial_grid_hierarchy);
  t8_forest_set_user_data (forest, &adapt_data);
  T8_ASSERT (t8_forest_is_committed (forest));
  t8_forest_ref (forest);
  forest_adapt = t8_adapt_forest (forest, t8_msa_adapt_callback, 0, 0, &adapt_data);

  adapt_data_new.ref_grid_data = initial_grid_hierarchy.lev_arr[0].data_arr;
  adapt_data_new.adapt_lmi_data = initialize_lmi_adapt_data_new (forest_adapt);
  t8_forest_set_user_data (forest_adapt, &adapt_data_new);
  t8_forest_iterate_replace (forest_adapt, forest, t8_forest_replace);
  /* Write the adapted forest to a vtu file */
  adapt_data_new = *(struct lmi_adapt *) t8_forest_get_user_data (forest_adapt);
  //t8_write_vtu (forest_adapt, adapt_data, "t8_step7_adapt_forest");

  /* Free the memory */

  T8_FREE (adapt_data.adapt_lmi_data);
  adapt_data.adapt_lmi_data
    = T8_ALLOC (struct t8_data_per_element_adapt, t8_forest_get_local_num_elements (forest_adapt));

  /* Save the new forest as old forest */
  //t8_scheme_cxx_ref (forest->scheme_cxx);
  //t8_cmesh_ref (forest->cmesh);
  t8_forest_ref (initial_grid_hierarchy.lev_arr[0].forest_arr);
  t8_forest_unref (&forest);
  forest = forest_adapt;
  adapt_data = adapt_data_new;

  for (int lev = 1; lev < max_level; lev++) {
    adapt_data.ref_grid_data = initial_grid_hierarchy.lev_arr[lev].data_arr;
    t8_forest_set_user_data (forest, &adapt_data);
    t8_forest_ref (forest);
    forest_adapt = t8_adapt_forest (forest, t8_msa_adapt_callback, 0, 0, &adapt_data);

    adapt_data_new.ref_grid_data = initial_grid_hierarchy.lev_arr[lev].data_arr;
    adapt_data_new.adapt_lmi_data = initialize_lmi_adapt_data_new (forest_adapt);

    t8_forest_set_user_data (forest_adapt, &adapt_data_new);
    t8_forest_iterate_replace (forest_adapt, forest, t8_forest_replace);

    /* Write the adapted forest to a vtu file */
    struct lmi_adapt *elem_data = (struct lmi_adapt *) t8_forest_get_user_data (forest_adapt);

    //t8_write_vtu (forest_adapt, adapt_data, "t8_step7_adapt_forest");
    /* Free the memory */
    T8_FREE (adapt_data.adapt_lmi_data);
    adapt_data.adapt_lmi_data
      = T8_ALLOC (struct t8_data_per_element_adapt, t8_forest_get_local_num_elements (forest_adapt));

    /* Save the new forest as old forest */
    //t8_forest_ref (initial_grid_hierarchy.lev_arr[0].forest_arr);
    t8_forest_unref (&forest);
    forest = forest_adapt;
    adapt_data = adapt_data_new;
  }
  //t8_forest_unref (&forest_adapt);
  t8_forest_write_vtk (forest, "adapted_forest");
  T8_FREE (adapt_data.adapt_lmi_data);
  T8_FREE (adapt_data.ref_grid_data);
  //T8_FREE (&adapt_data_new);
  //T8_FREE (&adapt_data);
  //t8_forest_unref (&forest_adapt);
}

/*
 * MSA: Multiscaleanalysis: We perform the steps
 * Optionen: Waveletfree oder classical
 */
void
t8_msa_spline_3d (spline eval_spline, const gsl_spline2d *spline, gsl_interp_accel *xacc, gsl_interp_accel *yacc,
                  struct grid_hierarchy initial_grid_hierarchy, double c_tresh)
{
  t8_forest_t forest = initial_grid_hierarchy.lev_arr[0].forest_arr;
  t8_forest_t forest_adapt;
  unsigned int anzahl_gesamt;
  unsigned int anzahl_klein;
  struct lmi_adapt adapt_data;
  struct lmi_adapt adapt_data_new;

  MultiScaleOperator (initial_grid_hierarchy);
  InverseMultiScaleOperator (initial_grid_hierarchy);
  HierarchischerThresholdOperator (initial_grid_hierarchy, c_tresh, 2.0, anzahl_gesamt, anzahl_klein);

  adapt_data.ref_grid_data = initial_grid_hierarchy.lev_arr[0].data_arr;
  adapt_data.adapt_lmi_data = initialize_lmi_adapt_data (initial_grid_hierarchy);
  t8_forest_set_user_data (forest, &adapt_data);
  T8_ASSERT (t8_forest_is_committed (forest));
  t8_forest_ref (forest);
  forest_adapt = t8_adapt_forest (forest, t8_msa_adapt_callback, 0, 0, &adapt_data);

  adapt_data_new.ref_grid_data = initial_grid_hierarchy.lev_arr[0].data_arr;
  adapt_data_new.adapt_lmi_data = initialize_lmi_adapt_data_new (forest_adapt);
  t8_forest_set_user_data (forest_adapt, &adapt_data_new);
  t8_forest_iterate_replace (forest_adapt, forest, t8_forest_replace);
  /* Write the adapted forest to a vtu file */
  adapt_data_new = *(struct lmi_adapt *) t8_forest_get_user_data (forest_adapt);
  //t8_write_vtu (forest_adapt, adapt_data, "t8_step7_adapt_forest");

  /* Free the memory */

  T8_FREE (adapt_data.adapt_lmi_data);
  adapt_data.adapt_lmi_data
    = T8_ALLOC (struct t8_data_per_element_adapt, t8_forest_get_local_num_elements (forest_adapt));

  /* Save the new forest as old forest */
  //t8_scheme_cxx_ref (forest->scheme_cxx);
  //t8_cmesh_ref (forest->cmesh);
  t8_forest_ref (initial_grid_hierarchy.lev_arr[0].forest_arr);
  t8_forest_unref (&forest);
  forest = forest_adapt;
  adapt_data = adapt_data_new;

  for (int lev = 1; lev < max_level; lev++) {
    adapt_data.ref_grid_data = initial_grid_hierarchy.lev_arr[lev].data_arr;
    t8_forest_set_user_data (forest, &adapt_data);
    t8_forest_ref (forest);
    forest_adapt = t8_adapt_forest (forest, t8_msa_adapt_callback, 0, 0, &adapt_data);

    adapt_data_new.ref_grid_data = initial_grid_hierarchy.lev_arr[lev].data_arr;
    adapt_data_new.adapt_lmi_data = initialize_lmi_adapt_data_new (forest_adapt);

    t8_forest_set_user_data (forest_adapt, &adapt_data_new);
    t8_forest_iterate_replace (forest_adapt, forest, t8_forest_replace);

    /* Write the adapted forest to a vtu file */
    struct lmi_adapt *elem_data = (struct lmi_adapt *) t8_forest_get_user_data (forest_adapt);

    //t8_write_vtu (forest_adapt, adapt_data, "t8_step7_adapt_forest");
    /* Free the memory */
    T8_FREE (adapt_data.adapt_lmi_data);
    adapt_data.adapt_lmi_data
      = T8_ALLOC (struct t8_data_per_element_adapt, t8_forest_get_local_num_elements (forest_adapt));

    /* Save the new forest as old forest */
    //t8_forest_ref (initial_grid_hierarchy.lev_arr[0].forest_arr);
    t8_forest_unref (&forest);
    forest = forest_adapt;
    adapt_data = adapt_data_new;
  }
  //t8_forest_unref (&forest_adapt);
  t8_forest_write_vtk (forest, "adapted_forest");
  T8_FREE (adapt_data.adapt_lmi_data);
  T8_FREE (adapt_data.ref_grid_data);
  //T8_FREE (&adapt_data_new);
  //T8_FREE (&adapt_data);
  //t8_forest_unref (&forest_adapt);
}

/*
 * MSA: Multiscaleanalysis: We perform the steps
 * Optionen: Waveletfree oder classical
 */
void
t8_msa_spline_wf (spline eval_spline, const gsl_spline2d *spline, gsl_interp_accel *xacc, gsl_interp_accel *yacc,
                  struct grid_hierarchy initial_grid_hierarchy, double c_tresh)
{
  t8_forest_t forest = initial_grid_hierarchy.lev_arr[0].forest_arr;
  t8_forest_t forest_adapt;
  unsigned int anzahl_gesamt;
  unsigned int anzahl_klein;
  struct lmi_adapt adapt_data;
  struct lmi_adapt adapt_data_new;

  MultiScaleOperator (initial_grid_hierarchy);
  InverseMultiScaleOperator (initial_grid_hierarchy);
  HierarchischerThresholdOperator (initial_grid_hierarchy, c_tresh, 2.0, anzahl_gesamt, anzahl_klein);

  adapt_data.ref_grid_data = initial_grid_hierarchy.lev_arr[0].data_arr;
  adapt_data.adapt_lmi_data = initialize_lmi_adapt_data (initial_grid_hierarchy);
  t8_forest_set_user_data (forest, &adapt_data);
  T8_ASSERT (t8_forest_is_committed (forest));
  t8_forest_ref (forest);
  forest_adapt = t8_adapt_forest (forest, t8_msa_adapt_callback, 0, 0, &adapt_data);

  adapt_data_new.ref_grid_data = initial_grid_hierarchy.lev_arr[0].data_arr;
  adapt_data_new.adapt_lmi_data = initialize_lmi_adapt_data_new (forest_adapt);
  t8_forest_set_user_data (forest_adapt, &adapt_data_new);
  t8_forest_iterate_replace (forest_adapt, forest, t8_forest_replace);
  /* Write the adapted forest to a vtu file */
  adapt_data_new = *(struct lmi_adapt *) t8_forest_get_user_data (forest_adapt);
  //t8_write_vtu (forest_adapt, adapt_data, "t8_step7_adapt_forest");

  /* Free the memory */

  T8_FREE (adapt_data.adapt_lmi_data);
  adapt_data.adapt_lmi_data
    = T8_ALLOC (struct t8_data_per_element_adapt, t8_forest_get_local_num_elements (forest_adapt));

  /* Save the new forest as old forest */
  //t8_scheme_cxx_ref (forest->scheme_cxx);
  //t8_cmesh_ref (forest->cmesh);
  t8_forest_ref (initial_grid_hierarchy.lev_arr[0].forest_arr);
  t8_forest_unref (&forest);
  forest = forest_adapt;
  adapt_data = adapt_data_new;

  for (int lev = 1; lev < max_level; lev++) {
    adapt_data.ref_grid_data = initial_grid_hierarchy.lev_arr[lev].data_arr;
    t8_forest_set_user_data (forest, &adapt_data);
    t8_forest_ref (forest);
    forest_adapt = t8_adapt_forest (forest, t8_msa_adapt_callback, 0, 0, &adapt_data);

    adapt_data_new.ref_grid_data = initial_grid_hierarchy.lev_arr[lev].data_arr;
    adapt_data_new.adapt_lmi_data = initialize_lmi_adapt_data_new (forest_adapt);

    t8_forest_set_user_data (forest_adapt, &adapt_data_new);
    t8_forest_iterate_replace (forest_adapt, forest, t8_forest_replace);

    /* Write the adapted forest to a vtu file */
    struct lmi_adapt *elem_data = (struct lmi_adapt *) t8_forest_get_user_data (forest_adapt);

    //t8_write_vtu (forest_adapt, adapt_data, "t8_step7_adapt_forest");
    /* Free the memory */
    T8_FREE (adapt_data.adapt_lmi_data);
    adapt_data.adapt_lmi_data
      = T8_ALLOC (struct t8_data_per_element_adapt, t8_forest_get_local_num_elements (forest_adapt));

    /* Save the new forest as old forest */
    //t8_forest_ref (initial_grid_hierarchy.lev_arr[0].forest_arr);
    t8_forest_unref (&forest);
    forest = forest_adapt;
    adapt_data = adapt_data_new;
  }
  //t8_forest_unref (&forest_adapt);
  t8_forest_write_vtk (forest, "adapted_forest");
  T8_FREE (adapt_data.adapt_lmi_data);
  T8_FREE (adapt_data.ref_grid_data);
  //T8_FREE (&adapt_data_new);
  //T8_FREE (&adapt_data);
  //t8_forest_unref (&forest_adapt);
}

/*
 * MSA: Multiscaleanalysis: We perform the steps
 * Optionen: Waveletfree oder classical
 */
void
t8_msa_spline_wf_3d (spline eval_spline, const gsl_spline2d *spline, gsl_interp_accel *xacc, gsl_interp_accel *yacc,
                     struct grid_hierarchy initial_grid_hierarchy, double c_tresh)
{
  t8_forest_t forest = initial_grid_hierarchy.lev_arr[0].forest_arr;
  t8_forest_t forest_adapt;
  unsigned int anzahl_gesamt;
  unsigned int anzahl_klein;
  struct lmi_adapt adapt_data;
  struct lmi_adapt adapt_data_new;

  MultiScaleOperator (initial_grid_hierarchy);
  InverseMultiScaleOperator (initial_grid_hierarchy);
  HierarchischerThresholdOperator (initial_grid_hierarchy, c_tresh, 2.0, anzahl_gesamt, anzahl_klein);

  adapt_data.ref_grid_data = initial_grid_hierarchy.lev_arr[0].data_arr;
  adapt_data.adapt_lmi_data = initialize_lmi_adapt_data (initial_grid_hierarchy);
  t8_forest_set_user_data (forest, &adapt_data);
  T8_ASSERT (t8_forest_is_committed (forest));
  t8_forest_ref (forest);
  forest_adapt = t8_adapt_forest (forest, t8_msa_adapt_callback, 0, 0, &adapt_data);

  adapt_data_new.ref_grid_data = initial_grid_hierarchy.lev_arr[0].data_arr;
  adapt_data_new.adapt_lmi_data = initialize_lmi_adapt_data_new (forest_adapt);
  t8_forest_set_user_data (forest_adapt, &adapt_data_new);
  t8_forest_iterate_replace (forest_adapt, forest, t8_forest_replace);
  /* Write the adapted forest to a vtu file */
  adapt_data_new = *(struct lmi_adapt *) t8_forest_get_user_data (forest_adapt);
  //t8_write_vtu (forest_adapt, adapt_data, "t8_step7_adapt_forest");

  /* Free the memory */

  T8_FREE (adapt_data.adapt_lmi_data);
  adapt_data.adapt_lmi_data
    = T8_ALLOC (struct t8_data_per_element_adapt, t8_forest_get_local_num_elements (forest_adapt));

  /* Save the new forest as old forest */
  //t8_scheme_cxx_ref (forest->scheme_cxx);
  //t8_cmesh_ref (forest->cmesh);
  t8_forest_ref (initial_grid_hierarchy.lev_arr[0].forest_arr);
  t8_forest_unref (&forest);
  forest = forest_adapt;
  adapt_data = adapt_data_new;

  for (int lev = 1; lev < max_level; lev++) {
    adapt_data.ref_grid_data = initial_grid_hierarchy.lev_arr[lev].data_arr;
    t8_forest_set_user_data (forest, &adapt_data);
    t8_forest_ref (forest);
    forest_adapt = t8_adapt_forest (forest, t8_msa_adapt_callback, 0, 0, &adapt_data);

    adapt_data_new.ref_grid_data = initial_grid_hierarchy.lev_arr[lev].data_arr;
    adapt_data_new.adapt_lmi_data = initialize_lmi_adapt_data_new (forest_adapt);

    t8_forest_set_user_data (forest_adapt, &adapt_data_new);
    t8_forest_iterate_replace (forest_adapt, forest, t8_forest_replace);

    /* Write the adapted forest to a vtu file */
    struct lmi_adapt *elem_data = (struct lmi_adapt *) t8_forest_get_user_data (forest_adapt);

    //t8_write_vtu (forest_adapt, adapt_data, "t8_step7_adapt_forest");
    /* Free the memory */
    T8_FREE (adapt_data.adapt_lmi_data);
    adapt_data.adapt_lmi_data
      = T8_ALLOC (struct t8_data_per_element_adapt, t8_forest_get_local_num_elements (forest_adapt));

    /* Save the new forest as old forest */
    //t8_forest_ref (initial_grid_hierarchy.lev_arr[0].forest_arr);
    t8_forest_unref (&forest);
    forest = forest_adapt;
    adapt_data = adapt_data_new;
  }
  //t8_forest_unref (&forest_adapt);
  t8_forest_write_vtk (forest, "adapted_forest");
  T8_FREE (adapt_data.adapt_lmi_data);
  T8_FREE (adapt_data.ref_grid_data);
  //T8_FREE (&adapt_data_new);
  //T8_FREE (&adapt_data);
  //t8_forest_unref (&forest_adapt);
}

/*
 * MSA: Multiscaleanalysis: We perform the steps
 * Optionen: Waveletfree oder classical
 */
int
VanishingMomentsUnitTest (struct grid_hierarchy grid_hierarchy)
{
  int num_vanishing_moments = 100;
  MultiScaleOperator (grid_hierarchy);
  InverseMultiScaleOperator (grid_hierarchy);
  MultiScaleOperator (grid_hierarchy);
  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    const t8_element_t *element;
    for (int l = 0; l < max_level; ++l) {
      t8_global_productionf ("levvel:%i \n", l);
      T8_ASSERT (t8_forest_is_committed (grid_hierarchy.lev_arr[l].forest_arr));
      num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy.lev_arr[l].forest_arr);
      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        /* Get the number of elements of this tree. */
        num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          int count = 0;
          double max_d = 0;
          for (int i = 0; i < 3 * M; ++i) {
            max_d = max (max_d, grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[i]);
            if (isZero (grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[i])) {
              count += 1;
            }
          }
          t8_global_productionf ("count:%f \n", max_d);
          t8_global_productionf ("count:%i \n", count);
          num_vanishing_moments = min (num_vanishing_moments, count);
        }
      }
    }
  }
  return num_vanishing_moments;
}

/*
 *
 */
bool
NumericalStabilityMSOperatorUnitTest (func F, struct grid_hierarchy initial_grid_hierarchy, double tol, int num_iter)
{
  struct grid_hierarchy grid_hierarchy_copy;
  grid_hierarchy_copy = initial_grid_hierarchy;
  double max_diff = 0;
  for (int i = 0; i < num_iter; ++i) {
    MultiScaleOperator (grid_hierarchy_copy);
    InverseMultiScaleOperator (grid_hierarchy_copy);
  }

  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    const t8_element_t *element;
    for (int l = 0; l < max_level; ++l) {
      T8_ASSERT (t8_forest_is_committed (grid_hierarchy_copy.lev_arr[l].forest_arr));
      num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy_copy.lev_arr[l].forest_arr);
      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        /* Get the number of elements of this tree. */
        num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy_copy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          for (int i = 0; i < 3 * M; ++i) {
            max_diff = max (max_diff, abs (grid_hierarchy_copy.lev_arr[l].data_arr[current_index].d_coeff[i]
                                           - initial_grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff[i]));
          }
        }
      }
    }
  }
  t8_global_productionf ("max_diff:%f \n", max_diff);
  return abs (max_diff) <= tol;
}

/*
 *
 */
bool
NumericalStabilityMSOperatorWaveletfreeUnitTest (func F, struct grid_hierarchy_waveletfree initial_grid_hierarchy,
                                                 const int num_iter, const double tol)
{
  struct grid_hierarchy_waveletfree grid_hierarchy_copy;
  grid_hierarchy_copy = initial_grid_hierarchy;
  double max_diff = 0;
  for (int i = 0; i < num_iter; ++i) {
    MultiScaleOperatorWaveletFree (initial_grid_hierarchy);
    InverseMultiScaleOperatorwaveletfree (initial_grid_hierarchy);
  }

  {
    t8_locidx_t itree, num_local_trees;
    t8_locidx_t current_index;
    t8_locidx_t ielement, num_elements_in_tree;
    const t8_element_t *element;
    for (int l = 0; l < max_level; ++l) {
      T8_ASSERT (t8_forest_is_committed (grid_hierarchy_copy.lev_arr[l].forest_arr));
      num_local_trees = t8_forest_get_num_local_trees (grid_hierarchy_copy.lev_arr[l].forest_arr);
      for (itree = 0, current_index = 0; itree < num_local_trees; ++itree) {
        /* Get the number of elements of this tree. */
        num_elements_in_tree = t8_forest_get_tree_num_elements (grid_hierarchy_copy.lev_arr[l].forest_arr, itree);
        for (ielement = 0; ielement < num_elements_in_tree; ++ielement, ++current_index) {
          for (int i = 0; i < M; ++i) {
            for (int j = 0; j < 4; ++j) {
              max_diff = max (
                max_diff, abs (grid_hierarchy_copy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][j]
                               - initial_grid_hierarchy.lev_arr[l].data_arr[current_index].d_coeff_wavelet_free[i][j]));
            }
          }
        }
      }
    }
  }
  t8_global_productionf ("max_diff:%f \n", max_diff);
  return abs (max_diff) <= tol;
}

/* These are six test functions. */
double
F (double x, double y)
{
  if ((x == -1.) && (y == -1.))
    return 1.;
  return sin (2. * M_PI * x) * sin (2. * M_PI * y);
}

double
G (double x, double y)
{
  if ((x == -1.) && (y == -1.))
    return 2.;
  double r = x * x + y * y;
  return (r < 0.25) ? 1.0 : 0.0;
}

double
H (double x, double y)
{
  if ((x == -1.) && (y == -1.))
    return 3.;
  double r = x * x + y * y;
  return (r < 0.25) ? (x * y + x + 3.) : (x * x * y - 2. * x * y * y + 3. * x);
}

double
I (double x, double y)
{
  if ((x == -1.) && (y == -1.))
    return 4.;
  return sin (1 / (1.001 - x * y));
}

double
J (double x, double y)
{
  if ((x == -1.) && (y == -1.))
    return 5.;
  return ((x < (0.5 - 0.001953125)) ? sin (x) : sin (y));
}

double
K (double x, double y)
{
  if ((x == -1.) && (y == -1.))
    return 6.;
  if (x < 0.41)
    return 0.;
  double r4 = (x - 0.5) * (x - 0.5) + (y - 0.5) * (y - 0.5);
  double r = sqrt (r4);
  if (r > 1. / 3.)
    return 0.;
  r *= 3.;
  r4 *= 9.;
  r4 *= r4;
  double rm1 = r - 1.;
  double rm1h2 = rm1 * rm1;
  double rm1h3 = rm1 * rm1h2;
  return 1. - r4 + 4. * r4 * rm1 - 10. * r4 * rm1h2 + 20 * r4 * rm1h3;
}

/* Testing jump sizes */
double
L (double x, double y)
{
  double val = 0.0;
  if ((y >= 0.191))
    val += 1e-8;
  if ((y >= 0.291))
    val += 1e-7;
  if ((y >= 0.391))
    val += 1e-6;
  if ((y >= 0.491))
    val += 1e-5;
  if ((y >= 0.591))
    val += 1e-4;
  if ((y >= 0.691))
    val += 1e-3;
  if ((y >= 0.791))
    val += 1e-2;
  if ((y >= 0.891))
    val += 1e-1;
  if ((y >= 0.991))
    val += 1.0;
  return val;
}
//The polynomials are used to verify the Vanishing Moments
double
zero_degree_const (double x, double y)
{
  return 1;
}

double
first_degree_x (double x, double y)
{
  return x;
}

double
first_degree_y (double x, double y)
{
  return y;
}

double
second_degree_x (double x, double y)
{
  return x * x;
}

double
second_degree_y (double x, double y)
{
  return y * y;
}

double
second_degree_xy (double x, double y)
{
  return x * y;
}

double
third_degree_x (double x, double y)
{
  return x * x * x;
}

double
third_degree_xxy (double x, double y)
{
  return x * x * y;
}

double
third_degree_xyy (double x, double y)
{
  return x * y * y;
}

double
third_degree_y (double x, double y)
{
  return y * y * y;
}

double
high_degree (double x, double y)
{
  return y * y * y * y * y * y * y * x * x * x * x * x * x * x;
}

double
AuswertungSpline (const gsl_spline2d *spline, const double x, const double y, gsl_interp_accel *xacc,
                  gsl_interp_accel *yacc)
{
  return gsl_spline2d_eval (spline, x, y, xacc, yacc);
}

int
t8_tutorial_build_cmesh_main (int argc, char **argv)
{

  /* hier kommentierte Testausführungen festhalten */

  InitialisiereKoeff (p, M0, M1, M2, M3, N0, N1, N2, N3);
  double c_tresh = 1e-10;
  cout << setprecision (14);
  initialize_pow4 ();
  /* The prefix for our output files. */
  const char *prefix_basic = "basic_mesh_tri";
  const char *prefix_octagon = "octagonal_mesh";
  const char *prefix_polygonal = "complex_polygonal_mesh";
  const char *prefix_l_shape = "l_shape_mesh";
  const char *prefix_basic_forest = "forest_basic_mesh_tri";
  const char *prefix_octagon_forest = "forest_octagonal_mesh";
  const char *prefix_polygonal_forest = "forest_complex_polygonal_mesh";
  const char *prefix_l_shape_forest = "forest_l_shape_mesh";
  const char *prefix_adapted = "forest_adapted";

  const char *prefix = prefix_l_shape_forest;
  /* The element data */
  t8_data_per_element *data;

  //read the binary file for the MPTRAC data
  float fileData[721801];
  ifstream inputFileStream ("orig_data.h2o.lev.1000.bin", ios::in | ios::binary);
  cout << "File data is\n";
  inputFileStream.read ((char *) &fileData, 721801 * sizeof (float));

  /* If you want to output the data array
  for (int i=0; i<721801; i++)
      cout << fileData[i] << ", " << endl;
  cout << "\nFinished reading\n";
  */
  inputFileStream.close ();

  //2d interpolation using gsl library gsl_interp2d_bicubic
  const gsl_interp2d_type *T = gsl_interp2d_bilinear;  //or gsl_interp2d_bicubic
  const size_t N = 100;                                /* number of points to interpolate */
  const size_t nx = 1201;                              /* x grid points */
  const size_t ny = 601;                               /* y grid points */
  double xa[nx];                                       /* define unit square */
  for (int i = 0; i < nx; i++) {
    xa[i] = 0.3 * i;
  }
  double ya[ny];
  for (int i = 0; i < ny; i++) {
    ya[i] = -90.0 + 0.3 * i;
  }
  double *za = T8_ALLOC (double, nx *ny);
  gsl_spline2d *spline = gsl_spline2d_alloc (T, nx, ny);
  gsl_interp_accel *xacc = gsl_interp_accel_alloc ();
  gsl_interp_accel *yacc = gsl_interp_accel_alloc ();

  /* set z grid values */
  for (int iy = 600, i = 0; iy >= 0 && i < ny; --iy, ++i) {
    for (int ix = 0; ix < nx; ++ix) {
      gsl_spline2d_set (spline, za, ix, i, fileData[ix * ny + iy]);
    }
  }

  /* initialize interpolation */
  gsl_spline2d_init (spline, xa, ya, za, nx, ny);
  size_t i, j;
  /* interpolate N values in x and y and print out grid for plotting */
  for (i = 0; i < nx; ++i) {
    double xi = 0.3 * i;

    for (j = 0; j < ny; ++j) {
      double yj = -90.0 + 0.3 * j;
      double zij = gsl_spline2d_eval (spline, xi, yj, xacc, yacc);
      //AuswertungSpline(spline, xi, yj, xacc, yacc);
      //t8_global_productionf (" Differenz: %f", zij-AuswertungSpline(spline, xi, yj, xacc, yacc));
      //printf("%f %f %f\n", xi, yj, zij);
    }
    //printf("\n");
  }

  /*
   * Initialization.
   */

  /* Initialize MPI. This has to happen before we initialize sc or t8code. */
  int mpiret = sc_MPI_Init (&argc, &argv);
  /* Error check the MPI return value. */
  SC_CHECK_MPI (mpiret);

  /* Initialize the sc library, has to happen before we initialize t8code. */
  sc_init (sc_MPI_COMM_WORLD, 1, 1, NULL, SC_LP_PRODUCTION);
  /* Initialize t8code with log level SC_LP_PRODUCTION. See sc.h for more info on the log levels. */
  t8_init (SC_LP_PRODUCTION);

  /* We will use MPI_COMM_WORLD as a communicator. */
  sc_MPI_Comm comm = sc_MPI_COMM_WORLD;

  /*
   * Definition of the meshes.
   */
  /* Creation of a basic two dimensional cmesh. */
  t8_cmesh_t cmesh = t8_cmesh_new_debugging (comm);

  const t8_scheme *scheme = t8_scheme_new_default ();
  int level = max_level;
  //struct grid_hierarchy initial_grid_hierarchy=initialize_grid_hierarchy_spline(cmesh,scheme,AuswertungSpline,spline,xacc,yacc,comm,10,max_level);

  //struct grid_hierarchy initial_grid_hierarchy=initialize_grid_hierarchy(cmesh,scheme,first_degree_x,comm,10,max_level);
  // struct grid_hierarchy initial_grid_hierarchy=initialize_grid_hierarchy(cmesh,scheme,F,comm, 10,max_level);
  //
  //
  // t8_msa (F,initial_grid_hierarchy,c_tresh,max_level);
  struct grid_hierarchy_waveletfree initial_grid_hierarchy
    = initialize_grid_hierarchy_waveletfree (cmesh, scheme, F, comm, 10, max_level);

  t8_msa_wf (F, initial_grid_hierarchy, c_tresh, max_level, "L2");
  //zero_degree_const
  //t8_global_productionf ("Numerical Stability Test %i \n",NumericalStabilityMSOperatorUnitTest(first_degree_x,initial_grid_hierarchy,1e-16, 100));
  //t8_global_productionf ("Number of vanishing moments: %i \n",VanishingMomentsUnitTest (initial_grid_hierarchy));

  //t8_msa_spline (AuswertungSpline,spline,xacc,yacc,initial_grid_hierarchy,c_tresh);

  /*
  double err, errthr;
  double eps = 1.;
  double div = pow(2.,p);
  for (int levels = 1; levels < 11; ++levels) {
    initializeGrid(Gitter, levels);
    ProjektionSinglescale(Gitter, H, 10);
    MultiScaleOperator(Gitter);
    unsigned int anzahl_gesamt, anzahl_klein;
    ThresholdOperator(Gitter, eps, 2.0, anzahl_gesamt, anzahl_klein);
    InverseMultiScaleOperator(Gitter);
    double err_new = ErrorSinglescale(Gitter, H, 10);
    cout << eps << " & " << err_new << " &";
    if (levels > 1) cout << " " << log2(err/err_new);
    cout << " & " << ((((double) anzahl_klein)/((double) anzahl_gesamt))*100.) << " \\\\" << endl;
    err = err_new;
    eps /= div;
  */

  //deref_grid_hierarchy(initial_grid_hierarchy);
  /* Output the meshes to vtu files. */
  t8_cmesh_vtk_write_file (cmesh, prefix);
  //
  //   t8_forest_t forest = t8_forest_new_uniform (cmesh, scheme, level, 0, comm);
  //   /*
  //    * Build data array and gather data for the local elements.
  //    */
  //   data = t8_create_element_data_spline(forest, AuswertungSpline,spline, xacc, yacc, 10,level);
  //
  //   /*
  //    * Exchange the data values of the ghost elements
  //    */
  //
  //   /*
  //    * Output the volume data to vtu.
  //    */
  //   t8_output_data_to_vtu (forest, data, prefix);
  //
  // /*
  //    * Clean-up
  //    */
  //   /* Deallocate the cmeshes */
  //   /* Free the data array. */
  //   T8_FREE (data);
  gsl_spline2d_free (spline);
  gsl_interp_accel_free (xacc);
  gsl_interp_accel_free (yacc);
  T8_FREE (za);

  /* This also destroys the cmeshes. */
  //t8_forest_unref (&forest);

  /* Finalize the sc library */
  sc_finalize ();

  mpiret = sc_MPI_Finalize ();
  SC_CHECK_MPI (mpiret);

  return 0;
}

int
main (int argc, char **argv)
{
  return t8_tutorial_build_cmesh_main (argc, argv);
}

//T8_EXTERN_C_END ();
