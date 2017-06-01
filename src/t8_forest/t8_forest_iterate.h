/*
  This file is part of t8code.
  t8code is a C library to manage a collection (a forest) of multiple
  connected adaptive space-trees of general element classes in parallel.

  Copyright (C) 2015 the developers

  t8code is free software; you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation; either version 2 of the License, or
  (at your option) any later version.

  t8code is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with t8code; if not, write to the Free Software Foundation, Inc.,
  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
*/

/** \file t8_forest_iterate.h
 * We define variuos routines to iterate through (parts of) a forest and execute
 * callback functions on the leaf elements,
 */

/* TODO: begin documenting this file: make doxygen 2>&1 | grep t8_forest_iterate */

#ifndef T8_FOREST_ITERATE_H
#define T8_FOREST_ITERATE_H

#include <t8.h>
#include <t8_forest.h>

typedef int         (*t8_forest_iterate_face_fn) (t8_forest_t forest,
                                                  t8_locidx_t ltreeid,
                                                  const t8_element_t *
                                                  element, int face,
                                                  void *user_data,
                                                  t8_locidx_t
                                                  tree_leaf_index);

typedef int         (*t8_forest_search_query_fn) (t8_forest_t forest,
                                                  t8_locidx_t ltreeid,
                                                  const t8_element_t *
                                                  element,
                                                  sc_array_t * leaf_elements,
                                                  void *user_data,
                                                  t8_locidx_t
                                                  tree_leaf_index);

T8_EXTERN_C_BEGIN ();

/* TODO: Document */
void                t8_forest_split_array (const t8_element_t * element,
                                           sc_array_t * leaf_elements,
                                           t8_eclass_scheme_c * ts,
                                           size_t * offsets);

/* TODO: comment */
/* Iterate over all leafs of an element that touch a given face of the element */
/* Callback is called in each recursive step with element as input.
 * leaf_index is only not negative if element is a leaf, in which case it indicates
 * the index of the leaf in the leafs of the tree. If it is negative, it is
 * - (index + 1) */
/* Top-down iteration and callback is called on each intermediate level.
 * It it returns false, the current element is not traversed further */
void                t8_forest_iterate_faces (t8_forest_t forest,
                                             t8_locidx_t ltreeid,
                                             const t8_element_t * element,
                                             int face,
                                             sc_array_t * leaf_elements,
                                             void *user_data,
                                             t8_locidx_t
                                             tree_lindex_of_first_leaf,
                                             t8_forest_iterate_face_fn
                                             callback);

/* Perform a top-down search of the forest, executing a callback on each
 * intermediate element. The search will enter each tree at least once.
 * If the callback returns false for an element, its descendants
 * are not further searched.
 */
void                t8_forest_search (t8_forest_t forest,
                                      t8_forest_search_query_fn search_fn,
                                      void *user_data);

T8_EXTERN_C_END ();

#endif /* !T8_FOREST_ITERATE_H! */