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

typedef void        (*t8_forest_iterate_face_fn) (t8_forest_t forest,
                                                  t8_locidx_t ltreeid,
                                                  const t8_element_t * leaf,
                                                  int face, void *user_data);

T8_EXTERN_C_BEGIN ();

/* TODO: Document */
void                t8_forest_split_array (t8_element_t * element,
                                           sc_array_t * leaf_elements,
                                           t8_eclass_scheme_c * ts,
                                           size_t * offsets);

/* TODO: comment */
/* Iterate over all leafs of an element that touch a given face of the element */
void                t8_forest_iterate_faces (t8_forest_t forest,
                                             t8_locidx_t ltreeid,
                                             t8_element_t * element,
                                             int face,
                                             sc_array_t * leaf_elements,
                                             void *user_data,
                                             t8_forest_iterate_face_fn
                                             callback);

T8_EXTERN_C_END ();

#endif /* !T8_FOREST_ITERATE_H! */
