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

#include <p4est_bits.h>
#include "t8_default_common.h"
#include "t8_default_common_cxx.hxx"
#include "t8_default_quad_cxx.hxx"

/* This function is used by other element functions and we thus need to
 * declare it up here */
uint64_t            t8_element_get_linear_id (const t8_element_t * elem,
                                              int level);

#ifdef T8_ENABLE_DEBUG

int
t8_element_surround_matches (const p4est_quadrant_t * q,
                             const p4est_quadrant_t * r)
{
  return T8_QUAD_GET_TDIM (q) == T8_QUAD_GET_TDIM (r) &&
    (T8_QUAD_GET_TDIM (q) == -1 ||
     (T8_QUAD_GET_TNORMAL (q) == T8_QUAD_GET_TNORMAL (r) &&
      T8_QUAD_GET_TCOORD (q) == T8_QUAD_GET_TCOORD (r)));
}

#endif /* T8_ENABLE_DEBUG */

int
t8_default_scheme_quad_c::t8_element_maxlevel (void)
{
  return P4EST_QMAXLEVEL;
}

/* *INDENT-OFF* */
t8_eclass_t
t8_default_scheme_quad_c::t8_element_child_eclass (int childid)
/* *INDENT-ON* */

{
  T8_ASSERT (0 <= childid && childid < P4EST_CHILDREN);

  return T8_ECLASS_QUAD;
}

int
t8_default_scheme_quad_c::t8_element_level (const t8_element_t * elem)
{
  return (int) ((const p4est_quadrant_t *) elem)->level;
}

static void
t8_element_copy_surround (const p4est_quadrant_t * q, p4est_quadrant_t * r)
{
  T8_QUAD_SET_TDIM (r, T8_QUAD_GET_TDIM (q));
  if (T8_QUAD_GET_TDIM (q) == 3) {
    T8_QUAD_SET_TNORMAL (r, T8_QUAD_GET_TNORMAL (q));
    T8_QUAD_SET_TCOORD (r, T8_QUAD_GET_TCOORD (q));
  }
}

void
t8_default_scheme_quad_c::t8_element_copy (const t8_element_t * source,
                                           t8_element_t * dest)
{
  const p4est_quadrant_t *q = (const p4est_quadrant_t *) source;
  p4est_quadrant_t   *r = (p4est_quadrant_t *) dest;

  *r = *q;
  t8_element_copy_surround (q, r);
}

int
t8_default_scheme_quad_c::t8_element_compare (const t8_element_t * elem1,
                                              const t8_element_t * elem2)
{
  int                 maxlvl;
  u_int64_t           id1, id2;

  /* Compute the bigger level of the two */
  maxlvl = SC_MAX (t8_element_level (elem1), t8_element_level (elem2));
  /* Compute the linear ids of the elements */
  id1 = t8_element_get_linear_id (elem1, maxlvl);
  id2 = t8_element_get_linear_id (elem2, maxlvl);
  /* return negativ if id1 < id2, zero if id1 = id2, positive if id1 > id2 */
  return id1 < id2 ? -1 : id1 != id2;
}

void
t8_default_scheme_quad_c::t8_element_parent (const t8_element_t * elem,
                                             t8_element_t * parent)
{
  const p4est_quadrant_t *q = (const p4est_quadrant_t *) elem;
  p4est_quadrant_t   *r = (p4est_quadrant_t *) parent;

  p4est_quadrant_parent (q, r);
  t8_element_copy_surround (q, r);
}

void
t8_default_scheme_quad_c::t8_element_sibling (const t8_element_t * elem,
                                              int sibid,
                                              t8_element_t * sibling)
{
  const p4est_quadrant_t *q = (const p4est_quadrant_t *) elem;
  p4est_quadrant_t   *r = (p4est_quadrant_t *) sibling;

  p4est_quadrant_sibling (q, r, sibid);
  t8_element_copy_surround (q, r);
}

void
t8_default_scheme_quad_c::t8_element_child (const t8_element_t * elem,
                                            int childid, t8_element_t * child)
{
  const p4est_quadrant_t *q = (const p4est_quadrant_t *) elem;
  const p4est_qcoord_t shift = P4EST_QUADRANT_LEN (q->level + 1);
  p4est_quadrant_t   *r = (p4est_quadrant_t *) child;

  T8_ASSERT (p4est_quadrant_is_extended (q));
  T8_ASSERT (q->level < P4EST_QMAXLEVEL);
  T8_ASSERT (childid >= 0 && childid < P4EST_CHILDREN);

  r->x = childid & 0x01 ? (q->x | shift) : q->x;
  r->y = childid & 0x02 ? (q->y | shift) : q->y;
  r->level = q->level + 1;
  T8_ASSERT (p4est_quadrant_is_parent (q, r));

  t8_element_copy_surround (q, r);
}

void
t8_default_scheme_quad_c::t8_element_children (const t8_element_t * elem,
                                               int length, t8_element_t * c[])
{
  const p4est_quadrant_t *q = (const p4est_quadrant_t *) elem;
  int                 i;

  T8_ASSERT (length == P4EST_CHILDREN);

  p4est_quadrant_childrenpv (q, (p4est_quadrant_t **) c);
  for (i = 0; i < P4EST_CHILDREN; ++i) {
    t8_element_copy_surround (q, (p4est_quadrant_t *) c[i]);
  }
}

int
t8_default_scheme_quad_c::t8_element_child_id (const t8_element_t * elem)
{
  return p4est_quadrant_child_id ((p4est_quadrant_t *) elem);
}

int
t8_default_scheme_quad_c::t8_element_is_family (t8_element_t ** fam)
{
  return p4est_quadrant_is_familypv ((p4est_quadrant_t **) fam);
}

void
t8_default_scheme_quad_c::t8_element_set_linear_id (t8_element_t * elem,
                                                    int level, uint64_t id)
{
  T8_ASSERT (0 <= level && level <= P4EST_QMAXLEVEL);
  T8_ASSERT (0 <= id && id < ((uint64_t) 1) << P4EST_DIM * level);

  p4est_quadrant_set_morton ((p4est_quadrant_t *) elem, level, id);
  T8_QUAD_SET_TDIM ((p4est_quadrant_t *) elem, 2);
}

uint64_t
  t8_default_scheme_quad_c::t8_element_get_linear_id (const t8_element_t *
                                                      elem, int level)
{
  T8_ASSERT (0 <= level && level <= P4EST_QMAXLEVEL);

  return p4est_quadrant_linear_id ((p4est_quadrant_t *) elem, level);
}

void
t8_default_scheme_quad_c::t8_element_first_descendant (const t8_element_t *
                                                       elem,
                                                       t8_element_t * desc)
{
  p4est_quadrant_first_descendant ((p4est_quadrant_t *) elem,
                                   (p4est_quadrant_t *) desc,
                                   P4EST_QMAXLEVEL);
}

void
t8_default_scheme_quad_c::t8_element_last_descendant (const t8_element_t *
                                                      elem,
                                                      t8_element_t * desc)
{
  p4est_quadrant_last_descendant ((p4est_quadrant_t *) elem,
                                  (p4est_quadrant_t *) desc, P4EST_QMAXLEVEL);
}

void
t8_default_scheme_quad_c::t8_element_successor (const t8_element_t * elem1,
                                                t8_element_t * elem2,
                                                int level)
{
  uint64_t            id;
  T8_ASSERT (0 <= level && level <= P4EST_QMAXLEVEL);

  id = p4est_quadrant_linear_id ((const p4est_quadrant_t *) elem1, level);
  T8_ASSERT (id + 1 < ((uint64_t) 1) << P4EST_DIM * level);
  p4est_quadrant_set_morton ((p4est_quadrant_t *) elem2, level, id + 1);
  t8_element_copy_surround ((const p4est_quadrant_t *) elem1,
                            (p4est_quadrant_t *) elem2);
}

void
t8_default_scheme_quad_c::t8_element_nca (const t8_element_t * elem1,
                                          const t8_element_t * elem2,
                                          t8_element_t * nca)
{
  const p4est_quadrant_t *q1 = (const p4est_quadrant_t *) elem1;
  const p4est_quadrant_t *q2 = (const p4est_quadrant_t *) elem2;
  p4est_quadrant_t   *r = (p4est_quadrant_t *) nca;

  T8_ASSERT (t8_element_surround_matches (q1, q2));

  p4est_nearest_common_ancestor (q1, q2, r);
  t8_element_copy_surround (q1, r);
}

void
t8_default_scheme_quad_c::t8_element_boundary (const t8_element_t * elem,
                                               int min_dim, int length,
                                               t8_element_t ** boundary)
{
#ifdef T8_ENABLE_DEBUG
  int                 per_eclass[T8_ECLASS_COUNT];
#endif

  T8_ASSERT (length ==
             t8_eclass_count_boundary (T8_ECLASS_QUAD, min_dim, per_eclass));

  /* TODO: write this function */
  SC_ABORT_NOT_REACHED ();
}

void
t8_default_scheme_quad_c::t8_element_anchor (const t8_element_t * elem,
                                             int coord[3])
{
  p4est_quadrant_t   *q;

  q = (p4est_quadrant_t *) elem;
  coord[0] = q->x;
  coord[1] = q->y;
  coord[2] = 0;
}

int
t8_default_scheme_quad_c::t8_element_root_len (const t8_element_t * elem)
{
  return P4EST_ROOT_LEN;
}

/* Constructor */
t8_default_scheme_quad_c::t8_default_scheme_quad_c (void)
{
  eclass = T8_ECLASS_QUAD;
  element_size = sizeof (t8_pquad_t);
  ts_context = sc_mempool_new (sizeof (element_size));
}

t8_default_scheme_quad_c::~t8_default_scheme_quad_c ()
{
  /* This destructor is empty since the destructor of the
   * default_common scheme is called automatically and it
   * suffices to destroy the quad_scheme.
   * However we need to provide an implementation of the destructor
   * and hence this empty function. */
}
