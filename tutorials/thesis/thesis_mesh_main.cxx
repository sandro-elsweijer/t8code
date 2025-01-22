/*
  This file is part of t8code.
  t8code is a C library to manage a collection (a forest) of multiple
  connected adaptive space-trees of general element types in parallel.

  Copyright (C) 2023 the developers

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

/* This file is part of the t8code tutorial regarding the creation of a scmesh.
 * Here, we only wrap the main function. See t8_tutorial_build_cmesh.cxx for the documentation of
 * the 'build-cmesh'-tutorial
 */
#include <tutorials/thesis/thesis_mesh.hpp>

int
main (int argc, char **argv)
{
  return t8_tutorial_build_cmesh_main (argc, argv);
}
