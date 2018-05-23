/*
Copyright (C) 2006 Pedro Felzenszwalb

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

#ifndef SEGMENT_GRAPH
#define SEGMENT_GRAPH

#include <algorithm>
#include <cstdlib>
#include <math.h>
#include "image.h"
#include "disjoint-set.h"
#include "caffe/blob.hpp"

// threshold function
#define THRESHOLD(size, c) (c/size)

typedef struct {
  float w;
  int a, b;
} edge;

bool operator<(const edge &a, const edge &b) {
  return a.w < b.w;
}

rgb random_rgb(){
  rgb c;
  c.r = (uchar)random();
  c.g = (uchar)random();
  c.b = (uchar)random();

  return c;
}

/*
 * Segment a graph
 *
 * Returns a disjoint-set forest representing the segmentation.
 *
 * num_vertices: number of vertices in graph.
 * num_edges: number of edges in graph
 * edges: array of edges.
 * c: constant for treshold function.
 */
 template <typename Dtype>
 universe *segment_graph(int num_vertices, int num_edges, Dtype *edges,
			Dtype threshold) {
  // make a disjoint-set forest
  universe *u = new universe(num_vertices);

  // for each edge, in non-decreasing weight order...
  int idx = 0;
  for (int i = 0; i < num_edges; i++) {
    // components conected by this edge
    int a = u->find(int(edges[idx++]));
    int b = u->find(int(edges[idx++]));
    if (a != b) {
      if (edges[idx] <= threshold) {
	       u->join(a, b);
      }
    }
    idx++;
  }

  return u;
}

template <typename Dtype>
universe *segment_graph_egb(int num_vertices, int num_edges, Dtype *edges,
     Dtype threshold) {
 edge *edge_list = new edge[num_edges];
 int idx = 0;
 for (int i = 0; i < num_edges; i++) {
   edge_list[i].a = edges[idx++];
   edge_list[i].b = edges[idx++];
   edge_list[i].w = edges[idx++];
 }

 // sort edges by weight
 std::sort(edge_list, edge_list + num_edges);

 // make a disjoint-set forest
 universe *u = new universe(num_vertices);

 // init thresholds
 float *thresh = new float[num_vertices];
 for (int i = 0; i < num_vertices; i++)
   thresh[i] = THRESHOLD(1, threshold);

 // for each edge, in non-decreasing weight order...
 for (int i = 0; i < num_edges; i++) {
   edge *pedge = &edge_list[i];

   // components conected by this edge
   int a = u->find(pedge->a);
   int b = u->find(pedge->b);
   if (a != b) {
     if ((pedge->w <= thresh[a]) &&
       (pedge->w <= thresh[b])) {
       u->join(a, b);
       a = u->find(a);
       thresh[a] = pedge->w + THRESHOLD(u->size(a), threshold);
     }
   }
 }

 delete thresh;
 return u;
}

#endif
