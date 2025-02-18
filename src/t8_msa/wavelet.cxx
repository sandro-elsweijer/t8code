/**
 * @file   wavelet.cpp
 * @author Nils Gerhard <gerhard@igpm.rwth-aachen.de>
 * @date   Tue Jun 4 2013
 *
 * @brief L2-Projection and Multiscale transformation on Triangles
 *
 *
 */

#include "wavelet.hxx"
#include "basis_functions.hxx"
#include "mask_coefficients.hxx"

int p = 4;
int M = (p * (p + 1)) / 2;

mat M0, M1, M2, M3, N0, N1, N2, N3;

/*
void HierarchischerThresholdOperator(GlobalGrid & Grid, double eps, double a, unsigned int & anzahl_gesamt, unsigned int & anzahl_signifikant) {
  const int levels = Grid.size();
  anzahl_gesamt = 0; anzahl_signifikant = 0;
  for (int l = 1; l < levels; ++l) {
    eps /= a;
    for (GridMap::iterator it = (Grid[l]).begin(); it != (Grid[l]).end(); ++it){
      ++anzahl_gesamt;
      it->second.adaptiert = false;
      it->second.signifikant = false;
    }
  }
  for (int l = levels-2; l>=0; --l) {
    eps /= a;
    for (GridMap::iterator it = (Grid[l]).begin(); it != (Grid[l]).end(); ++it){
      assert((it->second).d_coeff.size() == 3*M);
      if ((it.second)->signifikant) continue;
      for (int i = 0; i < 3*M; ++i) {
        if (abs(it->second.d_coeff[i]) > sqrt(2.0*(it->second.flaeche))*eps) {
          Grid[l+1][it->second.childs[0]].adaptiert = true;
          Grid[l+1][it->second.childs[1]].adaptiert = true;
          Grid[l+1][it->second.childs[2]].adaptiert = true;
          Grid[l+1][it->second.childs[3]].adaptiert = true;
          (it->second).signifikant = true;
          ++anzahl_signifikant;
          int level = l;
          int index = it->first;
          while (Grid[level][index].hasFather) {
            ++anzahl_signifikant;
            index = Grid[level][index].Fkey;
            --level;
            Grid[level][index].signifikant = true;
          }
          break;
        }
      }
      if (!(it.second)->signifikant) (it->second).d_coeff = 0.;
    }
  }
  for (GridMap::iterator it = (Grid[0]).begin(); it != (Grid[0]).end(); ++it){
    if (!((it->second).signifikant)) (it->second).adaptiert = true;
  }
}
*/

void
ThresholdOperator (GlobalGrid &Grid, double eps, double a, unsigned int &anzahl_gesamt, unsigned int &anzahl_klein)
{
  const int levels = Grid.size ();
  anzahl_gesamt = 0;
  anzahl_klein = 0;
  for (int l = levels - 2; l >= 0; --l) {
    eps /= a;
    for (GridMap::iterator it = (Grid[l]).begin (); it != (Grid[l]).end (); ++it) {
      assert ((it->second).d_coeff.size () == 3 * M);
      for (int i = 0; i < 3 * M; ++i) {
        ++anzahl_gesamt;
        if (abs (it->second.d_coeff[i]) <= sqrt (2.0 * (it->second.flaeche)) * eps) {
          (it->second).d_coeff[i] = 0.;
          ++anzahl_klein;
        }
      }
    }
  }
}

// L2-Fehler zwischen Projektion und Ergebnis des Thresholdings
double
err_thr (GlobalGrid &Grid, double eps, double a)
{
  const int levels = Grid.size ();
  double sum = 0;
  for (int l = levels - 2; l >= 0; --l) {
    eps /= a;
    for (GridMap::iterator it = (Grid[l]).begin (); it != (Grid[l]).end (); ++it) {
      assert ((it->second).d_coeff.size () == 3 * M);
      for (int i = 0; i < 3 * M; ++i) {
        if (abs (it->second.d_coeff[i]) <= sqrt (2.0 * (it->second.flaeche)) * eps) {
          sum += (it->second).d_coeff[i] * (it->second).d_coeff[i];
        }
      }
    }
  }
  return sqrt (sum);
}

// Kleinstes eps, sodass kein Koeffizient mehr signifikant,
// also größte sinnvolle Wahl für eps
double
max_eps (GlobalGrid &Grid, double a)
{
  const int levels = Grid.size ();
  double max = 0.;
  double b = 1.;
  for (int l = levels - 2; l >= 0; --l) {
    b /= a;
    for (GridMap::iterator it = (Grid[l]).begin (); it != (Grid[l]).end (); ++it) {
      assert ((it->second).d_coeff.size () == 3 * M);
      for (int i = 0; i < 3 * M; ++i) {
        double value = abs (it->second.d_coeff[i]) / (sqrt (2.0 * (it->second.flaeche)) * b);
        if (value > max)
          max = value;
      }
    }
  }
  return max;
}

double
eps_opt (GlobalGrid &Grid, double err, double a)
{
  double delta = pow (10., floor (log10 (err)) - 1);
  double eps_l = 0.;
  double eps_r = max_eps (Grid, a);
  double err_l = 0.;
  double err_r = err_thr (Grid, eps_r, a);
  if (err_r < err)
    return eps_r;
  while ((eps_r - eps_l) >= delta) {
    double eps_m = (eps_l + eps_r) / 2.;
    double err_m = err_thr (Grid, eps_m, a);
    if (err_m == err)
      return eps_m;
    if (err_m < err) {
      eps_l = eps_m;
      err_l = err_m;
    }
    else {
      eps_r = eps_m;
      err_r = err_m;
    }
  }
  return eps_r;
}

void
MultiScaleOperator (GlobalGrid &Grid)
{
  const int levels = Grid.size ();
  for (int l = levels - 2; l >= 0; --l) {
    for (GridMap::iterator it = (Grid[l]).begin (); it != (Grid[l]).end (); ++it) {
      (it->second).u_coeff.resize (M);
      (it->second).d_coeff.resize (3 * M);
      for (int i = 0; i < M; ++i) {
        double u_sum = 0., d_sum = 0.;
        for (int j = 0; j < M; ++j) {
          double v0 = Grid[l + 1][it->second.childs[0]].u_coeff[j];
          double v1 = Grid[l + 1][it->second.childs[1]].u_coeff[j];
          double v2 = Grid[l + 1][it->second.childs[2]].u_coeff[j];
          double v3 = Grid[l + 1][it->second.childs[3]].u_coeff[j];
          u_sum += M0 (i, j) * v0;
          u_sum += M1 (i, j) * v1;
          u_sum += M2 (i, j) * v2;
          u_sum += M3 (i, j) * v3;
          d_sum += N0 (i, j) * v0;
          d_sum += N1 (i, j) * v1;
          d_sum += N2 (i, j) * v2;
          d_sum += N3 (i, j) * v3;
        }
        it->second.u_coeff[i] = u_sum;
        it->second.d_coeff[i] = d_sum;
      }
      for (int i = M; i < 3 * M; ++i) {
        double sum = 0.;
        for (int j = 0; j < M; ++j) {
          sum += N0 (i, j) * Grid[l + 1][it->second.childs[0]].u_coeff[j];
          sum += N1 (i, j) * Grid[l + 1][it->second.childs[1]].u_coeff[j];
          sum += N2 (i, j) * Grid[l + 1][it->second.childs[2]].u_coeff[j];
          sum += N3 (i, j) * Grid[l + 1][it->second.childs[3]].u_coeff[j];
        }
        it->second.d_coeff[i] = sum;
      }
    }
  }
}

void
InverseMultiScaleOperator (GlobalGrid &Grid)
{
  const int levels = Grid.size ();
  for (int l = 0; l < levels - 1; ++l) {
    for (GridMap::iterator it = (Grid[l]).begin (); it != (Grid[l]).end (); ++it) {
      Grid[l + 1][it->second.childs[0]].u_coeff.resize (M);
      Grid[l + 1][it->second.childs[1]].u_coeff.resize (M);
      Grid[l + 1][it->second.childs[2]].u_coeff.resize (M);
      Grid[l + 1][it->second.childs[3]].u_coeff.resize (M);

      for (int i = 0; i < M; ++i) {
        double sum0 = 0., sum1 = 0., sum2 = 0., sum3 = 0.;
        for (int j = 0; j < M; ++j) {
          sum0 += M0 (j, i) * it->second.u_coeff[j];
          sum1 += M1 (j, i) * it->second.u_coeff[j];
          sum2 += M2 (j, i) * it->second.u_coeff[j];
          sum3 += M3 (j, i) * it->second.u_coeff[j];
        }
        for (int j = 0; j < 3 * M; ++j) {
          sum0 += N0 (j, i) * it->second.d_coeff[j];
          sum1 += N1 (j, i) * it->second.d_coeff[j];
          sum2 += N2 (j, i) * it->second.d_coeff[j];
          sum3 += N3 (j, i) * it->second.d_coeff[j];
        }
        Grid[l + 1][it->second.childs[0]].u_coeff[i] = sum0;
        Grid[l + 1][it->second.childs[1]].u_coeff[i] = sum1;
        Grid[l + 1][it->second.childs[2]].u_coeff[i] = sum2;
        Grid[l + 1][it->second.childs[3]].u_coeff[i] = sum3;
      }
    }
  }
}

void
ProjektionSinglescale (GlobalGrid &Grid, func F, int rule)
{
  const int levels = Grid.size ();
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  order_num = dunavant_order_num (rule);
  wtab = new double[order_num];
  xytab = new double[2 * order_num];
  xytab_ref = new double[2 * order_num];
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  for (GridMap::iterator it = (Grid[levels - 1]).begin (); it != (Grid[levels - 1]).end (); ++it) {
    (it->second).u_coeff.resize (M);
    double eckpunkte[6] = { (it->second.nodes)[0].x, (it->second.nodes)[0].y, (it->second.nodes)[1].x,
                            (it->second.nodes)[1].y, (it->second.nodes)[2].x, (it->second.nodes)[2].y };
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
        lr_solve (it->second.A, it->second.r, tau);
        quad
          += wtab[order] * F (x, y) * sqrt (1. / (2. * it->second.flaeche)) * skalierungsfunktion (i, tau (0), tau (1));
      }
      quad *= it->second.flaeche;
      (it->second).u_coeff[i] = quad;
    }
  }
}

// nicht sinnvoll
void
ProjektionMultiscale (GlobalGrid &Grid, func F, int rule)
{
  const int levels = Grid.size ();
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  order_num = dunavant_order_num (rule);
  wtab = new double[order_num];
  xytab = new double[2 * order_num];
  xytab_ref = new double[2 * order_num];
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  for (int l = levels - 2; l >= 0; --l) {
    for (GridMap::iterator it = (Grid[l]).begin (); it != (Grid[l]).end (); ++it) {
      (it->second).d_coeff.resize (3 * M);
      double eckpunkte[6] = { (it->second.nodes)[0].x, (it->second.nodes)[0].y, (it->second.nodes)[1].x,
                              (it->second.nodes)[1].y, (it->second.nodes)[2].x, (it->second.nodes)[2].y };
      reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
      for (int i = 0; i < 3 * M; ++i) {
        double quad = 0.;
        for (int order = 0; order < order_num; ++order) {
          double x = xytab[order * 2];
          double y = xytab[1 + order * 2];

          vec tau (3);
          tau (0) = x;
          tau (1) = y;
          tau (2) = 1.;
          lr_solve (it->second.A, it->second.r, tau);
          quad += wtab[order] * F (x, y) * sqrt (1. / (2. * it->second.flaeche))
                  * muttermultiwavelet (p, i, tau (0), tau (1));
        }
        quad *= it->second.flaeche;
        (it->second).d_coeff[i] = quad;
      }
    }
  }
  for (GridMap::iterator it = (Grid[0]).begin (); it != (Grid[0]).end (); ++it) {
    (it->second).u_coeff.resize (M);
    double eckpunkte[6] = { (it->second.nodes)[0].x, (it->second.nodes)[0].y, (it->second.nodes)[1].x,
                            (it->second.nodes)[1].y, (it->second.nodes)[2].x, (it->second.nodes)[2].y };
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
        lr_solve (it->second.A, it->second.r, tau);
        quad
          += wtab[order] * F (x, y) * sqrt (1. / (2. * it->second.flaeche)) * skalierungsfunktion (i, tau (0), tau (1));
      }
      quad *= it->second.flaeche;
      (it->second).u_coeff[i] = quad;
    }
  }
}

double
AuswertungSinglescale (GlobalGrid &Grid, double x, double y, int level, Key index)
{
  vec tau (3);
  tau (0) = x;
  tau (1) = y;
  tau (2) = 1.;
  lr_solve (Grid[level][index].A, Grid[level][index].r, tau);
  assert (Grid[level][index].u_coeff.size () == M);
  // assert((tau(0)>=0.) && (tau(1)>=0.) && (tau(0)+tau(1)<=1.));
  double sum = 0.;
  for (int i = 0; i < M; ++i) {
    sum += Grid[level][index].u_coeff[i] * sqrt (1. / (2. * Grid[level][index].flaeche))
           * skalierungsfunktion (i, tau (0), tau (1));
  }
  return sum;
}

double
AuswertungMultiscale (GlobalGrid &Grid, double x, double y, int level, Key index)
{
  vec tau (3);
  tau (0) = x;
  tau (1) = y;
  tau (2) = 1.;
  lr_solve (Grid[level][index].A, Grid[level][index].r, tau);
  // assert((tau(0)>=0.) && (tau(1)>=0.) && (tau(0)+tau(1)<=1.));
  double sum = 0.;
  while (Grid[level][index].hasFather) {
    index = Grid[level][index].Fkey;
    --level;
    assert (Grid[level][index].d_coeff.size () == 3 * M);
    tau (0) = x;
    tau (1) = y;
    tau (2) = 1.;
    lr_solve (Grid[level][index].A, Grid[level][index].r, tau);
    for (int i = 0; i < 3 * M; ++i) {
      sum += Grid[level][index].d_coeff[i] * sqrt (1. / (2. * Grid[level][index].flaeche))
             * muttermultiwavelet (p, i, tau (0), tau (1));
    }
  }
  assert (Grid[level][index].u_coeff.size () == M);
  for (int i = 0; i < M; ++i) {
    sum += Grid[level][index].u_coeff[i] * sqrt (1. / (2. * Grid[level][index].flaeche))
           * skalierungsfunktion (i, tau (0), tau (1));
  }
  return sum;
}

void
PlotSinglescale (GlobalGrid &Grid, int n, func F)
{
  const int levels = Grid.size ();
  const int points = ((n + 2) * (n + 3)) / 2;
  const int cells = (n + 1) * (n + 1);
  char filename[80];
  sprintf (filename, "PlotSinglescale.plt");
  fstream output;
  output.open (filename, ios::out);
  output << "TITLE = PlotSinglescale" << endl;
  output << "VARIABLES = X Y F U" << endl;
  for (GridMap::iterator it = (Grid[levels - 1]).begin (); it != (Grid[levels - 1]).end (); ++it) {
    char res_out[160];
    sprintf (res_out, "Zone T=INTERIOR N=%i, E=%i, F=FEPOINT, ET=TRIANGLE", points, cells);
    output << res_out << endl;
    for (int i = 0; i < n + 2; ++i) {
      for (int j = 0; j < n + 2 - i; ++j) {
        double tau2 = (((double) j) / ((double) (n + 1)));
        double tau3 = (((double) i) / ((double) (n + 1)));
        double tau1 = 1.0 - tau2 - tau3;
        double x = tau1 * (it->second.nodes)[0].x + tau2 * (it->second.nodes)[1].x + tau3 * (it->second.nodes)[2].x;
        double y = tau1 * (it->second.nodes)[0].y + tau2 * (it->second.nodes)[1].y + tau3 * (it->second.nodes)[2].y;
        double u = AuswertungSinglescale (Grid, x, y, levels - 1, it->first);
        double f = F (x, y);
        sprintf (res_out, "%10.10f %10.10f %10.10f %10.10f", x, y, f, u);
        output << res_out << endl;
      }
    }
    for (int k = 0; k <= n; ++k) {
      int p1 = k + 1;
      int weiter = n + 2;
      for (int l = 0; l < n - k; ++l) {
        int p2 = p1 + 1;
        int p3 = p1 + weiter;
        sprintf (res_out, "%i %i %i", p1, p2, p3);
        output << res_out << endl;
        p1 = p3 + 1;
        sprintf (res_out, "%i %i %i", p1, p2, p3);
        output << res_out << endl;
        p1 = p3;
        --weiter;
      }
      int p2 = p1 + 1;
      int p3 = p1 + weiter;
      sprintf (res_out, "%i %i %i", p1, p2, p3);
      output << res_out << endl;
    }
  }
  output << endl;
  output.close ();
}

void
PlotAdaptiveGrid (GlobalGrid &Grid, int n, func F)
{
  const int levels = Grid.size ();
  int bsp = (int) F (-1., -1.);
  int points = ((n + 2) * (n + 3)) / 2;
  int cells = (n + 1) * (n + 1);
  char filename[80];
  sprintf (filename, "AdGr_f%i_p=%i_L=%i.plt", bsp, p, levels - 1);
  fstream output;
  output.open (filename, ios::out);
  output << "TITLE = PlotAdaptiveGrid" << endl;
  output << "VARIABLES = X Y U" << endl;
  for (int lvl = levels - 1; lvl >= 0; --lvl) {
    for (GridMap::iterator it = (Grid[lvl]).begin (); it != (Grid[lvl]).end (); ++it) {
      if (it->second.adaptiert) {
        char res_out[160];
        sprintf (res_out, "Zone T=INTERIOR N=%i, E=%i, F=FEPOINT, ET=TRIANGLE", points, cells);
        output << res_out << endl;
        for (int i = 0; i < n + 2; ++i) {
          for (int j = 0; j < n + 2 - i; ++j) {
            double tau2 = (((double) j) / ((double) (n + 1)));
            double tau3 = (((double) i) / ((double) (n + 1)));
            double tau1 = 1.0 - tau2 - tau3;
            double x = tau1 * (it->second.nodes)[0].x + tau2 * (it->second.nodes)[1].x + tau3 * (it->second.nodes)[2].x;
            double y = tau1 * (it->second.nodes)[0].y + tau2 * (it->second.nodes)[1].y + tau3 * (it->second.nodes)[2].y;
            double u = AuswertungSinglescale (Grid, x, y, lvl, it->first);
            // double f = F(x,y);
            // sprintf(res_out, "%10.10f %10.10f %10.10f %10.10f", x, y, f, u);
            sprintf (res_out, "%10.10f %10.10f %10.10f", x, y, u);
            output << res_out << endl;
          }
        }
        for (int k = 0; k <= n; ++k) {
          int p1 = k + 1;
          int weiter = n + 2;
          for (int l = 0; l < n - k; ++l) {
            int p2 = p1 + 1;
            int p3 = p1 + weiter;
            sprintf (res_out, "%i %i %i", p1, p2, p3);
            output << res_out << endl;
            p1 = p3 + 1;
            sprintf (res_out, "%i %i %i", p1, p2, p3);
            output << res_out << endl;
            p1 = p3;
            --weiter;
          }
          int p2 = p1 + 1;
          int p3 = p1 + weiter;
          sprintf (res_out, "%i %i %i", p1, p2, p3);
          output << res_out << endl;
        }
      }
    }
    n *= 2;
    n += 1;
    points = ((n + 2) * (n + 3)) / 2;
    cells = (n + 1) * (n + 1);
  }
  output << endl;
  output.close ();
}

void
PlotFunction (GlobalGrid &Grid, int n, func F)
{
  const int levels = Grid.size ();
  int bsp = (int) F (-1., -1.);
  int points = ((n + 2) * (n + 3)) / 2;
  int cells = (n + 1) * (n + 1);
  char filename[80];
  sprintf (filename, "f%i.plt", bsp, p, levels - 1);
  fstream output;
  output.open (filename, ios::out);
  output << "TITLE = PlotFunction" << endl;
  output << "VARIABLES = X Y F" << endl;
  int lvl = levels - 1;
  for (GridMap::iterator it = (Grid[lvl]).begin (); it != (Grid[lvl]).end (); ++it) {
    char res_out[160];
    sprintf (res_out, "Zone T=INTERIOR N=%i, E=%i, F=FEPOINT, ET=TRIANGLE", points, cells);
    output << res_out << endl;
    for (int i = 0; i < n + 2; ++i) {
      for (int j = 0; j < n + 2 - i; ++j) {
        double tau2 = (((double) j) / ((double) (n + 1)));
        double tau3 = (((double) i) / ((double) (n + 1)));
        double tau1 = 1.0 - tau2 - tau3;
        double x = tau1 * (it->second.nodes)[0].x + tau2 * (it->second.nodes)[1].x + tau3 * (it->second.nodes)[2].x;
        double y = tau1 * (it->second.nodes)[0].y + tau2 * (it->second.nodes)[1].y + tau3 * (it->second.nodes)[2].y;
        double f = F (x, y);
        sprintf (res_out, "%10.10f %10.10f %10.10f", x, y, f);
        output << res_out << endl;
      }
    }
    for (int k = 0; k <= n; ++k) {
      int p1 = k + 1;
      int weiter = n + 2;
      for (int l = 0; l < n - k; ++l) {
        int p2 = p1 + 1;
        int p3 = p1 + weiter;
        sprintf (res_out, "%i %i %i", p1, p2, p3);
        output << res_out << endl;
        p1 = p3 + 1;
        sprintf (res_out, "%i %i %i", p1, p2, p3);
        output << res_out << endl;
        p1 = p3;
        --weiter;
      }
      int p2 = p1 + 1;
      int p3 = p1 + weiter;
      sprintf (res_out, "%i %i %i", p1, p2, p3);
      output << res_out << endl;
    }
  }
  output << endl;
  output.close ();
}

void
PlotSkalierungsfkt (int m, int n)
{
  n *= 2;
  n += 1;
  int points = ((n + 2) * (n + 3)) / 2;
  int cells = (n + 1) * (n + 1);
  char filename[80];
  sprintf (filename, "PlotSkalierungsfkt_p=%i_i=%i.plt", p, m);
  fstream output;
  output.open (filename, ios::out);
  output << "TITLE = PlotSkalierungsfkt" << endl;
  output << "VARIABLES = X Y Phi" << endl;
  char res_out[160];
  sprintf (res_out, "Zone T=INTERIOR N=%i, E=%i, F=FEPOINT, ET=TRIANGLE", points, cells);
  output << res_out << endl;
  double a = (2. - sqrt (3.)) / 4.;
  double b = (2. + sqrt (3.)) / 4.;
  double c = sqrt (2.) / sqrt (sqrt (3.));
  for (int i = 0; i < n + 2; ++i) {
    for (int j = 0; j < n + 2 - i; ++j) {
      double tau2 = (((double) j) / ((double) (n + 1)));
      double tau3 = (((double) i) / ((double) (n + 1)));
      double tau1 = 1.0 - tau2 - tau3;
      double x = tau2 + tau3 * 0.5;
      double y = tau1 * a + tau2 * a + tau3 * b;
      double Phi = c * skalierungsfunktion (m, tau1, tau2);
      sprintf (res_out, "%10.10f %10.10f %10.10f", x, y, Phi);
      output << res_out << endl;
    }
  }
  for (int k = 0; k <= n; ++k) {
    int p1 = k + 1;
    int weiter = n + 2;
    for (int l = 0; l < n - k; ++l) {
      int p2 = p1 + 1;
      int p3 = p1 + weiter;
      sprintf (res_out, "%i %i %i", p1, p2, p3);
      output << res_out << endl;
      p1 = p3 + 1;
      sprintf (res_out, "%i %i %i", p1, p2, p3);
      output << res_out << endl;
      p1 = p3;
      --weiter;
    }
    int p2 = p1 + 1;
    int p3 = p1 + weiter;
    sprintf (res_out, "%i %i %i", p1, p2, p3);
    output << res_out << endl;
  }
  output << endl << "Zone T=INTERIOR N=9, E=6, F=FEPOINT, ET=TRIANGLE" << endl;
  sprintf (res_out, "%10.10f %10.10f %10.10f", 0., 1., 0.);
  output << res_out << endl;
  sprintf (res_out, "%10.10f %10.10f %10.10f", 1., 1., 0.);
  output << res_out << endl;
  sprintf (res_out, "%10.10f %10.10f %10.10f", 0.0, b, 0.);
  output << res_out << endl;
  sprintf (res_out, "%10.10f %10.10f %10.10f", 0.5, b, 0.);
  output << res_out << endl;
  sprintf (res_out, "%10.10f %10.10f %10.10f", 1.0, b, 0.);
  output << res_out << endl;
  sprintf (res_out, "%10.10f %10.10f %10.10f", 0.0, a, 0.);
  output << res_out << endl;
  sprintf (res_out, "%10.10f %10.10f %10.10f", 1.0, a, 0.);
  output << res_out << endl;
  sprintf (res_out, "%10.10f %10.10f %10.10f", 0., 0., 0.);
  output << res_out << endl;
  sprintf (res_out, "%10.10f %10.10f %10.10f", 1., 0., 0.);
  output << res_out << endl;
  output << "1 2 3" << endl;
  output << "2 3 5" << endl;
  output << "3 4 6" << endl;
  output << "4 5 7" << endl;
  output << "6 7 8" << endl;
  output << "7 8 9" << endl;
}

void
PlotMultiwavelet (int m, int n)
{
  int points = ((n + 2) * (n + 3)) / 2;
  int cells = (n + 1) * (n + 1);
  char filename[80];
  sprintf (filename, "PlotMultiwavelet_p=%i_i=%i.plt", p, m);
  fstream output;
  output.open (filename, ios::out);
  output << "TITLE = PlotMultiwavelet" << endl;
  output << "VARIABLES = X Y Psi" << endl;
  double a = (2. - sqrt (3.)) / 4.;
  double b = (2. + sqrt (3.)) / 4.;
  double c = sqrt (2.) / sqrt (sqrt (3.));
  char res_out[160];
  sprintf (res_out, "Zone T=INTERIOR N=%i, E=%i, F=FEPOINT, ET=TRIANGLE", points, cells);
  output << res_out << endl;
  for (int i = 0; i < n + 2; ++i) {
    for (int j = 0; j < n + 2 - i; ++j) {
      double tau2 = (((double) j) / ((double) (n + 1)));
      double tau3 = (((double) i) / ((double) (n + 1)));
      double tau1 = 1.0 - tau2 - tau3;
      double x = tau1 * 0.5 + tau2 * 0.75 + tau3 * 0.25;
      double y = tau1 * a + tau2 * 0.5 + tau3 * 0.5;
      double Psi = c * muttermultiwavelet (p, m, 0.5 - 0.5 * tau2, 0.5 * tau1 + 0.5 * tau2, 0);
      sprintf (res_out, "%10.10f %10.10f %10.10f", x, y, Psi);
      output << res_out << endl;
    }
  }
  for (int k = 0; k <= n; ++k) {
    int p1 = k + 1;
    int weiter = n + 2;
    for (int l = 0; l < n - k; ++l) {
      int p2 = p1 + 1;
      int p3 = p1 + weiter;
      sprintf (res_out, "%i %i %i", p1, p2, p3);
      output << res_out << endl;
      p1 = p3 + 1;
      sprintf (res_out, "%i %i %i", p1, p2, p3);
      output << res_out << endl;
      p1 = p3;
      --weiter;
    }
    int p2 = p1 + 1;
    int p3 = p1 + weiter;
    sprintf (res_out, "%i %i %i", p1, p2, p3);
    output << res_out << endl;
  }
  output << endl;

  sprintf (res_out, "Zone T=INTERIOR N=%i, E=%i, F=FEPOINT, ET=TRIANGLE", points, cells);
  output << res_out << endl;
  for (int i = 0; i < n + 2; ++i) {
    for (int j = 0; j < n + 2 - i; ++j) {
      double tau2 = (((double) j) / ((double) (n + 1)));
      double tau3 = (((double) i) / ((double) (n + 1)));
      double tau1 = 1.0 - tau2 - tau3;
      double x = tau2 * 0.5 + tau3 * 0.25;
      double y = tau1 * a + tau2 * a + tau3 * 0.5;
      double Psi = c * muttermultiwavelet (p, m, 0.5 + 0.5 * tau1, 0.5 * tau2, 1);
      sprintf (res_out, "%10.10f %10.10f %10.10f", x, y, Psi);
      output << res_out << endl;
    }
  }
  for (int k = 0; k <= n; ++k) {
    int p1 = k + 1;
    int weiter = n + 2;
    for (int l = 0; l < n - k; ++l) {
      int p2 = p1 + 1;
      int p3 = p1 + weiter;
      sprintf (res_out, "%i %i %i", p1, p2, p3);
      output << res_out << endl;
      p1 = p3 + 1;
      sprintf (res_out, "%i %i %i", p1, p2, p3);
      output << res_out << endl;
      p1 = p3;
      --weiter;
    }
    int p2 = p1 + 1;
    int p3 = p1 + weiter;
    sprintf (res_out, "%i %i %i", p1, p2, p3);
    output << res_out << endl;
  }
  output << endl;

  sprintf (res_out, "Zone T=INTERIOR N=%i, E=%i, F=FEPOINT, ET=TRIANGLE", points, cells);
  output << res_out << endl;
  for (int i = 0; i < n + 2; ++i) {
    for (int j = 0; j < n + 2 - i; ++j) {
      double tau2 = (((double) j) / ((double) (n + 1)));
      double tau3 = (((double) i) / ((double) (n + 1)));
      double tau1 = 1.0 - tau2 - tau3;
      double x = tau1 + tau2 * 0.75 + tau3 * 0.5;
      double y = tau1 * a + tau2 * 0.5 + tau3 * a;
      double Psi = c * muttermultiwavelet (p, m, 0.5 - 0.5 * tau1 - 0.5 * tau2, 0.5 + 0.5 * tau1, 2);
      sprintf (res_out, "%10.10f %10.10f %10.10f", x, y, Psi);
      output << res_out << endl;
    }
  }
  for (int k = 0; k <= n; ++k) {
    int p1 = k + 1;
    int weiter = n + 2;
    for (int l = 0; l < n - k; ++l) {
      int p2 = p1 + 1;
      int p3 = p1 + weiter;
      sprintf (res_out, "%i %i %i", p1, p2, p3);
      output << res_out << endl;
      p1 = p3 + 1;
      sprintf (res_out, "%i %i %i", p1, p2, p3);
      output << res_out << endl;
      p1 = p3;
      --weiter;
    }
    int p2 = p1 + 1;
    int p3 = p1 + weiter;
    sprintf (res_out, "%i %i %i", p1, p2, p3);
    output << res_out << endl;
  }
  output << endl;

  sprintf (res_out, "Zone T=INTERIOR N=%i, E=%i, F=FEPOINT, ET=TRIANGLE", points, cells);
  output << res_out << endl;
  for (int i = 0; i < n + 2; ++i) {
    for (int j = 0; j < n + 2 - i; ++j) {
      double tau2 = (((double) j) / ((double) (n + 1)));
      double tau3 = (((double) i) / ((double) (n + 1)));
      double tau1 = 1.0 - tau2 - tau3;
      double x = tau1 * 0.5 + tau2 * 0.25 + tau3 * 0.75;
      double y = tau1 * b + tau2 * 0.5 + tau3 * 0.5;
      double Psi = c * muttermultiwavelet (p, m, 0.5 * tau2, 0.5 - 0.5 * tau1 - 0.5 * tau2, 3);
      sprintf (res_out, "%10.10f %10.10f %10.10f", x, y, Psi);
      output << res_out << endl;
    }
  }
  for (int k = 0; k <= n; ++k) {
    int p1 = k + 1;
    int weiter = n + 2;
    for (int l = 0; l < n - k; ++l) {
      int p2 = p1 + 1;
      int p3 = p1 + weiter;
      sprintf (res_out, "%i %i %i", p1, p2, p3);
      output << res_out << endl;
      p1 = p3 + 1;
      sprintf (res_out, "%i %i %i", p1, p2, p3);
      output << res_out << endl;
      p1 = p3;
      --weiter;
    }
    int p2 = p1 + 1;
    int p3 = p1 + weiter;
    sprintf (res_out, "%i %i %i", p1, p2, p3);
    output << res_out << endl;
  }
  output << endl;

  output << "Zone T=INTERIOR N=9, E=6, F=FEPOINT, ET=TRIANGLE" << endl;
  sprintf (res_out, "%10.10f %10.10f %10.10f", 0., 1., 0.);
  output << res_out << endl;
  sprintf (res_out, "%10.10f %10.10f %10.10f", 1., 1., 0.);
  output << res_out << endl;
  sprintf (res_out, "%10.10f %10.10f %10.10f", 0.0, b, 0.);
  output << res_out << endl;
  sprintf (res_out, "%10.10f %10.10f %10.10f", 0.5, b, 0.);
  output << res_out << endl;
  sprintf (res_out, "%10.10f %10.10f %10.10f", 1.0, b, 0.);
  output << res_out << endl;
  sprintf (res_out, "%10.10f %10.10f %10.10f", 0.0, a, 0.);
  output << res_out << endl;
  sprintf (res_out, "%10.10f %10.10f %10.10f", 1.0, a, 0.);
  output << res_out << endl;
  sprintf (res_out, "%10.10f %10.10f %10.10f", 0., 0., 0.);
  output << res_out << endl;
  sprintf (res_out, "%10.10f %10.10f %10.10f", 1., 0., 0.);
  output << res_out << endl;
  output << "1 2 3" << endl;
  output << "2 3 5" << endl;
  output << "3 4 6" << endl;
  output << "4 5 7" << endl;
  output << "6 7 8" << endl;
  output << "7 8 9" << endl;
}

void
GridAdaptation (GlobalGrid &Grid, double eps, double a)
{
  const int levels = Grid.size ();
  for (int l = 1; l < levels; ++l) {
    eps /= a;
    for (GridMap::iterator it = (Grid[l]).begin (); it != (Grid[l]).end (); ++it) {
      it->second.adaptiert = false;
    }
  }
  for (GridMap::iterator it = (Grid[0]).begin (); it != (Grid[0]).end (); ++it) {
    it->second.adaptiert = true;
  }
  for (int l = 0; l < levels - 1; ++l) {
    for (GridMap::iterator it = (Grid[l]).begin (); it != (Grid[l]).end (); ++it) {
      if (it->second.adaptiert) {
        assert ((it->second).d_coeff.size () == 3 * M);
        for (int i = 0; i < 3 * M; ++i) {
          if (abs (it->second.d_coeff[i]) > sqrt (2.0 * (it->second.flaeche)) * eps) {
            it->second.adaptiert = false;
            Grid[l + 1][it->second.childs[0]].adaptiert = true;
            Grid[l + 1][it->second.childs[1]].adaptiert = true;
            Grid[l + 1][it->second.childs[2]].adaptiert = true;
            Grid[l + 1][it->second.childs[3]].adaptiert = true;
            break;
          }
        }
      }
    }
    eps *= a;
  }
}

// Erzeugt ein Gitter auf [0,1]^2 mit 2 Dreiecken auf Level 0
void
initializeGrid (GlobalGrid &Grid, int levels)
{
  Grid.clear ();
  Grid.resize (levels);

  // Initialisierung auf Level 0:
  Key IK;
  CellData ID;
  // Erstes Dreieck
  IK = 0;

  ID.hasFather = false;
  ID.hasChilds = true;
  ID.nodes.resize (3);
  ID.level = 0;
  ID.Fkey = -1;

  ID.adaptiert = false;

  ID.nodes[0].x = 0.;
  ID.nodes[0].y = 0.;

  ID.nodes[1].x = 0.;
  ID.nodes[1].y = 1.;

  ID.nodes[2].x = 1.;
  ID.nodes[2].y = 0.;

  ID.flaeche = 0.5
               * abs ((ID.nodes[0].x - ID.nodes[2].x) * (ID.nodes[1].y - ID.nodes[2].y)
                      - (ID.nodes[1].x - ID.nodes[2].x) * (ID.nodes[0].y - ID.nodes[2].y));
  ID.A.resize (3, 3);
  ID.A (0, 0) = ID.nodes[0].x;
  ID.A (0, 1) = ID.nodes[1].x;
  ID.A (0, 2) = ID.nodes[2].x;
  ID.A (1, 0) = ID.nodes[0].y;
  ID.A (1, 1) = ID.nodes[1].y;
  ID.A (1, 2) = ID.nodes[2].y;
  ID.A (2, 0) = 1.;
  ID.A (2, 1) = 1.;
  ID.A (2, 2) = 1.;
  lr_factors (ID.A, ID.r);

  Grid[0][IK] = ID;

  // Zweites Dreieck
  IK = 1;

  ID.nodes[0].x = 1.;
  ID.nodes[0].y = 1.;

  ID.nodes[1].x = 1.;
  ID.nodes[1].y = 0.;

  ID.nodes[2].x = 0.;
  ID.nodes[2].y = 1.;

  ID.flaeche = 0.5
               * abs ((ID.nodes[0].x - ID.nodes[2].x) * (ID.nodes[1].y - ID.nodes[2].y)
                      - (ID.nodes[1].x - ID.nodes[2].x) * (ID.nodes[0].y - ID.nodes[2].y));
  ID.A.resize (3, 3);
  ID.A (0, 0) = ID.nodes[0].x;
  ID.A (0, 1) = ID.nodes[1].x;
  ID.A (0, 2) = ID.nodes[2].x;
  ID.A (1, 0) = ID.nodes[0].y;
  ID.A (1, 1) = ID.nodes[1].y;
  ID.A (1, 2) = ID.nodes[2].y;
  ID.A (2, 0) = 1.;
  ID.A (2, 1) = 1.;
  ID.A (2, 2) = 1.;
  lr_factors (ID.A, ID.r);

  Grid[0][IK] = ID;

  // Fuege Kinder hinzu:
  for (int i = 0; i < levels - 1; i++) {
    // Nummerierung der Kinder auf Level i+1
    int index = 0;
    // Laufe ueber alle Zellen auf Level i
    for (GridMap::iterator it = (Grid[i]).begin (); it != (Grid[i]).end (); ++it) {
      // Berechne Mittelpunkte der Zellkanten
      node new1, new2, new3;
      node old1, old2, old3;
      old1 = it->second.nodes[0];
      old2 = it->second.nodes[1];
      old3 = it->second.nodes[2];

      new1.x = 0.5 * (it->second.nodes[0].x + it->second.nodes[1].x);
      new1.y = 0.5 * (it->second.nodes[0].y + it->second.nodes[1].y);
      new2.x = 0.5 * (it->second.nodes[1].x + it->second.nodes[2].x);
      new2.y = 0.5 * (it->second.nodes[1].y + it->second.nodes[2].y);
      new3.x = 0.5 * (it->second.nodes[0].x + it->second.nodes[2].x);
      new3.y = 0.5 * (it->second.nodes[0].y + it->second.nodes[2].y);

      it->second.childs.resize (4);
      // Fuege 1. Kind hinzu
      CellData Insert;
      Insert.level = i + 1;
      Insert.nodes.resize (3);
      // Knoten des Dreiecks
      Insert.nodes[0] = new1;
      Insert.nodes[1] = new2;
      Insert.nodes[2] = new3;
      // Vaterzelle
      Insert.hasFather = true;
      Insert.Fkey = it->first;
      // Hat Kinderzellen?
      if (i == levels - 2) {
        Insert.hasChilds = false;
        Insert.adaptiert = true;
      }
      else {
        Insert.hasChilds = true;
        Insert.adaptiert = false;
      }

      Insert.flaeche = 0.25 * it->second.flaeche;
      Insert.A.resize (3, 3);

      Insert.A (0, 0) = Insert.nodes[0].x;
      Insert.A (0, 1) = Insert.nodes[1].x;
      Insert.A (0, 2) = Insert.nodes[2].x;
      Insert.A (1, 0) = Insert.nodes[0].y;
      Insert.A (1, 1) = Insert.nodes[1].y;
      Insert.A (1, 2) = Insert.nodes[2].y;
      Insert.A (2, 0) = 1.;
      Insert.A (2, 1) = 1.;
      Insert.A (2, 2) = 1.;
      lr_factors (Insert.A, Insert.r);

      // Fuege Zelle ins Gitter ein
      Grid[i + 1][index] = Insert;
      // Teile Vater das Kind mit
      it->second.childs[0] = index;
      index++;

      // Fuege 2. Kind hinzu
      Insert.nodes[0] = old1;
      Insert.nodes[1] = new1;
      Insert.nodes[2] = new3;
      Insert.A (0, 0) = Insert.nodes[0].x;
      Insert.A (0, 1) = Insert.nodes[1].x;
      Insert.A (0, 2) = Insert.nodes[2].x;
      Insert.A (1, 0) = Insert.nodes[0].y;
      Insert.A (1, 1) = Insert.nodes[1].y;
      Insert.A (1, 2) = Insert.nodes[2].y;
      Insert.A (2, 0) = 1.;
      Insert.A (2, 1) = 1.;
      Insert.A (2, 2) = 1.;
      lr_factors (Insert.A, Insert.r);
      // Fuege Zelle ins Gitter ein
      Grid[i + 1][index] = Insert;
      // Teile Vater das Kind mit
      it->second.childs[1] = index;
      index++;

      // Fuege 3. Kind hinzu
      Insert.nodes[0] = old2;
      Insert.nodes[1] = new2;
      Insert.nodes[2] = new1;
      Insert.A (0, 0) = Insert.nodes[0].x;
      Insert.A (0, 1) = Insert.nodes[1].x;
      Insert.A (0, 2) = Insert.nodes[2].x;
      Insert.A (1, 0) = Insert.nodes[0].y;
      Insert.A (1, 1) = Insert.nodes[1].y;
      Insert.A (1, 2) = Insert.nodes[2].y;
      Insert.A (2, 0) = 1.;
      Insert.A (2, 1) = 1.;
      Insert.A (2, 2) = 1.;
      lr_factors (Insert.A, Insert.r);
      // Fuege Zelle ins Gitter ein
      Grid[i + 1][index] = Insert;
      // Teile Vater das Kind mit
      it->second.childs[2] = index;
      index++;

      // Fuege 4. Kind hinzu
      Insert.nodes[0] = old3;
      Insert.nodes[1] = new3;
      Insert.nodes[2] = new2;
      Insert.A (0, 0) = Insert.nodes[0].x;
      Insert.A (0, 1) = Insert.nodes[1].x;
      Insert.A (0, 2) = Insert.nodes[2].x;
      Insert.A (1, 0) = Insert.nodes[0].y;
      Insert.A (1, 1) = Insert.nodes[1].y;
      Insert.A (1, 2) = Insert.nodes[2].y;
      Insert.A (2, 0) = 1.;
      Insert.A (2, 1) = 1.;
      Insert.A (2, 2) = 1.;
      lr_factors (Insert.A, Insert.r);
      // Fuege Zelle ins Gitter ein
      Grid[i + 1][index] = Insert;
      // Teile Vater das Kind mit
      it->second.childs[3] = index;
      index++;
    }
  }

  // cout << "Test: " << endl;
  // cout << Grid << endl;
}

double
ErrorSinglescale (GlobalGrid &Grid, func F, int rule)
{
  const int levels = Grid.size ();
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  order_num = dunavant_order_num (rule);
  wtab = new double[order_num];
  xytab = new double[2 * order_num];
  xytab_ref = new double[2 * order_num];
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  double sum = 0.0;
  for (GridMap::iterator it = (Grid[levels - 1]).begin (); it != (Grid[levels - 1]).end (); ++it) {
    (it->second).u_coeff.resize (M);
    double eckpunkte[6] = { (it->second.nodes)[0].x, (it->second.nodes)[0].y, (it->second.nodes)[1].x,
                            (it->second.nodes)[1].y, (it->second.nodes)[2].x, (it->second.nodes)[2].y };
    reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
    double quad = 0.;
    for (int order = 0; order < order_num; ++order) {
      double x = xytab[order * 2];
      double y = xytab[1 + order * 2];
      double value = F (x, y) - AuswertungSinglescale (Grid, x, y, levels - 1, it->first);
      quad += wtab[order] * value * value;
    }
    quad *= it->second.flaeche;
    sum += quad;
  }
  return sqrt (sum);
}

double
ErrorMultiscale (GlobalGrid &Grid, func F, int rule)
{
  const int levels = Grid.size ();
  int order_num;
  double *wtab;
  double *xytab;
  double *xytab_ref;
  order_num = dunavant_order_num (rule);
  wtab = new double[order_num];
  xytab = new double[2 * order_num];
  xytab_ref = new double[2 * order_num];
  dunavant_rule (rule, order_num, xytab_ref, wtab);
  double sum = 0.0;
  for (GridMap::iterator it = (Grid[levels - 1]).begin (); it != (Grid[levels - 1]).end (); ++it) {
    (it->second).u_coeff.resize (M);
    double eckpunkte[6] = { (it->second.nodes)[0].x, (it->second.nodes)[0].y, (it->second.nodes)[1].x,
                            (it->second.nodes)[1].y, (it->second.nodes)[2].x, (it->second.nodes)[2].y };
    reference_to_physical_t3 (eckpunkte, order_num, xytab_ref, xytab);
    double quad = 0.;
    for (int order = 0; order < order_num; ++order) {
      double x = xytab[order * 2];
      double y = xytab[1 + order * 2];
      double value = F (x, y) - AuswertungMultiscale (Grid, x, y, levels - 1, it->first);
      quad += wtab[order] * value * value;
    }
    quad *= it->second.flaeche;
    sum += quad;
  }
  return sqrt (sum);
}

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

int
main ()
{
  InitialisiereKoeff (p, M0, M1, M2, M3, N0, N1, N2, N3);
  GlobalGrid Gitter;
  cout << setprecision (10);

  /////////////////////////////
  // Basisfunktionen plotten //
  /////////////////////////////
  /*
  for (int i = 0; i < M; ++i) PlotSkalierungsfkt(i, 100);
  for (int i = 0; i < 3*M; ++i) PlotMultiwavelet(i, 100);
*/

  //////////////////////
  // Funktion plotten //
  //////////////////////
  /*
  initializeGrid(Gitter, 9);
  ProjektionSinglescale(Gitter, K, 10);
  PlotFunction(Gitter, 0, K);
*/

  //////////////////////////////
  // Adaptives Gitter plotten //
  //////////////////////////////
  /*
  initializeGrid(Gitter, 8);
  double eps = 1e-2;

  ProjektionSinglescale(Gitter, F, 10);
  MultiScaleOperator(Gitter);
  GridAdaptation(Gitter, eps, 2.0);
  PlotAdaptiveGrid(Gitter, 0, F);
*/

  ////////////////////////////////////////////
  // Numerische Konvergenzordnung bei eps^L //
  ////////////////////////////////////////////
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
  }
*/

  ////////////////////////////////////////////////
  // Numerische Konvergenzordnung bei eps_opt^L //
  ////////////////////////////////////////////////
  /*
  double err, errthr, eps;
  for (int levels = 1; levels < 11; ++levels) {
    initializeGrid(Gitter, levels);
    ProjektionSinglescale(Gitter, K, 10);
    double err_new = ErrorSinglescale(Gitter, K, 10);
    MultiScaleOperator(Gitter);
    double eps_new = eps_opt(Gitter, err_new, 2.0);
    unsigned int anzahl_gesamt, anzahl_klein;
    ThresholdOperator(Gitter, eps_new, 2.0, anzahl_gesamt, anzahl_klein);
    InverseMultiScaleOperator(Gitter);
    double errthr_new = ErrorSinglescale(Gitter, K, 10);
    cout << (levels-1) << " & " << err_new << " &";
    if (levels > 1) cout << " " << log2(err/err_new);
    cout << " &";
    if (levels > 1) cout << " " << eps_new;
    cout << " & " << errthr_new << " &";
    if (levels > 1) cout << " " << log2(errthr/errthr_new);
    cout << " &";
    if (levels > 2) cout << " " << log2(eps/eps_new);
    cout << " &";
    if (levels > 1) cout << " " << ((((double) anzahl_klein)/((double) anzahl_gesamt))*100.);
    cout << " \\\\" << endl;
    eps = eps_new;
    err = err_new;
    errthr = errthr_new;
  }
*/

  return 0;
};
