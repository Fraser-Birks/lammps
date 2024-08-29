// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "pair_hybrid_overlay_mlml.h"

#include "atom.h"
#include "error.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "pair.h"
#include "respa.h"
#include "suffix.h"
#include "update.h"

#include <cstring>
#include <iostream>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairHybridOverlayMLML::PairHybridOverlayMLML(LAMMPS *lmp) : PairHybridOverlay(lmp) {
  f_copy = nullptr;
  ilist_temp = nullptr;
  ilist_copy = nullptr;
  pot_eval_arr = nullptr;
  last_ntot = -1;
  last_nlocal = -1;
}

PairHybridOverlayMLML::~PairHybridOverlayMLML() {
  // std::cout<< "destroying pairhybridoverlaymlml"<<std::endl;
  memory->destroy(f_copy);
  if (ilist_temp) memory->destroy(ilist_temp);
  if (ilist_copy) memory->destroy(ilist_copy);
  if (pot_eval_arr) memory->destroy(pot_eval_arr);
}


void PairHybridOverlayMLML::resize_arrays(){
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int ntot = nlocal + nghost;
  if (ntot > last_ntot) {
    // std::cout<<"resizing fcopy"<<std::endl;
    memory->grow(f_copy, ntot, 3, "pair:f_copy");
    last_ntot = ntot;
  }
  if (nlocal > last_nlocal){
    // std::cout<<"resizing ilists"<<std::endl;
    memory->grow(ilist_temp, nlocal, "pair:ilist_temp");
    memory->grow(ilist_copy, nlocal, "pair:ilist_copy");
    last_nlocal = nlocal;
  }
}


void PairHybridOverlayMLML::settings(int narg, char **arg){
  PairHybridOverlay::PairHybrid::settings(narg, arg);
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int ntot = nlocal + nghost;
  memory->create(ilist_copy, nlocal, "pair:ilist_copy");
  memory->create(ilist_temp, nlocal, "pair:ilist_temp");
  memory->create(f_copy, ntot, 3, "pair:f_copy");
  memory->create(pot_eval_arr, nstyles, "pair:pot_eval_arr");
  resize_arrays();

}


/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairHybridOverlayMLML::coeff(int narg, char **arg)
{
  if (narg < 3) error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  // 3rd arg = pair sub-style name
  // 4th arg = pair style pot_for_eval index
  // 5th arg = pair sub-style index if name used multiple times
  // allow for "none" as valid sub-style name

  int multflag = 0;
  int m;

  for (m = 0; m < nstyles; m++) {
    multflag = 0;
    if (strcmp(arg[2],keywords[m]) == 0) {
      if (multiple[m]) {
        multflag = 1;
        if (narg < 5) error->all(FLERR,"Incorrect args for pair coefficients");
        if (multiple[m] == utils::inumeric(FLERR,arg[4],false,lmp)) break;
        else continue;
      } else break;
    }
  }

  // once multiples dealt with, set pot eval arr
  pot_eval_arr[m] = utils::inumeric(FLERR,arg[3],false,lmp);
  // std::cout << "pot_eval_arr[" << m << "] = " << pot_eval_arr[m] << std::endl;
  if (pot_eval_arr[m] < 1) error->all(FLERR,"Index for potential evaluation must be greater than 1");


  int none = 0;
  if (m == nstyles) {
    if (strcmp(arg[2],"none") == 0)
      none = 1;
    else
      error->all(FLERR,"Expected hybrid sub-style instead of {} in pair_coeff command", arg[2]);
  }

  // move 1st/2nd args to 3rd/4th args
  // if multflag: move 1st/2nd args to 4th/5th args
  // just copy ptrs, since arg[] points into original input line

  arg[3+multflag] = arg[1];
  arg[2+multflag] = arg[0];

  // ensure that one_coeff flag is honored

  if (!none && styles[m]->one_coeff)
    if ((strcmp(arg[0],"*") != 0) || (strcmp(arg[1],"*") != 0))
      error->all(FLERR,"Incorrect args for pair coefficients");

  // invoke sub-style coeff() starting with 1st remaining arg

  if (!none) styles[m]->coeff(narg-2-multflag,arg+2+multflag);

  // set setflag and which type pairs map to which sub-style
  // if sub-style is none: set hybrid subflag, wipe out map
  // else: set hybrid setflag & map only if substyle setflag is set
  //       if sub-style is new for type pair, add as multiple mapping
  //       if sub-style exists for type pair, don't add, just update coeffs

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      if (none) {
        setflag[i][j] = 1;
        nmap[i][j] = 0;
        count++;
      } else if (styles[m]->setflag[i][j]) {
        int k;
        for (k = 0; k < nmap[i][j]; k++)
          if (map[i][j][k] == m) break;
        if (k == nmap[i][j]) map[i][j][nmap[i][j]++] = m;
        setflag[i][j] = 1;
        count++;
      }
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}


void PairHybridOverlayMLML::compute(int eflag, int vflag)
{
  int i,j,m,n;
  double** f = atom->f;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  NeighList *list_m;
  int* ilist;
   

  //std::cout << "PairHybridOverlayMLML::compute" << std::endl;
  
  resize_arrays();
  
  // get the i2_potential and d2_eval properties per atom
  int** i2_potential = (int**)atom->extract("i2_potential");
  double** d2_eval = (double**)atom->extract("d2_eval");


  // check if no_virial_fdotr_compute is set and global component of
  //   incoming vflag = VIRIAL_FDOTR
  // if so, reset vflag as if global component were VIRIAL_PAIR
  // necessary since one or more sub-styles cannot compute virial as F dot r

  if (no_virial_fdotr_compute && (vflag & VIRIAL_FDOTR))
    vflag = VIRIAL_PAIR | (vflag & ~VIRIAL_FDOTR);

  ev_init(eflag,vflag);

  // check if global component of incoming vflag = VIRIAL_FDOTR
  // if so, reset vflag passed to substyle so VIRIAL_FDOTR is turned off
  // necessary so substyle will not invoke virial_fdotr_compute()

  int vflag_substyle;
  if (vflag & VIRIAL_FDOTR) vflag_substyle = vflag & ~VIRIAL_FDOTR;
  else vflag_substyle = vflag;

  double *saved_special = save_special();

  // check if we are running with r-RESPA using the hybrid keyword

  Respa *respa = nullptr;
  respaflag = 0;
  if (utils::strmatch(update->integrate_style,"^respa")) {
    respa = dynamic_cast<Respa *>(update->integrate);
    if (respa->nhybrid_styles > 0) respaflag = 1;
  }

  for (m = 0; m < nstyles; m++) {

    set_special(m);

    if (!respaflag || (respaflag && respa->hybrid_compute[m])) {
      
      // invoke compute() unless compute flag is turned off or
      // outerflag is set and sub-style has a compute_outer() method
      list_m = styles[m]->list;
      ilist = list_m->ilist;
      inum_copy = list_m->inum;
      inum_new = 0;

      // manually copy the data over
      for (int i = 0; i < inum_copy; i++) {
        ilist_copy[i] = ilist[i];
      }

      // print ilist pointer address
      // copy forces
      for (int i = 0; i < nlocal+nghost; i++) {
        f_copy[i][0] = f[i][0];
        f_copy[i][1] = f[i][1];
        f_copy[i][2] = f[i][2];
      }

      // zero original forces
      for (int i = 0; i < nlocal+nghost; i++) {
        f[i][0] = 0.0;
        f[i][1] = 0.0;
        f[i][2] = 0.0;
      }


      for (i = 0; i < nlocal; i++){
        ilist_temp[i] = 0;
        if (i2_potential[i][pot_eval_arr[m]-1] == 1){
          ilist_temp[inum_new] = ilist[i];
          inum_new++;
        }
      }


      list_m->inum = inum_new;
      for (int i = 0; i < nlocal; i++) {
        ilist[i] = ilist_temp[i];
      }
      
      if (styles[m]->compute_flag == 0) continue;
      if (outerflag && styles[m]->respa_enable)
        styles[m]->compute_outer(eflag,vflag_substyle);
      else styles[m]->compute(eflag,vflag_substyle);


      // restore forces
      for (int i = 0; i < nlocal+nghost; i++) {
        f_copy[i][0] += f[i][0] * d2_eval[i][pot_eval_arr[m]-1];
        f_copy[i][1] += f[i][1] * d2_eval[i][pot_eval_arr[m]-1];
        f_copy[i][2] += f[i][2] * d2_eval[i][pot_eval_arr[m]-1];

        f[i][0] = f_copy[i][0];
        f[i][1] = f_copy[i][1];
        f[i][2] = f_copy[i][2];
      }

      // restore neigh list
      list_m->inum = inum_copy;
      // restore ilist
      for (int i = 0; i < nlocal; i++) {
        ilist[i] = ilist_copy[i];
      }
    }

    restore_special(saved_special);

    // jump to next sub-style if r-RESPA does not want global accumulated data

    if (respaflag && !respa->tally_global) continue;

    if (eflag_global) {
      eng_vdwl += styles[m]->eng_vdwl;
      eng_coul += styles[m]->eng_coul;
    }
    if (vflag_global) {
      for (n = 0; n < 6; n++) virial[n] += styles[m]->virial[n];
    }
    if (eflag_atom) {
      n = atom->nlocal;
      if (force->newton_pair) n += atom->nghost;
      double *eatom_substyle = styles[m]->eatom;
      for (i = 0; i < n; i++) eatom[i] += eatom_substyle[i];
    }
    if (vflag_atom) {
      n = atom->nlocal;
      if (force->newton_pair) n += atom->nghost;
      double **vatom_substyle = styles[m]->vatom;
      for (i = 0; i < n; i++)
        for (j = 0; j < 6; j++)
          vatom[i][j] += vatom_substyle[i][j];
    }

    // substyles may be CENTROID_SAME or CENTROID_AVAIL

    if (cvflag_atom) {
      n = atom->nlocal;
      if (force->newton_pair) n += atom->nghost;
      if (styles[m]->centroidstressflag == CENTROID_AVAIL) {
        double **cvatom_substyle = styles[m]->cvatom;
        for (i = 0; i < n; i++)
          for (j = 0; j < 9; j++)
            cvatom[i][j] += cvatom_substyle[i][j];
      } else {
        double **vatom_substyle = styles[m]->vatom;
        for (i = 0; i < n; i++) {
          for (j = 0; j < 6; j++) {
            cvatom[i][j] += vatom_substyle[i][j];
          }
          for (j = 6; j < 9; j++) {
            cvatom[i][j] += vatom_substyle[i][j-3];
          }
        }
      }
    }

  }

  delete[] saved_special;


  if (vflag_fdotr) virial_fdotr_compute();
}
