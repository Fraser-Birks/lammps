#include "fix_mlml.h"
#include "atom.h"
#include "error.h"
#include "force.h"
#include "respa.h"
#include "update.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "comm.h"
#include <cstring>

#include <iostream>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixMLML::FixMLML(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  comm_forward = 3;
  // fix 1 all mlml nevery rqm bw rblend type
  if (narg < 8) utils::missing_cmd_args(FLERR, "fix mlml", error);

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0)
    error->all(FLERR,"Illegal fix mlml nevery value: {}", nevery);

  rqm = utils::numeric(FLERR,arg[4],false,lmp);
  std::cout << "rqm: " << rqm << std::endl;
  if (rqm < 0.0)
    error->all(FLERR,"Illegal fix mlml rqm value: {}", rqm);

  bw = utils::numeric(FLERR,arg[5],false,lmp);
  if (bw < 0.0)
    error->all(FLERR,"Illegal fix mlml bw value: {}", bw);
  
  rblend = utils::numeric(FLERR,arg[6],false,lmp);
  if (rblend < 0.0)
    error->all(FLERR,"Illegal fix mlml rblend value: {}", rblend);
  
  type_val = utils::inumeric(FLERR,arg[7],false,lmp);
  is_type = utils::is_type(arg[7]);
  std::cout << "type_val: " << type_val << std::endl;
  std::cout << "is_type: " << is_type << std::endl;
  // This is pretty restrictive and should be updated
  // to handle proper type descriptors, e.g *, -1 etc.
  if (is_type != 0)
    error->all(FLERR,"Illegal fix mlml type value: {}", type_val);
}

/* ---------------------------------------------------------------------- */

int FixMLML::setmask()
{
  int mask = 0;
  // only call the end of the step here.
  mask |= FixConst::END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixMLML::init()
{
  // request neighbor list - for now just use 1
  // requestQM = 
  neighbor->add_request(this, NeighConst::REQ_FULL);
  // requestQM->set_cutoff(rqm);
  // requestQM->set_id(1);
  // requestBL = neighbor->add_request(this, NeighConst::REQ_FULL);
  // requestBL->set_cutoff(rblend);
  // requestBL->set_id(2);
  // requestB = neighbor->add_request(this, NeighConst::REQ_FULL);
  // requestB->set_cutoff(bw);
  // requestB->set_id(3);

}

void FixMLML::init_list(int id, NeighList *ptr)
{

  list = ptr;
  // if (id == 1){
  //   std::cout << "init_list: " << id << std::endl;
  //   rqm_list = ptr;
  //   }
  // else if (id == 2){
  //   std::cout << "init_list: " << id << std::endl;
  //   rbl_list = ptr;
  //   }
  // else if (id == 3){
  //   std::cout << "init_list: " << id << std::endl;
  //   bw_list = ptr;
  //   }
}

void FixMLML::end_of_step()
{
  // at the end of the timestep this is called
  // we perform the re-allocation of
  // i2_potential and d2_eval
  int **i2_potential = (int**)atom->extract("i2_potential");
  double **d2_eval = (double**)atom->extract("d2_eval");
  int *ilist = list->ilist;
  int *jlist;
  int *num_neigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  int inum = list->inum;
  

  // start off setting all i2_potential[0] to 0
  // and all of i2_potential[1] to 1
  for (int ii = 0; ii < inum; ii++){
    int i = ilist[ii];
    i2_potential[i][0] = 0;
    i2_potential[i][1] = 1;
    d2_eval[i][0] = 0.0;
    d2_eval[i][1] = 0.0;
  }

  // pre allocate qm_idx to size n_local
  // loop over nlocal getting index of all atoms where type == type_val
  // if an atom is QM then set i2_potential[0][i] = 1
  // and loop over it's neighbours and set i2_potential[0][j] = 1
  // if the neighbour is within rqm cutoff
  for (int ii = 0; ii < inum; ii++){
    int i = ilist[ii];
    if (type[i] == type_val){
      i2_potential[i][0] = 1;
      d2_eval[i][0] = 1.0;
      jlist = firstneigh[i];
      for (int jj = 0; jj < num_neigh[i]; jj++){
        int j = jlist[jj];
        if (check_cutoff(atom->x[i], atom->x[j], rqm)){
          i2_potential[j][0] = 1;
          d2_eval[j][0] = 1.0;
        }
      }
    }
  }
  // communicate the core QM atoms to all other processors
  comm->forward_comm(this); // 'this' is necessary to call pack and unpack functions


  // loop over inum and if i2_potential[i][0] == 1 
  // then add i to the core qm atoms and loop over it's neighbours
  std::vector<int> core_qm_idx;
  bool just_qm;
  int n_core_qm = 0;
  for (int ii = 0; ii < inum; ii++){
    int i = ilist[ii];
    if (i2_potential[i][0] == 1){
      core_qm_idx.push_back(i);
      n_core_qm++;
      just_qm = true;
      jlist = firstneigh[i];
      for (int jj = 0; jj < num_neigh[i]; jj++){
        int j = jlist[jj];
        if (check_cutoff(atom->x[i], atom->x[j], bw)){
          if (i2_potential[j][0] == 0){
            just_qm = false;
            break;
          }
        }
      }
      if (just_qm){
        // this atom is outside the MM buffer
        i2_potential[i][1] = 0;
      }
    }
  }

  // loop over core_qm_idx and set d2_eval to the linear blend
  for (int ii = 0; ii < n_core_qm; ii++){
    int i = core_qm_idx[ii];
    for (int jj = 0; jj < num_neigh[i]; jj++){
      int j = jlist[jj];
      if (check_cutoff(atom->x[i], atom->x[j], rqm)){
        d2_eval[j][0] = fmax(d2_eval[j][0], linear_blend(atom->x[i], atom->x[j]));
        i2_potential[j][0] = 1;
      }
    }
  }

  // communicate QM blending atoms and atoms outside MM buffer
  comm->forward_comm(this);

  // now repeat again for QM buffer
  std::vector<int> qm_and_blend_idx;
  int n_qm_and_blend_idx = 0;
  for (int ii = 0; ii < inum; ii++){
    int i = ilist[ii];
    if (i2_potential[i][0] == 1){
      qm_and_blend_idx.push_back(i);
      n_qm_and_blend_idx++;
    }
  }

  for (int ii = 0; ii < n_qm_and_blend_idx; ii++){
    int i = qm_and_blend_idx[ii];
    for (int jj = 0; jj < num_neigh[i]; jj++){
      int j = jlist[jj];
      if (check_cutoff(atom->x[i], atom->x[j], bw)){
        i2_potential[j][0] = 1;
      }
    }
  }

  // communicate QM buffer atoms
  comm->forward_comm(this);
}

int FixMLML::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i,j,m;
  int **i2_potential = (int**)atom->extract("i2_potential");
  int **d2_eval = (int**)atom->extract("d2_eval");

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(i2_potential[j][0]).d;
    buf[m++] = ubuf(i2_potential[j][1]).d;
    buf[m++] = d2_eval[0][j];
  }
  return m;
}

void FixMLML::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;
  int **i2_potential = (int**)atom->extract("i2_potential");
  int **d2_eval = (int**)atom->extract("d2_eval");
  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    i2_potential[i][0] = (int) ubuf(buf[m++]).i;
    i2_potential[i][1] = (int) ubuf(buf[m++]).i;
    d2_eval[i][1] = buf[m++];
  }
}

bool FixMLML::check_cutoff(double *x1, double *x2, double cutoff)
{
  double dx = x1[0] - x2[0];
  double dy = x1[1] - x2[1];
  double dz = x1[2] - x2[2];
  double rsq = dx*dx + dy*dy + dz*dz;
  if (rsq < cutoff*cutoff) return true;
  return false;
}

double FixMLML::linear_blend(double *x1, double *x2)
{
  double dx = x1[0] - x2[0];
  double dy = x1[1] - x2[1];
  double dz = x1[2] - x2[2];
  double r = sqrt(dx*dx + dy*dy + dz*dz);
  return 1.0 - (r/rblend);
}
