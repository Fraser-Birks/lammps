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
#include "domain.h"
#include "group.h"
#include "modify.h"
#include "memory.h"
#include <cstring>

#include <iostream>

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixMLML::FixMLML(LAMMPS *lmp, int narg, char **arg) : Fix(lmp, narg, arg)
{
  local_qm_atom_list = nullptr;
  all_qm = nullptr;
  gflag = false;
  fflag = false;
  init_flag = false;
  comm_forward = 4;
  comm_reverse = 4;
  setup_only=false;
  all_pot_one_flag=false;
  prev_nlocal = -1;
  prev_qm_tot = -1;
  first_set = true;

  // fix 1 all mlml nevery rqm bw rblend type
  if (narg < 9) utils::missing_cmd_args(FLERR, "fix mlml", error);

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0){
    if (nevery == -1) setup_only = true;
    else error->all(FLERR,"Illegal fix mlml nevery value: {}", nevery);
  }

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
  // std::cout << "rblend: " << rblend << std::endl;
  // now check for keyword arguments
  // group

  int iarg = 7;
  if (strcmp(arg[iarg],"group") == 0) {
    if (iarg + 2 != narg) error->all(FLERR,"Illegal fix mlml group command");
    group2 = utils::strdup(arg[iarg+1]);
    igroup2 = group->find(arg[iarg+1]);
    if (igroup2 == -1)
      error->all(FLERR,"Group ID does not exist");
    group2bit = group->bitmask[igroup2];    
    gflag = true;
  
  // classify using the output of a different fix
  } else if (strcmp(arg[iarg], "fix_classify") == 0){
    if (iarg + 5 > narg) error->all(FLERR,"Illegal fix mlml fix_classify command");
    fix_id = utils::strdup(arg[iarg+1]);
    // error checking for fix is done when it is needed

    // next argument is nfreq, integer (how often to check connected fix)
    nfreq = utils::inumeric(FLERR,arg[iarg+2],false,lmp);

    // pass remaining arguments as doubles
    
    // check if lb is -inf
    if (strcmp(arg[iarg+3], "-inf") == 0){
      lb = -1.0e100; // switch to flag system later if this doesn't work
    } else lb = utils::numeric(FLERR,arg[iarg+3],false,lmp);

    // check if ub is inf
    if (strcmp(arg[iarg+4], "inf") == 0){
      ub = 1.0e100; // switch to flag system later if this doesn't work
    } else ub = utils::numeric(FLERR,arg[iarg+4],false,lmp);

    // check coord_ub > coord_lb
    if (ub < lb){
      error->all(FLERR, "FixMLML: fix upper bound must be greater than lower bound");
    }
    fflag=true;
    if (narg>12){
      if (narg == 14){
        // now check if we are using an initialisation group
        if (strcmp(arg[iarg+5], "init_group")==0){
          init_flag=true;
          group2 = utils::strdup(arg[iarg+6]);
          igroup2 = group->find(group2);
          if (igroup2 == -1)
            error->all(FLERR,"Group ID does not exist");
          group2bit = group->bitmask[igroup2];
        }else error->all(FLERR,"Illegal fix mlml fix_classify command");
      }else error->all(FLERR,"Illegal fix mlml fix_classify command");
    }else{
      all_pot_one_flag = true;
      first_set = false;
    }
  } else error->all(FLERR,"Illegal fix mlml command");

}

FixMLML::~FixMLML()
{
  if (local_qm_atom_list) memory->destroy(local_qm_atom_list);
}

/* ---------------------------------------------------------------------- */

int FixMLML::setmask()
{
  int mask = 0;
  // only call the end of the step here.
  mask |= FixConst::PRE_FORCE;
  if (!setup_only) mask |= FixConst::END_OF_STEP;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixMLML::init()
{
  // request neighbor list - for now just use 1
  // requestQM = 
  double max_cutoff = fmax(rqm, fmax(bw, rblend));
  neighbor->add_request(this, NeighConst::REQ_FULL)->set_cutoff(max_cutoff);
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

void FixMLML::setup_pre_force(int){
  // called right at the start of simulation
  int nlocal = atom->nlocal;
  memory->create(local_qm_atom_list, nlocal, "FixMLML: local_qm_atom_list");
  memory->create(core_qm_atom_idx, nlocal, "FixMLML: core_qm_atom_idx");
  prev_nlocal = nlocal;

  // if no initial group set, then start all atoms as QM
  if (all_pot_one_flag){
    int **i2_potential = (int**)atom->extract("i2_potential");
    double **d2_eval = (double**)atom->extract("d2_eval");
    int nlocal = atom->nlocal;
    int nghost = atom->nghost;
    for (int i = 0; i < nlocal + nghost; i++){
      i2_potential[i][0] = 1;
      i2_potential[i][1] = 0;
      d2_eval[i][0] = 1.0;
      d2_eval[i][1] = 0.0;
    }
    
  }else{

    // if there is an initial group, switch at the start
    // to doing it based on group. If not, then do it
    // with the fix
    if (init_flag){
      fflag = false;
      gflag = true;
    }

    this->allocate_regions();

    if (init_flag){
      fflag = true;
      gflag = false;
    }
  }
}

void FixMLML::end_of_step()
{
  // at the start of the timestep this is called
  this->allocate_regions();
}


void FixMLML::allocate_regions(){
  // we perform the re-allocation of
  // i2_potential and d2_eval
  int **i2_potential = (int**)atom->extract("i2_potential");
  double **d2_eval = (double**)atom->extract("d2_eval");
  int *ilist = list->ilist;
  int *jlist;
  int *num_neigh = list->numneigh;
  int **firstneigh = list->firstneigh;
  // int *type = atom->type;
  //int *tag = atom->tag;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int inum = list->inum;
  int core_tot = 0;

  bigint current_timestep = update->ntimestep;
  bigint natoms = atom->natoms;

  if (i2_potential == nullptr || d2_eval == nullptr){
    error->all(FLERR, "FixMLML: both i2_potential and d2_eval must be allocated");
  }

  if (prev_nlocal<nlocal){
    prev_nlocal = nlocal;
    memory->grow(local_qm_atom_list, nlocal, "FixMLML: local_qm_atom_list");
    memory->grow(core_qm_atom_idx, nlocal, "FixMLML: core_qm_atom_idx");
  }

  if (fflag){
    // check if this is a timestep where global qm list is updated
    if (current_timestep % nfreq == 0){
      // if it is, get fix vector and update list
      first_set = true;
      int ifix = modify->find_fix(fix_id);
      if (ifix == -1){
        error->all(FLERR, "FixMLML: fix id does not exist");
      }
      classify_fix = modify->fix[ifix];
      classify_vec = classify_fix->vector_atom;
      if (classify_vec == nullptr){
        error->all(FLERR, "FixMLML: fix does not output compatible vector");
      }
      update_global_QM_list();
    }
    // // now iterate through all_qm and 
    // // add atom->map[all_qm[i]] to core_qm_atom_idx
    // int rank;
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // for (int i = 0; i < prev_qm_tot; i++){
    //   // if all_qm[i] is a local atom
    //   std::cout<<"rank: "<<rank<<" global index: "<<all_qm[i]<<" local index: "<<atom->map(all_qm[i])<<std::endl;
    //   if (atom->map(all_qm[i]) != -1){
    //     core_qm_atom_idx[core_tot] = atom->map(all_qm[i]);
    //     core_tot++;
    //     std::cout<<"rank: "<<rank<<" added "<< atom->map(all_qm[i])<<" to core_qm_atom_idx"<<std::endl;
    //   }
    //}
  }


  // start off setting all i2_potential[0] to 0
  // and all of i2_potential[1] to 1
  if (first_set){
    for (int i = 0; i < nlocal + nghost; i++){
      i2_potential[i][0] = 0;
      i2_potential[i][1] = 1;
      d2_eval[i][0] = 0.0;
      d2_eval[i][1] = 0.0;
    }
  }
  // send this to all ghost atoms on other processors
  comm->forward_comm(this);

  // core_qm_atoms_idx is a local set of 
  // indices grouped or classified QM atoms.

  bool atom_is_qm;

  for (int ii = 0; ii < inum; ii++){
    atom_is_qm = false;
    int i = ilist[ii];

    if (gflag){
      if (group2bit & atom->mask[i]){
        atom_is_qm = true;
      }
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (fflag){
      for (int jj = 0; jj < tot_qm; jj++){
        //std::cout << "rank: " << rank << " all_qm[" << jj << "]: " << all_qm[jj] << std::endl;
        if (all_qm[jj] == atom->tag[i]){
          atom_is_qm = true;
          // if (atom_is_qm){
          //   std::cout<<"rank: "<<rank<<" atom: "<<i<<" with tag  " << atom->tag[i] <<" is QM"<<std::endl;
          // }
          break;
        }
      }
    }

    if (atom_is_qm){
      i2_potential[i][0] = 1;
      d2_eval[i][0] = 1.0;
      jlist = firstneigh[i];
      // std:: cout << "i: " << i << std::endl;
      for (int jj = 0; jj < num_neigh[i]; jj++){
        int j = jlist[jj];
        j &= NEIGHMASK;
        //int j_mapped = tag[j];
        //std::cout<<"j_mapped: "<<j_mapped<<std::endl;
        //std::cout << "j: " << j << std::endl;
        if (check_cutoff(atom->x[i], atom->x[j], rqm)){
          i2_potential[j][0] = 1;
          d2_eval[j][0] = 1.0;
        }
      }
    }
  }
  
  //std::cout <<"here2"<<std::endl;

  // now do a backward communication followed by a forward communication
  // communicate the core QM atoms to all other processors
  comm->reverse_comm(this);
  comm->forward_comm(this); // 'this' is necessary to call pack and unpack functions

  //std::cout <<"here3"<<std::endl;

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
        j &= NEIGHMASK;
        // std::cout << "j: " << j << std::endl;
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

  //std::cout <<"here4"<<std::endl;

  // loop over core_qm_idx and set d2_eval to the linear blend
  for (int ii = 0; ii < n_core_qm; ii++){
    int i = core_qm_idx[ii];
    // std::cout << "i: " << i << std::endl;
    for (int jj = 0; jj < num_neigh[i]; jj++){
      jlist = firstneigh[i];
      int j = jlist[jj];
      j &= NEIGHMASK;
      if (check_cutoff(atom->x[i], atom->x[j], rblend)){
        // std::cout<<"j: "<<j<<std::endl;
        d2_eval[j][0] = fmax(d2_eval[j][0], linear_blend(atom->x[i], atom->x[j]));
        i2_potential[j][0] = 1;
      }
    }
  }

  //std::cout <<"here5"<<std::endl;

  // communicate QM blending atoms and atoms outside MM buffer
  comm->reverse_comm(this);
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

  //std::cout <<"here6"<<std::endl;

  for (int ii = 0; ii < n_qm_and_blend_idx; ii++){
    int i = qm_and_blend_idx[ii];
    for (int jj = 0; jj < num_neigh[i]; jj++){
      jlist = firstneigh[i];
      int j = jlist[jj];
      j &= NEIGHMASK;
      if (check_cutoff(atom->x[i], atom->x[j], bw)){
        i2_potential[j][0] = 1;
      }
    }
  }

  //std::cout <<"here7"<<std::endl;

  // communicate QM buffer atoms
  comm->reverse_comm(this);
  comm->forward_comm(this);

  //std::cout <<"here8"<<std::endl;

  // final thing is to make d2_eval[i][1] = 1.0-d2_eval[i][0] for all atoms
  for (int ii = 0; ii < inum; ii++){
    int i = ilist[ii];
    d2_eval[i][1] = 1.0 - d2_eval[i][0];
  }
  // now forward comm to ensure d2_eval is correct on ghost atoms
  comm->forward_comm(this);
}


int FixMLML::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;
  int **i2_potential = (int**)atom->extract("i2_potential");
  double **d2_eval = (double**)atom->extract("d2_eval");

  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    buf[m++] = ubuf(i2_potential[i][0]).d;
    buf[m++] = ubuf(i2_potential[i][1]).d;
    buf[m++] = d2_eval[i][0];
    buf[m++] = d2_eval[i][1];
  }

  return m;
}

void FixMLML::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;
  int **i2_potential = (int**)atom->extract("i2_potential");
  double **d2_eval = (double**)atom->extract("d2_eval");

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    i2_potential[j][0] = std::max((int) ubuf(buf[m++]).i, i2_potential[j][0]);
    i2_potential[j][1] = std::max((int) ubuf(buf[m++]).i, i2_potential[j][1]);
    d2_eval[j][0] = fmax(buf[m++], d2_eval[j][0]);
    d2_eval[j][1] = fmax(buf[m++], d2_eval[j][1]);
  }
}

int FixMLML::pack_forward_comm(int n, int *list, double *buf, int pbc_flag, int *pbc)
{
  int i,j,m;
  int **i2_potential = (int**)atom->extract("i2_potential");
  double **d2_eval = (double**)atom->extract("d2_eval");

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(i2_potential[j][0]).d;
    buf[m++] = ubuf(i2_potential[j][1]).d;
    buf[m++] = d2_eval[j][0];
    buf[m++] = d2_eval[j][1];
  }
  return m;
}

void FixMLML::unpack_forward_comm(int n, int first, double *buf)
{
  int i,m,last;
  int **i2_potential = (int**)atom->extract("i2_potential");
  double **d2_eval = (double**)atom->extract("d2_eval");
  m = 0;
  last = first + n;
  for (i = first; i < last; i++) {
    i2_potential[i][0] = (int) ubuf(buf[m++]).i;
    i2_potential[i][1] = (int) ubuf(buf[m++]).i;
    d2_eval[i][0] = buf[m++];
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

  double delta[3];

  //domain->closest_image(x1, x2, delta);
  delta[0] = x1[0] - x2[0];
  delta[1] = x1[1] - x2[1];
  delta[2] = x1[2] - x2[2];
  double r = sqrt(delta[0]*delta[0] + delta[1]*delta[1] + delta[2]*delta[2]);
  
  // std::cout<<"delta[0]: "<<delta[0]<<std::endl;
  // std::cout<<"delta[1]: "<<delta[1]<<std::endl;
  // std::cout<<"delta[2]: "<<delta[2]<<std::endl;
  // std::cout<<"r: "<<r<<std::endl;
  // std::cout<<"rblend: "<<rblend<<std::endl;
  // std::cout<<"1.0 - (r/rblend): "<<1.0 - (r/rblend)<<std::endl;
  return 1.0 - (r/rblend);
}


void FixMLML::update_global_QM_list()
{
  int nlocal = atom->nlocal;
  int *atom_ids = atom->tag;
  int tot_qm_loc;
  int world_size;
  tot_qm_loc = 0;
  // first store all local global qm atom indices
  // std::cout<<"local qm atom list"<<std::endl;
  for (int i = 0; i < nlocal; i++){
    if (classify_vec[i] >= lb && classify_vec[i] <= ub){
      std::cout<< "local index: " << i << " global index: " << atom_ids[i] << std::endl;
      std::cout<< "classify_vec: " << classify_vec[i] << std::endl;
      local_qm_atom_list[tot_qm_loc] = atom_ids[i];
      tot_qm_loc++;
    }
  }
  // now gather all global qm atom indices from all processors
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  // std::cout<<"world size: "<<world_size<<std::endl;
  int *tot_qm_glob = new int[world_size];
  MPI_Allgather(&tot_qm_loc, 1, MPI_INT, tot_qm_glob, 1, MPI_INT, MPI_COMM_WORLD);
  tot_qm = 0;
  // std::cout<<"total qm atom list"<<std::endl;
  for (int i = 0; i < world_size; i++){
    // std::cout<<"processor: "<<i<<" total qm atoms: "<<tot_qm_glob[i]<<std::endl;
    tot_qm += tot_qm_glob[i];
  }
  // std::cout<<"total qm atoms: "<<tot_qm<<std::endl;

  if (prev_qm_tot < tot_qm){
    memory->grow(all_qm, tot_qm, "FixMLML: all_qm");
    prev_qm_tot = tot_qm;
  }
  int *displs = new int[world_size];
  displs[0] = 0;
  for (int i = 1; i < world_size; i++){
    displs[i] = displs[i-1] + tot_qm_glob[i-1];
    // std::cout<<"displs["<<i<<"]: "<<displs[i]<<std::endl;
  }
  // print everything going into allgatherv
  // std::cout<<"total qm loc: "<<tot_qm_loc<<std::endl;
  // std::cout<<"local qm atom list"<<std::endl;
  // for (int i = 0; i < tot_qm_loc; i++){
  //   std::cout<< "global index: " << local_qm_atom_list[i] << std::endl;
  // }
  // std::cout<<"total qm glob: "<<tot_qm<<std::endl;

  // std::cout<<"displs: "<<std::endl;
  // for (int i = 0; i < world_size; i++){
  //   std::cout<<displs[i]<<std::endl;
  // }

  MPI_Allgatherv(local_qm_atom_list, tot_qm_loc, MPI_INT, all_qm, tot_qm_glob, displs, MPI_INT, MPI_COMM_WORLD);

  // get proc
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // print out all_qm
  // std::cout<<"all qm atom list"<<std::endl;
  // for (int i = 0; i < tot_qm; i++){
  //   std::cout<< "rank: " << rank << "global index: " << all_qm[i] << std::endl;
  // }
  delete[] tot_qm_glob;
  delete[] displs;
}