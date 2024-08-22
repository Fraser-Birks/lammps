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

/*
Copyright 2021 Yury Lysogorskiy^1, Cas van der Oord^2, Anton Bochkarev^1,
 Sarath Menon^1, Matteo Rinaldi^1, Thomas Hammerschmidt^1, Matous Mrovec^1,
 Aidan Thompson^3, Gabor Csanyi^2, Christoph Ortner^4, Ralf Drautz^1

^1: Ruhr-University Bochum, Bochum, Germany
^2: University of Cambridge, Cambridge, United Kingdom
^3: Sandia National Laboratories, Albuquerque, New Mexico, USA
^4: University of British Columbia, Vancouver, BC, Canada
*/

//
// Created by Lysogorskiy Yury on 27.02.20.
//

#include "pair_pace_mix.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "math_const.h"
#include "memory.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "update.h"

#include <cstring>
#include <exception>

#include "ace-evaluator/ace_c_basis.h"
#include "ace-evaluator/ace_evaluator.h"
#include "ace-evaluator/ace_recursive.h"
#include "ace-evaluator/ace_version.h"

namespace LAMMPS_NS {
struct ACEImpl {
  ACEImpl() : basis_set(nullptr), ace(nullptr) {}
  ~ACEImpl()
  {
    delete basis_set;
    delete ace;
  }
  ACECTildeBasisSet *basis_set;
  ACERecursiveEvaluator *ace;
};
}    // namespace LAMMPS_NS

using namespace LAMMPS_NS;
using namespace MathConst;

static char const *const elements_pace[] = {
    "X",  "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne", "Na", "Mg", "Al", "Si",
    "P",  "S",  "Cl", "Ar", "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
    "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr",
    "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac",
    "Th", "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"};
static constexpr int elements_num_pace = sizeof(elements_pace) / sizeof(const char *);

static int AtomicNumberByName_pace(char *elname)
{
  for (int i = 1; i < elements_num_pace; i++)
    if (strcmp(elname, elements_pace[i]) == 0) return i;
  return -1;
}

/* ---------------------------------------------------------------------- */
PairPACEMix::PairPACEMix(LAMMPS *lmp) : Pair(lmp)
{
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;

  aceimpl = new ACEImpl;
  recursive = false;

  scale = nullptr;

  chunksize = 4096;
}

/* ----------------------------------------------------------------------
   check if allocated, since class can be destructed when incomplete
------------------------------------------------------------------------- */

PairPACEMix::~PairPACEMix()
{
  if (copymode) return;

  delete aceimpl;

  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
    memory->destroy(scale);
  }
}

/* ---------------------------------------------------------------------- */

void PairPACEMix::compute(int eflag, int vflag)
{
  int i, j, ii, jj, inum, jnum, k, mmi, mmii, mmsize;
  double delx, dely, delz, evdwl;
  double fij[3];
  int *ilist, *jlist, *numneigh, **firstneigh;

  ev_init(eflag, vflag);

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int **i2_potential = (int**)atom->extract("i2_potential");
  // int **i2_buffer = (int**)atom->extract("i2_buffer");
  double **d2_eval = (double**)atom->extract("d2_eval");
  // double **d2_n3lerr = (double**)atom->extract("d2_n3lerr");
  double n3lerr_p_a[3];


  // check if both variables could be read, if not throw an exception
  if (i2_potential == nullptr || d2_eval == nullptr) { // || d2_n3lerr == nullptr
    error->all(FLERR, "pair style pace/mix requires 'i2_potential' and 'd2_eval' property/atom attributes");
  }


  // number of atoms in cell
  int nlocal = atom->nlocal;
  int newton_pair = force->newton_pair;

  // inum: length of the neighborlists list
  inum = list->inum;

  // ilist: list of "i" atoms for which neighbor lists exist
  ilist = list->ilist;

  //numneigh: the length of each these neigbor list
  numneigh = list->numneigh;

  // the pointer to the list of neighbors of "i"
  firstneigh = list->firstneigh;

  //determine the maximum number of neighbours
  int max_jnum = 0;
  int nei = 0;
  for (ii = 0; ii < list->inum; ii++) {
    i = ilist[ii];
    jnum = numneigh[i];
    nei = nei + jnum;
    if (jnum > max_jnum) max_jnum = jnum;
  }

  aceimpl->ace->resize_neighbours_cache(max_jnum);
  
  // reduce the size of the i neighbor lists by only considering
  // atoms of the specified type
  // get the temp forces array from the atom object created when pair_style was initialised

  std::vector<int> reduced_neigh_indices;
  reduced_neigh_indices.reserve(list->inum);
  for (int ii = 0; ii < list->inum; ii++) {
      i = list->ilist[ii];
      if (i2_potential[i][this->pot_for_eval-1] == 1) {
          reduced_neigh_indices.push_back(i);
      }
  }

  // // set entries of d2_n3lerr to 0.0
  // for (int ii = 0; ii < list->inum; ii++) {
  //   i = list->ilist[ii];
  //   for (int j = 0; j < 3; j++)
  //     d2_n3lerr[i][j] = 0.0;
  // }

  // make boolean variable which  represents a bond that crosses the border 
  // initially false
  bool cross_border = false;
  
  // get the correct length of the reduced_neigh_indices array
  int reduced_neigh_length = reduced_neigh_indices.size();
  //loop over atoms
  for (ii = 0; ii < reduced_neigh_length; ii++) {
    i = reduced_neigh_indices[ii];
    const int itype = type[i];

    const double xtmp = x[i][0];
    const double ytmp = x[i][1];
    const double ztmp = x[i][2];

    jlist = firstneigh[i];
    jnum = numneigh[i];

    // checking if neighbours are actually within cutoff range is done inside compute_atom
    // mapping from LAMMPS atom types ('type' array) to ACE species is done inside compute_atom
    //      by using 'aceimpl->ace->element_type_mapping' array
    // x: [r0 ,r1, r2, ..., r100]
    // i = 0 ,1
    // jnum(0) = 50
    // jlist(neigh ind of 0-atom) = [1,2,10,7,99,25, .. 50 element in total]

    try {
      aceimpl->ace->compute_atom(i, x, type, jnum, jlist);
    } catch (std::exception &e) {
      error->one(FLERR, e.what());
    }

    // 'compute_atom' will update the `aceimpl->ace->e_atom` and `aceimpl->ace->neighbours_forces(jj, alpha)` arrays
    std::vector<int> MM_neigh_indices;
    reduced_neigh_indices.reserve(jnum);
    for (k = 0; k < 3; k++) {
      n3lerr_p_a[k] = 0.0;
    }
    // std::cout<<"i: "<<i<<"\n";
    // std::cout<<"n3lerr_p_a before:"<<"\n";
    // std::cout<<n3lerr_p_a[0]<<" "<<n3lerr_p_a[1]<<" "<<n3lerr_p_a[2]<<"\n";
    // std::cout<<"d2_n3lerr before:"<<"\n";
    // std::cout<<d2_n3lerr[i][0]<<" "<<d2_n3lerr[i][1]<<" "<<d2_n3lerr[i][2]<<"\n";

    // loop over the neighbors of atom i
    cross_border = false;
    for (jj = 0; jj < jnum; jj++) { 

      j = jlist[jj];
      j &= NEIGHMASK;

      // if (i_potential[j] == 2){
      //   MM_neigh_indices.push_back(j);
      // }

      delx = x[j][0] - xtmp;
      dely = x[j][1] - ytmp;
      delz = x[j][2] - ztmp;

      fij[0] = scale[itype][itype] * aceimpl->ace->neighbours_forces(jj, 0);
      fij[1] = scale[itype][itype] * aceimpl->ace->neighbours_forces(jj, 1);
      fij[2] = scale[itype][itype] * aceimpl->ace->neighbours_forces(jj, 2);

      f[i][0] += fij[0]*d2_eval[i][this->pot_for_eval-1];
      f[i][1] += fij[1]*d2_eval[i][this->pot_for_eval-1];
      f[i][2] += fij[2]*d2_eval[i][this->pot_for_eval-1];
      f[j][0] -= fij[0]*d2_eval[j][this->pot_for_eval-1];
      f[j][1] -= fij[1]*d2_eval[j][this->pot_for_eval-1];
      f[j][2] -= fij[2]*d2_eval[j][this->pot_for_eval-1];


      // having these if statements is worrying from a performance point of view

      // // this part is activated if your 'i' atom is in the buffer region
      // // it throws away the force components on i
      // if (i_potential[i] != this->pot_for_eval) {
      //     //std::cout<<"1 activated for i: "<<i<<"\n";
      //     f[i][0] -= fij[0];
      //     f[i][1] -= fij[1];
      //     f[i][2] -= fij[2];
      // //     d2_n3lerr[i][0] -= fij[0];
      // //     d2_n3lerr[i][1] -= fij[1];
      // //     d2_n3lerr[i][2] -= fij[2];
      // //     n3lerr_p_a[0] -= fij[0];
      // //     n3lerr_p_a[1] -= fij[1];
      // //     n3lerr_p_a[2] -= fij[2];
      // //     cross_border = true;
      // }

      // // // this part is activated if your 'j' atom is in the buffer region
      // // // it throws away the force components on j 
      // if (i_potential[j] != this->pot_for_eval) {
      //     //std::cout<<"2 activated for i: "<<i<<"\n";
      //     f[j][0] += fij[0];
      //     f[j][1] += fij[1];
      //     f[j][2] += fij[2];
      // //     d2_n3lerr[i][0] += fij[0];
      // //     d2_n3lerr[i][1] += fij[1];
      // //     d2_n3lerr[i][2] += fij[2];
      // //     n3lerr_p_a[0] += fij[0];
      // //     n3lerr_p_a[1] += fij[1];
      // //     n3lerr_p_a[2] += fij[2];
      // //     cross_border = true;
      // }

      // tally per-atom virial contribution
      if (vflag_either){
        ev_tally_xyz(i, j, nlocal, newton_pair, 0.0, 0.0, fij[0], fij[1], fij[2], -delx, -dely,
                     -delz);
      }
    }

    // tally energy contribution
    if (eflag_either) {
      // evdwl = energy of atom I
      // if (i_potential[i] == this->pot_for_eval) {
      evdwl = scale[itype][itype] * aceimpl->ace->e_atom;
      ev_tally_full(i, 2.0 * evdwl, 0.0, 0.0, 0.0, 0.0, 0.0);
      }
    // iterate back over the MM atoms and add the negative of the average
    // of the n3l error
    // std::cout<<"n3lerr_p_a after:"<<"\n";
    // std::cout<<n3lerr_p_a[0]<<" "<<n3lerr_p_a[1]<<" "<<n3lerr_p_a[2]<<"\n";
    // std::cout<<"d2_n3lerr after:"<<"\n";
    // std::cout<<d2_n3lerr[i][0]<<" "<<d2_n3lerr[i][1]<<" "<<d2_n3lerr[i][2]<<"\n";
    // if (cross_border) {
    //   mmsize = MM_neigh_indices.size();
    //   // print out the MM_neigh_indices
    //   // std::cout<<"neigh_indices_: ";
    //   // print out the size of the MM_neigh_indices
    //   // std::cout<<"size: "<<mmsize<<"\n";
    //   // std::cout<<"change in forces: ";
    //   // std::cout<<n3lerr_p_a[0]/mmsize<<" "<<n3lerr_p_a[1]/mmsize<<" "<<n3lerr_p_a[2]/mmsize<<"\n";
    //   for (int mmii = 0; mmii < mmsize; mmii++) {
    //     mmi = MM_neigh_indices[mmii];
    //     for (int k = 0; k < 3; k++) {
    //       f[mmi][k] -= n3lerr_p_a[k] / mmsize;
    //     }
    //   }
    // }
  }

  if (vflag_fdotr) virial_fdotr_compute();

  // end modifications YL
}

/* ---------------------------------------------------------------------- */

void PairPACEMix::allocate()
{
  allocated = 1;
  int n = atom->ntypes + 1;

  memory->create(setflag, n, n, "pair:setflag");
  memory->create(cutsq, n, n, "pair:cutsq");
  memory->create(scale, n, n, "pair:scale");
  map = new int[n];
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairPACEMix::settings(int narg, char **arg)
{
  if (narg > 3) utils::missing_cmd_args(FLERR, "pair_style pace/mix", error);

  // ACE potentials are parameterized in metal units
  if (strcmp("metal", update->unit_style) != 0)
    error->all(FLERR, "ACE potentials require 'metal' units");

  recursive = true;    // default evaluator style: RECURSIVE

  int iarg = 0;
  while (iarg < narg) {
    if (strcmp(arg[iarg], "recursive") == 0) {
      recursive = true;
      iarg += 1;
    } else if (strcmp(arg[iarg], "product") == 0) {
      recursive = false;
      iarg += 1;
    } else if (strcmp(arg[iarg], "chunksize") == 0) {
      chunksize = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    } else
      error->all(FLERR, "Unknown pair_style pace/mix keyword: {}", arg[iarg]);
  }

  if (comm->me == 0) {
    utils::logmesg(lmp, "ACE version: {}.{}.{}\n", VERSION_YEAR, VERSION_MONTH, VERSION_DAY);
    if (recursive)
      utils::logmesg(lmp, "Recursive evaluator is used\n");
    else
      utils::logmesg(lmp, "Product evaluator is used\n");
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairPACEMix::coeff(int narg, char **arg)
{

  if (!allocated) allocate();

  map_element2type(narg - 4, arg + 4);

  auto potential_file_name = utils::get_potential_file_path(arg[2]);

  //set pot_for_eval to be arg[3] -> this is the lammps type that will be used to make the neighbour list smaller
  int pot_for_eval = std::stoi(arg[3]);
  // store pot_for_eval as an attribute such that it
  // can be used in the compute method
  //initialize pot_for_eval
  this->pot_for_eval = pot_for_eval;

  // output for debugging
  if (comm->me == 0) {
    utils::logmesg(lmp, "Potential for evaluation: {}\n", pot_for_eval);
  }

  //load potential file
  delete aceimpl->basis_set;
  if (comm->me == 0) utils::logmesg(lmp, "Loading {}\n", potential_file_name);
  aceimpl->basis_set = new ACECTildeBasisSet(potential_file_name);

  if (comm->me == 0) {
    utils::logmesg(lmp, "Total number of basis functions\n");

    for (SPECIES_TYPE mu = 0; mu < aceimpl->basis_set->nelements; mu++) {
      int n_r1 = aceimpl->basis_set->total_basis_size_rank1[mu];
      int n = aceimpl->basis_set->total_basis_size[mu];
      utils::logmesg(lmp, "\t{}: {} (r=1) {} (r>1)\n", aceimpl->basis_set->elements_name[mu], n_r1,
                     n);
    }
  }

  // read args that map atom types to PACE elements
  // map[i] = which element the Ith atom type is, -1 if not mapped
  // map[0] is not used

  delete aceimpl->ace;
  aceimpl->ace = new ACERecursiveEvaluator();
  aceimpl->ace->set_recursive(recursive);
  aceimpl->ace->element_type_mapping.init(atom->ntypes + 1);

  const int n = atom->ntypes;
  for (int i = 1; i <= n; i++) {
    char *elemname = arg[3 + i];
    if (strcmp(elemname, "NULL") == 0) {
      // species_type=-1 value will not reach ACE Evaluator::compute_atom,
      // but if it will ,then error will be thrown there
      aceimpl->ace->element_type_mapping(i) = -1;
      map[i] = -1;
      if (comm->me == 0) utils::logmesg(lmp, "Skipping LAMMPS atom type #{}(NULL)\n", i);
    } else {
      int atomic_number = AtomicNumberByName_pace(elemname);
      if (atomic_number == -1) error->all(FLERR, "'{}' is not a valid element\n", elemname);
      SPECIES_TYPE mu = aceimpl->basis_set->get_species_index_by_name(elemname);
      if (mu != -1) {
        if (comm->me == 0)
          utils::logmesg(lmp, "Mapping LAMMPS atom type #{}({}) -> ACE species type #{}\n", i,
                         elemname, mu);
        map[i] = mu;
        // set up LAMMPS atom type to ACE species  mapping for ace evaluator
        aceimpl->ace->element_type_mapping(i) = mu;
      } else {
        error->all(FLERR, "Element {} is not supported by ACE-potential from file {}", elemname,
                   potential_file_name);
      }
    }
  }

  // initialize scale factor
  for (int i = 1; i <= n; i++) {
    for (int j = i; j <= n; j++) scale[i][j] = 1.0;
  }

  aceimpl->ace->set_basis(*aceimpl->basis_set, 1);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairPACEMix::init_style()
{
  if (atom->tag_enable == 0) error->all(FLERR, "Pair style pace/mix requires atom IDs");
  if (force->newton_pair == 0) error->all(FLERR, "Pair style pace/mix requires newton pair on");

  // request a full neighbor list
  neighbor->add_request(this, NeighConst::REQ_FULL);
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairPACEMix::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "All pair coeffs are not set");
  //cutoff from the basis set's radial functions settings
  scale[j][i] = scale[i][j];
  return aceimpl->basis_set->radial_functions->cut(map[i], map[j]);
}

/* ----------------------------------------------------------------------
    extract method for extracting value of scale variable
 ---------------------------------------------------------------------- */
void *PairPACEMix::extract(const char *str, int &dim)
{
  dim = 2;
  if (strcmp(str, "scale") == 0) return (void *) scale;
  return nullptr;
}
