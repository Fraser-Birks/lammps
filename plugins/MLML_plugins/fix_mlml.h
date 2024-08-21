#ifndef LMP_FIX_MLML_H
#define LMP_FIX_MLML_H

#include "fix.h"

namespace LAMMPS_NS {

class FixMLML : public Fix {
 public:
  FixMLML(class LAMMPS *, int, char **);

  int setmask() override;
  void init() override;
  void init_list(int, class NeighList *) override;
  void end_of_step() override;
  void setup_pre_force(int) override;
  int pack_reverse_comm(int, int, double *) override;
  void unpack_reverse_comm(int, int *, double *) override;
  int pack_forward_comm(int, int *, double *, int, int *) override;
  void unpack_forward_comm(int, int, double *) override;

  bool check_cutoff(double *, double *, double);
  double linear_blend(double *, double *);
  void allocate_regions();

 protected:
  class NeighList *list;
  // class NeighList *rqm_list;
  // class NeighList *rbl_list;
  // class NeighList *bw_list;

  // class NeighRequest *requestQM;
  // class NeighRequest *requestBL;
  // class NeighRequest *requestB;


  double dtv, dtf;
  double *step_respa;
  int mass_require, is_type, type_val;
  double rqm, bw, rblend;
};

}    // namespace LAMMPS_NS

#endif

    /* ERROR/WARNING messages:

E: Illegal ... command

Self-explanatory.  Check the input script syntax and compare to the
documentation for the command.  You can use -echo screen as a
command-line option when running LAMMPS to see the offending line.

*/
