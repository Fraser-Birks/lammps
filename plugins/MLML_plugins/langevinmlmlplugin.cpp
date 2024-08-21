
#include "lammpsplugin.h"

#include "version.h"

#include <cstring>

#include "fix_langevin_mlml.h"

using namespace LAMMPS_NS;

static Fix *mlmlcreator(LAMMPS *lmp, int argc, char **argv)
{
  return new FixLangevinMLML(lmp, argc, argv);
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin;
  lammpsplugin_regfunc register_plugin = (lammpsplugin_regfunc) regfunc;

  plugin.version = LAMMPS_VERSION;
  plugin.style = "fix";
  plugin.name = "langevin/mlml";
  plugin.info = "Langevin MLML fix style v0.1";
  plugin.author = "Fraser Birks (fraser.birks1@gmail.com)";
  plugin.creator.v2 = (lammpsplugin_factory2 *) &mlmlcreator;
  plugin.handle = handle;
  (*register_plugin)(&plugin, lmp);
}
