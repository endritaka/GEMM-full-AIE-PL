
#include <adf.h>
#include "kernels.h"
#include "project.h"

using namespace adf;

simpleGraph mygraph;

//This main() function runs only for AIESIM and X86Sim targets. 
//Emulation uses a different host code
#if defined(__AIESIM__) || defined(__X86SIM__)
int main(void) {
  mygraph.init();
  mygraph.run(10);
  mygraph.end();
  return 0;
}
#endif
