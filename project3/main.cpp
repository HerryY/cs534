
#include "Util/CLP.h"
#include <stdint.h>
#include <iostream>
#include <fstream>

#include "MLTree/MLTreeMain.h"


int main(int argc, char** argv)
{

	CLP cmdLine;
	cmdLine.parse(argc, argv);

	cmdLine.setDefault("ii", "-1");
	int ii = cmdLine.getInt("ii");

	MLTreeMain(ii);
	return 0;
}

