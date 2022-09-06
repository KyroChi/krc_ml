#include <stdio.h>
#include "utilities.h"

void
krc_ml_error (char* message, ERROR_E code)
{
	fflush(stdout);
        fprintf(stderr, "%s\n", message);
        fflush(stderr);
        exit(exit_code);
}
