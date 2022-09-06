#ifndef UTILITIES_H
#define UTILITIES_H

typedef enum {
	UNSPECEFIED,
} ERROR_E;

void krc_ml_error (char* message, ERROR_E code);

#endif
