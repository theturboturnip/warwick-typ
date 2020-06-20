//
// Created by samuel on 20/06/2020.
//

#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include "fatal_error.h"

void fatal_error(const char* filepath, int line, const char* fmt, ...) {
    fprintf(stderr, "Fatal Error at %s:%d\n", filepath, line);

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    exit(1);
}