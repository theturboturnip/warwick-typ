//
// Created by samuel on 20/06/2020.
//

#pragma once

// Fatal error function - exits the program with the given error message.
// Useful for bailing out without requiring exceptions
void fatal_error(const char* filepath, int line, const char* fmt, ...);

#define FATAL_ERROR(format, ...) fatal_error(__FILE__, __LINE__, format, __VA_ARGS__)
