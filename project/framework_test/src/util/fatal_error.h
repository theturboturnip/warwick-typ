//
// Created by samuel on 20/06/2020.
//

#pragma once

// Fatal error function - exits the program with the given error message.
// Useful for bailing out without requiring exceptions
__attribute__((__format__ (__printf__, 3, 4))) // Tell the compiler to treat *fmt as a format string.
void fatal_error(const char* filepath, int line, const char* fmt, ...);

#define FATAL_ERROR(...) fatal_error(__FILE__, __LINE__, __VA_ARGS__)

#define FATAL_ERROR_IF(cond, ...) do { if (cond) FATAL_ERROR(__VA_ARGS__); } while(0);
#define FATAL_ERROR_UNLESS(cond, ...) do { if (!cond) FATAL_ERROR(__VA_ARGS__); } while(0);

#define STR(x) #x

#ifndef NDEBUG
// NDEBUG is not defined => We're in debug mode

// Debug assertion which prints a custom message
// and exits the program when the condition is false.
#define DASSERT_M(cond, ...) FATAL_ERROR_UNLESS(cond, __VA_ARGS__)
// Debug assertion which prints the violated condition
// and exits the program when the condition is false
#define DASSERT(cond) FATAL_ERROR_UNLESS(cond, "Assertion failed: %s\n", STR(cond))

#else
// In Release mode, disable DASSERT

// Debug assertion, disabled in release mode
#define DASSERT_M(cond, ...)
// Debug assertion, disabled in release mode
#define DASSERT(cond)
#endif