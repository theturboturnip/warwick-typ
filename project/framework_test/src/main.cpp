#include "cmd_line/CommandLineParser.h"

int main(int argc, const char* argv[]) {
    return CommandLineParser().parseArguments(argc, argv);
}