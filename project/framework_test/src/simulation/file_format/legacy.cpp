//
// Created by samuel on 20/06/2020.
//

#include "legacy.h"

#include <fstream>
#include <limits>
#include <cerrno>
#include <cstring>
#include <sstream>
#include "util/fatal_error.h"

LegacySimDump LegacySimDump::fromFile(std::string path) {
    LegacySimDump dump;

    FILE* fp = fopen(path.c_str(), "rb");
    if (fp == nullptr) {
        FATAL_ERROR("Couldn't open file '%s': %s\n", path.c_str(), strerror(errno));
    }

    fread(&dump.imax, sizeof(int), 1, fp);
    fread(&dump.jmax, sizeof(int), 1, fp);
    fread(&dump.xlength, sizeof(float), 1, fp);
    fread(&dump.ylength, sizeof(float), 1, fp);

    const int totalElements = dump.totalElements();
    // TODO: Check if the file has enough data for totalElements
    dump.u = std::vector<float>(totalElements);
    dump.v = std::vector<float>(totalElements);
    dump.p = std::vector<float>(totalElements);
    dump.flag = std::vector<char>(totalElements);

    for (int i=0; i < dump.imax+2; i++) {
        fread(&dump.u[i * (dump.jmax+2)], sizeof(float), dump.jmax+2, fp);
        fread(&dump.v[i * (dump.jmax+2)], sizeof(float), dump.jmax+2, fp);
        fread(&dump.p[i * (dump.jmax+2)], sizeof(float), dump.jmax+2, fp);
        fread(&dump.flag[i * (dump.jmax+2)], sizeof(char), dump.jmax+2, fp);
    }
    fclose(fp);

    return dump;

    /*std::ifstream input_file(path, std::ios::binary);
    // Determine total file length, from https://stackoverflow.com/a/22986486
    // Read until the EOF
    input_file.ignore( std::numeric_limits<std::streamsize>::max() );
    std::streamsize actual_filesize = input_file.gcount();
    input_file.clear();   //  Since ignore will have set eof.
    // Go back to the beginning.
    input_file.seekg( 0, std::ios_base::beg );

    input_file >> dump.imax >> dump.jmax;
    input_file >> dump.ylength >> dump.ylength;

    const int totalElements = dump.totalElements();
    // Check if the file has enough data for totalElements
    size_t expected_filesize =
            totalElements * ((sizeof(float) * 3) + sizeof(char))  // Each element has three floats (u,v,p) and char flag.
            + 2 * (sizeof(int) + sizeof(float)); // Include the file header, two ints and two floats
    if (expected_filesize > actual_filesize) {
        FATAL_ERROR("File %s has %zu bytes, but %zu are required", path.c_str(), actual_filesize, expected_filesize);
    }

    dump.u = std::vector<float>(totalElements);
    dump.v = std::vector<float>(totalElements);
    dump.p = std::vector<float>(totalElements);
    dump.flag = std::vector<char>(totalElements);

    float* u_data = dump.u.data();
    float* v_data = dump.v.data();
    float* p_data = dump.p.data();
    char* flag_data = dump.flag.data();

    for (int i = 0; i < dump.imax + 2; i++) {
        input_file.read(u_data, sizeof(float) * (dump.jmax+2));
    }*/
}

void LegacySimDump::saveToFile(std::string path) {
    FILE *fp = fopen(path.c_str(), "wb");

    if (fp == nullptr) {
        FATAL_ERROR("Couldn't open file '%s': %s\n", path.c_str(), strerror(errno));
    }

    fwrite(&imax, sizeof(int), 1, fp);
    fwrite(&jmax, sizeof(int), 1, fp);
    fwrite(&xlength, sizeof(float), 1, fp);
    fwrite(&ylength, sizeof(float), 1, fp);

    for (int i = 0; i < imax+2; i++) {
        fwrite(&u[i * (jmax+2)], sizeof(float), jmax+2, fp);
        fwrite(&v[i * (jmax+2)], sizeof(float), jmax+2, fp);
        fwrite(&p[i * (jmax+2)], sizeof(float), jmax+2, fp);
        fwrite(&flag[i * (jmax+2)], sizeof(char), jmax+2, fp);
    }
    fclose(fp);
}

std::string LegacySimDump::debugString() {
    std::stringstream str;
    str << "imax: " << imax << " jmax: " << jmax << '\n';
    str << "xlength: " << xlength << " ylength: " << ylength << '\n';
    return str.str();
}
