#version 450

in f_uv;

void main() {
    gl_FragColor = vec4(f_uv.x, f_uv.y, 0.0, 1.0);
}
