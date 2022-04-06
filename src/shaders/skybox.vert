#version 330 core

layout (location = 0) in vec3 position;
out vec3 fragTexCoord;
uniform mat4 modelviewprojection;

void main() {
    fragTexCoord = position;
    gl_Position = modelviewprojection * vec4(position, 1.0);
}