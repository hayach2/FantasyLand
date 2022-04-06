#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
out vec3 fragNormal;
out vec3 fragView;

// COLOR.VERT FROM TP1

void main() {
    gl_Position = projection * view * model * vec4(position, 1);
    //fragNormal = normal; // fixed light
    // light 'rotates' with camera movement
    fragNormal = transpose(inverse(mat3(view * model))) * normal;
    fragView = normalize((view * model * vec4(position, 1)).xyz);
}