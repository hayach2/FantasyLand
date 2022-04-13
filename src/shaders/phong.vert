#version 330 core

const int NUM_LIGHT_SRC = 4;

// Fog visibility variables
const float density = 0.007;
const float gradient = 1.5;

layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uvs;
layout(location = 2) in vec3 normal;

uniform mat4 model, view, projection;
uniform vec3 light_position[NUM_LIGHT_SRC];

// position and normal for the fragment shader, in WORLD coordinates
out vec3 w_position, w_normal;
out vec2 frag_uv;
out float visibility;
out vec3 to_light_vector[NUM_LIGHT_SRC];
out vec3 surface_normal;


void main() {

    vec4 worldPosition = model * vec4(position, 1.0);
    vec4 positionRelativeToCam = view * worldPosition;

    gl_Position = projection * positionRelativeToCam;
    frag_uv = vec2(uvs.x, uvs.y);

    // shadow calculation
    w_position =  worldPosition.xyz / worldPosition.w;

    w_normal = transpose(inverse(mat3(model))) * normal;
    for(int i = 0;i < NUM_LIGHT_SRC; i++)
    {
        to_light_vector[i] = light_position[i] - worldPosition.xyz;
    }

    float distance = length(positionRelativeToCam.xyz);
    visibility = exp(-pow((distance * density), gradient));
    visibility = clamp(visibility, 0.0, 1.0);
}
