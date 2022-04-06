#version 330 core
in vec3 fragNormal;
in vec3 fragView;

vec3 v, l, n, r, ka, kd, ks, color;
float diff, spec;

out vec4 outColor;

void main() {
    v = normalize(fragView);
    l = normalize(vec3(0, 0, 1));
    n = normalize(fragNormal);
    r = reflect(l, n);
    ka = vec3(0, 0, 1);
    kd = vec3(1, 0.5, 0.5);
    ks = vec3(1, 0, 0);
    diff = max(dot(n, l), 0);
    spec = pow(max(dot(r, v), 0), 50); // shape of the specular lobe
    //color = diff * vec3(0, 1, 0); // lambertian model
    color = ka + kd * diff + ks * spec;
    outColor = vec4(color, 1);
}