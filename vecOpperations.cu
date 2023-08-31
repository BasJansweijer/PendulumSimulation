

__device__ double3 subtract(double3 a, double3 b){
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

__device__ double3 add(double3 a, double3 b){
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

__device__ double3 scale(double3 a, float s){
    return {a.x * s, a.y * s, a.z * s};
}

__device__ double dot(double3 a, double3 b){
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

__device__ float magnitude3D(double3 v){
    return sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}

__device__ double3 normalize3D(double3 v){
    float m = magnitude3D(v);
    return {v.x/m, v.y/m, v.z/m};
}

__device__ float magnitude2D(double3 v){
    return sqrt(v.x*v.x + v.y*v.y);
}

__device__ float distance(double3 v, double3 w){
    return magnitude3D(subtract(v,w));
}

__device__ void printDouble3(double3 v, const char *name){
    printf("%s: %lf, %lf, %lf\n", name, v.x, v.y, v.z);
}

__host__ void printDouble3Host(double3 v, const char *name){
    printf("%s: %lf, %lf, %lf\n", name, v.x, v.y, v.z);
}


