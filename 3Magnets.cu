#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vecOpperations.cu"
#include <cuda_runtime.h>

#define MAX_ITTERATIONS 1000
// N determines the coordinate system. The image will show the coordinates -N to N in both directions
#define N 10
#define Dt 0.05
#define FGRAVITY 9.81
#define FMAGNET 100
#define PENDULUM_ORIGIN {0,0, 10}
#define PENDULUM_ARM 10
#define DAMPING 0.99
#define PIXELSPERUNIT 50
#define PIXELS_PER_ROW (N * 2 * PIXELSPERUNIT)
#define TOTAL_PIXELS (int)pow(PIXELS_PER_ROW, 2)
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))


typedef struct pendulum
{
    double3 origin;
    double phi;
    double theta;
    float armLength;
    // angle contains phi, theta in radians
    double phiVelocity;
    double thetaVelocity;
    double3 ballPos;
    float ballMass;
} pendulum;


__device__ void calcAngleFromBallPos(pendulum *p){
    double3 vec = subtract(p->ballPos, p->origin);
    p->phi = atan2f(vec.y, vec.x); // Result in radians
    double rXY = sqrtf(vec.x * vec.x + vec.y * vec.y); // Magnitude of the vector projected onto the x-y plane
    p->theta = atan2f(rXY, vec.z); // Result in radians
}

// Theta represents the angle measured from the positive z-axis to the vector projected onto the xy-plane
// Phi represents the angle measured in the xy-plane from the positive x-axis to the projection of the vector onto the xy-plane. 
__device__ void calcBallPosFromAngle(pendulum * p){
    double x = p->armLength * sinf(p->theta) * cosf(p->phi);
    double y = p->armLength * sinf(p->theta) * sinf(p->phi);
    double z = p->armLength * cosf(p->theta);
    p->ballPos = add({x,y,z}, p->origin);
}

// Given a pixel index this function calculates the coresponding position on the XY plane.
__device__ double2 getPosFromPixel(int pixelIndex){
    return {(double)(pixelIndex % PIXELS_PER_ROW) / PIXELSPERUNIT - N, (double)((int)(pixelIndex / PIXELS_PER_ROW)) / PIXELSPERUNIT - N};
}

// Given a Position on the XY plane this function returns the pixel we are in.
__device__ int getPixelFromPos(double2 pos){
    float pixelInterval = 1.0f/PIXELSPERUNIT;
    int colNum = (pos.x+N) / pixelInterval;
    int rowNum = (pos.y+N) / pixelInterval;
    return colNum + rowNum * PIXELS_PER_ROW;
}

/*
Creates a pendulum object from the x and z coordinate of the ball.
*/
__device__ void createPendulumFrom2DBallPos(pendulum *penPointer, double2 Ball2D, double3 origin, float armLength)
{
    // calculate at what height the ball should connect
    Ball2D.x -= origin.x;
    Ball2D.y -= origin.y;

    // We know that: sqrt(BallX^2 + BallZ^2 + BallY^2) = Armlength
    // we need the - solution for ballY as we want to hang as low as posible.
    double BallZ = -sqrt(pow(armLength, 2) - pow(Ball2D.x, 2) - pow(Ball2D.y, 2));

    if(isnan(BallZ)){
        penPointer = NULL;
        return;
    }


    penPointer->ballPos = add({Ball2D.x, Ball2D.y, BallZ} , origin);
    penPointer->origin = origin;
    penPointer->armLength = armLength;
    penPointer->phiVelocity = 0.5;
    penPointer->thetaVelocity = 0;
    penPointer->ballMass = 1;
    // Now we need to calculate the angles of our pendulum

    calcAngleFromBallPos(penPointer);
}


/*
This function updates the pendulum given the external force on the ball of the pendulum.
The external force should contain all forces (except the tension force of the pendulum cord).
*/
__device__ void updatePendulum(pendulum* pen, double3 Fexternal)
{
    // We need to calculate the tension force
    // we use the fact that v dot w = length of v projected on w * length of w
    // thus when normaling the direction of Ftens and taking the dot we get the length of Ftension
    double3 tensionUnitVec = normalize3D(subtract(pen->ballPos, pen->origin));
    // -1 from the dot product means they point in opposite direction which is what we want.
    // Thus the actual magnitude is - the dot product
    double TensMagnitude = -dot(tensionUnitVec, Fexternal);
    double3 Ftension = scale(tensionUnitVec, TensMagnitude);
    double3 Fres = add(Ftension, Fexternal);
    // We now need to calculat phi hat and theta hat to decompose Fres. (https://en.wikipedia.org/wiki/Spherical_coordinate_system)
    double3 thetaHat = {cosf(pen->theta)*cosf(pen->phi), cosf(pen->theta)*sinf(pen->phi), -sinf(pen->theta)};
    double3 phiHat = {-sinf(pen->phi), cosf(pen->phi), 0};
    double Fphi = dot(Fres, phiHat);
    double Ftheta = dot(Fres, thetaHat);

    // F/armLen = m*a
    double phiVelChange = Fphi/(pen->armLength*pen->ballMass);
    double thetaVelChange = Ftheta/(pen->armLength*pen->ballMass);
    pen->phiVelocity *= DAMPING;
    pen->thetaVelocity *= DAMPING;
    pen->phiVelocity += phiVelChange * Dt;
    pen->thetaVelocity += thetaVelChange * Dt;
    pen->phi += pen->phiVelocity * Dt;
    pen->theta += pen->thetaVelocity * Dt;
    // Ensure angles are wrapped within the appropriate ranges
    pen->phi = fmodf(pen->phi,  2.0 * M_PI);
    if (pen->phi < 0)
        pen->phi +=  2.0 * M_PI;
    // theta could be 0 through PI to hit all posible positions but that would require us to add PI to phi
    // once theta gets above PI and is reset to 0.
    pen->theta = fmodf(pen->theta, 2.0 * M_PI);
    if (pen->theta < 0)
        pen->theta += 2.0 * M_PI;

    calcBallPosFromAngle(pen);
}

__global__ void pendulumPathKernel(int3 *outputPixels, pendulum *pendulums, double2 start)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= TOTAL_PIXELS)
    {
        return;
    }else if(tid != 0){
        outputPixels[tid] = {0,0,0};
        return;
    }
    double2 ballPos = start;

    pendulum * pen = &pendulums[tid];
    createPendulumFrom2DBallPos(pen, ballPos, PENDULUM_ORIGIN, PENDULUM_ARM);

    if(pen == NULL){
        return;
    }
    int iter = 0;
    while(iter < MAX_ITTERATIONS){
        int pixelIndex = getPixelFromPos({pen->ballPos.x, pen->ballPos.y});
        outputPixels[pixelIndex] = {255,255,255};
        iter += 1;
        updatePendulum(pen, {0, 0, -1});
    }

    free(pen);
    return;
}


__device__ float pythagoras(double2 pointA, double2 pointB)
{
    return sqrt(pow(pointA.x - pointB.x, 2) + pow(pointA.y - pointB.y, 2));
}
/*
This function takes in the ball position and magnet position.
It then calculates this magnets force on the ball and adds it to the Fres
*/
__device__ double3 calcMagnetsForce(double3 ballPos, double3 magnetPos)
{
    double3 betweenVec = subtract(magnetPos, ballPos);
    double dist = magnitude3D(betweenVec);
    double3 unitVec = normalize3D(betweenVec);
    return scale(unitVec, FMAGNET/(dist*dist));
}

__device__ float minMagnetDistance(double3 ballPos, double3 *magnets, int numMagnets)
{
    float best_dist = 9999999999999;
    for (int i = 0; i < numMagnets; i++)
    {
        float dist = magnitude3D(subtract(magnets[i], ballPos));
        if (dist < best_dist)
        {
            best_dist = dist;
        }
    }
    return best_dist;
}

__global__ void magnetKernel(int3 *outputPixels, pendulum *pendulums, double3 *magnets, int numMagnets)
{
    int pixel_i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (pixel_i >= TOTAL_PIXELS)
    {
        return;
    }
    double2 ballPos = getPosFromPixel(pixel_i);

    pendulum * pen = &pendulums[pixel_i];
    createPendulumFrom2DBallPos(pen, ballPos, PENDULUM_ORIGIN, PENDULUM_ARM);
    if(pen == NULL){
        outputPixels[pixel_i] = {0,0,0};
        return;
    }

    if (minMagnetDistance({ballPos.x, ballPos.y, 0}, magnets, numMagnets) < 0.25)
    {
        outputPixels[pixel_i] = {255, 255, 255};
        return;
    }

    int itteration = 0;
    while (itteration < MAX_ITTERATIONS)
    {
        double3 Fres = {0, 0, -FGRAVITY * pen->ballMass};
        for (int i = 0; i < numMagnets; i++)
        {
            double3 force = calcMagnetsForce(pen->ballPos, magnets[i]);
            Fres = add(force, Fres);
        }
        updatePendulum(pen, Fres);
        itteration += 1;
    }


    int closest = 0;
    float best_dist = 9999999999999;
    for (int i = 0; i < numMagnets; i++)
    {
        double3 betweenVec = subtract(pen->ballPos, magnets[i]);
        float dist = magnitude2D({betweenVec.x, betweenVec.y});
        if (dist < best_dist)
        {
            best_dist = dist;
            closest = i + 1;
        }
    }
    int intensity = 255 - ((int)best_dist * 10);
    switch (closest)
    {
    case 1:
        outputPixels[pixel_i] = {intensity, 0, 0};
        return;
    case 2:
        outputPixels[pixel_i] = {0, intensity, 0};
        return;
    case 3:
        outputPixels[pixel_i] = {0, 0, intensity};
        return;
    default:
        outputPixels[pixel_i] = {0, 0, 0};
        return;
    }
}

int main()
{
    int totalThreads = pow(N * 2 * PIXELSPERUNIT, 2);
    int threadsPerBlock = 1024;
    int numBlocks = (int)ceil((float)totalThreads / threadsPerBlock);

    // Initialize d_magnets on the host
    int numMagnets = 4;
    double3 h_magnets[numMagnets] = {
        {5, 5, 0},
        {-5, 5, 0},
        {-5, -5, 0},
        {5, -5, 0},
    };
    double3 *d_magnets;
    // 1gb
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, pow(2, 30));

    int3 *d_result;
    cudaMalloc((void **)&d_result, TOTAL_PIXELS * sizeof(int3));
    pendulum *d_pendulums;
    cudaMalloc((void **)&d_pendulums, TOTAL_PIXELS * sizeof(pendulum));
    if(d_result == NULL || d_pendulums == NULL){
        printf("NULLLPOINTER1\n");
        exit(1);
    }
    cudaMalloc((void **)&d_magnets, numMagnets * sizeof(double3));
    cudaMemcpy(d_magnets, h_magnets, numMagnets * sizeof(double3), cudaMemcpyHostToDevice);
    printf("calling kernel <<<%d, %d>>>\n", numBlocks, threadsPerBlock);
    //pendulumPathKernel<<<numBlocks, threadsPerBlock>>>(d_result, d_pendulums, {5, 2});
    magnetKernel<<<numBlocks, threadsPerBlock>>>(d_result, d_pendulums, d_magnets, numMagnets);

    cudaDeviceSynchronize();

    int3 *out = (int3 *)malloc(TOTAL_PIXELS * sizeof(int3));
    if(out == NULL){
        printf("NULLLPOINTER2\n");
        exit(1);
    }

    cudaMemcpy(out, d_result, TOTAL_PIXELS * sizeof(int3), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    cudaFree(d_magnets);

    // converting our result to the ppm format
    printf("total Pixels: %d, (%dx%d)\n", TOTAL_PIXELS, PIXELS_PER_ROW, PIXELS_PER_ROW);
    FILE *fp = fopen("out.ppm", "w");
    fprintf(fp, "P3 %d %d 255", PIXELS_PER_ROW, PIXELS_PER_ROW);
    for (int i = 0; i < TOTAL_PIXELS; i++)
    {
        if (i % PIXELS_PER_ROW == 0)
        {
            fprintf(fp, "\n");
        }
        fprintf(fp, "%d %d %d   ", out[i].x, out[i].y, out[i].z);
    }

    fclose(fp);
    free(out);
    return 0;
}