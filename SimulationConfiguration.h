
// IMAGE QUALITY SETTINGS (higher quality increases runtime of the program)
//  N determines the coordinate system. The image will show the coordinates -N to N in both directions
#define N 8
#define PIXELSPERUNIT 5
// total number of pixels = (N*2*PIXELSPERUNIT)^2

// Defines the timestep per itteration
#define Dt 0.01
#define FGRAVITY 9.81
// each itteration the angular velocity is multiplied with the DAMPING value.
#define DAMPING 0.999

// 2^30 is 1 gb
#define MAX_VRAM_USAGE pow(2, 30)

// MAGNET SETTINGS
#define FMAGNET 50
#define MAGNET_VISUAL_RADIUS 0.2

// "magnets" contains the coordinates (x, y, z) for al the magnets.
// Make sure that you have NUM_MAGNETS sets of coordinates.
// Also, please keep in mind that the simulation is not designed to account for forces pulling upwards on the pendulum.
// The pendulum arm is assumed to be always taut.
#define NUM_MAGNETS 3
double3 magnets[NUM_MAGNETS] = {
    {3 * cos(0), 3 * sin(0), 0},
    {3 * cos(0 + (2 * M_PI / 3)), 3 * sin(0 + (2 * M_PI / 3)), 0},
    {3 * cos(0 + (4 * M_PI / 3)), 3 * sin(0 + (4 * M_PI / 3)), 0},
};

// PENDULUM SETTINGS
#define PENDULUM_ORIGIN \
    {                   \
        0, 0, 10        \
    }
#define PENDULUM_ARM_LENGTH 10

// SIMULATION END CRITERIA
// POS_VICINITY is used to determine when to stop. It determines when the previous position is no longer close enough.
#define POS_VICINITY 0.000001
#define UNMOVED_ITTERATION_THRESHOLD 10
#define MAX_ITTERATIONS 100000