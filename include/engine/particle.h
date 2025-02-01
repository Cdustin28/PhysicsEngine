//
// Created by Cole Dustin on 1/26/25.
//

#ifndef PARTICLE_H
#define PARTICLE_H

#include "Vector3d.h"

namespace engine{
    /**
    *  The simplest object that can be simulated in a physics system
    */
    class Particle{
      public:
        /**
        *  Hold the linear position of the particle in 3d space
        */
        Vector3d position;

        /**
        *  Hold the linear velocity og the particle in 3d space
        */
        Vector3d velocity;


        /**
        * Holds the 3d acceloration of the particle in 3d space
        */
        Vector3d acceleration;


        /**
        *  This is the amount of energy lost through numerical instability in the integrator.
        *  Holds the amount of daming applied toi the liner motion.
        */
        real damping;

        real inverseMass;



    }

}



#endif //PARTICLE_H
