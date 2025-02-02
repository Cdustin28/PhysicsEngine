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

      protected:
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

    public:

        Particle() {
            position = Vector3d();
            velocity = Vector3d();
            acceleration = Vector3d();
            damping = 1;
            inverseMass = 1;
        }

        Particle(const Vector3d& position, const Vector3d& velocity, const Vector3d& acceleration, const real damping, const real inverseMass) {
            this->position = position;
            this->velocity = velocity;
            this->acceleration = acceleration;
            this->damping = damping;
            this->inverseMass = inverseMass;
        }

        void integrate(real duration);

        void setMass(const real mass);

        real GetMass();

        void setInverseMass(const real inverseMass);

        real GetInverseMass();

        bool hasFiniteMass() const;

        void setDamping(const real damping);

        void getDamping() const;

        void setPosition(const Vector3d& position);

        void setPosition(const real x, const real y, const real z);

        void getPosition(Vector3d& position) const;

        Vector3d getPosistion() const;

        void setVelocity(const Vector3d& velocity);

        void setVelocity(const real x, const real y, const real z);

        void getVelocity(Vector3d& velocity) const;

        Vector3d getVelocity() const;

        void setAcceleration(const Vector3d& acceleration);

        void setAcceleration(const real x, const real y, const real z);

        void getAcceleration(Vector3d& acceleration) const;

        Vector3d getAcceloration() const;

        void clearAccumulator();

        void addForce(const Vector3d& force);



    };

}



#endif //PARTICLE_H
