//
// Created by Cole Dustin on 2/1/25.
//

#include "../include/engine/Vector3d.h"
#include "../include/engine/particle.h"
#include "../include/engine/precision.h"
#include <assert.h>

using namespace engine;

void Particle::integrate(const real duration)
{

  assert(duration > 0.0);

  position.addScaledVector(velocity, duration);

  Vector3d resultingAcc = acceleration;
  resultingAcc.addScaledVector(acceleration, duration);

  velocity.addScaledVector(resultingAcc, duration);

  velocity *= real_pow(damping, duration);

 }
/**
 * Sets the mass of the particle. The mass is converted to its reciprocal to calculate and store the inverse mass.
 *
 * @param mass The mass of the particle. Must be non-zero.
 */
void Particle::setMass(const real mass)
 {

    assert(mass != 0);
    this->inverseMass = 1/mass;
}
/**
 * Retrieves the mass of the particle. The mass is calculated as the reciprocal of its inverse mass.
 *
 * @return The mass of the particle as a real number.
 */
real Particle::GetMass() const
{
    return 1/inverseMass;
}
/**
 * Sets the inverse mass of the particle.
 *
 * The inverse mass is stored directly. A value of zero indicates
 * that the particle has infinite mass, meaning it is immovable.
 *
 * @param inverseMass The inverse mass to be assigned to the particle.
 */
void Particle::setInverseMass(const real inverseMass)
{
    this->inverseMass = inverseMass;
}

/**
 * Retrieves the precomputed inverse of the particle's mass.
 *
 * @return The inverse mass of the particle. Returns zero if the particle is considered to have infinite mass.
 */
real Particle::GetInverseMass() const
{
    return inverseMass;
}
/**
 * Determines whether the particle has a finite mass by checking if its inverse mass is non-zero.
 *
 * @return True if the particle has a finite mass, false otherwise.
 */
bool Particle::hasFiniteMass() const
{
   return inverseMass != 0;
}

/**
 * Sets the damping factor for the particle. The damping factor is used to reduce the velocity over time during
 * simulation.
 *
 * @param damping The damping factor of the particle. Should be a positive real value.
 */
void Particle::setDamping(const real damping)
{
    this->damping = damping;
}

/**
 * Retrieves the damping factor of the particle, which is used to reduce its velocity over time.
 *
 * @return The damping factor of the particle.
 */
real Particle::getDamping() const
{
    return this->damping;
}

/**
 * Sets the position of the particle.
 *
 * @param position The new position of the particle represented as a 3D vector.
 */
void Particle::setPosition(const Vector3d& position)
{
    this->position = position;
}

/**
 * Sets the position of the particle in a 3D space.
 *
 * @param x The x-coordinate of the position.
 * @param y The y-coordinate of the position.
 * @param z The z-coordinate of the position.
 */
void Particle::setPosition(const real x, const real y, const real z)
{
    this->position.x = x;
    this->position.y = y;
    this->position.z = z;
}
/**
 * Retrieves the current position of the particle.
 *
 * @param position Pointer to a Vector3d object where the particle's position will be stored.
 */
void Particle::getPosition(Vector3d *position) const
{
    *position  = this->position;
}

/**
 * Retrieves the current position of the particle.
 *
 * @return The position of the particle as a Vector3d object.
 */
Vector3d Particle::getPosistion() const
{
    return this->position;
}

/**
 * Sets the velocity of the particle.
 *
 * @param velocity The new velocity of the particle as a 3D vector.
 */
void Particle::setVelocity(const Vector3d& velocity)
{
    this->velocity = velocity;
}

/**
 * Sets the velocity of the particle by defining its components along the x, y, and z axes.
 *
 * @param x The velocity component along the x-axis.
 * @param y The velocity component along the y-axis.
 * @param z The velocity component along the z-axis.
 */
void Particle::setVelocity(const real x, const real y, const real z)
{
    this->velocity.x = x;
    this->velocity.y = y;
    this->velocity.z = z;
}

/**
 * Retrieves the velocity of the particle.
 *
 * @param velocity Pointer to a Vector3d object where the velocity of the particle will be stored.
 */
void Particle::getVelocity(Vector3d *velocity) const
{
    *velocity = this->velocity;
}

/**
 * Retrieves the current velocity of the particle.
 *
 * @return A Vector3d representing the velocity of the particle.
 */
Vector3d Particle::getVelocity() const
{
    return this->velocity;
}

/**
 * Sets the acceleration of the particle.
 *
 * @param acceleration The acceleration vector to be assigned to the particle.
 */
void Particle::setAcceleration(const Vector3d& acceleration)
{
    this->acceleration = acceleration;
}

/**
 * Sets the acceleration of the particle using the specified components.
 *
 * @param x The acceleration along the x-axis.
 * @param y The acceleration along the y-axis.
 * @param z The acceleration along the z-axis.
 */
void Particle::setAcceleration(const real x, const real y, const real z)
{
    this->acceleration.x = x;
    this->acceleration.y = y;
    this->acceleration.z = z;
}

/**
 * Retrieves the current acceleration of the particle.
 *
 * @param acceleration Pointer to a Vector3d object where the acceleration will be stored.
 */
void Particle::getAcceleration(Vector3d *acceleration) const
{
    *acceleration = this->acceleration;
}

/**
 * Retrieves the current acceleration of the particle.
 *
 * @return The acceleration of the particle as a Vector3d.
 */
Vector3d Particle::getAcceloration() const
{
    return this->acceleration;
}

/**
 *
 */
void Particle::clearAccumulator()
{
    forceAcum.clear();
}

/**
 * Adds a force to the particle's accumulated force.
 *
 * @param force The vector representing the force to be added.
 */
void Particle::addForce(const Vector3d &force)
{
    forceAcum += force;
}

