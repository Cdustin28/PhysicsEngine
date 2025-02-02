//
// Created by Cole Dustin on 2/1/25.
//

#include "../include/engine/Vector3d.h"
#include "../include/engine/particle.h"
#include "../include/engine/precision.h"
#include <assert.h>

using namespace engine;

void Particle::integrate(real duration){

  assert(duration > 0.0);

  position.addScaledVector(velocity, duration);

  Vector3d resultingAcc = acceleration;
  resultingAcc.addScaledVector(acceleration, duration);

  velocity.addScaledVector(resultingAcc, duration);

  velocity *= real_pow(damping, duration);

 }

void setMass(const real mass){

}

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

