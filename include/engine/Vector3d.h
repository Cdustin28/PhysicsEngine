//
// Created by Cole Dustin on 1/26/25.
//

#ifndef VECTOR3D_H
#define VECTOR3D_H

#include "precision.h"

namespace engine {
    /**
     * A class for representing three-dimensional vectors and performing various
     * vector operations such as addition, subtraction, scaling, normalization,
     * and computing dot or cross products.
     */
    class Vector3d {

    public:

        /** Holds the value along the x-axis. */
        real x;
        /** Holds the value along the y-axis. */
        real y;
        /** Holds the value along the z-axis. */
        real z;

    private:

        /**
        * padding to ensure the 4-word alignment
        */
        real pad;

    public:

        /**
        *
        */
        Vector3d() : x(), y(), z() {}

        /**
         *
         * @param x
         * @param y
         * @param z
         *
         * The explicit constructor that takes values from the function parameters
         */
        Vector3d(const real x, const real y, const real z) : x(x), y(y), z(z) {}


        const static Vector3d GRAVITY;
        const static Vector3d HIGH_GRAVITY;
        const static Vector3d UP;
        const static Vector3d RIGHT;
        const static Vector3d OUT_OF_SCREEN;
        const static Vector3d X;
        const static Vector3d Y;
        const static Vector3d Z;
        /**
         *  Flips all the components around the 3d point of origin
         */
        void invert() {
            x = -x;
            y = -y;
            z = -z;
        }

        /**
         * Returns the magnitude of the vector
         */
        real magnitude() const
        {
            /**
            * This should return a real_sqrt, but the damn flag keep showing up. I don't
            * understand why when it is defined to the exact same thing that is passes as.
            */
            return real_sqrt(x*x + y*y + z*z);

        }


        real squareMagnitude() const{
            return x*x + y*y + z*z;
        }

        /**
         * This will make the magnitude of the vector one unit length while keeping the same direction
         */
        void normalize() {

            real magnitude = 0;
            if (magnitude != 0) {
                x /= magnitude;
                y /= magnitude;
                z /= magnitude;
            }
        }

        /**
         *
         * @param scalar
         *
         *  Overrides the *= to be a scalar multiplication on a given vector
         */
        void operator*=(const real scalar) {

            x *= scalar;
            y *= scalar;
            z *= scalar;
        }

        /**
         *
         * @param scalar
         * @return New vector with the scaled value
         *
         * Returns a new vector with the scaled value
         */
        Vector3d operator*(const real scalar) const {

            return Vector3d(x*scalar, y*scalar, z*scalar);
        }

        /**
         *
         * @param scalar
         *
         * Overrides the /= to be a scalr divsion on a given vector
         */
        void operator/=(const real scalar) {

            x /= scalar;
            y /= scalar;
            z /= scalar;
        }

        /**
         *
         * @param scalar
         * @return New vector with the scaled value
         *
         * Returns a new vector with the scaled values
         */
        Vector3d operator/(const real scalar) const {

            return Vector3d(x/scalar, y/scalar, z/scalar);
        }

        Vector3d operator+(const Vector3d& vector) {

            return Vector3d(x + vector.x, y + vector.y, z + vector.z);
        }

        void operator+=(const Vector3d& vector)
        {

            x += vector.x;
            y += vector.y;
            z += vector.z;
        }

        Vector3d operator-(const Vector3d& vector) const{

            return Vector3d(x - vector.x, y - vector.y, z - vector.z);
        }

        void operator-=(const Vector3d& vector) {

            x -= vector.x;
            y -= vector.y;
            z -= vector.z;
        }

        void addScaledVector(const Vector3d& vector, real scale) {

            x += vector.x * scale;
            y += vector.y * scale;
            z += vector.z * scale;

        }

        Vector3d componentProduct(const Vector3d& vector) const {

            return Vector3d(x*vector.x, y*vector.y, z*vector.z);
        }

        void conpenenetProductUpdate(const Vector3d& vector) {

            x *= vector.x;
            y *= vector.y;
            z *= vector.z;
        }

        /**
         *
         * @param vector
         * @return dot product of the two vectors
         */
        real dotProduct(const Vector3d& vector) const {

            return x*vector.x + y*vector.y + z*vector.z;
        }

        /**
         *
         * @param vector
         * @return Dot Prodcut of the two vectors
         */
        real operator *(const Vector3d& vector) {

            return x*vector.x + y*vector.y + z*vector.z;
        }

        Vector3d crossProduct(const Vector3d& vector) const {

            return Vector3d(y*vector.z-z*vector.y,
                            z*vector.x-x*vector.z,
                            x*vector.y-y*vector.x);
        }

        /**
        * Updates this vector to be the vector product of its current
        * value and the given vector.
        */
        void operator %=(const Vector3d &vector)
        {
            *this = crossProduct(vector);
        }

        /**
        * Calculates and returns the vector product of this vector
        * with the given vector.
        */
        Vector3d operator%(const Vector3d &vector) const
        {
            return Vector3d(y*vector.z-z*vector.y,
                            z*vector.x-x*vector.z,
                            x*vector.y-y*vector.x);
        }

        /**
         * Resets the values of x, y, and z to zero, clearing any existing state.
         */
        void clear() {
            x = y = z = 0;
        }

        void trim(real size) {
            if (squareMagnitude() > size*size) {
                normalize();
                x *= size;
                y *= size;
                z *= size;
            }
        }

    };


}

#endif //VECTOR3D_H
