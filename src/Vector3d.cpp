//
// Created by Cole Dustin on 2/1/25.
//

#include "../include/engine/precision.h"
#include "../include/engine/Vector3d.h"

using namespace engine;

const Vector3d Vector3d::GRAVITY = Vector3d(0, -9.81, 0);
const Vector3d Vector3d::HIGH_GRAVITY = Vector3d(0, -19.62, 0);
const Vector3d Vector3d::UP = Vector3d(0, 1, 0);
const Vector3d Vector3d::RIGHT = Vector3d(1, 0, 0);
const Vector3d Vector3d::OUT_OF_SCREEN = Vector3d(0, 0, 1);
const Vector3d Vector3d::X = Vector3d(0, 1, 0);
const Vector3d Vector3d::Y = Vector3d(1, 0, 0);
const Vector3d Vector3d::Z = Vector3d(0, 0, 1);