//
// Created by Cole Dustin on 1/26/25.
//
// TODO Set up with Github
// TODO Set up NotMAKEFILE

#include <stdio.h>

#include <iostream>
#include "include/engine/Vector3d.h"
#include "include/engine/particle.h"
#include "include/Window/Window.h"

#include <GL/glew.h>   // For managing modern OpenGL function pointers
#pragma comment(lib, "opengl32.lib") // Ensure OpenGL library is linked on Windows
#include <GLFW/glfw3.h> // For window and context management

using namespace std;
using namespace engine;

int testCases();
int OpenGLWindow();
void framebuffer_size_callback(GLFWwindow*, int, int);





int main()
{
    Particle myParticle = Particle();
    myParticle.setMass(10);
    myParticle.setVelocity(Vector3d(1, 1, 1));


    OpenGLWindow();
    testCases();
    return 0;
}

int testCases() {
    Vector3d myVector(10, 10, 10);

    cout << "The x, z, y is: " << myVector.x << ", " << myVector.y << ", " << myVector.z << std::endl;
    cout << "The magnitude of the vector is: " << myVector.magnitude() << endl;

    return 0;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}
int OpenGLWindow() {
    if (!glfwInit()) {
        std::cerr << "Ensure that the GLFW library is properly linked and installed\n";
        std::cerr << "Failed to initialize GLFW\n";
        return -1;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // For MacOS compatibility
#endif

    // Create window
    GLFWwindow* window = glfwCreateWindow(1600, 1200, "OpenGL Window", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return -1;
    }

    // Make the OpenGL context current
    glfwMakeContextCurrent(window);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW\n";
        return -1;
    }

    // Set the viewport and callback for resizing
    glViewport(0, 0, 1600, 1200);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Main render loop
    while (!glfwWindowShouldClose(window)) {
        // Clear the screen with a color
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}