# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.31

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /opt/homebrew/bin/cmake

# The command to remove a file.
RM = /opt/homebrew/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /Users/coledustin/Desktop/Coding/PhysicsEngine

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /Users/coledustin/Desktop/Coding/PhysicsEngine/Build

# Include any dependencies generated for this target.
include CMakeFiles/PhysicsEngine.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/PhysicsEngine.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/PhysicsEngine.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/PhysicsEngine.dir/flags.make

CMakeFiles/PhysicsEngine.dir/codegen:
.PHONY : CMakeFiles/PhysicsEngine.dir/codegen

CMakeFiles/PhysicsEngine.dir/main.cpp.o: CMakeFiles/PhysicsEngine.dir/flags.make
CMakeFiles/PhysicsEngine.dir/main.cpp.o: /Users/coledustin/Desktop/Coding/PhysicsEngine/main.cpp
CMakeFiles/PhysicsEngine.dir/main.cpp.o: CMakeFiles/PhysicsEngine.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/Users/coledustin/Desktop/Coding/PhysicsEngine/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/PhysicsEngine.dir/main.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/PhysicsEngine.dir/main.cpp.o -MF CMakeFiles/PhysicsEngine.dir/main.cpp.o.d -o CMakeFiles/PhysicsEngine.dir/main.cpp.o -c /Users/coledustin/Desktop/Coding/PhysicsEngine/main.cpp

CMakeFiles/PhysicsEngine.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/PhysicsEngine.dir/main.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /Users/coledustin/Desktop/Coding/PhysicsEngine/main.cpp > CMakeFiles/PhysicsEngine.dir/main.cpp.i

CMakeFiles/PhysicsEngine.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/PhysicsEngine.dir/main.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /Users/coledustin/Desktop/Coding/PhysicsEngine/main.cpp -o CMakeFiles/PhysicsEngine.dir/main.cpp.s

# Object files for target PhysicsEngine
PhysicsEngine_OBJECTS = \
"CMakeFiles/PhysicsEngine.dir/main.cpp.o"

# External object files for target PhysicsEngine
PhysicsEngine_EXTERNAL_OBJECTS =

PhysicsEngine: CMakeFiles/PhysicsEngine.dir/main.cpp.o
PhysicsEngine: CMakeFiles/PhysicsEngine.dir/build.make
PhysicsEngine: /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX15.2.sdk/System/Library/Frameworks/OpenGL.framework
PhysicsEngine: /opt/homebrew/lib/libglfw.3.4.dylib
PhysicsEngine: /opt/homebrew/lib/libGLEW.2.2.0.dylib
PhysicsEngine: CMakeFiles/PhysicsEngine.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/Users/coledustin/Desktop/Coding/PhysicsEngine/Build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable PhysicsEngine"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/PhysicsEngine.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/PhysicsEngine.dir/build: PhysicsEngine
.PHONY : CMakeFiles/PhysicsEngine.dir/build

CMakeFiles/PhysicsEngine.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/PhysicsEngine.dir/cmake_clean.cmake
.PHONY : CMakeFiles/PhysicsEngine.dir/clean

CMakeFiles/PhysicsEngine.dir/depend:
	cd /Users/coledustin/Desktop/Coding/PhysicsEngine/Build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /Users/coledustin/Desktop/Coding/PhysicsEngine /Users/coledustin/Desktop/Coding/PhysicsEngine /Users/coledustin/Desktop/Coding/PhysicsEngine/Build /Users/coledustin/Desktop/Coding/PhysicsEngine/Build /Users/coledustin/Desktop/Coding/PhysicsEngine/Build/CMakeFiles/PhysicsEngine.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/PhysicsEngine.dir/depend

