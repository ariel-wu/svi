# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The program to use to edit the cache.
CMAKE_EDIT_COMMAND = /usr/bin/ccmake

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/arielwu/svi/noninf

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/arielwu/svi/noninf/build

# Include any dependencies generated for this target.
include CMakeFiles/noninf.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/noninf.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/noninf.dir/flags.make

CMakeFiles/noninf.dir/src/noninf.cpp.o: CMakeFiles/noninf.dir/flags.make
CMakeFiles/noninf.dir/src/noninf.cpp.o: ../src/noninf.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/arielwu/svi/noninf/build/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/noninf.dir/src/noninf.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/noninf.dir/src/noninf.cpp.o -c /home/arielwu/svi/noninf/src/noninf.cpp

CMakeFiles/noninf.dir/src/noninf.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/noninf.dir/src/noninf.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/arielwu/svi/noninf/src/noninf.cpp > CMakeFiles/noninf.dir/src/noninf.cpp.i

CMakeFiles/noninf.dir/src/noninf.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/noninf.dir/src/noninf.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/arielwu/svi/noninf/src/noninf.cpp -o CMakeFiles/noninf.dir/src/noninf.cpp.s

CMakeFiles/noninf.dir/src/noninf.cpp.o.requires:
.PHONY : CMakeFiles/noninf.dir/src/noninf.cpp.o.requires

CMakeFiles/noninf.dir/src/noninf.cpp.o.provides: CMakeFiles/noninf.dir/src/noninf.cpp.o.requires
	$(MAKE) -f CMakeFiles/noninf.dir/build.make CMakeFiles/noninf.dir/src/noninf.cpp.o.provides.build
.PHONY : CMakeFiles/noninf.dir/src/noninf.cpp.o.provides

CMakeFiles/noninf.dir/src/noninf.cpp.o.provides.build: CMakeFiles/noninf.dir/src/noninf.cpp.o

CMakeFiles/noninf.dir/src/genotype.cpp.o: CMakeFiles/noninf.dir/flags.make
CMakeFiles/noninf.dir/src/genotype.cpp.o: ../src/genotype.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/arielwu/svi/noninf/build/CMakeFiles $(CMAKE_PROGRESS_2)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/noninf.dir/src/genotype.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/noninf.dir/src/genotype.cpp.o -c /home/arielwu/svi/noninf/src/genotype.cpp

CMakeFiles/noninf.dir/src/genotype.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/noninf.dir/src/genotype.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/arielwu/svi/noninf/src/genotype.cpp > CMakeFiles/noninf.dir/src/genotype.cpp.i

CMakeFiles/noninf.dir/src/genotype.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/noninf.dir/src/genotype.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/arielwu/svi/noninf/src/genotype.cpp -o CMakeFiles/noninf.dir/src/genotype.cpp.s

CMakeFiles/noninf.dir/src/genotype.cpp.o.requires:
.PHONY : CMakeFiles/noninf.dir/src/genotype.cpp.o.requires

CMakeFiles/noninf.dir/src/genotype.cpp.o.provides: CMakeFiles/noninf.dir/src/genotype.cpp.o.requires
	$(MAKE) -f CMakeFiles/noninf.dir/build.make CMakeFiles/noninf.dir/src/genotype.cpp.o.provides.build
.PHONY : CMakeFiles/noninf.dir/src/genotype.cpp.o.provides

CMakeFiles/noninf.dir/src/genotype.cpp.o.provides.build: CMakeFiles/noninf.dir/src/genotype.cpp.o

CMakeFiles/noninf.dir/src/storage.cpp.o: CMakeFiles/noninf.dir/flags.make
CMakeFiles/noninf.dir/src/storage.cpp.o: ../src/storage.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/arielwu/svi/noninf/build/CMakeFiles $(CMAKE_PROGRESS_3)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/noninf.dir/src/storage.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/noninf.dir/src/storage.cpp.o -c /home/arielwu/svi/noninf/src/storage.cpp

CMakeFiles/noninf.dir/src/storage.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/noninf.dir/src/storage.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/arielwu/svi/noninf/src/storage.cpp > CMakeFiles/noninf.dir/src/storage.cpp.i

CMakeFiles/noninf.dir/src/storage.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/noninf.dir/src/storage.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/arielwu/svi/noninf/src/storage.cpp -o CMakeFiles/noninf.dir/src/storage.cpp.s

CMakeFiles/noninf.dir/src/storage.cpp.o.requires:
.PHONY : CMakeFiles/noninf.dir/src/storage.cpp.o.requires

CMakeFiles/noninf.dir/src/storage.cpp.o.provides: CMakeFiles/noninf.dir/src/storage.cpp.o.requires
	$(MAKE) -f CMakeFiles/noninf.dir/build.make CMakeFiles/noninf.dir/src/storage.cpp.o.provides.build
.PHONY : CMakeFiles/noninf.dir/src/storage.cpp.o.provides

CMakeFiles/noninf.dir/src/storage.cpp.o.provides.build: CMakeFiles/noninf.dir/src/storage.cpp.o

# Object files for target noninf
noninf_OBJECTS = \
"CMakeFiles/noninf.dir/src/noninf.cpp.o" \
"CMakeFiles/noninf.dir/src/genotype.cpp.o" \
"CMakeFiles/noninf.dir/src/storage.cpp.o"

# External object files for target noninf
noninf_EXTERNAL_OBJECTS =

noninf: CMakeFiles/noninf.dir/src/noninf.cpp.o
noninf: CMakeFiles/noninf.dir/src/genotype.cpp.o
noninf: CMakeFiles/noninf.dir/src/storage.cpp.o
noninf: CMakeFiles/noninf.dir/build.make
noninf: CMakeFiles/noninf.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable noninf"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/noninf.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/noninf.dir/build: noninf
.PHONY : CMakeFiles/noninf.dir/build

CMakeFiles/noninf.dir/requires: CMakeFiles/noninf.dir/src/noninf.cpp.o.requires
CMakeFiles/noninf.dir/requires: CMakeFiles/noninf.dir/src/genotype.cpp.o.requires
CMakeFiles/noninf.dir/requires: CMakeFiles/noninf.dir/src/storage.cpp.o.requires
.PHONY : CMakeFiles/noninf.dir/requires

CMakeFiles/noninf.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/noninf.dir/cmake_clean.cmake
.PHONY : CMakeFiles/noninf.dir/clean

CMakeFiles/noninf.dir/depend:
	cd /home/arielwu/svi/noninf/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/arielwu/svi/noninf /home/arielwu/svi/noninf /home/arielwu/svi/noninf/build /home/arielwu/svi/noninf/build /home/arielwu/svi/noninf/build/CMakeFiles/noninf.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/noninf.dir/depend

