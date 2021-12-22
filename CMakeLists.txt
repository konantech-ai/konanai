cmake_minimum_required (VERSION 3.9)
project ("kai")

# options
option(BUILD_DOCUMENTATION "Create and install the HTML based API documentation")
option(NO_RANDOM "Disable random(shuffle, layer initialize)")

# ctest enabled
enable_testing()

# add subdirectory
add_subdirectory(kai)
if(BUILD_DOCUMENTATION) 
	#add_subdirectory(docs)
endif()