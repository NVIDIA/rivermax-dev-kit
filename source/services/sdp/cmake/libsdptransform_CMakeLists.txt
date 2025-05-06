cmake_minimum_required(VERSION 3.19)

project(sdptransform VERSION 1.2.10 LANGUAGES CXX)

# Set C++ standard:
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Setup the library:
add_library(${PROJECT_NAME} STATIC)

target_sources(${PROJECT_NAME}
    PRIVATE
        src/grammar.cpp
        src/parser.cpp
        src/writer.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC include)

# Add subdirectories for tests and other helper tools:
add_subdirectory(test)
add_subdirectory(readme-helper)
