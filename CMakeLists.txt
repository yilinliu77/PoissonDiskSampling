cmake_minimum_required (VERSION 3.10)
PROJECT(cgal_poisson)

find_package(Boost REQUIRED filesystem serialization )
find_package(Eigen3 CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(CGAL CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


function(link_general arg)
    target_include_directories(${ARGV0} PRIVATE ${Boost_INCLUDE_DIRS})
    target_link_libraries(${ARGV0} general CGAL::CGAL)
    target_link_libraries(${ARGV0} general TBB::tbb TBB::tbbmalloc)
    target_link_libraries(${ARGV0} general glog::glog)
endfunction()

FILE(GLOB SOURCE_FILE "*.cpp" "*.h")

add_executable(cgal_poisson ${SOURCE_FILE})
link_general(cgal_poisson)

