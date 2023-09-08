cmake_minimum_required(VERSION 2.8)
project( DisplayImage )
# for debugging using gcc/gdb, idk about how it works with other compilers
add_compile_options(-g)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( extract-bbs extract-bbs.cpp )
target_link_libraries( extract-bbs ${OpenCV_LIBS} )
