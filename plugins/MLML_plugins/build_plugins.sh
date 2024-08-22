mkdir build
cd build
cmake .. -D BUILD_SHARED_LIBS=yes -D CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV \
         -D BUILD_MPI=yes

cmake --build . -j 1

