for directory in cart2int cart2SASIC; do
    echo
    echo "Entre "$directory
    cd $directory
    # build
    if [ -d build ]; then rm -r build; fi
    mkdir build
    cd build
    cmake -DCMAKE_C_COMPILER=icc -DCMAKE_CXX_COMPILER=icpc -DCMAKE_Fortran_COMPILER=ifort ..
    cmake --build .
    cd ..
    # link exe
    if [ -f $directory.exe ]; then rm $directory.exe; fi
    ln -s build/$directory.exe
    # finish
    cd ..
done
