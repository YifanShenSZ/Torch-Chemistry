for directory in polynomial linalg gaussian intcoord chemistry SAS; do
    cd $directory/build
    rm -f test.exe
    cmake --build .
    ./test.exe
    cd ../..
done