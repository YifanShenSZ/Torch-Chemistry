for directory in linalg polynomial SApolynomial \
gaussian intcoord SASintcoord chemistry; do
    cd $directory/build
    rm -f test.exe
    cmake --build .
    ./test.exe
    cd ../..
done