for directory in linalg polynomial SApolynomial \
gaussian intcoord SASintcoord \
chemistry phaser normal_mode; do
    echo
    echo "Entre "$directory
    cd $directory/build
    rm -f test.exe
    cmake --build .
    ./test.exe
    cd ../..
done