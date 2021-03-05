for directory in linalg polynomial SApolynomial phaser \
intcoord SASintcoord gaussian chemistry; do
    echo
    echo "Entre "$directory
    cd $directory/build
    rm -f test.exe
    cmake --build .
    ./test.exe
    cd ../..
done