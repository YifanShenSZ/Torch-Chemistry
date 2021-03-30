for directory in linalg \
intcoord SASintcoord chemistry phaser normal_mode \
polynomial SApolynomial gaussian; do
    echo
    echo "Entre "$directory
    cd $directory/build
    rm -f test.exe
    cmake --build .
    ./test.exe
    cd ../..
done