for directory in linalg \
intcoord SASintcoord chemistry phaser normal_mode \
polynomial SApolynomial gaussian; do
    echo
    echo "Entre "$directory
    cd $directory/build
    rm -f test.exe
    cmake --build .
    cd ..
    if [ -d input ]; then
        cd input
        ../build/test.exe
        cd ../..
    else
       ./build/test.exe
       cd ..
    fi
done