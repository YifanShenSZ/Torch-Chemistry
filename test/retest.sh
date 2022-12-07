for directory in linalg \
intcoord chemistry phaser orderer normal_mode \
polynomial SApolynomial gaussian; do
    echo
    echo "Entre "$directory
    # build
    cd $directory/build
    cmake --build .
    cd ..
    # run
    if [ -d input ]; then
        cd input
        ../build/test.exe
        cd ../..
    else
       ./build/test.exe
       cd ..
    fi
done
