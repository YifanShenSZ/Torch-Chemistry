for directory in linalg \
intcoord chemistry phaser normal_mode \
polynomial SApolynomial gaussian; do
    echo
    echo "Entre "$directory
    # build
    cd $directory/build
    rm test.exe
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
