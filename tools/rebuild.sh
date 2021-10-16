for directory in cart2int cart2SASIC; do
    echo
    echo "Entre "$directory
    cd $directory/build
    cmake --build .
    cd ../..
done
