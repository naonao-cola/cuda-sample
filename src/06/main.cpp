#include "test_06.cuh"


int main(int argc ,char**argv)
{

    //asyncAPI();
    //simpleCallback();
    //simpleHyperqBreadth(atoi(argv[1]), atoi(argv[2]));
    simpleHyperqDependence(atoi(argv[1]), atoi(argv[2]));
    return EXIT_SUCCESS;
}