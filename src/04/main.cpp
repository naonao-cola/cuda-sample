#include "test_04.cuh"

int main(int argc, char** argv)
{

    //globalVariable();
    //memTransfer();
    //pinMemTransfer();
    //readSegment(atoi(argv[1]));
    //readSegmentUnroll(atoi(argv[1]));
    //simpleMathAoS();
    //simpleMathSoA();
    sumArrayZerocpy(atoi(argv[1]));
    return EXIT_SUCCESS;
}