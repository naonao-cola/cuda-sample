#include "test_04.cuh"

int main(int argc, char** argv)
{

    //globalVariable();
    //memTransfer();
    //pinMemTransfer();
    //readSegment(atoi(argv[1]));
    //readSegmentUnroll(atoi(argv[1]));
    //simpleMathAoS();
    simpleMathSoA();
    //sumArrayZerocpy(atoi(argv[1]));
    //sumMatrixGPUManaged(atoi(argv[1]));
    //sumMatrixGPUManual(0);
    //writeSegment(atoi(argv[1]));
    //transpose(atoi(argv[1]));
    return EXIT_SUCCESS;
}