// MP Reduction
// Given a list (lst) of length n
// Output its sum = lst[0] + lst[1] + ... + lst[n-1];

#include    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this
// #define BLOCK_SIZE 5120 //@@ You can change this
#define SECTION_SIZE 2*BLOCK_SIZE

// Trace    main::36    Device 1 name: Tesla C2050
// Trace    main::37    Computational Capabilities: 2.0
// Trace    main::38    Maximum global memory size: 2817982464
// Trace    main::39    Maximum constant memory size: 65536
// Trace    main::40    Maximum shared memory size per block: 49152
// Trace    main::43    Maximum block dimensions: 1024 x 1024 x 64
// Trace    main::46    Maximum grid dimensions: 65535 x 65535 x 65535
// Trace    main::47    Warp size: 32

// 0 256
// 1 512
// 2 600
// 3 1024
// 4 9000
// 5 12670

#define wbCheck(stmt) do {                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
        wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
        wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
        return -1;                                                        \
    }                                                                     \
} while(0)

__global__ void  total(float * input, float * output, int len) {
    //@@ Load a segment of the input vector into shared memory
    //@@ Traverse the reduction tree
    //@@ Write the computed sum of the block to the output vector at the 
    //@@ correct index
    __shared__ float partialSum[2*BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = 2*blockIdx.x*blockDim.x;

    partialSum[t] = start + t >= len ? 0.0 : input[start + t];
    partialSum[blockDim.x+t] = start + blockDim.x+t >= len ? 0.0 : input[start + blockDim.x+t];
    //printf("len %d start %d start + blockDim.x+t %d partialSum[t] %g partialSum[blockDim.x+t] %g\n", len, start, start + blockDim.x+t, partialSum[t], partialSum[blockDim.x+t]);
    // XY[2*BLOCK_SIZE] is in shared memory
    // for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2)
    // {
    //     __syncthreads();
    //     if (t % stride == 0)
    //         partialSum[2*t] += partialSum[2*t+stride];
    // 
}
__syncthreads();
for (unsigned int stride = blockDim.x; stride > 0;  stride /= 2)
{
    __syncthreads();
    if (t < stride)
        partialSum[t] += partialSum[t+stride]; 

}
// printf("[%i, %i, %i]: %f + %f = %f\n", blockIdx.x, tx, stride, f1, f2, partialSum[tx]);
__syncthreads();
if (t == 0)
{
    output[0] = partialSum[0];

}

}

int main(int argc, char ** argv) {
    int ii;
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    int numInputElements; // number of elements in the input list
    int numOutputElements; // number of elements in the output list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numInputElements);

    numOutputElements = numInputElements / (BLOCK_SIZE<<1);
    if (numInputElements % (BLOCK_SIZE<<1)) {
        numOutputElements++;

    }
    hostOutput = (float*) malloc(numOutputElements * sizeof(float));

    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numInputElements);
    wbLog(TRACE, "The number of output elements in the input is ", numOutputElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    cudaMalloc((void **) &deviceInput, numInputElements * sizeof(float));
    cudaMalloc((void **) &deviceOutput, numOutputElements * sizeof(float));


    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    cudaMemcpy(deviceInput, hostInput, numInputElements * sizeof(float), cudaMemcpyHostToDevice);

    wbTime_stop(GPU, "Copying input memory to the GPU.");
    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid((numInputElements + BLOCK_SIZE - 1)/BLOCK_SIZE, 1, 1);
    wbLog(TRACE, "GridSize ", (numInputElements + BLOCK_SIZE - 1)/BLOCK_SIZE);
    wbLog(TRACE, "numOutputElements ", numOutputElements);
    // dim3 DimGrid(ceil(numInputElements/BLOCK_SIZE), 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    for (int i = 0; i < numOutputElements; ++i)
    {
        total<<<1,DimBlock>>>(
                // deviceInput, 
                &deviceInput[BLOCK_SIZE*i*2], 
                // deviceOutput, 
                &deviceOutput[i], 
                numInputElements - BLOCK_SIZE*(i+1)*2 > 0 ? BLOCK_SIZE*2 : numInputElements - BLOCK_SIZE*(i)*2

                );
        //wbLog(TRACE, "len ", numInputElements - BLOCK_SIZE*(i+1)*2);
        //float temp;
        //cudaMemcpy(&temp, &deviceOutput[i], 1 * sizeof(float), cudaMemcpyDeviceToHost);
        //wbLog(TRACE, "output ", temp);
        // cudaDeviceSynchronize();
        cudaThreadSynchronize();

    }
    // total<<<DimGrid,DimBlock>>>(deviceInput, deviceOutput, numInputElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    cudaMemcpy(hostOutput, deviceOutput, numOutputElements * sizeof(float), cudaMemcpyDeviceToHost);

    wbTime_stop(Copy, "Copying output memory to the CPU");

    /********************************************************************
     * Reduce output vector on the host
     * NOTE: One could also perform the reduction of the output vector
     * recursively and support any size input. For simplicity, we do not
     * require that for this lab.
     ********************************************************************/
    for (ii = 1; ii < numOutputElements; ii++) {
        hostOutput[0] += hostOutput[ii];

    }

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    cudaFree(deviceInput);
    cudaFree(deviceOutput);    

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, 1);

    free(hostInput);
    free(hostOutput);

    return 0;

}
