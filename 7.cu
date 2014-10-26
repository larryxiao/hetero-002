nclude    <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt) do {                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
        wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
        wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
        return -1;                                                        \
    }                                                                     \
} while(0)

// len unknown
// __device__ float blocksum[len/BLOCK_SIZE];
// [lecture6.pdf](http://cs.nyu.edu/courses/spring12/CSCI-GA.3033-012/lecture6.pdf)
// __global__ void scan(float * input, float * output, int len) {
__global__ void scan(float * input, float * output, float * blocksum, int len) {
    //@@ Modify the body of this function to complete the functionality of
    //@@ the scan on the device
    //@@ You may need multiple kernel calls; write your kernels before this
    //@@ function and call them from here
    __shared__ float XY[2*BLOCK_SIZE];
    unsigned int t = threadIdx.x;
    unsigned int start = blockIdx.x*blockDim.x;
    XY[t] = start + t > len ? 0.0 : input[start + t];
    XY[blockDim.x+t] = start + blockDim.x+t > len ? 0.0 : input[start + blockDim.x+t];

    for (int stride = 1;stride <= BLOCK_SIZE; stride *= 2) {
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index < 2*BLOCK_SIZE)
            XY[index] += XY[index-stride];
        __syncthreads();

    }
    for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2) {
        __syncthreads();
        int index = (threadIdx.x+1)*stride*2 - 1;
        if(index+stride < 2*BLOCK_SIZE) {
            XY[index + stride] += XY[index];

        }

    }
    __syncthreads();
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < len) {
        output[i] = XY[threadIdx.x];
        if ((i+1)%blockDim.x == 0) blocksum[i/blockDim.x]=output[i];
        // printf("%d\t%d\t%d\t%g\n", i, threadIdx.x, XY[threadIdx.x], output[i]);
        // __syncthreads();
        // [cuda - Does __syncthreads() synchronize all threads in the grid? - Stack Overflow](http://stackoverflow.com/questions/15240432/does-syncthreads-synchronize-all-threads-in-the-grid)
        // To synchronize all threads in a grid currently there is not native API call.
        // not very good memory access pattern
        // need shared memory
        // for (int j=0; j<i/blockDim.x; j++){
        // output[i] += blocksum[j];
        // be careful about the index
        // if (j+1==i/blockDim.x) output[i] += output[j*blockDim.x+blockDim.x-1];
        // it's not going to work (it's for same block)
        // __syncthreads();
        // 
    }
    // __syncthreads();
    // printf("%d\t%d\t%d\t%g\n", i, threadIdx.x, XY[threadIdx.x], output[i]);

    }

}

__global__ void helper(float * output, float * blocksum, int len) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < len){
        for (int j=0; j<i/blockDim.x; j++)
            output[i] += blocksum[j];

    }

}

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostInput; // The input 1D list
    float * hostOutput; // The output list
    float * deviceInput;
    float * deviceOutput;
    float * deviceTemp;
    int numElements; // number of elements in the list

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostInput = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float*) malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The number of input elements in the input is ", numElements);

    wbTime_start(GPU, "Allocating GPU memory.");
    wbCheck(cudaMalloc((void**)&deviceInput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceOutput, numElements*sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceTemp, numElements/BLOCK_SIZE*sizeof(float)));
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Clearing output memory.");
    wbCheck(cudaMemset(deviceOutput, 0, numElements*sizeof(float)));
    wbCheck(cudaMemset(deviceTemp, 0, numElements/BLOCK_SIZE*sizeof(float)));
    wbTime_stop(GPU, "Clearing output memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements*sizeof(float), cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid((numElements + BLOCK_SIZE - 1)/BLOCK_SIZE, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    wbLog(TRACE, "DimGrid ", (numElements + BLOCK_SIZE - 1)/BLOCK_SIZE);
    wbLog(TRACE, "DimBlock ", BLOCK_SIZE);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce
    // scan<<<DimGrid,DimBlock>>>(deviceInput, deviceOutput, numElements);
    scan<<<DimGrid,DimBlock>>>(deviceInput, deviceOutput, deviceTemp, numElements);
    helper<<<DimGrid,DimBlock>>>(deviceOutput, deviceTemp, numElements);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements*sizeof(float), cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    cudaFree(deviceTemp);
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;

}


}
