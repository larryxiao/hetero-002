#include    <wb.h>


#define wbCheck(stmt) do {                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
        wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
        wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
        return -1;                                                        \
    }                                                                     \
} while(0)

#define MASK_WIDTH  5
#define MASK_RADIUS MASK_WIDTH/2

//@@ INSERT CODE HERE

#define O_TILE_WIDTH 12
#define BLOCK_WIDTH (O_TILE_WIDTH + 4)
#define MAX(a, b)  (((a) > (b)) ? (a) : (b))
#define MIN(a, b)  (((a) < (b)) ? (a) : (b)) 
#define clamp(x, start, end) MIN(MAX(x, start), end)

__global__ void convolution_2D_kernel(float *P, float *N, int height, int width, int channels, const float *M) {
    // __global__ void convolution_2D_kernel(float *P, float *N, int height, int width, int channels, const float __restrict__ *M) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row_o = blockIdx.y*O_TILE_WIDTH + ty;
    int col_o = blockIdx.x*O_TILE_WIDTH + tx;
    int row_i = row_o - 2;
    int col_i = col_o - 2;

    // notice
    // __shared__ float Ns[width][height];
    // __shared__ float Ns[width][height][channels];  // channels === 3
    __shared__ float Ns[BLOCK_WIDTH][BLOCK_WIDTH][3];  // channels === 3
    if((row_i >= 0) && (row_i < height) && (col_i >= 0)  && (col_i < width))
    {
        // Ns[ty][tx] = N[row_i*width + col_i];
        // reorganized the layout (good thing?)
        Ns[ty][tx][0] = N[(row_i*width + col_i)*3 + 0];
        Ns[ty][tx][1] = N[(row_i*width + col_i)*3 + 1];
        Ns[ty][tx][2] = N[(row_i*width + col_i)*3 + 2];

    } else {
        Ns[ty][tx][0] = 0.0f;
        Ns[ty][tx][1] = 0.0f;
        Ns[ty][tx][2] = 0.0f;

    }
    __syncthreads();

    float output0 = 0.0f;
    float output1 = 0.0f;
    float output2 = 0.0f;
    if(ty < O_TILE_WIDTH && tx < O_TILE_WIDTH){
        for(int i = 0; i < MASK_WIDTH; i++) {
            for(int j = 0; j < MASK_WIDTH; j++) {
                // output += M[i][j] * Ns[i+ty][j+tx];
                // output0 += M[(i*MASK_WIDTH+j)*3 + 0] * Ns[i+ty-2][j+tx-2][0];
                // output1 += M[(i*MASK_WIDTH+j)*3 + 1] * Ns[i+ty-2][j+tx-2][1];
                // output2 += M[(i*MASK_WIDTH+j)*3 + 2] * Ns[i+ty-2][j+tx-2][2];
                // output0 += M[(i-MASK_RADIUS)*MASK_WIDTH+j-MASK_RADIUS] * Ns[i+ty-MASK_RADIUS][j+tx-MASK_RADIUS][0];
                // output1 += M[(i-MASK_RADIUS)*MASK_WIDTH+j-MASK_RADIUS] * Ns[i+ty-MASK_RADIUS][j+tx-MASK_RADIUS][1];
                // output2 += M[(i-MASK_RADIUS)*MASK_WIDTH+j-MASK_RADIUS] * Ns[i+ty-MASK_RADIUS][j+tx-MASK_RADIUS][2];
                // output0 += M[(i)*MASK_WIDTH+j] * Ns[i+ty-MASK_RADIUS][j+tx-MASK_RADIUS][0];
                // output1 += M[(i)*MASK_WIDTH+j] * Ns[i+ty-MASK_RADIUS][j+tx-MASK_RADIUS][1];
                // output2 += M[(i)*MASK_WIDTH+j] * Ns[i+ty-MASK_RADIUS][j+tx-MASK_RADIUS][2];
                output0 += M[(i)*MASK_WIDTH+j] * Ns[i+ty][j+tx][0];
                output1 += M[(i)*MASK_WIDTH+j] * Ns[i+ty][j+tx][1];
                output2 += M[(i)*MASK_WIDTH+j] * Ns[i+ty][j+tx][2];
                // output0 += 0.2f;
                // output1 += 0.3f;
                // output2 += 0.5f;

            }

        }
        __syncthreads();

        if(row_o < height && col_o < width){
            // P[row_o*width + col_o] = clamp(output);
            P[(row_o*width + col_o)*3 + 0] = output0;
            P[(row_o*width + col_o)*3 + 1] = output1;
            P[(row_o*width + col_o)*3 + 2] = output2;
            // P[(row_o*width + col_o)*3 + 0] = clamp(output0, 0, 1);
            // P[(row_o*width + col_o)*3 + 1] = clamp(output1, 0, 1);
            // P[(row_o*width + col_o)*3 + 2] = clamp(output2, 0, 1);

        }

    }

}

int main(int argc, char* argv[]) {
    wbArg_t args;
    int maskRows;
    int maskColumns;
    int imageChannels;
    int imageWidth;
    int imageHeight;
    char * inputImageFile;
    char * inputMaskFile;
    wbImage_t inputImage;
    wbImage_t outputImage;
    float * hostInputImageData;
    float * hostOutputImageData;
    float * hostMaskData;
    float * deviceInputImageData;
    float * deviceOutputImageData;
    float * deviceMaskData;

    args = wbArg_read(argc, argv); /* parse the input arguments */

    inputImageFile = wbArg_getInputFile(args, 0);
    inputMaskFile = wbArg_getInputFile(args, 1);

    inputImage = wbImport(inputImageFile);
    hostMaskData = (float *) wbImport(inputMaskFile, &maskRows, &maskColumns);

    assert(maskRows == 5); /* mask height is fixed to 5 in this mp */
    assert(maskColumns == 5); /* mask width is fixed to 5 in this mp */

    imageWidth = wbImage_getWidth(inputImage);
    imageHeight = wbImage_getHeight(inputImage);
    imageChannels = wbImage_getChannels(inputImage);

    outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);

    hostInputImageData = wbImage_getData(inputImage);
    hostOutputImageData = wbImage_getData(outputImage);

    wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

    wbTime_start(GPU, "Doing GPU memory allocation");
    cudaMalloc((void **) &deviceInputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceOutputImageData, imageWidth * imageHeight * imageChannels * sizeof(float));
    cudaMalloc((void **) &deviceMaskData, maskRows * maskColumns * sizeof(float));
    wbTime_stop(GPU, "Doing GPU memory allocation");


    wbTime_start(Copy, "Copying data to the GPU");
    cudaMemcpy(deviceInputImageData,
            hostInputImageData,
            imageWidth * imageHeight * imageChannels * sizeof(float),
            cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMaskData,
            hostMaskData,
            maskRows * maskColumns * sizeof(float),
            cudaMemcpyHostToDevice);
    wbTime_stop(Copy, "Copying data to the GPU");


    wbTime_start(Compute, "Doing the computation on the GPU");
    //@@ INSERT CODE HERE
    dim3 dimBlock(BLOCK_WIDTH,BLOCK_WIDTH);
    dim3 dimGrid((wbImage_getWidth(inputImage)-1)/O_TILE_WIDTH+1, (wbImage_getHeight(inputImage)-1)/O_TILE_WIDTH+1, 1);
    convolution_2D_kernel<<<dimGrid,dimBlock>>>(deviceOutputImageData, deviceInputImageData, imageHeight, imageWidth, imageChannels, deviceMaskData);

    wbTime_stop(Compute, "Doing the computation on the GPU");


    wbTime_start(Copy, "Copying data from the GPU");
    cudaMemcpy(hostOutputImageData,
            deviceOutputImageData,
            imageWidth * imageHeight * imageChannels * sizeof(float),
            cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying data from the GPU");

    wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

    wbSolution(args, outputImage);

    cudaFree(deviceInputImageData);
    cudaFree(deviceOutputImageData);
    cudaFree(deviceMaskData);

    free(hostMaskData);
    wbImage_delete(outputImage);
    wbImage_delete(inputImage);

    return 0;

}

}
