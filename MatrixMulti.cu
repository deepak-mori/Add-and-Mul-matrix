#include<iostream>
#include<sys/time.h>
#include<cuda.h>
using namespace std;

__global__ void kernel_ab(int* A, int* B, int* AB, int A_row, int A_col, int B_col) {
    
    __shared__ int mulAB[1024];

	if(threadIdx.x == 0){
        for(int j=0; j<1024; j++){
            mulAB[j] = 0;
        }
    }
    __syncthreads();
	
    for(int i=0; i<A_col; i++){

        mulAB[threadIdx.x] += A[A_col*blockIdx.x + i] * B[threadIdx.x + i*B_col];
        __syncthreads();

    }
	int AB_index = B_col*blockIdx.x + threadIdx.x;
    AB[AB_index] = mulAB[threadIdx.x];   
}

__global__ void kernel_cd(int* C, int* D, int* E, int C_row, int C_col, int D_row){
    
    __shared__ int mulAB[1024];

    if(threadIdx.x == 0){
        for(int j=0; j<1024; j++){
            mulAB[j] = 0;
        }
    }
    __syncthreads();

	int C_index = C_col*blockIdx.x + threadIdx.x;
	int D_index = C_col*blockIdx.y + threadIdx.x;
	mulAB[threadIdx.x] = C[C_index] * D[D_index];
	__syncthreads();

	if(threadIdx.x == 0){
    E[D_row*blockIdx.x + blockIdx.y] = 0;
		for(int i=0; i<C_col; i++){
			E[D_row*blockIdx.x + blockIdx.y] += mulAB[i];
    	}
	}
}

__global__ void kernel_sum(int* AB, int* E, int p, int q, int r){
    
	int E_index = r*blockIdx.x + threadIdx.x;
	int AB_index = r*blockIdx.x + threadIdx.x;
    E[E_index] += AB[AB_index];

}

// function to compute the output matrix
void computE(int p, int q, int r, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixE){
	// Device variables declarations...
	int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixE, *d_matrixAB;
	// allocate memory...
	cudaMalloc(&d_matrixA, p * q * sizeof(int));
	cudaMalloc(&d_matrixB, q * r * sizeof(int));
	cudaMalloc(&d_matrixC, p * q * sizeof(int));
	cudaMalloc(&d_matrixD, r * q * sizeof(int));
	cudaMalloc(&d_matrixE, p * r * sizeof(int));
    cudaMalloc(&d_matrixAB, p * r * sizeof(int));

	// copy the values...
	cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixB, h_matrixB, q * r * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixC, h_matrixC, p * q * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_matrixD, h_matrixD, r * q * sizeof(int), cudaMemcpyHostToDevice);

	/* ****************************************************************** */
	/* Write your code here */
	/* Configure and launch kernels */

	dim3 grid(p,r,1);

    kernel_ab<<<p,r>>>(d_matrixA, d_matrixB, d_matrixAB, p, q, r);
	cudaDeviceSynchronize();
    kernel_cd<<<grid,q>>>(d_matrixC, d_matrixD, d_matrixE, p, q, r);
	cudaDeviceSynchronize();
    kernel_sum<<<p,r>>>(d_matrixAB, d_matrixE, p, q, r);

	/* ****************************************************************** */

	// copy the result back...
	cudaMemcpy(h_matrixE, d_matrixE, p * r * sizeof(int), cudaMemcpyDeviceToHost);

	// deallocate the memory...
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);
	cudaFree(d_matrixD);
	cudaFree(d_matrixE);
}

// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixE;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d", &p, &q, &r);

	// allocate memory and read input matrices
	matrixA = (int*) malloc(p * q * sizeof(int));
	matrixB = (int*) malloc(q * r * sizeof(int));
	matrixC = (int*) malloc(p * q * sizeof(int));
	matrixD = (int*) malloc(r * q * sizeof(int));
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, r);
	readMatrix(inputFilePtr, matrixC, p, q);
	readMatrix(inputFilePtr, matrixD, r, q);

	// allocate memory for output matrix
	matrixE = (int*) malloc(p * r * sizeof(int));

	// call the compute function
	gettimeofday(&t1, NULL);
	computE(p, q, r, matrixA, matrixB, matrixC, matrixD, matrixE);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixE, p, r);

	// close files
	fclose(inputFilePtr);
	fclose(outputFilePtr);

	// deallocate memory
	free(matrixA);
	free(matrixB);
	free(matrixC);
	free(matrixD);
	free(matrixE);

	return 0;
}
	