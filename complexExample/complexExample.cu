/*   Bluebird Library - High performance CPUs and GPUs computing library.
*    
*    Copyright (C) 2012-2013 Orange Owl Solutions.  
*
*    This file is part of Bluebird Library.
*    Bluebird Library is free software: you can redistribute it and/or modify
*    it under the terms of the Lesser GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*
*    Bluebird Library is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    Lesser GNU General Public License for more details.
*
*    You should have received a copy of the GNU General Public License
*    along with Bluebird Library.  If not, see <http://www.gnu.org/licenses/>.
*
*
*    For any request, question or bug reporting please visit http://www.orangeowlsolutions.com/
*    or send an e-mail to: info@orangeowlsolutions.com
*
*
*/

using namespace std;

// includes, system
#include <conio.h>
#include <iostream>

// includes, CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// includes, complex library
#include "ComplexTypes.cuh"										
using namespace BB;
#include "Result_Types.cuh"
#include "Operator_Overloads.cuh"
#include "Function_Overloads.cuh"

#include "TimingCPU.h"
#include "TimingCPU.cpp"
#include "TimingGPU.cuh"
#include "TimingGPU.cu"

/***************************/
/* GPU LIBRARY TEST KERNEL */
/***************************/
__global__ void test_library(const float2_* A, const float2_* B, const float2_* C, float2_* D, int N) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// --- Case 1
	//if (tid < N) D[tid] = 3.f * A[tid] + B[tid] * C[tid];

	// --- Case 2
	//if (tid < N) D[tid] = A[tid] * B[tid] - sin(C[tid]) + 3.f;

	// --- Case 3
	if (tid < N) D[tid] = A[tid] / B[tid] + 3.f * cos(C[tid]);
}

/*******************************/
/* GPU HAND CODING TEST KERNEL */
/*******************************/
__global__ void test(float2* A, float2* B, float2* C, float2* D, int N) {

	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < N) {
		// --- Case 1
		//D[tid].x = 3.f * A[tid].x + B[tid].x * C[tid].x - B[tid].y * C[tid].y;
		//D[tid].y = 3.f * A[tid].y + B[tid].x * C[tid].y + B[tid].y * C[tid].x;
		
		// --- Case 2
		//D[tid].x=(A[tid].x*B[tid].x-A[tid].y*B[tid].y)-sin(C[tid].x)*cosh(C[tid].y)+3.f;
		//D[tid].y=(A[tid].x*B[tid].y+A[tid].y*B[tid].x)-cos(C[tid].x)*sinh(C[tid].y);

		// --- Case 3
		float den = hypot(B[tid].x,B[tid].y); den = den * den;
		D[tid].x=(A[tid].x*B[tid].x+A[tid].y*B[tid].y)/den+3.f*cos(C[tid].x)*cosh(C[tid].y);
		D[tid].y=((A[tid].y*B[tid].x)-(A[tid].x*B[tid].y))/den-3.f*sin(C[tid].x)*sinh(C[tid].y);

	}

}

/********************************/
/* CPU INITIALIZATIONS FUNCTION */
/********************************/
void Initializations(float2_* A, float2_* B, float2_* C, float2_* D, float2* A2, float2* B2, float2* C2, float2* D2, int N) {
	for (int i=0; i<N; i++) {
		A[i]	= float2_(i*i,sqrt((float)i));
		A2[i].x	= (float)i*i;					A2[i].y = sqrt((float)i);
		B[i]	= float2_(log((float)(i+3)),i*i*i);
		B2[i].x	= (float)log((float)(i+3));		B2[i].y = i*i*i;
		C[i]	= float2_(3.f,0.f);
		C2[i].x	= 3.f;							C2[i].y = 0.f;
	}
}

int iDivUp(const int a, const int b) { return (a % b != 0) ? (a / b + 1) : (a / b); };

#define BLOCKSIZE 256

void main( int argc, char** argv) 
{
//	int N = 4194304;
	int N	= 10000;
	int Nit = 100;

	/*************/
	/* HOST SIDE */
	/*************/

	// --- CPU memory allocations
	float2_* A = new float2_[N];
	float2_* B = new float2_[N];
	float2_* C = new float2_[N];
	float2_* D = new float2_[N];

	float2* A2 = new float2[N];
	float2* B2 = new float2[N];
	float2* C2 = new float2[N];
	float2* D2 = new float2[N];

	// --- Initilizations
	Initializations(A,B,C,D,A2,B2,C2,D2,N);

	TimingCPU timer;
	float timer1 = 0.f;
	for (int k=0; k<Nit; k++)
		for (int i=0; i<N; i++) {
			timer.StartCounter();
			// --- Case 1
			//D[i] = 3.f * A[i] + B[i] * C[i];
			// --- Case 2
			//D[i] = A[i] * B[i] - sin(C[i]) + 3.f;
			// --- Case 2
			D[i] = A[i] / B[i] + 3.f * cos(C[i]);
			timer1=timer1+timer.GetCounter();
		}
	cout << "CPU timing using customized types: " << timer1 << " ms\n";
	
	float timer2 = 0.f;
	for (int k=0; k<Nit; k++)
		for (int i=0; i<N; i++) { 
			timer.StartCounter();
			// --- Case 1
			//D2[i].x = 3.f * A2[i].x + B2[i].x * C2[i].x - B2[i].y * C2[i].y;
			//D2[i].y = 3.f * A2[i].y + B2[i].x * C2[i].y + B2[i].y * C2[i].x;
			// --- Case 2
			//D2[i].x=(A2[i].x*B2[i].x-A2[i].y*B2[i].y)-sin(C2[i].x)*cosh(C2[i].y)+3.;
			//D2[i].y=(A2[i].x*B2[i].y+A2[i].y*B2[i].x)-cos(C2[i].x)*sinh(C2[i].y);
			// --- Case 3
			float den = hypot(B2[i].x,B2[i].y); den = den * den;
			D2[i].x=(A2[i].x*B2[i].x+A2[i].y*B2[i].y)/den+3.f*cos(C2[i].x)*cosh(C2[i].y);
			D2[i].y=((A2[i].y*B2[i].x)-(A2[i].x*B2[i].y))/den-3.f*sin(C2[i].x)*sinh(C2[i].y);
			timer2=timer2+timer.GetCounter();
		}
	cout << "CPU timing using CUDA types: " << timer1 << " ms\n";

	/***************/
	/* DEVICE SIDE */
	/***************/

	// --- GPU memory allocations
	float2_* d_A; cudaMalloc((void**)&d_A, N*sizeof(float2_));
	float2_* d_B; cudaMalloc((void**)&d_B, N*sizeof(float2_));
	float2_* d_C; cudaMalloc((void**)&d_C, N*sizeof(float2_));
	float2_* d_D; cudaMalloc((void**)&d_D, N*sizeof(float2_));

	float2* d_A2; cudaMalloc((void**)&d_A2, N*sizeof(float2));
	float2* d_B2; cudaMalloc((void**)&d_B2, N*sizeof(float2));
	float2* d_C2; cudaMalloc((void**)&d_C2, N*sizeof(float2));
	float2* d_D2; cudaMalloc((void**)&d_D2, N*sizeof(float2));

	// --- host-device memory transfers
	cudaMemcpy(d_A, A, N*sizeof(float2_), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, N*sizeof(float2_), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, C, N*sizeof(float2_), cudaMemcpyHostToDevice);
	cudaMemcpy(d_D, D, N*sizeof(float2_), cudaMemcpyHostToDevice);

	cudaMemcpy(d_A2, A2, N*sizeof(float2), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B2, B2, N*sizeof(float2), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C2, C2, N*sizeof(float2), cudaMemcpyHostToDevice);
	cudaMemcpy(d_D2, D2, N*sizeof(float2), cudaMemcpyHostToDevice);

	// --- kernel performance comparison
	dim3 dimBlock(BLOCKSIZE);
	dim3 dimGrid(iDivUp(N,dimBlock.x));

	TimingGPU d_timer;
	float d_timer1 = 0.f;
	for (int i=0; i<Nit; i++) {
		d_timer.StartCounter();
		test_library<<<dimGrid.x,dimBlock.x>>>(d_A,d_B,d_C,d_D,N);
		d_timer1 = d_timer1 + d_timer.GetCounter();
	}
	cout << "GPU timing using customized types: " << d_timer1 << " ms\n";

	float d_timer2 = 0.f;
	for (int i=0; i<Nit; i++) {
		d_timer.StartCounter();
		test<<<dimGrid.x,dimBlock.x>>>(d_A2,d_B2,d_C2,d_D2,N);
		d_timer2 = d_timer2 + d_timer.GetCounter();
	}
	cout << "GPU timing using CUDA types: " << d_timer2 << " ms\n";

	cudaMemcpy(A2, d_A2, N*sizeof(float2), cudaMemcpyDeviceToHost);
	cudaMemcpy(B2, d_B2, N*sizeof(float2), cudaMemcpyDeviceToHost);
	cudaMemcpy(C2, d_C2, N*sizeof(float2), cudaMemcpyDeviceToHost);
	cudaMemcpy(D2, d_D2, N*sizeof(float2), cudaMemcpyDeviceToHost);

	cudaMemcpy(A, d_A, N*sizeof(float2_), cudaMemcpyDeviceToHost);
	cudaMemcpy(B, d_B, N*sizeof(float2_), cudaMemcpyDeviceToHost);
	cudaMemcpy(C, d_C, N*sizeof(float2_), cudaMemcpyDeviceToHost);
	cudaMemcpy(D, d_D, N*sizeof(float2_), cudaMemcpyDeviceToHost);

	cout << "Handcoded result:\n";
	cout << "D[0] = (" << D2[0].x << "," << D2[0].y << ") \n";
	cout << "D[1] = (" << D2[1].x << "," << D2[1].y << ") \n";
	cout << "D[2] = (" << D2[2].x << "," << D2[2].y << ") \n";

	cout << "Library result:\n";
	cout << "D[0] = " << D[0] << "\n";
	cout << "D[1] = " << D[1] << "\n";
	cout << "D[2] = " << D[2] << "\n";

	std::cout << "Going to sleep" << std::endl;
	getch();
}

