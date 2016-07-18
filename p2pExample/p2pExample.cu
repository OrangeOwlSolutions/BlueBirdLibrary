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
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <conio.h>

#include <typeinfo>
#include <iostream>

// includes CUDA
#include <cuda.h>
#include <cuda_runtime.h>

#include "ComplexTypes.cuh"										
#include "ComplexTypes.cu"										
using namespace BB;

// includes Expression Templates library
#include "Macros.h"											
#include "HExceptions.h"										
#include "HExceptions.cpp"										
#include "DExceptions.cuh"										
#include "Constants.h"											
#include "DStreams.cuh"											
#include "DStreams.cu"											
#include "P2P.cu"
#include "SubIndicesAccessHandling.h"							
#include "SubIndicesAccessHandling.cpp"							
#include "DGlobals.cuh"											
#include "CudaExpressionMacros.cuh"								
////#include "CudaExpressionKernels.cuh"						
#include "Expression.cuh"										
//
//// includes CUDA Expression Templates library
#include "Scalar.cuh"											
#include "Promotion.cuh"									
#include "Hmatrix.h"												
#include "Hmatrix.cpp"												
#include "DMatrixExpressionKernels.cuh"						
#include "Dmatrix.cuh"										
#include "Dmatrix.cu"										
////#include "CudaMatrixExpression.cuh"
#include "Addition.cuh"
#include "Addition.cu"
#include "Subtraction.cuh"
#include "Subtraction.cu"
#include "Multiplication.cuh"
#include "Multiplication.cu"
#include "Division.cuh"
#include "Division.cu"
#include "Functions.cuh"
#include "Functions.cu"
#include "Utility.cuh"
#include "Grid.cuh"
#include "SubMatrixExpression.h"
#include "DFFT.cuh"
#include "DBLAS.cuh"
#include "TimingCPU.h"
#include "TimingCPU.cpp"
#include "TimingGPU.cuh"
#include "TimingGPU.cu"
#include "DReduction.cuh"
#include "HReduction.h"
#include "InputOutPut.h"
#include "Utility.h"

//#pragma comment(lib, "winmm.lib")

// Input/Output
// --- Overload of << for int2_
std::ostream & operator << (std::ostream&, const int2_&);
// --- Overload of << for float2_
std::ostream & operator << (std::ostream&, const float2_&);
// --- Overload of << for double2_
std::ostream & operator << (std::ostream&, const double2_&);
// --- Overload of << for type T Hmatrix (int, float, double, int2_, float2_, double2_)
template <class T> std::ostream& operator << (std::ostream&, const Hmatrix<T>&);
// --- Overload of << for type T Dmatrix (int, float, double, int2_, float2_, double2_)
template <class T> std::ostream & operator << (std::ostream&, const Dmatrix<T>&);

/**********************************/
/* OVERLOAD OF << FOR EXPRESSIONS */
/**********************************/
template <class Q, class T>
ostream & operator << (ostream & output, const Expr<Q,T> v)
{
	Hmatrix<T> a(v.GetRows(),v.GetColumns());
	if (v.IsDevice()) {
		Dmatrix<T> b(v.GetRows(),v.GetColumns());
		b = v;
		a = b; }
	else {
		a = v; }
	output << a;
	return output;   
}

// Constant to CPU Hmatrix assignment
template <class OutType>
const BB::Hmatrix<OutType>& BB::Hmatrix<OutType>::operator=(const OutType c)
{
	*this = BB::Expr<BB::Scalar<OutType>,OutType>(BB::Scalar<OutType>(c),Rows_,Columns_,ISHOST);
	return *this;
}

// Constant to GPU Hmatrix assignment
template <class OutType>
const BB::Dmatrix<OutType>& BB::Dmatrix<OutType>::operator=(const OutType c)
{
	*this = BB::Expr<BB::Scalar<OutType>,OutType>(BB::Scalar<OutType>(c),Rows_,Columns_,ISDEVICE);
	return *this;
}


//#include <helper_functions.h>  // helper for shared that are common to CUDA SDK samples

inline bool IsGPUCapableP2P(cudaDeviceProp *pProp)
{
#ifdef _WIN32
    return (bool)(pProp->tccDriver ? true : false);
#else
    return (bool)(pProp->major >= 2);
#endif
}

int main(int argc, char **argv)
{
	// --- Initializes peer-to-peer
	p2p.InitP2P();
	
	// --- Initializes the streams for all the GPUs supporting peer-to-peer
	for (int i = 0; i < p2p.GetNumP2PGPUs(); i++) {
		p2p.SetDevice(i);
		streams[i].InitStreams(1);
		streams[i].SetStream(0); }

	{

		// Size of the memory to be allocated
		const size_t buf_size = 1024 * 1024 * 16 * sizeof(float);

		// Allocate memory on the first p2p-supporting GPU
		p2p.SetDevice(p2p.gpuid[0]);
		Dmatrix<float> g0(SetZeros<float>(1,1024 * 1024 * 16));

		// Allocate memory on the second p2p-supporting GPU
		p2p.SetDevice(p2p.gpuid[1]);
		Dmatrix<float> g1(SetZeros<float>(1,1024 * 1024 * 16));

		// Allocate host memory
		Hmatrix<float> h0(1,1024 * 1024 * 16,PINNED);

 		TimingGPU timer;

		timer.StartCounterFlags();

		// Assignments between two device matrices residing on different devices
		/******************************************************************/
		/* ASSUMPTION!!! gpuid[0] and gpuid[1] can access peer each other */
		/******************************************************************/
		// This operation is performed by launching kernels on one GPU that use data residing on the other GPU
		cout << "Iterative kernel launches on GPU "<< p2p.gpuid[0] << " using data on GPU " << p2p.gpuid[1] << " and viceversa.\n\n";
		for (int i=0; i<100; i++)
			if (i % 2 == 0) g1 = g0 + 1;
			else g0 = g1 + 1;

		cout << "Average time: " << timer.GetCounter()/100 << "\n";
			
		// Copying data from CPU to GPU
		cout << "Copying data from CPU to GPU " << p2p.gpuid[0] << "\n";

		for (int i=0; i<buf_size / sizeof(float); i++) h0(i) = float(i % 4096);

		p2p.SetDevice(p2p.gpuid[0]);
		g0 = h0;

		// Run kernel on GPU 1, reading input from the GPU 0, writing output to the GPU 1 buffer
		cout << "Doubling the GPU 0 matrix on GPU 1\n";
		g1 = 2. * g0;

		CudaSafeCall(cudaDeviceSynchronize());

		// Run kernel on GPU 0, reading input from the GPU 1, writing output to the GPU 0 buffer
		cout << "Doubling the GPU 1 matrix on GPU 0\n";
		g0 = 2. * g1;

		CudaSafeCall(cudaDeviceSynchronize());

		// Copying data from CPU to GPU
		cout << "Copying data from GPU " << p2p.gpuid[0] << " back to CPU\n";
		h0 = g0;

		// Verifying the correctness of the results
		cout << "Verification...\n";
		
		int error_count = 0;
		for (int i=0; i<buf_size / sizeof(float); i++)
		{
			if (h0(i) != float(i % 4096) * 2.0f * 2.0f)
			{
				printf("Verification error @ element %i: val = %f, ref = %f\n", i, h0.GetDataPointer()[i], (float(i%4096)*2.0f*2.0f));

				if (error_count++ > 10) break;
			}
		}
		
		if (error_count != 0)
		{
			printf("Test failed!\n");
			exit(EXIT_FAILURE);
		}
		printf("Test passed\n");
		getch();
		exit(EXIT_SUCCESS);

   }

	p2p.ResetGPUs();
}
