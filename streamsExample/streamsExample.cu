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
#include "Dstreams.cuh"											
#include "Dstreams.cu"											
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

void main( int argc, char** argv) 
{
	streams[0].InitStreams(1);
	streams[0].SetStream(0);

	{
		int nstreams = 4;              // number of streams for CUDA calls
		int nreps = 20;                 // number of times each experiment is repeated
		int n = 16 * 1024 * 1024;       // number of ints in the data set

		TimingGPU	timer1, timer2, timer3, timer4, timer5, timer6;

		// allocate host memory

		Hmatrix<int>				h_a_matrix(1,n,PINNED);
		Dmatrix<int>			d_a_matrix(1,n);

		printf("\nTiming streams - measurements in ms\n\n");

		streams[0].InitStreams(1);

		/*******************************************************************/
		/* TIMING THE MEMCPY ASYNC ONLY - NO COMPUTATION - ONE STREAM ONLY */
		/*******************************************************************/
		timer1.StartCounter();
		streams[0].SetStream(0);
		for(int k = 0; k < nreps; k++)
		{
			h_a_matrix=d_a_matrix;
		}
		streams[0].SynchronizeAll();
		float timing1 = timer1.GetCounter() / nreps;
		printf("Memcopy async only (1 stream):\t\t\t\t %.2f\n", timing1);
    
		/*****************************************************************/
		/* TIMING THE KERNEL ONLY - NO MEMORY TRANSFER - ONE STREAM ONLY */
		/*****************************************************************/
		timer2.StartCounter();
		streams[0].SetStream(0);
		for(int k = 0; k < nreps; k++)
		{
			d_a_matrix = 5; 
		}
		streams[0].SynchronizeAll();
		float timing2 = timer2.GetCounter() / nreps;
		printf("Kernel only (1 stream):\t\t\t\t\t %.2f\n", timing2);

		/*************************************************************/
		/* TIMING THE KERNEL WITH MEMORY TRANSFERS - ONE STREAM ONLY */
		/*************************************************************/
 		timer3.StartCounter();
		for(int k = 0; k < nreps; k++)
		{
			d_a_matrix = 5;
			h_a_matrix = d_a_matrix;
		}
		streams[0].SynchronizeAll();
		printf("Kernel with memory transfers (1 stream):\t\t %.2f (expected %.2f)\n", timer3.GetCounter() / nreps, timing1+timing2);
		streams[0].DestroyStreams();

		/*****************************************************************/
		/* TIMING THE KERNEL WITHOUT MEMORY TRANSFERS - MULTIPLE STREAMS */
		/*****************************************************************/
		streams[0].InitStreams(nstreams);
 		timer4.StartCounter();
		for(int k = 0; k < nreps; k++)
		{
			// asynchronously launch nstreams kernels, each operating on its own portion of data
			for(int i = 0; i < nstreams; i++)
			{ 
				streams[0].SetStream(i);
				d_a_matrix(Range(i*n/nstreams,(i+1)*n/nstreams-1)) = 5;
			}
		}
		streams[0].SynchronizeAll();
		printf("Kernel without memory transfers (%d streams): \t\t %.2f \n", nstreams, timer4.GetCounter() / nreps);

		/****************************************************************************/
		/* TIMING THE KERNEL WITH MEMORY TRANSFERS - MULTIPLE STREAMS - APPROACH #1 */
		/****************************************************************************/
		h_a_matrix = SetZeros<int>(h_a_matrix.GetRows(),h_a_matrix.GetColumns());

		streams[0].InitStreams(nstreams);
 		timer5.StartCounter();
		for(int k = 0; k < nreps; k++)
		{
			// asynchronously launch nstreams kernels, each operating on its own portion of data
			for(int i = 0; i < nstreams; i++)
			{ 
				streams[0].SetStream(i);
				d_a_matrix(Range(i*n/nstreams,(i+1)*n/nstreams-1)) = 5;
				h_a_matrix(Range(i*n/nstreams,(i+1)*n/nstreams-1))=d_a_matrix(Range(i*n/nstreams,(i+1)*n/nstreams-1));
			}
		}
		streams[0].SynchronizeAll();
		printf("Kernel with memory transfers (%d streams) - approach #1: %.2f \n", nstreams, timer5.GetCounter() / nreps);

		/****************************************************************************/
		/* TIMING THE KERNEL WITH MEMORY TRANSFERS - MULTIPLE STREAMS - APPROACH #2 */
		/****************************************************************************/
		streams[0].InitStreams(nstreams);
 		timer6.StartCounter();
		for(int k = 0; k < nreps; k++)
		{
			// asynchronously launch nstreams kernels, each operating on its own portion of data
			for(int i = 0; i < nstreams; i++)
			{ 
				streams[0].SetStream(i);
				d_a_matrix(Range(i*n/nstreams,(i+1)*n/nstreams-1)) = 5;
			}
			// asynchronously launch nstreams memcopies.  Note that memcopy in stream x will only
			//   commence executing when all previous CUDA calls in stream x have completed
			for(int i = 0; i < nstreams; i++) {
				streams[0].SetStream(i);
				h_a_matrix(Range(i*n/nstreams,(i+1)*n/nstreams-1))=d_a_matrix(Range(i*n/nstreams,(i+1)*n/nstreams-1));
			}
		}
		streams[0].SynchronizeAll();
		printf("Kernel with memory transfers (%d streams) - approach #2: %.2f \n", nstreams, timer6.GetCounter() / nreps);

	}
	
	cudaDeviceReset();

	getch();

}
