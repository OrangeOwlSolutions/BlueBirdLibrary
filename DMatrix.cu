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


#include "BB.h"
namespace BB { extern dim3 dimBlock; }
#include "DExceptions.cuh"
#include "Dstreams.cuh"					// Needed for async-copies involving pinned memory
namespace BB { extern DStreams streams[MAX_NUM_GPUs]; }
//#include "Dmatrix.cuh"
//#include "DMatrixExpressionKernels.cuh"
#include "Hmatrix.h"

	// Eliminare
#include <iostream>

using namespace std;
using namespace BB;

#include "DExceptions.cuh"				// Needed for CudaSafeCall
#include "HExceptions.h"						// Needed for GenericError
#include "Expression.cuh"				// Needed for Dmatrix = const
//#include "Scalar.cuh"

//#include "ComplexTypes.cuh"					// Needed for complex types and conversions between types
//////#include "Constants.h"					// Needed for PINNED etc.
////#include "Macros.h"							// Needed for IDX
////#include "CudaMacros.cuh"					// Needed for CudaSafeCall
//#include "Dstreams[p2p.active_GPU].cuh"					// Needed for async-copies involving pinned memory
//extern BB::DStreams streams;
//#include "Constants.h"				// Needed for dimBlock
//extern dim3 dimBlock;
////#include "SubIndicesAccessHandling.h"
////#include "CudaSubMatrixExpression.cuh"	
////#include "Scalar.cuh"	
////
//#include "ComplexTypes.cuh"					// Needed for complex types and conversions between types
////#include "Constants.h"					// Needed for PINNED etc.
//#include "HExceptions.h"						// Needed for GenericError
//#include "Macros.h"							// Needed for IDX
//#include "SubIndicesAccessHandling.h"
////#include "CudaExpressionKernels.cuh"
////#include "Expression.cuh"
//#include "CudaSubMatrixExpression.cuh"	
//#include "Scalar.cuh"	
//#include "Constants.h"				// Needed for BLOCKSIZE

//// --- CUDA libraries
#include <cuda.h>		  // Needed for GPU2CPU 
#include <cuda_runtime.h> // Needed for float2_ and double2_

//int iDivUp(const int a, const int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }
//extern	int iDivUp(const int, const int);

/************************************************/
/* KERNEL FUNCTIONS & FUNCTIONS CALLING KERNELS */ 
/************************************************/

// --- 1D evaluation kernel function - Hmatrix = Hmatrix
template <class T1, class T2>
__global__ void evaluation_matrix(T1* data_, T2 ob, int NumElements)
{
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < NumElements) data_[i] = ob[i];
}

// --- 1D evaluation kernel function - Hmatrix = Hmatrix
template<class T1, class T2>
void eval_matrix_wrap(T1* data_, T2 ob, int NumElements)
{
	dim3 dimGrid(iDivUp(NumElements,dimBlock.x*(BB::streams[p2p.active_GPU].GetNumStreams())));
	evaluation_matrix<<<dimGrid,dimBlock,0,streams[p2p.active_GPU].GetActiveStream()>>>(data_,ob,NumElements);
    CudaCheckError();
}

// Explicit instantiations
//template __global__ void evaluation_matrix(int*,int*,int);
//template __global__ void evaluation_matrix(int*,float*,int);
//template __global__ void evaluation_matrix(int*,double*,int);
//template __global__ void evaluation_matrix(float*,int*,int);
//template __global__ void evaluation_matrix(float*,float*,int);
//template __global__ void evaluation_matrix(float*,double*,int);
//template __global__ void evaluation_matrix(double*,int*,int);
//template __global__ void evaluation_matrix(double*,float*,int);
//template __global__ void evaluation_matrix(double*,double*,int);
//template __global__ void evaluation_matrix(BB::int2_*,int*,int);
//template __global__ void evaluation_matrix(BB::int2_*,float*,int);
//template __global__ void evaluation_matrix(BB::int2_*,double*,int);
//template __global__ void evaluation_matrix(BB::int2_*,BB::int2_*,int);
//template __global__ void evaluation_matrix(BB::int2_*,BB::float2_*,int);
//template __global__ void evaluation_matrix(BB::int2_*,BB::double2_*,int);
//template __global__ void evaluation_matrix(BB::float2_*,int*,int);
//template __global__ void evaluation_matrix(BB::float2_*,float*,int);
//template __global__ void evaluation_matrix(BB::float2_*,double*,int);
//template __global__ void evaluation_matrix(BB::float2_*,BB::int2_*,int);
//template __global__ void evaluation_matrix(BB::float2_*,BB::float2_*,int);
//template __global__ void evaluation_matrix(BB::float2_*,BB::double2_*,int);
//template __global__ void evaluation_matrix(BB::double2_*,int*,int);
//template __global__ void evaluation_matrix(BB::double2_*,float*,int);
//template __global__ void evaluation_matrix(BB::double2_*,double*,int);
//template __global__ void evaluation_matrix(BB::double2_*,BB::int2_*,int);
//template __global__ void evaluation_matrix(BB::double2_*,BB::float2_*,int);
//template __global__ void evaluation_matrix(BB::double2_*,BB::double2_*,int);

//template void eval_matrix_wrap(int*,int*,int);
//template void eval_matrix_wrap(int*,float*,int);
//template void eval_matrix_wrap(int*,double*,int);
//template void eval_matrix_wrap(float*,int*,int);
//template void eval_matrix_wrap(float*,float*,int);
//template void eval_matrix_wrap(float*,double*,int);
//template void eval_matrix_wrap(double*,int*,int);
//template void eval_matrix_wrap(double*,float*,int);
//template void eval_matrix_wrap(double*,double*,int);
//template void eval_matrix_wrap(BB::int2_*,int*,int);
//template void eval_matrix_wrap(BB::int2_*,float*,int);
//template void eval_matrix_wrap(BB::int2_*,double*,int);
//template void eval_matrix_wrap(BB::int2_*,BB::int2_*,int);
//template void eval_matrix_wrap(BB::int2_*,BB::float2_*,int);
//template void eval_matrix_wrap(BB::int2_*,BB::double2_*,int);
//template void eval_matrix_wrap(BB::float2_*,int*,int);
//template void eval_matrix_wrap(BB::float2_*,float*,int);
//template void eval_matrix_wrap(BB::float2_*,double*,int);
//template void eval_matrix_wrap(BB::float2_*,BB::int2_*,int);
//template void eval_matrix_wrap(BB::float2_*,BB::float2_*,int);
//template void eval_matrix_wrap(BB::float2_*,BB::double2_*,int);
//template void eval_matrix_wrap(BB::double2_*,int*,int);
//template void eval_matrix_wrap(BB::double2_*,float*,int);
//template void eval_matrix_wrap(BB::double2_*,double*,int);
//template void eval_matrix_wrap(BB::double2_*,BB::int2_*,int);
//template void eval_matrix_wrap(BB::double2_*,BB::float2_*,int);
//template void eval_matrix_wrap(BB::double2_*,BB::double2_*,int);

template<class T1, class T2> void evaluation_matrix_function(T1* data, T2 ob, int NumElements)	  { eval_matrix_wrap(data,ob,NumElements); }

// Explicit instantiations
template void evaluation_matrix_function(int*,int*,int);
template void evaluation_matrix_function(int*,float*,int);
template void evaluation_matrix_function(int*,double*,int);
template void evaluation_matrix_function(float*,int*,int);
template void evaluation_matrix_function(float*,float*,int);
template void evaluation_matrix_function(float*,double*,int);
template void evaluation_matrix_function(double*,int*,int);
template void evaluation_matrix_function(double*,float*,int);
template void evaluation_matrix_function(double*,double*,int);
template void evaluation_matrix_function(BB::int2_*,int*,int);
template void evaluation_matrix_function(BB::int2_*,float*,int);
template void evaluation_matrix_function(BB::int2_*,double*,int);
template void evaluation_matrix_function(BB::int2_*,BB::int2_*,int);
template void evaluation_matrix_function(BB::int2_*,BB::float2_*,int);
template void evaluation_matrix_function(BB::int2_*,BB::double2_*,int);
template void evaluation_matrix_function(BB::float2_*,int*,int);
template void evaluation_matrix_function(BB::float2_*,float*,int);
template void evaluation_matrix_function(BB::float2_*,double*,int);
template void evaluation_matrix_function(BB::float2_*,BB::int2_*,int);
template void evaluation_matrix_function(BB::float2_*,BB::float2_*,int);
template void evaluation_matrix_function(BB::float2_*,BB::double2_*,int);
template void evaluation_matrix_function(BB::double2_*,int*,int);
template void evaluation_matrix_function(BB::double2_*,float*,int);
template void evaluation_matrix_function(BB::double2_*,double*,int);
template void evaluation_matrix_function(BB::double2_*,BB::int2_*,int);
template void evaluation_matrix_function(BB::double2_*,BB::float2_*,int);
template void evaluation_matrix_function(BB::double2_*,BB::double2_*,int);

/*******************************/
/* CONSTRUCTORS AND DESTRUCTOR */
/*******************************/

// --- Constructor
template <class OutType>
BB::Dmatrix<OutType>::Dmatrix(const int Rows,const int Columns)
{ 
	Rows_			= Rows;
	Columns_		= Columns;
	which_device_	= p2p.active_GPU;
	CudaSafeCall(cudaMalloc((void **)&data_, Rows_*Columns_*sizeof(OutType))); 
}
 
// --- Constructor & Inizializer on Dmatrix
template <class OutType>
BB::Dmatrix<OutType>::Dmatrix(const Dmatrix &ob)
{
	Rows_			= ob.GetRows();
	Columns_		= ob.GetColumns();
	which_device_	= p2p.active_GPU;
	CudaSafeCall(cudaMalloc((void **)&data_, Rows_*Columns_*sizeof(OutType)));
	if (which_device_ == ob.which_device_)
		evaluation_matrix_function(data_,ob.data_,GetNumElements());
	else
		cudaMemcpy(data_,ob.data_,Rows_*Columns_*sizeof(OutType),cudaMemcpyDefault);
}

// --- Constructor & Inizializer on Hmatrix
template <class OutType>
BB::Dmatrix<OutType>::Dmatrix(const BB::Hmatrix<OutType> &ob)
{
	Rows_			= ob.GetRows();
	Columns_		= ob.GetColumns();
	which_device_	= p2p.active_GPU;
	CudaSafeCall(cudaMalloc((void **)&data_, Rows_*Columns_*sizeof(OutType)));
	CudaSafeCall(cudaMemcpy(data_,ob.GetDataPointer(),GetNumElements()*sizeof(OutType),cudaMemcpyHostToDevice));
}

// --- Destructor
template <class OutType>
BB::Dmatrix<OutType>::~Dmatrix() { CudaSafeCall(cudaFree(data_)); }

/*****************/
/* QUERY METHODS */
/*****************/
// --- Gets the number of matrix rows
template <class OutType>
__host__ __device__ int BB::Dmatrix<OutType>::GetRows() const { return Rows_; };
// --- Gets the number of matrix columns
template <class OutType>
__host__ __device__ int BB::Dmatrix<OutType>::GetColumns() const { return Columns_; };
// --- Gets the number of matrix elements
template <class OutType>
__host__ __device__ int BB::Dmatrix<OutType>::GetNumElements() const {return Rows_*Columns_; } ;
// --- Checks if matrix is a vector
template <class OutType>
__host__ __device__ bool BB::Dmatrix<OutType>::IsVector() const { return (GetRows() == 1 || GetColumns() == 1) ? true : false; };
// --- Get data pointer
template <class OutType>
__host__ __device__ OutType		  * BB::Dmatrix<OutType>::GetDataPointer()		{ return data_; }
template <class OutType>
__host__ __device__ OutType const * BB::Dmatrix<OutType>::GetDataPointer() const { return data_; }

/***********/
/* RESHAPE */
/***********/
// --- Reshape matrix (simply changes row and column size, no memory movement)
template <class OutType>
__host__ void BB::Dmatrix<OutType>::Resize(int NewRows, int NewColumns)
{
	if (NewRows<=0 || NewColumns <=0 || (NewRows*NewColumns!=GetNumElements())) 
		{ char* str0 = "******************************\n"; 
	      char* str1 = "* Invalid GPU Hmatrix Resize *\n"; 
		  char* str2 = "Hmatrix size: "; 
		  char* str3 = "Reshape size: "; 
		  char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+15*sizeof(char)); 
		  sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,NewRows,NewColumns); 
		  throw  GenericError(catString,__FILE__,__LINE__); 
		} else {
			Rows_ = NewRows;
			Columns_ = NewColumns; }
}

/********************/
/* ACCESS OPERATORS */
/********************/
// --- Access from CPU with single index - return value only
//template <class OutType>
//__host__ OutType BB::Dmatrix<OutType>::operator()(const int i)	const { OutType d; CudaSafeCall(cudaMemcpy(&d,data_+i,sizeof(OutType),cudaMemcpyDeviceToHost)); return d; }
// --- Access from CPU with double index - return value only
template <class OutType>
__host__ OutType BB::Dmatrix<OutType>::operator()(const int i, const int j) const { OutType d; CudaSafeCall(cudaMemcpy(&d,data_+(i*Columns_+j),sizeof(OutType),cudaMemcpyDeviceToHost)); return d; }

/*******************************************/
/* SET INDIVIDUAL ELEMENTS OF GPU MATRICES */
/*******************************************/
template <class OutType>
__host__ void BB::Dmatrix<OutType>::Set(int i, OutType b) { CudaSafeCall(cudaMemcpy(&data_[i],&b,sizeof(OutType),cudaMemcpyHostToDevice)); }

template <class OutType>
__host__ void BB::Dmatrix<OutType>::Set(int i, int j, OutType b) { CudaSafeCall(cudaMemcpy(&data_[i*Columns_+j],&b,sizeof(OutType),cudaMemcpyHostToDevice)); }

/**********************/
/* ASSIGNMENT CPU2GPU */
/**********************/
template <class OutType>
const BB::Dmatrix<OutType>& BB::Dmatrix<OutType>::operator=(const BB::Hmatrix<OutType> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) { 
		if (ob.IsPinned()) 
			CudaSafeCall(cudaMemcpyAsync(data_,ob.GetDataPointer(),GetNumElements()*sizeof(OutType),cudaMemcpyHostToDevice,streams[p2p.active_GPU].GetActiveStream()));
		else 
			CudaSafeCall(cudaMemcpy(data_,ob.GetDataPointer(),GetNumElements()*sizeof(OutType),cudaMemcpyHostToDevice)); }
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

/***********************/
/* ASSIGNMENTS GPU2GPU */
/***********************/
// int = int
const BB::Dmatrix<int>& BB::Dmatrix<int>::operator=(const BB::Dmatrix<int>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) 
		if (which_device_ == ob.which_device_)
			evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
		else
			cudaMemcpy(data_,ob.data_,Rows_*Columns_*sizeof(int),cudaMemcpyDefault);
	else 
	{  	
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// int = float
const BB::Dmatrix<int>& BB::Dmatrix<int>::operator=(const BB::Dmatrix<float>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// int = double
const BB::Dmatrix<int>& BB::Dmatrix<int>::operator=(const BB::Dmatrix<double>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// float = int
const BB::Dmatrix<float>& BB::Dmatrix<float>::operator=(const BB::Dmatrix<int>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// float = float
const BB::Dmatrix<float>& BB::Dmatrix<float>::operator=(const BB::Dmatrix<float>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) 
		if (which_device_ == ob.which_device_)
			evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
		else
			cudaMemcpy(data_,ob.data_,Rows_*Columns_*sizeof(float),cudaMemcpyDefault);
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// float = double
const BB::Dmatrix<float>& BB::Dmatrix<float>::operator=(const BB::Dmatrix<double>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// double = int
const BB::Dmatrix<double>& BB::Dmatrix<double>::operator=(const BB::Dmatrix<int>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// double = float
const BB::Dmatrix<double>& BB::Dmatrix<double>::operator=(const BB::Dmatrix<float>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// double = double
const BB::Dmatrix<double>& BB::Dmatrix<double>::operator=(const BB::Dmatrix<double>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) 
		if (which_device_ == ob.which_device_)
			evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
		else
			cudaMemcpy(data_,ob.data_,Rows_*Columns_*sizeof(double),cudaMemcpyDefault);
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// int2_ = int
const BB::Dmatrix<BB::int2_>& BB::Dmatrix<BB::int2_>::operator=(const BB::Dmatrix<int>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// int2_ = float
const BB::Dmatrix<BB::int2_>& BB::Dmatrix<BB::int2_>::operator=(const BB::Dmatrix<float>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// int2_ = double
const BB::Dmatrix<BB::int2_>& BB::Dmatrix<BB::int2_>::operator=(const BB::Dmatrix<double>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// int2_ = int2_
const BB::Dmatrix<BB::int2_>& BB::Dmatrix<BB::int2_>::operator=(const BB::Dmatrix<BB::int2_>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) 
		if (which_device_ == ob.which_device_)
			evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
		else
			cudaMemcpy(data_,ob.data_,Rows_*Columns_*sizeof(int2_),cudaMemcpyDefault);
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// int2_ = float2_
const BB::Dmatrix<BB::int2_>& BB::Dmatrix<BB::int2_>::operator=(const BB::Dmatrix<BB::float2_>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// int2_ = double2_
const BB::Dmatrix<BB::int2_>& BB::Dmatrix<BB::int2_>::operator=(const BB::Dmatrix<BB::double2_>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// float2_ = int
const BB::Dmatrix<BB::float2_>& BB::Dmatrix<BB::float2_>::operator=(const BB::Dmatrix<int>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// float2_ = float
const BB::Dmatrix<BB::float2_>& BB::Dmatrix<BB::float2_>::operator=(const BB::Dmatrix<float>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// float2_ = double
const BB::Dmatrix<BB::float2_>& BB::Dmatrix<BB::float2_>::operator=(const BB::Dmatrix<double>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// float2_ = int2_
const BB::Dmatrix<BB::float2_>& BB::Dmatrix<BB::float2_>::operator=(const BB::Dmatrix<BB::int2_>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// float2_ = float2_
const BB::Dmatrix<BB::float2_>& BB::Dmatrix<BB::float2_>::operator=(const BB::Dmatrix<BB::float2_>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) 
		if (which_device_ == ob.which_device_)
			evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
		else
			cudaMemcpy(data_,ob.data_,Rows_*Columns_*sizeof(float2_),cudaMemcpyDefault);
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// float2_ = double2_
const BB::Dmatrix<BB::float2_>& BB::Dmatrix<BB::float2_>::operator=(const BB::Dmatrix<BB::double2_>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// double2_ = int
const BB::Dmatrix<BB::double2_>& BB::Dmatrix<BB::double2_>::operator=(const BB::Dmatrix<int>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// double2_ = float
const BB::Dmatrix<BB::double2_>& BB::Dmatrix<BB::double2_>::operator=(const BB::Dmatrix<float>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// double2_ = double
const BB::Dmatrix<BB::double2_>& BB::Dmatrix<BB::double2_>::operator=(const BB::Dmatrix<double>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// double2_ = int2_
const BB::Dmatrix<BB::double2_>& BB::Dmatrix<BB::double2_>::operator=(const BB::Dmatrix<BB::int2_>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// double2_ = float2_
const BB::Dmatrix<BB::double2_>& BB::Dmatrix<BB::double2_>::operator=(const BB::Dmatrix<BB::float2_>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// double2_ = double2_
const BB::Dmatrix<BB::double2_>& BB::Dmatrix<BB::double2_>::operator=(const BB::Dmatrix<BB::double2_>& ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) 
		if (which_device_ == ob.which_device_)
			evaluation_matrix_function(data_,ob.GetDataPointer(),GetNumElements());
		else
			cudaMemcpy(data_,ob.data_,Rows_*Columns_*sizeof(double2_),cudaMemcpyDefault);

	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// Hmatrix = constant
//template <class OutType>
//const BB::Dmatrix<OutType>& BB::Dmatrix<OutType>::operator=(const OutType c)
//{
//	*this = BB::Expr<BB::Scalar<OutType>,OutType>(BB::Scalar<OutType>(c),Rows_,Columns_,ISDEVICE);
//	return *this;
//}
//template const BB::Dmatrix<int>& BB::Dmatrix<int>::operator=(const int c);
//template const BB::Dmatrix<float>& BB::Dmatrix<float>::operator=(const float c);
//template const BB::Dmatrix<double>& BB::Dmatrix<double>::operator=(const double c);
//template const BB::Dmatrix<BB::int2_>& BB::Dmatrix<BB::int2_>::operator=(const BB::int2_ c);
//template const BB::Dmatrix<BB::float2_>& BB::Dmatrix<BB::float2_>::operator=(const BB::float2_ c);
//template const BB::Dmatrix<BB::double2_>& BB::Dmatrix<BB::double2_>::operator=(const BB::double2_ c);


// Dmatrix = Sub-Expression
//template <class OutType>
//const BB::Dmatrix<OutType>& BB::Dmatrix<OutType>::operator=(Expr<CudaSubMatrixExprRow<OutType*,OutType>,OutType> e)
//{   
//	if((e.GetRows() == Rows_) && (e.GetColumns() == Columns_)) 
//		for (int i=0; i<GetNumElements(); i++) data_[i] = e[i]; 
//	else 
//	{  
//		char* str0 = "**********************************************\n";
//		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
//		char* str2 = "Left operand size: ";
//		char* str3 = "Right operand size: ";
//		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
//		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,e.GetRows(),e.GetColumns());
//		throw  GenericError(catString,__FILE__,__LINE__);
//	} 
//	return *this;
//}

/*******************/
/* SUB-EXPRESSIONS */
/*******************/

// --- SubExpressions - Range
template <class OutType>		
BB::Expr<BB::SubMatrixExprRow<OutType*,OutType>,OutType> BB::Dmatrix<OutType>::operator()(BB::Range range)
{	if((range.GetStart() >= 0) && (range.GetNumElements() > 0) && (range.GetEnd() < Rows_*Columns_)) {	
		typedef BB::SubMatrixExprRow<OutType*,OutType> SExpr; 
		//return BB::Expr<SExpr,OutType>(SExpr(data_+range.GetStart(),0),1,range.GetNumElements(),ISDEVICE); 
		return BB::Expr<SExpr,OutType>(SExpr(data_,range.GetStart(),0),1,range.GetNumElements(),ISDEVICE); 
	} else	{	char* str0 = "****************************************\n"; 
				char* str1 = "* Invalid GPU SubMatrix access attempt *\n"; 
				char* str2 = "Dmatrix size: "; 
				char* str3 = "SubMatrix indices (a,b,c,d): "; 
				char* str4 = "SubMatrix size: "; 
				char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
				sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,0,0,range.GetStart(),range.GetEnd(),str4,1,range.GetNumElements()); 
				throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - index Range
template <class OutType>		
BB::Expr<BB::SubMatrixExprRow<OutType*,OutType>,OutType> BB::Dmatrix<OutType>::operator()(int index, BB::Range range2)
{	if((index >= 0) && (index < Rows_) && (range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Columns_)) {	
		typedef SubMatrixExprRow<OutType*,OutType> SExpr; 
		return Expr<SExpr,OutType>(SExpr(data_,index*Columns_+range2.GetStart(),0),1,range2.GetNumElements(),ISDEVICE); 
	} else	{	char* str0 = "****************************************\n"; 
				char* str1 = "* Invalid GPU SubMatrix access attempt *\n"; 
				char* str2 = "Dmatrix size: "; 
				char* str3 = "SubMatrix indices (a,b,c,d): "; 
				char* str4 = "SubMatrix size: "; 
				char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
				sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,index,index,range2.GetStart(),range2.GetEnd(),str4,1,range2.GetNumElements()); 
				throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - int RangeStep
template <class OutType>		
BB::Expr<BB::SubMatrixExprRowStep<OutType*,OutType>,OutType> BB::Dmatrix<OutType>::operator()(int index, BB::RangeStep range2)
{	if((index >= 0) && (index < Rows_) && (range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Columns_)) {	
		typedef BB::SubMatrixExprRowStep<OutType*,OutType> SExpr;
		return BB::Expr<SExpr,OutType>(SExpr(data_,index*Columns_+range2.GetStart(),range2.GetStep()),1,range2.GetNumElements(),ISDEVICE); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid GPU SubMatrix access attempt *\n"; 
			 char* str2 = "Hmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,index,index,range2.GetStart(),range2.GetEnd(),str4,1,range2.GetNumElements()); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - int Span
template <class OutType>		
BB::Expr<BB::SubMatrixExprRow<OutType*,OutType>,OutType> BB::Dmatrix<OutType>::operator()(int index, BB::SpanClass span2)
{	if((index >= 0) && (index < Rows_)) {	
		typedef BB::SubMatrixExprRow<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,index*Columns_,0),1,Columns_,ISDEVICE); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid GPU SubMatrix access attempt *\n"; 
			 char* str2 = "Hmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,index,index,0,Columns_-1,str4,1,Columns_); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - Range index
template <class OutType>		
BB::Expr<BB::SubMatrixExprColumn<OutType*,OutType>,OutType> BB::Dmatrix<OutType>::operator()(BB::Range range1, int index)
{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_) && 
	   (index >= 0) && (index < Columns_)) {	
		typedef BB::SubMatrixExprColumn<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,range1.GetStart()*Columns_+index,Columns_,0),range1.GetNumElements(),1,ISDEVICE); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid GPU SubMatrix access attempt *\n"; 
			 char* str2 = "Dmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),index,index,str4,range1.GetNumElements(),1); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}
			
// --- SubExpressions - RangeStep int
template <class OutType>		
BB::Expr<BB::SubMatrixExprColumnStep<OutType*,OutType>,OutType> BB::Dmatrix<OutType>::operator()(BB::RangeStep range1, int index)
{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_) && 
	   (index >= 0) && (index < Columns_)) {	
		typedef BB::SubMatrixExprColumnStep<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,range1.GetStart()*Columns_+index,Columns_,range1.GetStep(),0),range1.GetNumElements(),1,ISDEVICE); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid GPU SubMatrix access attempt *\n"; 
			 char* str2 = "Dmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),index,index,str4,range1.GetNumElements(),1); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - Span int
template <class OutType>		
BB::Expr<BB::SubMatrixExprColumn<OutType*,OutType>,OutType> BB::Dmatrix<OutType>::operator()(BB::SpanClass span1, int index)
{	if((index >= 0) && (index < Columns_)) {	
		typedef BB::SubMatrixExprColumn<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,index,Columns_,0),Rows_,1,ISDEVICE); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid GPU SubMatrix access attempt *\n"; 
			 char* str2 = "Hmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,0,Rows_-1,index,index,str4,Rows_,1); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

////// --- SubExpressions - Span Span
//////template <class OutType>		
//////Expr<CudaSubMatrixExpr<OutType*,OutType>,OutType> Dmatrix<OutType>::operator()(SpanClass span1, SpanClass span2)	
//////{	typedef CudaSubMatrixExpr<OutType*,OutType> SExpr; 
//////	return Expr<SExpr,OutType>(SExpr(data_,Rows_,Columns_,Rows_,Columns_,0,1,0,1),Rows_,Columns_,ISDEVICE); 
//////}
////
// --- SubExpressions - Range Range
template <class OutType>		
BB::Expr<BB::SubMatrixExprColumnRow<OutType*,OutType>,OutType> BB::Dmatrix<OutType>::operator()(BB::Range range1, BB::Range range2)
{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_) && 
	   (range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Rows_)) {	
		typedef BB::SubMatrixExprColumnRow<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,Rows_,Columns_,range1.GetNumElements(),range2.GetNumElements(),range1.GetStart(),range2.GetStart()),range1.GetNumElements(),range2.GetNumElements(),ISDEVICE); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid GPU SubMatrix access attempt *\n"; 
			 char* str2 = "Hmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),range1.GetEnd(),range2.GetEnd(),str4,range1.GetNumElements(),range2.GetNumElements()); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - Range RangeStep
template <class OutType>		
BB::Expr<BB::SubMatrixExprRowColumnStep<OutType*,OutType>,OutType> BB::Dmatrix<OutType>::operator()(BB::Range range1, BB::RangeStep range2)
{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_) && 
	   (range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Columns_)) {	
		typedef BB::SubMatrixExprRowColumnStep<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,Rows_,Columns_,range1.GetNumElements(),range2.GetNumElements(),range1.GetStart(),range2.GetStart(),range2.GetStep()),range1.GetNumElements(),range2.GetNumElements(),ISDEVICE); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid GPU SubMatrix access attempt *\n"; 
			 char* str2 = "Dmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),range1.GetEnd(),range2.GetEnd(),str4,range1.GetNumElements(),range2.GetNumElements()); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - RangeStep Range
template <class OutType>		
BB::Expr<BB::SubMatrixExprRowStepColumn<OutType*,OutType>,OutType> BB::Dmatrix<OutType>::operator()(BB::RangeStep range1, BB::Range range2)
{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_) && 
	   (range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Columns_)) {	
		typedef BB::SubMatrixExprRowStepColumn<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,Rows_,Columns_,range1.GetNumElements(),range2.GetNumElements(),range1.GetStart(),range1.GetStep(),range2.GetStart()),range1.GetNumElements(),range2.GetNumElements(),ISDEVICE); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid GPU SubMatrix access attempt *\n"; 
			 char* str2 = "Dmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),range1.GetEnd(),range2.GetEnd(),str4,range1.GetNumElements(),range2.GetNumElements()); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - RangeStep RangeStep
template <class OutType>		
BB::Expr<BB::SubMatrixExpr<OutType*,OutType>,OutType> BB::Dmatrix<OutType>::operator()(BB::RangeStep range1, BB::RangeStep range2)
{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_) && 
	   (range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Columns_)) {	
		typedef BB::SubMatrixExpr<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,Rows_,Columns_,range1.GetNumElements(),range2.GetNumElements(),range1.GetStart(),range1.GetStep(),range2.GetStart(),range2.GetStep()),
				range1.GetNumElements(),range2.GetNumElements(),ISDEVICE); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid GPU SubMatrix access attempt *\n"; 
			 char* str2 = "Dmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),range2.GetStart(),range2.GetEnd(),str4,range1.GetNumElements(),range2.GetNumElements()); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - Range Span
template <class OutType>		
BB::Expr<BB::SubMatrixExprColumnRow<OutType*,OutType>,OutType> BB::Dmatrix<OutType>::operator()(BB::Range range1, BB::SpanClass span2)
{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_)) {	
		typedef BB::SubMatrixExprColumnRow<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,Rows_,Columns_,range1.GetNumElements(),Columns_,range1.GetStart(),0),range1.GetNumElements(),Columns_,ISDEVICE); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid GPU SubMatrix access attempt *\n"; 
			 char* str2 = "Dmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),range1.GetEnd(),Columns_-1,str4,range1.GetNumElements(),Columns_-1); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - Span Range
template <class OutType>		
BB::Expr<BB::SubMatrixExprColumnRow<OutType*,OutType>,OutType> BB::Dmatrix<OutType>::operator()(BB::SpanClass span1, BB::Range range2)
{	if((range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Rows_)) {	
		typedef BB::SubMatrixExprColumnRow<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,Rows_,Columns_,Rows_,range2.GetNumElements(),0,range2.GetStart()),Rows_,range2.GetNumElements(),ISDEVICE); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid GPU SubMatrix access attempt *\n"; 
			 char* str2 = "Dmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,0,Rows_-1,Rows_-1,range2.GetEnd(),str4,Rows_,range2.GetNumElements()); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - Span RangeStep
template <class OutType>		
BB::Expr<BB::SubMatrixExprRowColumnStep<OutType*,OutType>,OutType> BB::Dmatrix<OutType>::operator()(BB::SpanClass span1, BB::RangeStep range2)
{	if((range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Columns_)) {	
		typedef BB::SubMatrixExprRowColumnStep<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,Rows_,Columns_,Rows_,range2.GetNumElements(),0,range2.GetStart(),range2.GetStep()),Rows_,range2.GetNumElements(),ISDEVICE); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid GPU SubMatrix access attempt *\n"; 
			 char* str2 = "Dmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,0,Rows_-1,Rows_-1,range2.GetEnd(),str4,Rows_,range2.GetNumElements()); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - RangeStep Span
template <class OutType>		
BB::Expr<BB::SubMatrixExprRowStepColumn<OutType*,OutType>,OutType> BB::Dmatrix<OutType>::operator()(BB::RangeStep range1, BB::SpanClass span2)
{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_)) {	
		typedef BB::SubMatrixExprRowStepColumn<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,Rows_,Columns_,range1.GetNumElements(),Columns_,range1.GetStart(),range1.GetStep(),0),range1.GetNumElements(),Columns_,ISDEVICE); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid GPU SubMatrix access attempt *\n"; 
			 char* str2 = "Dmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),range1.GetEnd(),Columns_-1,str4,range1.GetNumElements(),Columns_); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

////// --- SubExpressions - int Hmatrix
//////template <class OutType>
//////BB::Expr<BB::CudaSubMatrixExprRowPerm<OutType*,OutType>,OutType> BB::Dmatrix<OutType>::operator()(int index, BB::Dmatrix<int> indices2)
//////{	typedef BB::CudaSubMatrixExprRowPerm<OutType*,OutType> SExpr; 
//////	return	BB::Expr<SExpr,OutType>(SExpr(data_+index*Columns_,indices2,0),1,indices2.GetNumElements(),ISDEVICE); 
//////}

/******************/
/* MOVE SEMANTICS */
/******************/

// --- Move constructor
template <class OutType>
BB::Dmatrix<OutType>::Dmatrix(Dmatrix<OutType>&& other) throw() { 
	data_		= other.data_;
	Rows_		= other.Rows_;
	Columns_	= other.Columns_;
	other.Rows_ = other.Columns_ = 0; 
	other.data_ = nullptr; 
}

// --- Move assignment
template <class OutType>
BB::Dmatrix<OutType>& BB::Dmatrix<OutType>::operator=(BB::Dmatrix<OutType>&& other) throw() {
	using std::swap;
	swap(Rows_, other.Rows_);
	swap(Columns_, other.Columns_);
	swap(data_, other.data_); 
	return *this;
}

/***********************************************/
/* EXPLICIT INSTANTIATIONS OF THE MATRIX CLASS */
/***********************************************/
template class BB::Dmatrix<int>;
template class BB::Dmatrix<float>;
template class BB::Dmatrix<double>;
template class BB::Dmatrix<BB::int2_>;
template class BB::Dmatrix<BB::float2_>;
template class BB::Dmatrix<BB::double2_>;

/*******************************/
/* OVERLOAD OF THE << OPERATOR */
/*******************************/

// --- Overload of << for int Dmatrix
ostream  &operator << (ostream &output, const BB::Dmatrix<int> &v_d)
{
	BB::Hmatrix<int> v(v_d);
	for(int i=0; i<v.GetRows(); i++) {
		for(int j=0; j<v.GetColumns(); j++) {
			output << setw(prec_cout) << "Mat(" << i << "," << j << ") = " << v(i,j) << "\t"; }
		output << endl; }
	return output;
}

// --- Overload of << for float Dmatrix
ostream  &operator << (ostream &output, const BB::Dmatrix<float> &v_d)
{
	BB::Hmatrix<float> v(v_d);
	for(int i=0; i<v.GetRows(); i++) {
		for(int j=0; j<v.GetColumns(); j++) {
			output << setw(prec_cout) << "Mat(" << i << "," << j << ") = " << v(i,j) << "\t"; }
		output << endl; }
	return output;
}

// --- Overload of << for double Dmatrix
ostream  &operator << (ostream &output, const BB::Dmatrix<double> &v_d)
{
	BB::Hmatrix<double> v(v_d);
	for(int i=0; i<v.GetRows(); i++) {
		for(int j=0; j<v.GetColumns(); j++) {
			output << setw(prec_cout) << "Mat(" << i << "," << j << ") = " << v(i,j) << "\t"; }
		output << endl; }
	return output;
}

// --- Overload of << for int2_ Dmatrix
ostream  &operator << (ostream &output, const BB::Dmatrix<BB::int2_> &v_d)
{
	BB::Hmatrix<BB::int2_> v(v_d);
	for(int i=0; i<v.GetRows(); i++) {
		for(int j=0; j<v.GetColumns(); j++) {
			output << setw(prec_cout) << "Mat(" << i << "," << j << ") = " << v(i,j).c.x << "+ " << v(i,j).c.y << "*j \t"; }
		output << endl; }
	return output;
}

// --- Overload of << for float2_ Dmatrix
ostream  &operator << (ostream &output, const BB::Dmatrix<BB::float2_> &v_d)
{
	BB::Hmatrix<BB::float2_> v(v_d);
	for(int i=0; i<v.GetRows(); i++) {
		for(int j=0; j<v.GetColumns(); j++) {
			output << setw(prec_cout) << "Mat(" << i << "," << j << ") = " << v(i,j).c.x << "+ " << v(i,j).c.y << "*j \t"; }
		output << endl; }
	return output;
}

// --- Overload of << for double2_ Dmatrix
ostream  &operator << (ostream &output, const BB::Dmatrix<BB::double2_> &v_d)
{
	BB::Hmatrix<BB::double2_> v(v_d);
	for(int i=0; i<v.GetRows(); i++) {
		for(int j=0; j<v.GetColumns(); j++) {
			output << setw(prec_cout) << "Mat(" << i << "," << j << ") = " << v(i,j).c.x << "+ " << v(i,j).c.y << "*j \t"; }
		output << endl; }
	return output;
}

//////#include "FFT.h"
//////#include "Interpolation.h"
//////#include "DataPlotter.h"
////