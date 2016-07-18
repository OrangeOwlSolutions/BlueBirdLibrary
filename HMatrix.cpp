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
#include "Hmatrix.h"

using namespace std;
using namespace BB;

#include "Dstreams.cuh"					// Needed for async-copies involving pinned memory
namespace BB { extern DStreams streams[MAX_NUM_GPUs]; }

//#include <iomanip>							// Needed for setw
//
#include "DExceptions.cuh"					// Needed for CudaSafeCall
#include "Constants.h"							// Needed for precision in setw, PINNED, ISHOST, etc.
#include "Macros.h"								// Needed for IDX
#include "HExceptions.h"							// Needed for GenericError
#include "Scalar.cuh"						// Needed for Hmatrix = const assignment
#include "SubMatrixExpression.h"
#include "Expression.cuh"					// Needed for Hmatrix = const assignment
//#include "Dstreams[p2p.active_GPU].cuh"					// Needed for async-copies involving pinned memory
//namespace BB { extern DStreams streams; }
//
//#include "ComplexTypes.cuh"					// Needed for complex types and conversions between types
//#include "SubIndicesAccessHandling.h"
////#include "Expression.cuh"
////#include "CudaSubMatrixExpression.cuh"	
//#include "Scalar.cuh"	

// --- CUDA libraries
#include <cuda.h>		  // Needed for GPU2CPU 
#include <cuda_runtime.h> // Needed for float2_ and double2_

#include <iostream>

/*******************************/
/* CONSTRUCTORS AND DESTRUCTOR */
/*******************************/

// --- Constructor
template <class OutType>
BB::Hmatrix<OutType>::Hmatrix(const int Rows, const int Columns, const int IsPinned = 0) { 
	Rows_		= Rows;
	Columns_	= Columns;
	IsPinned_	= IsPinned;
	if (IsPinned_)	{ CudaSafeCall(cudaHostAlloc((void**)&data_,Rows_*Columns_*sizeof(OutType),cudaHostAllocDefault)); } 
	else			{ data_ = new OutType[Rows_*Columns_]; } 
}

// --- Constructor & Inizializer on Hmatrix
template <class OutType>
BB::Hmatrix<OutType>::Hmatrix(const Hmatrix &ob, const int IsPinned = 0) {
	Rows_		= ob.Rows_;
	Columns_	= ob.Columns_;
	IsPinned_	= IsPinned;
	if (IsPinned_)	{ CudaSafeCall(cudaHostAlloc((void**)&data_,ob.GetNumElements()*sizeof(OutType),cudaHostAllocDefault)); } 
	else			{ data_ = new OutType[ob.GetNumElements()]; }
	// Sostituire con memcpy ???
	for (int i=0; i<GetNumElements(); i++) data_[i] = ob.data_[i];
}

// --- Constructor & Inizializer on Dmatrix
template <class OutType>
BB::Hmatrix<OutType>::Hmatrix(const BB::Dmatrix<OutType> &ob, const int IsPinned = 0)
{
	Rows_		= ob.GetRows();
	Columns_	= ob.GetColumns();
	IsPinned_	= IsPinned;
	if (IsPinned_) { CudaSafeCall(cudaHostAlloc((void**)&data_,ob.GetNumElements()*sizeof(OutType),cudaHostAllocDefault));} 
	else { data_ = new OutType[ob.GetNumElements()]; }
	CudaSafeCall(cudaMemcpy(data_,ob.GetDataPointer(),ob.GetNumElements()*sizeof(OutType),cudaMemcpyDeviceToHost));
}

// --- Destructor
template <class OutType>
BB::Hmatrix<OutType>::~Hmatrix() { if (IsPinned_) CudaSafeCall(cudaFreeHost(data_)); else { delete [] data_; } }

/*****************/
/* QUERY METHODS */
/*****************/
// --- Gets the number of matrix rows
template <class OutType>
int BB::Hmatrix<OutType>::GetRows() const { return Rows_; };
// --- Gets the number of matrix columns
template <class OutType>
int BB::Hmatrix<OutType>::GetColumns() const { return Columns_; };
// --- Gets the number of matrix elements
template <class OutType>
int BB::Hmatrix<OutType>::GetNumElements() const {return Rows_*Columns_; } ;
// --- Returns if the memory has been allocated as PINNED memory
template <class OutType>
bool BB::Hmatrix<OutType>::IsPinned() const { return (IsPinned_ == PINNED) ? true : false; };
// --- Checks if matrix is a vector
template <class OutType>
bool BB::Hmatrix<OutType>::IsVector() const { return (GetRows() == 1 || GetColumns() == 1) ? true : false; };
// --- Get data pointer
template <class OutType>
OutType		  * BB::Hmatrix<OutType>::GetDataPointer()		{ return data_; }
template <class OutType>
OutType const * BB::Hmatrix<OutType>::GetDataPointer() const { return data_; }

/***********/
/* RESHAPE */
/***********/
// --- Reshape matrix (simply changes row and column size, no memory movement)
template <class OutType>
void BB::Hmatrix<OutType>::Resize(int NewRows, int NewColumns)
{
	if (NewRows<=0 || NewColumns <=0 || (NewRows*NewColumns!=GetNumElements())) 
		{ char* str0 = "******************************\n"; 
	      char* str1 = "* Invalid CPU Hmatrix Resize *\n"; 
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
// --- Access operators 
template <class OutType>
OutType & BB::Hmatrix<OutType>::operator()(const int i)						{ return data_[i]; }
template <class OutType>
OutType   BB::Hmatrix<OutType>::operator()(const int i)			  const		{ return data_[i]; }
template <class OutType>
OutType & BB::Hmatrix<OutType>::operator()(const int i, const int j)			{ return data_[IDX2R(i,j,GetColumns())]; }
template <class OutType>
OutType   BB::Hmatrix<OutType>::operator()(const int i, const int j) const		{ return data_[IDX2R(i,j,GetColumns())]; }
template <class OutType>
OutType & BB::Hmatrix<OutType>::operator[](const int i)						{ return data_[i]; }
template <class OutType>
OutType   BB::Hmatrix<OutType>::operator[](const int i) const					{ return data_[i]; }

/**********************/
/* ASSIGNMENT GPU2CPU */
/**********************/
template <class OutType>
const BB::Hmatrix<OutType>& BB::Hmatrix<OutType>::operator=(const BB::Dmatrix<OutType>& ob)
{
	if ((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) 
		{ if (IsPinned_) CudaSafeCall(cudaMemcpyAsync(data_,ob.GetDataPointer(),ob.GetNumElements()*sizeof(OutType),cudaMemcpyDeviceToHost,streams[p2p.active_GPU].GetActiveStream()));
		  else CudaSafeCall(cudaMemcpy(data_,ob.GetDataPointer(),ob.GetNumElements()*sizeof(OutType),cudaMemcpyDeviceToHost));}
	else
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=GPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

/***********************/
/* ASSIGNMENTS CPU2CPU */
/***********************/
// int = int
const BB::Hmatrix<int>& BB::Hmatrix<int>::operator=(const BB::Hmatrix<int> &ob)
{
	if((ob.Rows_ == Rows_) && (ob.Columns_ == Columns_)) memcpy(data_,ob.data_,GetNumElements()*sizeof(int));
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// int = float
const BB::Hmatrix<int>& BB::Hmatrix<int>::operator=(const BB::Hmatrix<float> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// int = double
const BB::Hmatrix<int>& BB::Hmatrix<int>::operator=(const BB::Hmatrix<double> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// float = int
const BB::Hmatrix<float>& BB::Hmatrix<float>::operator=(const BB::Hmatrix<int> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// float = float
const BB::Hmatrix<float>& BB::Hmatrix<float>::operator=(const BB::Hmatrix<float> &ob)
{
	if((ob.Rows_ == Rows_) && (ob.Columns_ == Columns_)) memcpy(data_,ob.data_,GetNumElements()*sizeof(float));
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// float = double
const BB::Hmatrix<float>& BB::Hmatrix<float>::operator=(const BB::Hmatrix<double> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// double = int
const BB::Hmatrix<double>& BB::Hmatrix<double>::operator=(const BB::Hmatrix<int> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// double = float
const BB::Hmatrix<double>& BB::Hmatrix<double>::operator=(const BB::Hmatrix<float> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// double = double
const BB::Hmatrix<double>& BB::Hmatrix<double>::operator=(const BB::Hmatrix<double> &ob)
{
	if((ob.Rows_ == Rows_) && (ob.Columns_ == Columns_)) memcpy(data_,ob.data_,GetNumElements()*sizeof(double));
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// int2_ = int
const BB::Hmatrix<BB::int2_>& BB::Hmatrix<BB::int2_>::operator=(const BB::Hmatrix<int> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// int2_ = float
const BB::Hmatrix<BB::int2_>& BB::Hmatrix<BB::int2_>::operator=(const BB::Hmatrix<float> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// int2_ = double
const BB::Hmatrix<BB::int2_>& BB::Hmatrix<BB::int2_>::operator=(const BB::Hmatrix<double> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// int2_ = int2_
const BB::Hmatrix<BB::int2_>& BB::Hmatrix<BB::int2_>::operator=(const BB::Hmatrix<BB::int2_> &ob)
{
	if((ob.Rows_ == Rows_) && (ob.Columns_ == Columns_)) memcpy(data_,ob.data_,GetNumElements()*sizeof(BB::int2_));
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// int2_ = float2_
const BB::Hmatrix<BB::int2_>& BB::Hmatrix<BB::int2_>::operator=(const BB::Hmatrix<BB::float2_> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
		{  
			char* str0 = "**********************************************\n";
			char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
			char* str2 = "Left operand size: ";
			char* str3 = "Right operand size: ";
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
			throw  GenericError(catString,__FILE__,__LINE__);
		} 
	return *this;
}

// int2_ = double2_
const BB::Hmatrix<BB::int2_>& BB::Hmatrix<BB::int2_>::operator=(const BB::Hmatrix<BB::double2_> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// float2_ = int
const BB::Hmatrix<BB::float2_>& BB::Hmatrix<BB::float2_>::operator=(const BB::Hmatrix<int> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// float2_ = float
const BB::Hmatrix<BB::float2_>& BB::Hmatrix<BB::float2_>::operator=(const BB::Hmatrix<float> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// float2_ = double
const BB::Hmatrix<BB::float2_>& BB::Hmatrix<BB::float2_>::operator=(const BB::Hmatrix<double> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// float2_ = int2_
const BB::Hmatrix<BB::float2_>& BB::Hmatrix<BB::float2_>::operator=(const BB::Hmatrix<BB::int2_> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// float2_ = float2_
const BB::Hmatrix<BB::float2_>& BB::Hmatrix<BB::float2_>::operator=(const BB::Hmatrix<BB::float2_> &ob)
{
	if((ob.Rows_ == Rows_) && (ob.Columns_ == Columns_)) memcpy(data_,ob.data_,GetNumElements()*sizeof(BB::float2_));
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// float2_ = double2_
const BB::Hmatrix<BB::float2_>& BB::Hmatrix<BB::float2_>::operator=(const BB::Hmatrix<BB::double2_> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// double2_ = int
const BB::Hmatrix<BB::double2_>& BB::Hmatrix<BB::double2_>::operator=(const BB::Hmatrix<int> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// double2_ = float
const BB::Hmatrix<BB::double2_>& BB::Hmatrix<BB::double2_>::operator=(const BB::Hmatrix<float> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// double2_ = double
const BB::Hmatrix<BB::double2_>& BB::Hmatrix<BB::double2_>::operator=(const BB::Hmatrix<double> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// double2_ = int2_
const BB::Hmatrix<BB::double2_>& BB::Hmatrix<BB::double2_>::operator=(const BB::Hmatrix<BB::int2_> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// double2_ = float2_
const BB::Hmatrix<BB::double2_>& BB::Hmatrix<BB::double2_>::operator=(const BB::Hmatrix<BB::float2_> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = ob.GetDataPointer()[i];
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// double2_ = double2_
const BB::Hmatrix<BB::double2_>& BB::Hmatrix<BB::double2_>::operator=(const BB::Hmatrix<BB::double2_> &ob)
{
	if((ob.GetRows() == Rows_) && (ob.GetColumns() == Columns_)) memcpy(data_,ob.GetDataPointer(),GetNumElements()*sizeof(double2_));
	else 
	{  
		char* str0 = "**********************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,ob.GetRows(),ob.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

// Constant to CPU Hmatrix assignment
//template <class OutType>
//const BB::Hmatrix<OutType>& BB::Hmatrix<OutType>::operator=(const OutType c)
//{
//	*this = Expr<Scalar<OutType>,OutType>(Scalar<OutType>(c),Rows_,Columns_,ISHOST);
//	return *this;
//}

// Hmatrix = Sub-Expression Row
template <class OutType>
const BB::Hmatrix<OutType>& BB::Hmatrix<OutType>::operator=(Expr<SubMatrixExprRow<OutType*,OutType>,OutType> e)
{   
	if((e.GetRows() == Rows_) && (e.GetColumns() == Columns_)) 
		if (e.IsDevice())
			if (IsPinned()) CudaSafeCall(cudaMemcpyAsync(GetDataPointer(),e.GetExpression().GetDataPointer(),GetNumElements()*sizeof(OutType),cudaMemcpyDeviceToHost,BB::streams[p2p.active_GPU].GetActiveStream()));
			else		    CudaSafeCall(cudaMemcpy(GetDataPointer(),e.GetExpression().GetDataPointer(),GetNumElements()*sizeof(OutType),cudaMemcpyDeviceToHost));
		else memcpy(data_,&e[0],GetNumElements()*sizeof(OutType));
		//else for (int i=0; i<GetNumElements(); i++) data_[i] = e[i]; 
	else 
	{  
		char* str0 = "********************************************************\n";
		char* str1 = "* Size mismatch in CPU=CPU-SubMatrix matrix assignment *\n";
		char* str2 = "Left operand size: ";
		char* str3 = "Right operand size: ";
		char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
		sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,e.GetRows(),e.GetColumns());
		throw  GenericError(catString,__FILE__,__LINE__);
	} 
	return *this;
}

/*******************/
/* SUB-EXPRESSIONS */
/*******************/

// --- SubExpressions - Range
template <class OutType>		
BB::Expr<BB::SubMatrixExprRow<OutType*,OutType>,OutType> BB::Hmatrix<OutType>::operator()(BB::Range range)
{	if((range.GetStart() >= 0) && (range.GetNumElements() > 0) && (range.GetEnd() < Rows_*Columns_)) {	
		typedef BB::SubMatrixExprRow<OutType*,OutType> SExpr; 
		//return BB::Expr<SExpr,OutType>(SExpr(data_+range.GetStart(),IsPinned_),1,range.GetNumElements(),ISHOST); 
		return BB::Expr<SExpr,OutType>(SExpr(data_,range.GetStart(),IsPinned_),1,range.GetNumElements(),ISHOST); 
	} else	{	char* str0 = "****************************************\n"; 
				char* str1 = "* Invalid CPU SubMatrix access attempt *\n"; 
				char* str2 = "Hmatrix size: "; 
				char* str3 = "SubMatrix indices (a,b,c,d): "; 
				char* str4 = "SubMatrix size: "; 
				char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
				sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,0,0,range.GetStart(),range.GetEnd(),str4,1,range.GetNumElements()); 
				throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - index Range
template <class OutType>		
BB::Expr<BB::SubMatrixExprRow<OutType*,OutType>,OutType> BB::Hmatrix<OutType>::operator()(int index, BB::Range range2)
{	if((index >= 0) && (index < Rows_) && (range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Columns_)) {	
		typedef SubMatrixExprRow<OutType*,OutType> SExpr; 
		return Expr<SExpr,OutType>(SExpr(data_,index*Columns_+range2.GetStart(),IsPinned_),1,range2.GetNumElements(),ISHOST); 
	} else	{	char* str0 = "****************************************\n"; 
				char* str1 = "* Invalid CPU SubMatrix access attempt *\n"; 
				char* str2 = "Hmatrix size: "; 
				char* str3 = "SubMatrix indices (a,b,c,d): "; 
				char* str4 = "SubMatrix size: "; 
				char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
				sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,index,index,range2.GetStart(),range2.GetEnd(),str4,1,range2.GetNumElements()); 
				throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - int RangeStep
template <class OutType>		
BB::Expr<BB::SubMatrixExprRowStep<OutType*,OutType>,OutType> BB::Hmatrix<OutType>::operator()(int index, BB::RangeStep range2)
{	if((index >= 0) && (index < Rows_) && (range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Columns_)) {	
		typedef BB::SubMatrixExprRowStep<OutType*,OutType> SExpr;
		return BB::Expr<SExpr,OutType>(SExpr(data_,index*Columns_+range2.GetStart(),range2.GetStep()),1,range2.GetNumElements(),ISHOST); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid CPU SubMatrix access attempt *\n"; 
			 char* str2 = "Hmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,index,index,range2.GetStart(),range2.GetEnd(),str4,1,range2.GetNumElements()); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - int Span
template <class OutType>		
BB::Expr<BB::SubMatrixExprRow<OutType*,OutType>,OutType> BB::Hmatrix<OutType>::operator()(int index, BB::SpanClass span2)
{	if((index >= 0) && (index < Rows_)) {	
		typedef BB::SubMatrixExprRow<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,index*Columns_,IsPinned_),1,Columns_,ISHOST); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid CPU SubMatrix access attempt *\n"; 
			 char* str2 = "Hmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,index,index,0,Columns_-1,str4,1,Columns_); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - Range index
template <class OutType>		
BB::Expr<BB::SubMatrixExprColumn<OutType*,OutType>,OutType> BB::Hmatrix<OutType>::operator()(BB::Range range1, int index)
{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_) && 
	   (index >= 0) && (index < Columns_)) {	
		typedef BB::SubMatrixExprColumn<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,range1.GetStart()*Columns_+index,Columns_,IsPinned_),range1.GetNumElements(),1,ISHOST); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid CPU SubMatrix access attempt *\n"; 
			 char* str2 = "Hmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),index,index,str4,range1.GetNumElements(),1); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}
			
// --- SubExpressions - RangeStep int
template <class OutType>		
BB::Expr<BB::SubMatrixExprColumnStep<OutType*,OutType>,OutType> BB::Hmatrix<OutType>::operator()(BB::RangeStep range1, int index)
{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_) && 
	   (index >= 0) && (index < Columns_)) {	
		typedef BB::SubMatrixExprColumnStep<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,range1.GetStart()*Columns_+index,Columns_,range1.GetStep(),IsPinned_),range1.GetNumElements(),1,ISHOST); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid CPU SubMatrix access attempt *\n"; 
			 char* str2 = "Hmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),index,index,str4,range1.GetNumElements(),1); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - Span int
template <class OutType>		
BB::Expr<BB::SubMatrixExprColumn<OutType*,OutType>,OutType> BB::Hmatrix<OutType>::operator()(BB::SpanClass span1, int index)
{	if((index >= 0) && (index < Columns_)) {	
		typedef BB::SubMatrixExprColumn<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,index,Columns_,IsPinned_),Rows_,1,ISHOST); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid CPU SubMatrix access attempt *\n"; 
			 char* str2 = "Hmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,0,Rows_-1,index,index,str4,Rows_,1); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

////// --- SubExpressions - Span Span
//////template <class OutType>		
//////Expr<CudaSubMatrixExpr<OutType*,OutType>,OutType> Hmatrix<OutType>::operator()(SpanClass span1, SpanClass span2)	
//////{	typedef CudaSubMatrixExpr<OutType*,OutType> SExpr; 
//////	return Expr<SExpr,OutType>(SExpr(data_,Rows_,Columns_,Rows_,Columns_,0,1,0,1),Rows_,Columns_,ISHOST); 
//////}
////
// --- SubExpressions - Range Range
template <class OutType>		
BB::Expr<BB::SubMatrixExprColumnRow<OutType*,OutType>,OutType> BB::Hmatrix<OutType>::operator()(BB::Range range1, BB::Range range2)
{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_) && 
	   (range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Rows_)) {	
		typedef BB::SubMatrixExprColumnRow<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,Rows_,Columns_,range1.GetNumElements(),range2.GetNumElements(),range1.GetStart(),range2.GetStart()),range1.GetNumElements(),range2.GetNumElements(),ISHOST); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid CPU SubMatrix access attempt *\n"; 
			 char* str2 = "Hmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),range1.GetEnd(),range2.GetEnd(),str4,range1.GetNumElements(),range2.GetNumElements()); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - Range RangeStep
template <class OutType>		
BB::Expr<BB::SubMatrixExprRowColumnStep<OutType*,OutType>,OutType> BB::Hmatrix<OutType>::operator()(BB::Range range1, BB::RangeStep range2)
{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_) && 
	   (range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Columns_)) {	
		typedef BB::SubMatrixExprRowColumnStep<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,Rows_,Columns_,range1.GetNumElements(),range2.GetNumElements(),range1.GetStart(),range2.GetStart(),range2.GetStep()),range1.GetNumElements(),range2.GetNumElements(),ISHOST); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid CPU SubMatrix access attempt *\n"; 
			 char* str2 = "Hmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),range1.GetEnd(),range2.GetEnd(),str4,range1.GetNumElements(),range2.GetNumElements()); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - RangeStep Range
template <class OutType>		
BB::Expr<BB::SubMatrixExprRowStepColumn<OutType*,OutType>,OutType> BB::Hmatrix<OutType>::operator()(BB::RangeStep range1, BB::Range range2)
{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_) && 
	   (range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Columns_)) {	
		typedef BB::SubMatrixExprRowStepColumn<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,Rows_,Columns_,range1.GetNumElements(),range2.GetNumElements(),range1.GetStart(),range1.GetStep(),range2.GetStart()),range1.GetNumElements(),range2.GetNumElements(),ISHOST); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid CPU SubMatrix access attempt *\n"; 
			 char* str2 = "Hmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),range1.GetEnd(),range2.GetEnd(),str4,range1.GetNumElements(),range2.GetNumElements()); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - RangeStep RangeStep
template <class OutType>		
BB::Expr<BB::SubMatrixExpr<OutType*,OutType>,OutType> BB::Hmatrix<OutType>::operator()(BB::RangeStep range1, BB::RangeStep range2)
{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_) && 
	   (range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Columns_)) {	
		typedef BB::SubMatrixExpr<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,Rows_,Columns_,range1.GetNumElements(),range2.GetNumElements(),range1.GetStart(),range1.GetStep(),range2.GetStart(),range2.GetStep()),
				range1.GetNumElements(),range2.GetNumElements(),ISHOST); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid CPU SubMatrix access attempt *\n"; 
			 char* str2 = "Hmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),range2.GetStart(),range2.GetEnd(),str4,range1.GetNumElements(),range2.GetNumElements()); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - Range Span
template <class OutType>		
BB::Expr<BB::SubMatrixExprColumnRow<OutType*,OutType>,OutType> BB::Hmatrix<OutType>::operator()(BB::Range range1, BB::SpanClass span2)
{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_)) {	
		typedef BB::SubMatrixExprColumnRow<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,Rows_,Columns_,range1.GetNumElements(),Columns_,range1.GetStart(),0),range1.GetNumElements(),Columns_,ISHOST); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid CPU SubMatrix access attempt *\n"; 
			 char* str2 = "Hmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),range1.GetEnd(),Columns_-1,str4,range1.GetNumElements(),Columns_-1); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - Span Range
template <class OutType>		
BB::Expr<BB::SubMatrixExprColumnRow<OutType*,OutType>,OutType> BB::Hmatrix<OutType>::operator()(BB::SpanClass span1, BB::Range range2)
{		if((range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Columns_)) {	
		typedef BB::SubMatrixExprColumnRow<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,Rows_,Columns_,Rows_,range2.GetNumElements(),0,range2.GetStart()),Rows_,range2.GetNumElements(),ISHOST); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid CPU SubMatrix access attempt *\n"; 
			 char* str2 = "Hmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,0,Rows_-1,Rows_-1,range2.GetEnd(),str4,Rows_,range2.GetNumElements()); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - Span RangeStep
template <class OutType>		
BB::Expr<BB::SubMatrixExprRowColumnStep<OutType*,OutType>,OutType> BB::Hmatrix<OutType>::operator()(BB::SpanClass span1, BB::RangeStep range2)
{	if((range2.GetStart() >= 0) && (range2.GetNumElements() > 0) && (range2.GetEnd() < Columns_)) {	
		typedef BB::SubMatrixExprRowColumnStep<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,Rows_,Columns_,Rows_,range2.GetNumElements(),0,range2.GetStart(),range2.GetStep()),Rows_,range2.GetNumElements(),ISHOST); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid CPU SubMatrix access attempt *\n"; 
			 char* str2 = "Hmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,0,Rows_-1,Rows_-1,range2.GetEnd(),str4,Rows_,range2.GetNumElements()); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

// --- SubExpressions - RangeStep Span
template <class OutType>		
BB::Expr<BB::SubMatrixExprRowStepColumn<OutType*,OutType>,OutType> BB::Hmatrix<OutType>::operator()(BB::RangeStep range1, BB::SpanClass span2)
{	if((range1.GetStart() >= 0) && (range1.GetNumElements() > 0) && (range1.GetEnd() < Rows_)) {	
		typedef BB::SubMatrixExprRowStepColumn<OutType*,OutType> SExpr; 
		return BB::Expr<SExpr,OutType>(SExpr(data_,Rows_,Columns_,range1.GetNumElements(),Columns_,range1.GetStart(),range1.GetStep(),0),range1.GetNumElements(),Columns_,ISHOST); 
	} else	{char* str0 = "****************************************\n"; 
			 char* str1 = "* Invalid CPU SubMatrix access attempt *\n"; 
			 char* str2 = "Hmatrix size: "; 
			 char* str3 = "SubMatrix indices (a,b,c,d): "; 
			 char* str4 = "SubMatrix size: "; 
			 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+15*sizeof(char)); 
			 sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n%s%i x %i\n",str0,str1,str0,str2,Rows_,Columns_,str3,range1.GetStart(),range1.GetEnd(),range1.GetEnd(),Columns_-1,str4,range1.GetNumElements(),Columns_); 
			 throw  GenericError(catString,__FILE__,__LINE__); }
}

////// --- SubExpressions - int Hmatrix
////template <class OutType>
////BB::Expr<BB::CudaSubMatrixExprRowPerm<OutType*,OutType>,OutType> BB::Hmatrix<OutType>::operator()(int index, BB::Hmatrix<int> indices2)
////{	typedef BB::CudaSubMatrixExprRowPerm<OutType*,OutType> SExpr; 
////	return	BB::Expr<SExpr,OutType>(SExpr(data_+index*Columns_,indices2,IsPinned_),1,indices2.GetNumElements(),ISHOST); 
////}
//
/******************/
/* MOVE SEMANTICS */
/******************/

// --- Move constructor
template <class OutType>
BB::Hmatrix<OutType>::Hmatrix(Hmatrix<OutType>&& other) throw() { 
	data_ = other.data_; Rows_ = other.Rows_; Columns_ = other.Columns_;
	other.Rows_ = other.Columns_ = 0; other.data_ = nullptr; 
}

// --- Move assignment
template <class OutType>
BB::Hmatrix<OutType>& BB::Hmatrix<OutType>::operator=(BB::Hmatrix<OutType>&& other) throw() {
	using std::swap;
	swap(Rows_, other.Rows_);
	swap(Columns_, other.Columns_);
	swap(data_, other.data_); 
	return *this;
}

/***********************************************/
/* EXPLICIT INSTANTIATIONS OF THE MATRIX CLASS */
/***********************************************/
template class BB::Hmatrix<int>;
template class BB::Hmatrix<float>;
template class BB::Hmatrix<double>;
template class BB::Hmatrix<BB::int2_>;
template class BB::Hmatrix<BB::float2_>;
template class BB::Hmatrix<BB::double2_>;

/*******************************/
/* OVERLOAD OF THE << OPERATOR */
/*******************************/

// --- Overload of << for type T Hmatrix (int, float and double)
template <class T>
ostream& operator << (ostream& output, const Hmatrix<T> & v) {
	for(int i=0; i<v.GetRows(); i++) {
		for(int j=0; j<v.GetColumns(); j++) {
			output << setw(prec_cout) << "Mat(" << i << "," << j << ") = " << v(i,j) << "\t"; }
		output << endl; }
	return output; }

template ostream& operator << (ostream & output, const Hmatrix<int> & v);
template ostream& operator << (ostream & output, const Hmatrix<float> & v);
template ostream& operator << (ostream & output, const Hmatrix<double> & v);

// --- Overload of << for int2_ Hmatrix
ostream& operator << (ostream &output, const Hmatrix<int2_> &v) {
	for(int i=0; i<v.GetRows(); i++) {
		for(int j=0; j<v.GetColumns(); j++) {
			output << setw(prec_cout) << "Mat(" << i << "," << j << ") = " << v(i,j).c.x << "+ " << v(i,j).c.y << "*j \t"; }
		output << endl; }
	return output;
}

// --- Overload of << for float2_ Hmatrix
ostream& operator << (ostream &output, const Hmatrix<float2_> &v) {
	for(int i=0; i<v.GetRows(); i++) {
		for(int j=0; j<v.GetColumns(); j++) {
			output << setw(prec_cout) << "Mat(" << i << "," << j << ") = " << v(i,j).c.x << "+ " << v(i,j).c.y << "*j \t"; }
		output << endl; }
	return output;
}

// --- Overload of << for double2_ Hmatrix
ostream& operator << (ostream &output, const Hmatrix<double2_> &v) {
	for(int i=0; i<v.GetRows(); i++) {
		for(int j=0; j<v.GetColumns(); j++) {
			output << setw(prec_cout) << "Mat(" << i << "," << j << ") = " << v(i,j).c.x << "+ " << v(i,j).c.y << "*j \t"; }
		output << endl; }
	return output;
}

//
//#include "FFT.h"
//#include "Interpolation.h"
//#include "DataPlotter.h"
