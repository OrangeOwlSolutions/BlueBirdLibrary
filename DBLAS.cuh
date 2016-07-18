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

#ifndef __DBLAS_CUH__
#define __DBLAS_CUH__

#include <cublas_v2.h>
#include "HExceptions.h"

namespace BB
{

	// --- cuBLAS initialization
	cublasHandle_t DEVICE_BLAS_CREATE_HANDLE()
	{
		cublasHandle_t handle;
		cublasSafeCall(cublasCreate(&handle));
		return handle;
	}

	// --- cuBLAS shutdown
	void DEVICE_BLAS_DESTROY_HANDLE(cublasHandle_t handle)
	{
		cublasSafeCall(cublasDestroy(handle));
	}

	/*********/
	/* FLOAT */
	/*********/

	// --- Matrix - Matrix
	Dmatrix<float> MatMul(const Dmatrix<float> &in1, const Dmatrix<float> &in2, cublasHandle_t handle = NULL)
	{
		if((in1.GetColumns() == in2.GetRows())) { 

			Dmatrix<float> out(in1.GetRows(),in2.GetColumns());

			float alpha = 1.0f;
			float beta = 0.0f;

			if (handle == NULL)
			{
				cublasSafeCall(cublasCreate(&handle));
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, in1.GetDataPointer(), 
					in1.GetRows(), in2.GetDataPointer(), 1, &beta, out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), in2.GetRows(), &beta, out.GetDataPointer(), out.GetRows()));
				cublasSafeCall(cublasDestroy(handle));
			}
			else
			{	
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), 1, &beta, out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), in2.GetRows(), &beta, out.GetDataPointer(), out.GetRows()));
			}
			return out;

		}
		else 
		{  
			char* str0 = "*********************************************\n";
			char* str1 = "* Size mismatch in GPU matrix multiplication *\n";
			char* str2 = "Left operand size: ";
			char* str3 = "Right operand size: ";
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,in1.GetRows(),in1.GetColumns(),str3,in2.GetRows(),in2.GetColumns());
			throw  GenericError(catString,__FILE__,__LINE__);
		} 
	}

	// --- Matrix - Expression
	template <class Q>
	Dmatrix<float> MatMul(const Dmatrix<float> &in1, const Expr<Q,float> &e2, cublasHandle_t handle = NULL)
	{
		Dmatrix<float> in2(e2.GetRows(),e2.GetColumns());
		in2 = e2;
		
		if((in1.GetColumns() == in2.GetRows())) { 

			Dmatrix<float> out(in1.GetRows(),in2.GetColumns());

			float alpha = 1.0f;
			float beta = 0.0f;

			if (handle == NULL)
			{
				cublasSafeCall(cublasCreate(&handle));
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, in1.GetDataPointer(), 
					in1.GetRows(), in2.GetDataPointer(), 1, &beta, out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), in2.GetRows(), &beta, out.GetDataPointer(), out.GetRows()));
				cublasSafeCall(cublasDestroy(handle));
			}
			else
			{	
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), 1, &beta, out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), in2.GetRows(), &beta, out.GetDataPointer(), out.GetRows()));
			}
			return out;

		}
		else 
		{  
			char* str0 = "*********************************************\n";
			char* str1 = "* Size mismatch in GPU matrix multiplication *\n";
			char* str2 = "Left operand size: ";
			char* str3 = "Right operand size: ";
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,in1.GetRows(),in1.GetColumns(),str3,in2.GetRows(),in2.GetColumns());
			throw  GenericError(catString,__FILE__,__LINE__);
		} 
	}

	// --- Expression - Matrix
	template <class Q>
	Dmatrix<float> MatMul(const Expr<Q,float> &e1, const Dmatrix<float> &in2, cublasHandle_t handle = NULL)
	{
		Dmatrix<float> in1(e1.GetRows(),e1.GetColumns());
		in1 = e1;
		
		if((in1.GetColumns() == in2.GetRows())) { 

			Dmatrix<float> out(in1.GetRows(),in2.GetColumns());

			float alpha = 1.0f;
			float beta = 0.0f;

			if (handle == NULL)
			{
				cublasSafeCall(cublasCreate(&handle));
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, in1.GetDataPointer(), 
					in1.GetRows(), in2.GetDataPointer(), 1, &beta, out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), in2.GetRows(), &beta, out.GetDataPointer(), out.GetRows()));
				cublasSafeCall(cublasDestroy(handle));
			}
			else
			{	
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), 1, &beta, out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), in2.GetRows(), &beta, out.GetDataPointer(), out.GetRows()));
			}
			return out;

		}
		else 
		{  
			char* str0 = "*********************************************\n";
			char* str1 = "* Size mismatch in GPU matrix multiplication *\n";
			char* str2 = "Left operand size: ";
			char* str3 = "Right operand size: ";
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,in1.GetRows(),in1.GetColumns(),str3,in2.GetRows(),in2.GetColumns());
			throw  GenericError(catString,__FILE__,__LINE__);
		} 
	}

	// --- Expression - Expression
	template <class Q1, class Q2>
	Dmatrix<float> MatMul(const Expr<Q1,float> &e1, const Expr<Q2,float> &e2, cublasHandle_t handle = NULL)
	{
		Dmatrix<float> in1(e1.GetRows(),e1.GetColumns());
		Dmatrix<float> in2(e2.GetRows(),e2.GetColumns());
		in1 = e1;
		in2 = e2;
		
		if((in1.GetColumns() == in2.GetRows())) { 

			Dmatrix<float> out(in1.GetRows(),in2.GetColumns());

			float alpha = 1.0f;
			float beta = 0.0f;

			if (handle == NULL)
			{
				cublasSafeCall(cublasCreate(&handle));
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, in1.GetDataPointer(), 
					in1.GetRows(), in2.GetDataPointer(), 1, &beta, out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), in2.GetRows(), &beta, out.GetDataPointer(), out.GetRows()));
				cublasSafeCall(cublasDestroy(handle));
			}
			else
			{	
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasSgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), 1, &beta, out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), in2.GetRows(), &beta, out.GetDataPointer(), out.GetRows()));
			}
			return out;

		}
		else 
		{  
			char* str0 = "*********************************************\n";
			char* str1 = "* Size mismatch in GPU matrix multiplication *\n";
			char* str2 = "Left operand size: ";
			char* str3 = "Right operand size: ";
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,in1.GetRows(),in1.GetColumns(),str3,in2.GetRows(),in2.GetColumns());
			throw  GenericError(catString,__FILE__,__LINE__);
		} 
	}

	/**********/
	/* DOUBLE */
	/**********/

	// --- Matrix - Matrix
	Dmatrix<double> MatMul(const Dmatrix<double> &in1, const Dmatrix<double> &in2, cublasHandle_t handle = NULL)
	{
		if((in1.GetColumns() == in2.GetRows())) { 

			Dmatrix<double> out(in1.GetRows(),in2.GetColumns());

			double alpha = 1.0;
			double beta = 0.0;

			if (handle == NULL)
			{
				cublasSafeCall(cublasCreate(&handle));
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasDgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, in1.GetDataPointer(), 
					in1.GetRows(), in2.GetDataPointer(), 1, &beta, out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), in2.GetRows(), &beta, out.GetDataPointer(), out.GetRows()));
				cublasSafeCall(cublasDestroy(handle));
			}
			else
			{	
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasDgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), 1, &beta, out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), in2.GetRows(), &beta, out.GetDataPointer(), out.GetRows()));
			}
			return out;

		}
		else 
		{  
			char* str0 = "*********************************************\n";
			char* str1 = "* Size mismatch in GPU matrix multiplication *\n";
			char* str2 = "Left operand size: ";
			char* str3 = "Right operand size: ";
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,in1.GetRows(),in1.GetColumns(),str3,in2.GetRows(),in2.GetColumns());
			throw  GenericError(catString,__FILE__,__LINE__);
		} 
	}

	// --- Matrix - Expression
	template <class Q>
	Dmatrix<double> MatMul(const Dmatrix<double> &in1, const Expr<Q,double> &e2, cublasHandle_t handle = NULL)
	{
		Dmatrix<double> in2(e2.GetRows(),e2.GetColumns());
		in2 = e2;
		
		if((in1.GetColumns() == in2.GetRows())) { 

			Dmatrix<double> out(in1.GetRows(),in2.GetColumns());

			double alpha = 1.0;
			double beta = 0.0;

			if (handle == NULL)
			{
				cublasSafeCall(cublasCreate(&handle));
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasDgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, in1.GetDataPointer(), 
					in1.GetRows(), in2.GetDataPointer(), 1, &beta, out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), in2.GetRows(), &beta, out.GetDataPointer(), out.GetRows()));
				cublasSafeCall(cublasDestroy(handle));
			}
			else
			{	
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasDgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), 1, &beta, out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), in2.GetRows(), &beta, out.GetDataPointer(), out.GetRows()));
			}
			return out;

		}
		else 
		{  
			char* str0 = "*********************************************\n";
			char* str1 = "* Size mismatch in GPU matrix multiplication *\n";
			char* str2 = "Left operand size: ";
			char* str3 = "Right operand size: ";
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,in1.GetRows(),in1.GetColumns(),str3,in2.GetRows(),in2.GetColumns());
			throw  GenericError(catString,__FILE__,__LINE__);
		} 
	}

	// --- Expression - Matrix
	template <class Q>
	Dmatrix<double> MatMul(const Expr<Q,double> &e1, const Dmatrix<double> &in2, cublasHandle_t handle = NULL)
	{
		Dmatrix<double> in1(e1.GetRows(),e1.GetColumns());
		in1 = e1;
		
		if((in1.GetColumns() == in2.GetRows())) { 

			Dmatrix<double> out(in1.GetRows(),in2.GetColumns());

			double alpha = 1.0;
			double beta = 0.0;

			if (handle == NULL)
			{
				cublasSafeCall(cublasCreate(&handle));
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasDgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, in1.GetDataPointer(), 
					in1.GetRows(), in2.GetDataPointer(), 1, &beta, out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), in2.GetRows(), &beta, out.GetDataPointer(), out.GetRows()));
				cublasSafeCall(cublasDestroy(handle));
			}
			else
			{	
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasDgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), 1, &beta, out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), in2.GetRows(), &beta, out.GetDataPointer(), out.GetRows()));
			}
			return out;

		}
		else 
		{  
			char* str0 = "*********************************************\n";
			char* str1 = "* Size mismatch in GPU matrix multiplication *\n";
			char* str2 = "Left operand size: ";
			char* str3 = "Right operand size: ";
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,in1.GetRows(),in1.GetColumns(),str3,in2.GetRows(),in2.GetColumns());
			throw  GenericError(catString,__FILE__,__LINE__);
		} 
	}

	// --- Expression - Expression
	template <class Q1, class Q2>
	Dmatrix<double> MatMul(const Expr<Q1,double> &e1, const Expr<Q2,double> &e2, cublasHandle_t handle = NULL)
	{
		Dmatrix<double> in1(e1.GetRows(),e1.GetColumns());
		Dmatrix<double> in2(e2.GetRows(),e2.GetColumns());
		in1 = e1;
		in2 = e2;
		
		if((in1.GetColumns() == in2.GetRows())) { 

			Dmatrix<double> out(in1.GetRows(),in2.GetColumns());

			double alpha = 1.0;
			double beta = 0.0;

			if (handle == NULL)
			{
				cublasSafeCall(cublasCreate(&handle));
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasDgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, in1.GetDataPointer(), 
					in1.GetRows(), in2.GetDataPointer(), 1, &beta, out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), in2.GetRows(), &beta, out.GetDataPointer(), out.GetRows()));
				cublasSafeCall(cublasDestroy(handle));
			}
			else
			{	
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasDgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), 1, &beta, out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, in1.GetDataPointer(), in1.GetRows(), in2.GetDataPointer(), in2.GetRows(), &beta, out.GetDataPointer(), out.GetRows()));
			}
			return out;

		}
		else 
		{  
			char* str0 = "*********************************************\n";
			char* str1 = "* Size mismatch in GPU matrix multiplication *\n";
			char* str2 = "Left operand size: ";
			char* str3 = "Right operand size: ";
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,in1.GetRows(),in1.GetColumns(),str3,in2.GetRows(),in2.GetColumns());
			throw  GenericError(catString,__FILE__,__LINE__);
		} 
	}

	/***********/
	/* FLOAT2_ */
	/***********/

	// --- Matrix - Matrix
	Dmatrix<float2_> MatMul(const Dmatrix<float2_> &in1, const Dmatrix<float2_> &in2, cublasHandle_t handle = NULL)
	{
		if((in1.GetColumns() == in2.GetRows())) { 

			Dmatrix<float2_> out(in1.GetRows(),in2.GetColumns());

			cuComplex alpha; alpha.x = 1.0f; alpha.y = 0.0f;
			cuComplex beta;  beta.x = 0.0f;  beta.y = 0.0f;

			if (handle == NULL)
			{
				cublasSafeCall(cublasCreate(&handle));
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasCgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, (cuComplex*)in1.GetDataPointer(), 
					in1.GetRows(), (cuComplex*)in2.GetDataPointer(), 1, &beta, (cuComplex*)out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, (cuComplex*)in1.GetDataPointer(), in1.GetRows(), (cuComplex*)in2.GetDataPointer(), in2.GetRows(), &beta, (cuComplex*)out.GetDataPointer(), out.GetRows()));
				cublasSafeCall(cublasDestroy(handle));
			}
			else
			{	
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasCgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, (cuComplex*)in1.GetDataPointer(), in1.GetRows(), (cuComplex*)in2.GetDataPointer(), 1, &beta, (cuComplex*)out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, (cuComplex*)in1.GetDataPointer(), in1.GetRows(), (cuComplex*)in2.GetDataPointer(), in2.GetRows(), &beta, (cuComplex*)out.GetDataPointer(), out.GetRows()));
			}
			return out;

		}
		else 
		{  
			char* str0 = "*********************************************\n";
			char* str1 = "* Size mismatch in GPU matrix multiplication *\n";
			char* str2 = "Left operand size: ";
			char* str3 = "Right operand size: ";
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,in1.GetRows(),in1.GetColumns(),str3,in2.GetRows(),in2.GetColumns());
			throw  GenericError(catString,__FILE__,__LINE__);
		} 
	}

	// --- Matrix - Expression
	template <class Q>
	Dmatrix<float2_> MatMul(const Dmatrix<float2_> &in1, const Expr<Q,float2_> &e2, cublasHandle_t handle = NULL)
	{
		Dmatrix<float2_> in2(e2.GetRows(),e2.GetColumns());
		in2 = e2;
		
		if((in1.GetColumns() == in2.GetRows())) { 

			Dmatrix<float2_> out(in1.GetRows(),in2.GetColumns());

			cuComplex alpha; alpha.x = 1.0f; alpha.y = 0.0f;
			cuComplex beta;  beta.x = 0.0f;  beta.y = 0.0f;

			if (handle == NULL)
			{
				cublasSafeCall(cublasCreate(&handle));
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasCgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, (cuComplex*)in1.GetDataPointer(), 
					in1.GetRows(), (cuComplex*)in2.GetDataPointer(), 1, &beta, (cuComplex*)out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, (cuComplex*)in1.GetDataPointer(), in1.GetRows(), (cuComplex*)in2.GetDataPointer(), in2.GetRows(), &beta, (cuComplex*)out.GetDataPointer(), out.GetRows()));
				cublasSafeCall(cublasDestroy(handle));
			}
			else
			{	
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasCgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, (cuComplex*)in1.GetDataPointer(), in1.GetRows(), (cuComplex*)in2.GetDataPointer(), 1, &beta, (cuComplex*)out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, (cuComplex*)in1.GetDataPointer(), in1.GetRows(), (cuComplex*)in2.GetDataPointer(), in2.GetRows(), &beta, (cuComplex*)out.GetDataPointer(), out.GetRows()));
			}
			return out;

		}
		else 
		{  
			char* str0 = "*********************************************\n";
			char* str1 = "* Size mismatch in GPU matrix multiplication *\n";
			char* str2 = "Left operand size: ";
			char* str3 = "Right operand size: ";
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,in1.GetRows(),in1.GetColumns(),str3,in2.GetRows(),in2.GetColumns());
			throw  GenericError(catString,__FILE__,__LINE__);
		} 
	}

	// --- Expression - Matrix
	template <class Q>
	Dmatrix<float2_> MatMul(const Expr<Q,float2_> &e1, const Dmatrix<float2_> &in2, cublasHandle_t handle = NULL)
	{
		Dmatrix<float2_> in1(e1.GetRows(),e1.GetColumns());
		in1 = e1;
		
		if((in1.GetColumns() == in2.GetRows())) { 

			Dmatrix<float2_> out(in1.GetRows(),in2.GetColumns());

			cuComplex alpha; alpha.x = 1.0f; alpha.y = 0.0f;
			cuComplex beta;  beta.x = 0.0f;  beta.y = 0.0f;

			if (handle == NULL)
			{
				cublasSafeCall(cublasCreate(&handle));
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasCgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, (cuComplex*)in1.GetDataPointer(), 
					in1.GetRows(), (cuComplex*)in2.GetDataPointer(), 1, &beta, (cuComplex*)out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, (cuComplex*)in1.GetDataPointer(), in1.GetRows(), (cuComplex*)in2.GetDataPointer(), in2.GetRows(), &beta, (cuComplex*)out.GetDataPointer(), out.GetRows()));
				cublasSafeCall(cublasDestroy(handle));
			}
			else
			{	
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasCgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, (cuComplex*)in1.GetDataPointer(), in1.GetRows(), (cuComplex*)in2.GetDataPointer(), 1, &beta, (cuComplex*)out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, (cuComplex*)in1.GetDataPointer(), in1.GetRows(), (cuComplex*)in2.GetDataPointer(), in2.GetRows(), &beta, (cuComplex*)out.GetDataPointer(), out.GetRows()));
			}
			return out;

		}
		else 
		{  
			char* str0 = "*********************************************\n";
			char* str1 = "* Size mismatch in GPU matrix multiplication *\n";
			char* str2 = "Left operand size: ";
			char* str3 = "Right operand size: ";
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,in1.GetRows(),in1.GetColumns(),str3,in2.GetRows(),in2.GetColumns());
			throw  GenericError(catString,__FILE__,__LINE__);
		} 
	}

	// --- Expression - Expression
	template <class Q1, class Q2>
	Dmatrix<float2_> MatMul(const Expr<Q1,float2_> &e1, const Expr<Q2,float2_> &e2, cublasHandle_t handle = NULL)
	{
		Dmatrix<float2_> in1(e1.GetRows(),e1.GetColumns());
		Dmatrix<float2_> in2(e2.GetRows(),e2.GetColumns());
		in1 = e1;
		in2 = e2;
		
		if((in1.GetColumns() == in2.GetRows())) { 

			Dmatrix<float2_> out(in1.GetRows(),in2.GetColumns());

			cuComplex alpha; alpha.x = 1.0f; alpha.y = 0.0f;
			cuComplex beta;  beta.x = 0.0f;  beta.y = 0.0f;

			if (handle == NULL)
			{
				cublasSafeCall(cublasCreate(&handle));
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasCgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, (cuComplex*)in1.GetDataPointer(), 
					in1.GetRows(), (cuComplex*)in2.GetDataPointer(), 1, &beta, (cuComplex*)out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, (cuComplex*)in1.GetDataPointer(), in1.GetRows(), (cuComplex*)in2.GetDataPointer(), in2.GetRows(), &beta, (cuComplex*)out.GetDataPointer(), out.GetRows()));
				cublasSafeCall(cublasDestroy(handle));
			}
			else
			{	
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasCgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, (cuComplex*)in1.GetDataPointer(), in1.GetRows(), (cuComplex*)in2.GetDataPointer(), 1, &beta, (cuComplex*)out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasCgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, (cuComplex*)in1.GetDataPointer(), in1.GetRows(), (cuComplex*)in2.GetDataPointer(), in2.GetRows(), &beta, (cuComplex*)out.GetDataPointer(), out.GetRows()));
			}
			return out;

		}
		else 
		{  
			char* str0 = "*********************************************\n";
			char* str1 = "* Size mismatch in GPU matrix multiplication *\n";
			char* str2 = "Left operand size: ";
			char* str3 = "Right operand size: ";
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,in1.GetRows(),in1.GetColumns(),str3,in2.GetRows(),in2.GetColumns());
			throw  GenericError(catString,__FILE__,__LINE__);
		} 
	}

	/************/
	/* DOUBLE2_ */
	/************/

	// --- Matrix - Matrix
	Dmatrix<double2_> MatMul(const Dmatrix<double2_> &in1, const Dmatrix<double2_> &in2, cublasHandle_t handle = NULL)
	{
		if((in1.GetColumns() == in2.GetRows())) { 

			Dmatrix<double2_> out(in1.GetRows(),in2.GetColumns());

			cuDoubleComplex alpha; alpha.x = 1.0f; alpha.y = 0.0f;
			cuDoubleComplex beta;  beta.x = 0.0f;  beta.y = 0.0f;

			if (handle == NULL)
			{
				cublasSafeCall(cublasCreate(&handle));
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasZgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, (cuDoubleComplex*)in1.GetDataPointer(), 
					in1.GetRows(), (cuDoubleComplex*)in2.GetDataPointer(), 1, &beta, (cuDoubleComplex*)out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, (cuDoubleComplex*)in1.GetDataPointer(), in1.GetRows(), (cuDoubleComplex*)in2.GetDataPointer(), in2.GetRows(), &beta, (cuDoubleComplex*)out.GetDataPointer(), out.GetRows()));
				cublasSafeCall(cublasDestroy(handle));
			}
			else
			{	
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasZgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, (cuDoubleComplex*)in1.GetDataPointer(), in1.GetRows(), (cuDoubleComplex*)in2.GetDataPointer(), 1, &beta, (cuDoubleComplex*)out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, (cuDoubleComplex*)in1.GetDataPointer(), in1.GetRows(), (cuDoubleComplex*)in2.GetDataPointer(), in2.GetRows(), &beta, (cuDoubleComplex*)out.GetDataPointer(), out.GetRows()));
			}
			return out;

		}
		else 
		{  
			char* str0 = "*********************************************\n";
			char* str1 = "* Size mismatch in GPU matrix multiplication *\n";
			char* str2 = "Left operand size: ";
			char* str3 = "Right operand size: ";
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,in1.GetRows(),in1.GetColumns(),str3,in2.GetRows(),in2.GetColumns());
			throw  GenericError(catString,__FILE__,__LINE__);
		} 
	}

	// --- Matrix - Expression
	template <class Q>
	Dmatrix<double2_> MatMul(const Dmatrix<double2_> &in1, const Expr<Q,double2_> &e2, cublasHandle_t handle = NULL)
	{
		Dmatrix<double2_> in2(e2.GetRows(),e2.GetColumns());
		in2 = e2;
		
		if((in1.GetColumns() == in2.GetRows())) { 

			Dmatrix<double2_> out(in1.GetRows(),in2.GetColumns());

			cuDoubleComplex alpha; alpha.x = 1.0f; alpha.y = 0.0f;
			cuDoubleComplex beta;  beta.x = 0.0f;  beta.y = 0.0f;

			if (handle == NULL)
			{
				cublasSafeCall(cublasCreate(&handle));
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasZgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, (cuDoubleComplex*)in1.GetDataPointer(), 
					in1.GetRows(), (cuDoubleComplex*)in2.GetDataPointer(), 1, &beta, (cuDoubleComplex*)out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, (cuDoubleComplex*)in1.GetDataPointer(), in1.GetRows(), (cuDoubleComplex*)in2.GetDataPointer(), in2.GetRows(), &beta, (cuDoubleComplex*)out.GetDataPointer(), out.GetRows()));
				cublasSafeCall(cublasDestroy(handle));
			}
			else
			{	
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasZgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, (cuDoubleComplex*)in1.GetDataPointer(), in1.GetRows(), (cuDoubleComplex*)in2.GetDataPointer(), 1, &beta, (cuDoubleComplex*)out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, (cuDoubleComplex*)in1.GetDataPointer(), in1.GetRows(), (cuDoubleComplex*)in2.GetDataPointer(), in2.GetRows(), &beta, (cuDoubleComplex*)out.GetDataPointer(), out.GetRows()));
			}
			return out;

		}
		else 
		{  
			char* str0 = "*********************************************\n";
			char* str1 = "* Size mismatch in GPU matrix multiplication *\n";
			char* str2 = "Left operand size: ";
			char* str3 = "Right operand size: ";
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,in1.GetRows(),in1.GetColumns(),str3,in2.GetRows(),in2.GetColumns());
			throw  GenericError(catString,__FILE__,__LINE__);
		} 
	}

	// --- Expression - Matrix
	template <class Q>
	Dmatrix<double2_> MatMul(const Expr<Q,double2_> &e1, const Dmatrix<double2_> &in2, cublasHandle_t handle = NULL)
	{
		Dmatrix<double2_> in1(e1.GetRows(),e1.GetColumns());
		in1 = e1;
		
		if((in1.GetColumns() == in2.GetRows())) { 

			Dmatrix<double2_> out(in1.GetRows(),in2.GetColumns());

			cuDoubleComplex alpha; alpha.x = 1.0f; alpha.y = 0.0f;
			cuDoubleComplex beta;  beta.x = 0.0f;  beta.y = 0.0f;

			if (handle == NULL)
			{
				cublasSafeCall(cublasCreate(&handle));
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasZgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, (cuDoubleComplex*)in1.GetDataPointer(), 
					in1.GetRows(), (cuDoubleComplex*)in2.GetDataPointer(), 1, &beta, (cuDoubleComplex*)out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, (cuDoubleComplex*)in1.GetDataPointer(), in1.GetRows(), (cuDoubleComplex*)in2.GetDataPointer(), in2.GetRows(), &beta, (cuDoubleComplex*)out.GetDataPointer(), out.GetRows()));
				cublasSafeCall(cublasDestroy(handle));
			}
			else
			{	
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasZgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, (cuDoubleComplex*)in1.GetDataPointer(), in1.GetRows(), (cuDoubleComplex*)in2.GetDataPointer(), 1, &beta, (cuDoubleComplex*)out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, (cuDoubleComplex*)in1.GetDataPointer(), in1.GetRows(), (cuDoubleComplex*)in2.GetDataPointer(), in2.GetRows(), &beta, (cuDoubleComplex*)out.GetDataPointer(), out.GetRows()));
			}
			return out;

		}
		else 
		{  
			char* str0 = "*********************************************\n";
			char* str1 = "* Size mismatch in GPU matrix multiplication *\n";
			char* str2 = "Left operand size: ";
			char* str3 = "Right operand size: ";
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,in1.GetRows(),in1.GetColumns(),str3,in2.GetRows(),in2.GetColumns());
			throw  GenericError(catString,__FILE__,__LINE__);
		} 
	}

	// --- Expression - Expression
	template <class Q1, class Q2>
	Dmatrix<double2_> MatMul(const Expr<Q1,double2_> &e1, const Expr<Q2,double2_> &e2, cublasHandle_t handle = NULL)
	{
		Dmatrix<double2_> in1(e1.GetRows(),e1.GetColumns());
		Dmatrix<double2_> in2(e2.GetRows(),e2.GetColumns());
		in1 = e1;
		in2 = e2;
		
		if((in1.GetColumns() == in2.GetRows())) { 

			Dmatrix<double2_> out(in1.GetRows(),in2.GetColumns());

			cuDoubleComplex alpha; alpha.x = 1.0f; alpha.y = 0.0f;
			cuDoubleComplex beta;  beta.x = 0.0f;  beta.y = 0.0f;

			if (handle == NULL)
			{
				cublasSafeCall(cublasCreate(&handle));
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasZgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, (cuDoubleComplex*)in1.GetDataPointer(), 
					in1.GetRows(), (cuDoubleComplex*)in2.GetDataPointer(), 1, &beta, (cuDoubleComplex*)out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, (cuDoubleComplex*)in1.GetDataPointer(), in1.GetRows(), (cuDoubleComplex*)in2.GetDataPointer(), in2.GetRows(), &beta, (cuDoubleComplex*)out.GetDataPointer(), out.GetRows()));
				cublasSafeCall(cublasDestroy(handle));
			}
			else
			{	
				cublasSafeCall(cublasSetStream(handle,streams[p2p.active_GPU].GetActiveStream()));
				if (in2.IsVector()) cublasSafeCall(cublasZgemv(handle, CUBLAS_OP_N, in1.GetRows(), in1.GetColumns(), &alpha, (cuDoubleComplex*)in1.GetDataPointer(), in1.GetRows(), (cuDoubleComplex*)in2.GetDataPointer(), 1, &beta, (cuDoubleComplex*)out.GetDataPointer(), 1));					
				else cublasSafeCall(cublasZgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, in1.GetRows(), in2.GetColumns(), in1.GetColumns(), &alpha, (cuDoubleComplex*)in1.GetDataPointer(), in1.GetRows(), (cuDoubleComplex*)in2.GetDataPointer(), in2.GetRows(), &beta, (cuDoubleComplex*)out.GetDataPointer(), out.GetRows()));
			}
			return out;

		}
		else 
		{  
			char* str0 = "*********************************************\n";
			char* str1 = "* Size mismatch in GPU matrix multiplication *\n";
			char* str2 = "Left operand size: ";
			char* str3 = "Right operand size: ";
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,in1.GetRows(),in1.GetColumns(),str3,in2.GetRows(),in2.GetColumns());
			throw  GenericError(catString,__FILE__,__LINE__);
		} 
	}

}

#endif