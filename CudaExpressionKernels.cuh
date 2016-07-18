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


#ifndef __CUDAEXPRESSIONKERNELS_CUH__
#define __CUDAEXPRESSIONKERNELS_CUH__

#include "Constants.h"				// Needed for BLOCKSIZE

namespace BB
{

	//// --- 1D evaluation kernel function - Hmatrix = Expression
	//template <class A, class T1, class T2>
	//__global__ inline void evaluation_matrix_expr(T1 *data_, const Expr<A,T2> e, int NumElements)
	//{
	//	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	//	if(i < NumElements) data_[i] = e[i];
	//}

	//// --- 1D evaluation kernel call - Hmatrix = Expression
	//template<class A, class T1, class T2>
	//void eval_matrix_wrap_expr(T1 *data_, const Expr<A,T2> e, int NumElements)
	//{
	//	//dim3 dimGrid(iDivUp(NumElements,dimBlock.x*streams.GetNumStreams()));
	//	dim3 dimGrid(iDivUp(NumElements,dimBlock.x));
	//	evaluation_matrix_expr<<<dimGrid,dimBlock,0,streams.GetActiveStream()>>>(data_,e,NumElements);
	//	CudaCheckError();
	//	//gpuErrchk( cudaPeekAtLastError() );
	//	//gpuErrchk( cudaDeviceSynchronize() );
	//}

	//template<class A, class T1, class T2> void evaluation_matrix_function_expr(T1* data, const Expr<A,T2> e, int NumElements)	  { eval_matrix_wrap_expr(data, e, NumElements); }

	// --- 1D evaluation kernel function - Expression = Dmatrix (needed for SubExpression assignments)
	template <class B>
	__global__ inline void evaluation_submatrix(CudaSubMatrixExpr<B*,B> e, B* data_, int NumElements)
	{ 
		const int i = blockDim.x * blockIdx.x + threadIdx.x;
		if(i < NumElements) e[i] = data_[i];
	}

	// --- 1D evaluation kernel function - Expression = CudaMatrix_Row - unitary step - (needed for SubExpression assignments)
	template <class B>
	__global__ inline void evaluation_submatrix_row(CudaSubMatrixExprRow<B*,B> e, B* data_, int NumElements)
	{ 
		const int i = blockDim.x * blockIdx.x + threadIdx.x;
		if(i < NumElements) e[i] = data_[i];
	}

	// --- 1D evaluation kernel function - Expression_Row = CudaMatrix_Row - unitary step - (needed for SubExpression assignments)
	template <class A, class B, class C>
	__global__ inline void evaluation_submatrix_row(Expr<A,B> e, Expr<C,B> data_, int NumElements)
	{ 
		const int i = blockDim.x * blockIdx.x + threadIdx.x;
		if(i < NumElements) e[i] = data_[i];
	}

	// --- 1D evaluation kernel function - Expression = CudaMatrix_Row - nonunitary step - (needed for SubExpression assignments)
	template <class B>
	__global__ inline void evaluation_submatrix_row_step(CudaSubMatrixExprRowStep<B*,B> e, B* data_, int NumElements)
	{ 
		const int i = blockDim.x * blockIdx.x + threadIdx.x;
		if(i < NumElements) e[i] = data_[i];
	}

	// --- 1D evaluation kernel function - Expression = constant (needed for SubExpression assignments)
	template <class B>
	__global__ inline void evaluation_submatrix_constant(CudaSubMatrixExpr<B*,B> e, B c, int NumElements)
	{ 
		const int i = blockDim.x * blockIdx.x + threadIdx.x;
		if(i < NumElements) e[i] = c;
	}

	// --- 1D evaluation kernel call - Expression = Dmatrix (needed for SubExpression assignments)
	template <class B>		
	inline void eval_submatrix_wrap(CudaSubMatrixExpr<B*,B> e, B *data_, int NumElements)
	{
		dim3 dimGrid(iDivUp(NumElements,dimBlock.x*streams.GetNumStreams()));
		evaluation_submatrix<<<dimGrid,dimBlock,0,streams.GetActiveStream()>>>(e,data_,NumElements);
		CudaCheckError();
		//gpuErrchk( cudaPeekAtLastError() );
		//gpuErrchk( cudaDeviceSynchronize() );
	}			
	
	// --- 1D evaluation kernel call - Expression = CudaMatrix_Row - unitary step - (needed for SubExpression assignments)
	template <class B>		
	inline void eval_submatrix_row_wrap(CudaSubMatrixExprRow<B*,B> e, B *data_, int NumElements)
	{
		dim3 dimGrid(iDivUp(NumElements,dimBlock.x));
		evaluation_submatrix_row<<<dimGrid,dimBlock,0,streams.GetActiveStream()>>>(e,data_,NumElements);
		CudaCheckError();
		//gpuErrchk( cudaPeekAtLastError() );
		//gpuErrchk( cudaDeviceSynchronize() );
	}			

	// --- 1D evaluation kernel call - Expression_Row = Expression - unitary step - (needed for SubExpression assignments)
	template <class A, class B, class C>		
	inline void eval_submatrix_row_wrap(Expr<A,B> e, Expr<C,B> data_, int NumElements)
	{
		dim3 dimGrid(iDivUp(NumElements,dimBlock.x));
		evaluation_submatrix_row<<<dimGrid,dimBlock,0,streams.GetActiveStream()>>>(e,data_,NumElements);
		CudaCheckError();
		//gpuErrchk( cudaPeekAtLastError() );
		//gpuErrchk( cudaDeviceSynchronize() );
	}			

	// --- 1D evaluation kernel call - Expression = CudaMatrix_Row - nonunitary- step (needed for SubExpression assignments)
	template <class B>		
	inline void eval_submatrix_row_step_wrap(CudaSubMatrixExprRowStep<B*,B> e, B *data_, int NumElements)
	{
		dim3 dimGrid(iDivUp(NumElements,dimBlock.x*streams.GetNumStreams()));
		evaluation_submatrix_row_step<<<dimGrid,dimBlock,0,streams.GetActiveStream()>>>(e,data_,NumElements);
		CudaCheckError();
		//gpuErrchk( cudaPeekAtLastError() );
		//gpuErrchk( cudaDeviceSynchronize() );
	}			

	// --- Assignment Expression = Scalar
	template <class B>		
	inline void eval_submatrix_constant_wrap(SubMatrixExpr<B*,B> e, B c, int NumElements)
	{
		dim3 dimGrid(iDivUp(NumElements,dimBlock.x*streams.GetNumStreams()));
		evaluation_submatrix_constant<<<dimGrid,dimBlock,0,streams.GetActiveStream()>>>(e,c,NumElements);
		CudaCheckError();
	}			

	// --- Controllare se non è inutile
	template <class B>
	void evaluation_submatrix_function(CudaSubMatrixExpr<B*,B> e, B* data, int NumElements) { eval_submatrix_wrap(e, data, NumElements); }
			
	// --- Controllare se non è inutile
	template <class B>
	void evaluation_submatrix_function(CudaSubMatrixExprRow<B*,B> e, B* data, int NumElements) { eval_submatrix_row_wrap(e, data, NumElements); }

	template <class A, class B, class C>
	void evaluation_submatrix_function_expression(Expr<A,B> e, Expr<C,B> data, int NumElements) { eval_submatrix_row_wrap(e, data, NumElements); }

	// --- Controllare se non è inutile
	template <class B>
	void evaluation_submatrix_function(CudaSubMatrixExprRowStep<B*,B> e, B* data, int NumElements) { eval_submatrix_row_step_wrap(e, data, NumElements); }

	// --- Assignment Expression = Scalar
	template <class B>
	void evaluation_submatrix_function_constant(CudaSubMatrixExpr<B*,B> e, B c, int NumElements) { eval_submatrix_constant_wrap(e, c, NumElements); }

	template <class B>
	void assign_cpu_matrix_to_gpu_expression(CudaSubMatrixExprRow<B*,B> e, const Hmatrix<B> &ob) 
	{ 					
		//Dmatrix<B> temp(ob);
		//evaluation_submatrix_function(e,temp.data_,ob.GetNumElements()); 
		//std::cout << "pointer " << e.GetDataPointer() << "\n";
		if (ob.IsPinned()) { CudaSafeCall(cudaMemcpyAsync(e.GetDataPointer(),ob.GetDataPointer(),ob.GetNumElements()*sizeof(B),cudaMemcpyHostToDevice,streams.GetActiveStream())); }
		else				CudaSafeCall(cudaMemcpy(e.GetDataPointer(),ob.GetDataPointer(),ob.GetNumElements()*sizeof(B),cudaMemcpyHostToDevice));
		//CudaSafeCall(cudaMemcpy(e.GetDataPointer(),ob.GetDataPointer(),ob.GetNumElements()*sizeof(B),cudaMemcpyHostToDevice));
	}
	
	template <class B>
	void assign_gpu_expression_to_cpu_expression(CudaSubMatrixExprRow<B*,B> e, CudaSubMatrixExprRow<B*,B> ob, const int NumElements) 
	{ 					
		if (e.IsPinned()) CudaSafeCall(cudaMemcpyAsync(e.GetDataPointer(),ob.GetDataPointer(),NumElements*sizeof(B),cudaMemcpyDeviceToHost,streams.GetActiveStream()));
		else			  CudaSafeCall(cudaMemcpy(e.GetDataPointer(),ob.GetDataPointer(),NumElements*sizeof(B),cudaMemcpyDeviceToHost));
		//CudaSafeCall(cudaMemcpy(e.GetDataPointer(),ob.GetDataPointer(),ob.GetNumElements()*sizeof(B),cudaMemcpyHostToDevice));
	}

	template <class B>
	void assign_cpu_expression_to_gpu_expression(CudaSubMatrixExprRow<B*,B> e, CudaSubMatrixExprRow<B*,B> ob, const int NumElements) 
	{ 					
		if (ob.IsPinned()) { CudaSafeCall(cudaMemcpyAsync(e.GetDataPointer(),ob.GetDataPointer(),NumElements*sizeof(B),cudaMemcpyHostToDevice,streams.GetActiveStream())); }
		else			   CudaSafeCall(cudaMemcpy(e.GetDataPointer(),ob.GetDataPointer(),NumElements*sizeof(B),cudaMemcpyHostToDevice));
		//CudaSafeCall(cudaMemcpy(e.GetDataPointer(),ob.GetDataPointer(),ob.GetNumElements()*sizeof(B),cudaMemcpyHostToDevice));
	}

	template <class B>
	void assign_cpu_expression_to_gpu_expression(CudaSubMatrixExprColumn<B*,B> e, CudaSubMatrixExprColumn<B*,B> ob, const int NumElements) 
	{ 					
	}

}

#endif