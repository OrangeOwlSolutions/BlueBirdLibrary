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


#ifndef __DMATRIXEXPRESSIONKERNELS_CUH__
#define __DMATRIXEXPRESSIONKERNELS_CUH__

namespace BB
{
	//int iDivUp(const int a, const int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

	// --- 1D evaluation kernel function - Dmatrix = Expression
	template <class A, class T1, class T2>
	__global__ void inline evaluation_matrix_expr(T1 *data_, const Expr<A,T2> e, int NumElements)
	{
		const int i = blockDim.x * blockIdx.x + threadIdx.x;
		if(i < NumElements) data_[i] = e[i];
	}

	//template <class A>
	//__global__ void inline evaluation_matrix_expr(double2_ *data_, const Expr<A,double2_> e, int NumElements)
	//{
	//	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	//	if(i < NumElements) { data_[i].x = e[i].x; data_[i].y = e[i].y; }
	//}

	// --- 1D evaluation kernel call - Dmatrix = Expression
	template<class A, class T1, class T2>
	void eval_matrix_wrap_expr(T1 *data_, const Expr<A,T2> e, int NumElements)
	{
		dim3 dimGrid(iDivUp(NumElements,dimBlock.x*streams[p2p.active_GPU].GetNumStreams()));
		evaluation_matrix_expr<<<dimGrid,dimBlock,0,streams[p2p.active_GPU].GetActiveStream()>>>(data_,e,NumElements);
		CudaCheckError();
	}

	//template<class A, class T1, class T2> void evaluation_matrix_function_expr(T1* data, const Expr<A,T2> e, int NumElements)	  { printf("Wrapper function evaluation\n"); eval_matrix_wrap_expr(data, e, NumElements); }
	template<class A, class T1, class T2> void evaluation_matrix_function_expr(T1* data, const Expr<A,T2> e, int NumElements)	  { eval_matrix_wrap_expr(data, e, NumElements); }
	
} // namespace

#endif