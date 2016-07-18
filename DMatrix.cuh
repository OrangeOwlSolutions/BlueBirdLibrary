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


#ifndef __DMATRIX_CUH__
#define __DMATRIX_CUH__

#include "BB.h"

//#include <ostream>							// Needed for ostream

#include "ComplexTypes.cuh"					// Needed for complex types and conversions between types, DefaultType
#include "Constants.h"						// Needed for DefaultType
#include "HExceptions.h"						// Needed for GenericError
//
//#include "Scalar.cuh"	
//////#include "Expression.cuh"
//#include "DStreams.cuh"
//#include "Constants.h"				// Needed for BLOCKSIZE
//#include "DExceptions.cuh"				// Needed for CudaCheckError
////
////int iDivUp(const int a, const int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

//namespace BB { extern dim3 dimBlock(BLOCKSIZE); }

/*****************************/
/* KERNEL FUNCTIONS		     */
/* FUNCTIONS CALLING KERNELS */ 
/*****************************/

//// --- 1D evaluation kernel function - Hmatrix = Expression
//template <class A, class T1, class T2>
//__global__ inline void evaluation_matrix(T1 *data_, const Expr<A,T2> e, int NumElements)
//{
//	const int i = blockDim.x * blockIdx.x + threadIdx.x;
//	if(i < NumElements) data_[i] = e[i];
//}
//
//// --- 1D evaluation kernel call - Hmatrix = Expression
//template<class A, class T1, class T2>
//inline void eval_matrix_wrap(T1 *data_, const Expr<A,T2> e, int NumElements)
//{
//	//dim3 dimGrid(iDivUp(NumElements,dimBlock.x*streams.GetNumStreams()));
//	dim3 dimGrid(iDivUp(NumElements,dimBlock.x));
//	evaluation_matrix<<<dimGrid,dimBlock,0,streams.GetActiveStream()>>>(data_,e,NumElements);
//    CudaCheckError();
//	//gpuErrchk( cudaPeekAtLastError() );
//	//gpuErrchk( cudaDeviceSynchronize() );
//}

////// --- 1D evaluation kernel function - Expression = Hmatrix (needed for SubExpression assignments)
////template <class A, class T1, class T2>
////__global__ inline void evaluation_submatrix(CudaSubMatrixExpr<T1*,T1> e, T2 *data_, int NumElements)
////{
////	const int i = blockDim.x * blockIdx.x + threadIdx.x;
////	if(i < NumElements) e[i] = data_[i];
////}
////
////// --- 1D evaluation kernel call - Expression = Hmatrix (needed for SubExpression assignments)
////template<class A, class T1, class T2>
////inline void eval_submatrix_wrap(CudaSubMatrixExpr<T1*,T1> e, T2 *data_, int NumElements)
////{
////	dim3 dimGrid(iDivUp(NumElements,dimBlock.x));
////	evaluation_submatrix<<<dimGrid,dimBlock>>>(e,data_,NumElements);
////    CudaCheckError();
////	//gpuErrchk( cudaPeekAtLastError() );
////	//gpuErrchk( cudaDeviceSynchronize() );
////}

//template<class A, class T1, class T2> void evaluation_matrix_function(T1* data, const Expr<A,T2> e, int NumElements)	  { eval_matrix_wrap(data, e, NumElements); }
////template<class T1, class T2>		  void evaluation_submatrix_function(CudaSubMatrixExpr<T1*,T1> e, T2* data, int NumElements) { eval_submatrix_wrap(e, data, NumElements); }
//template<class T1, class T2>		  void evaluation_matrix_function(T1* data, T2*					ob, int NumElements)	  { eval_matrix_wrap(data, ob,NumElements); }
//template void evaluation_matrix_function(BB::int2_*,BB::int2_*,int);

#include <iostream>

namespace BB
{
	
	template <typename OutType=DefaultType>
	class DeviceReferenceWrapper
	{
		public:
			explicit DeviceReferenceWrapper(void* ptr) : ptr_(ptr) {}

			DeviceReferenceWrapper& operator=(OutType val)
			{
				cudaMemcpy(ptr_,&val,sizeof(OutType),cudaMemcpyHostToDevice);
				return *this;
			}

			operator OutType() const
			{
				OutType val;
				cudaMemcpy(&val,ptr_,sizeof(OutType),cudaMemcpyDeviceToHost);
				return val;
			}

		private:
			void* ptr_;
	};
	
	/*********************/
	/* CUDA MATRIX CLASS */
	/*********************/
	template <typename OutType=DefaultType>
	class Dmatrix
	{

		//template <class OutType> class Hmatrix;
		//template <typename T> friend ostream  & operator << (ostream &, const Dmatrix<T> &) ;

		private :
			int which_device_;				// Device where the data reside
			int Rows_;						// Number of rows
			int Columns_;					// Number of columns
			OutType *data_;					// Row-major order allocation
		
		public:
    
			/** @name Constructors and Destructor
			 */
			//@{	
			/*! \brief Constructor. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int> foo(3,4);
			 *
			 *		defines an integer, 3x4, non-pinned device matrix. 
			 *
			 *      Dmatrix<int> foo(3,4);
			 *
			 *		defines an integer, 3x4, device matrix. 
			*/
			Dmatrix(const int Rows,const int Columns);
			/*! \brief Constructor and inizializer on Dmatrix. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int> foo1(3,4);
			 *      Dmatrix<int> foo2(foo1);
			 *
			 *		defines and initializes an integer device matrix foo2 according to size and values of foo1. 
			 *
			*/
			Dmatrix(const Dmatrix &ob);
			/*! \brief Constructor and inizializer on Hmatrix. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int> foo1(3,4);
			 *      Dmatrix<int> foo2(foo1);
			 *
			 *		defines and initializes an integer device matrix foo2 according to size and values of foo1. The operation inclues
			 *      data memory transfers from host to device.
			 *
			*/
			Dmatrix(const BB::Hmatrix<OutType>&);
			/*! \brief Constructor and inizializer on expression. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int> foo1(3,4), foo2(3,4), foo3(3,4);
			 *      Dmatrix<int> foo4(foo1*foo2+sin(foo3));
			 *
			 *		defines and initializes an integer device matrix foo4 according to size and values of the input expression. 
			 *      The expression must be a device expression.
			 *
			*/
			template<class A, class B>
			Dmatrix(const Expr<A,B>&e) {
				Rows_ = e.GetRows();
				Columns_ = e.GetColumns();
				cout << "Dmatrix = Expr\n";
				if (e.IsDevice()) {
					CudaSafeCall(cudaMalloc((void **)&data_, Rows_*Columns_*sizeof(OutType)));
					evaluation_matrix_function(data_,e,GetNumElements()); 
				}
				else {
					Hmatrix<OutType> temp(e);
					CudaSafeCall(cudaMalloc((void **)&data_, Rows_*Columns_*sizeof(OutType)));
					CudaSafeCall(cudaMemcpy(data_,temp.GetDataPointer(),GetNumElements()*sizeof(OutType),cudaMemcpyHostToDevice));
				}	
				//else {char* str0 = "*******************************************************\n"; 
				//	  char* str1 = "* Cannot construct a GPU matrix from a CPU expression *\n"; 
				//	  char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)); 
				//	  sprintf(catString, "%s%s%s\n",str0,str1,str0); 
				//	  throw  GenericError(catString,__FILE__,__LINE__); }	
			}
			/*! \brief Destructor. */ 
			~Dmatrix();
    		//@}

			/** @name Utility methods
			 */
			//@{	
			/*! \brief Gets the number of matrix rows. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int> foo(3,4);
			 *      
			 *      foo.GetRows() 
			 *
			 *		returns 3. 
			*/
			__host__ __device__ int GetRows() const;
			/*! \brief Gets the number of matrix columns. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int> foo(3,4);
			 *      
			 *      foo.GetColumns() 
			 *
			 *		returns 4. 
			*/
			__host__ __device__ int GetColumns() const;
			/*! \brief Gets the number of matrix elements. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int> foo(3,4);
			 *      
			 *      foo.GetNumElements() 
			 *
			 *		returns 12. 
			*/
			__host__ __device__ int GetNumElements() const;
			/*! \brief Checks if matrix is a vector. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int> foo(3,4);
			 *      
			 *      foo.IsVector() 
			 *
			 *		returns 0 (false). 
			*/
			__host__ __device__ inline bool IsVector() const;
 			/*! \brief Get data pointer. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int> foo(3,4);
			 *      
			 *      foo.GetDataPointer()[0] 
			 *
			 *		returns foo(0,0). 
			*/
			__host__ __device__ OutType		  * GetDataPointer();
 			/*! \brief Get data pointer. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int> foo(3,4);
			 *      
			 *      foo.GetDataPointer()[0] 
			 *
			 *		returns foo(0,0). 
			*/
			__host__ __device__ OutType const * GetDataPointer() const;
			/*! \brief Reshape matrix (simply changes row and column size, no memory movement). 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int> foo(3,4);
			 *      
			 *      foo.Resize(2,6)
			 *
			 *		reshapes foo from a 3x4 matrix to a 2x6 matrix by taking the elements row-wise. 
			*/
			__host__ void Resize(int NewRows, int NewColumns);
			//@}

			/** @name Access operators
			 */
			//@{	
			// --- Access operators (i commentati sono da fare)
			//__device__ inline OutType & operator[](const int i)							{ return data_[i]; }
			//  __device__ inline OutType   operator[](const int i)				const		{ return data_[i]; }
			//__device__ inline OutType   operator[](const int i)				const		{ return (*this).GetDataPointer()[0]; }
			//__host__ __device__ inline OutType   operator[](const int i)				const		{ return 3; }
			//__host__  inline OutType& operator()(const int i)								{ OutType d; CudaSafeCall(cudaMemcpy(&d,data_+i,sizeof(OutType),cudaMemcpyDeviceToHost)); return d; }
			// --- Access from CPU with single index - return value only

			// Set individual elements of a matrix
			__host__ void Set(int, OutType);
			__host__ void Set(int, int, OutType);

			DeviceReferenceWrapper<OutType> operator()(const int i) { return DeviceReferenceWrapper<OutType>(data_+i); }

			/*! \brief One-index access operator - return value only. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int> foo(3,4);
			 *      
			 *      cout << foo(3) << endl;
			 *
			 *		uses the value returned by foo(3). 
			*/
			__host__ OutType operator()(const int i) const;
			/*! \brief Two-indices access operator - return value only. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int> foo(3,4);
			 *      
			 *      cout << foo(2,1) << endl;
			 *
			 *		uses the value returned by foo(2,1). 
			*/
			__host__ OutType operator()(const int i, const int j) const;
			//__host__   inline OutType *	operator()(const int i, const int j)				{ return &data_[IDX2R(i,j,Columns_)]; }
			//@}

			/** @name Hmatrix assignments
			 */
			//@{	
			/*! \brief CPU Hmatrix to GPU Dmatrix assignment. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int> d_foo(3,4);
			 *      Hmatrix<int> h_foo(3,4);
			 *      
			 *      d_foo = h_foo;
			 *
			 *		assigns the content of h_foo to d_foo. d_foo and h_foo must be of the same type. CPU to GPU memory movements are included in this
			 *      operation.
			*/
			const Dmatrix<OutType> &operator=(const BB::Hmatrix<OutType> &ob);
			/*! \brief GPU integer Dmatrix to GPU Dmatrix assignment. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int2_> foo1(3,4);
			 *      Dmatrix<int> foo2(3,4);
			 *      
			 *      foo1 = foo2;
			 *
			 *		assigns the content of foo2 to foo1.  
			*/
			const Dmatrix& operator=(const Dmatrix<int>&);
			/*! \brief GPU float Dmatrix to GPU Dmatrix assignment. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int2_> foo1(3,4);
			 *      Dmatrix<float> foo2(3,4);
			 *      
			 *      foo1 = foo2;
			 *
			 *		assigns the content of foo2 to foo1.  
			*/
			const Dmatrix& operator=(const Dmatrix<float>&);
			/*! \brief GPU double Dmatrix to GPU Dmatrix assignment. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int2_> foo1(3,4);
			 *      Dmatrix<double> foo2(3,4);
			 *      
			 *      foo1 = foo2;
			 *
			 *		assigns the content of foo2 to foo1. 
			*/
			const Dmatrix& operator=(const Dmatrix<double>&);
			/*! \brief GPU int2_ Dmatrix to GPU Dmatrix assignment. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int2_> foo1(3,4);
			 *      Dmatrix<int2_> foo2(3,4);
			 *      
			 *      foo1 = foo2;
			 *
			 *		assigns the content of foo2 to foo1. foo1 cannot be a real Dmatrix.
			*/
			const Dmatrix& operator=(const Dmatrix<BB::int2_>&);
			/*! \brief GPU float2_ Dmatrix to GPU Dmatrix assignment. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int2_> foo1(3,4);
			 *      Dmatrix<float2_> foo2(3,4);
			 *      
			 *      foo1 = foo2;
			 *
			 *		assigns the content of foo2 to foo1. foo1 cannot be a real Dmatrix.
			*/
			const Dmatrix& operator=(const Dmatrix<BB::float2_>&);
			/*! \brief GPU double2_ Dmatrix to GPU Dmatrix assignment. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int2_> foo1(3,4);
			 *      Dmatrix<double2_> foo2(3,4);
			 *      
			 *      foo1 = foo2;
			 *
			 *		assigns the content of foo2 to foo1. foo1 cannot be a real Dmatrix.
			*/
			const Dmatrix& operator=(const Dmatrix<BB::double2_>&);
			/*! \brief GPU Expression to GPU Dmatrix assignment. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int2_> foo1(3,4);
			 *      Dmatrix<int2_> foo2(3,4);
			 *      Dmatrix<double2_> foo3(3,4);
			 *      
			 *      foo3 = foo1+3*sin(foo2);
			 *
			 *		assigns the content of expression foo1+3*sin(foo2) to foo3. Dmatrix types must be such that the operations between heterogeneous
			 *      types and type castings are defined.
			*/
			template <class A, class T>
			__forceinline__ const Dmatrix<OutType>& operator=(Expr<A,T> e)
			{   
				if((e.GetRows() == Rows_) && (e.GetColumns() == Columns_)) evaluation_matrix_function_expr(data_,e,GetNumElements());
				else 
				{  
					char* str0 = "**********************************************\n";
					char* str1 = "* Size mismatch in GPU=GPU matrix assignment *\n";
					char* str2 = "Left operand size: ";
					char* str3 = "Right operand size: ";
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,e.GetRows(),e.GetColumns());
					throw  GenericError(catString,__FILE__,__LINE__);
				} 
				return *this;
			}
			/*! \brief Constant to GPU Dmatrix assignment. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int2_> foo(3,4);
			 *      int2_ constant(2,3);
			 *      
			 *      foo = constant;
			 *
			 *		assigns the content of constant (real part 2 and imaginary part 3) to foo. Constant and foo must have the same type.
			*/
			const Dmatrix<OutType> & operator=(const OutType);
			/*! \brief GPU SubMatrix to GPU Dmatrix assignment. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int2_> foo1(3,4);
			 *      Dmatrix<int2_> foo2(5,7);
			 *      
			 *      foo1 = foo2(Range(0,2),Range(0,3));
			 *
			 *		assigns the SubMatrix foo2(Range(0,2),Range(0,3)) to foo1.
			*/
			//const Dmatrix<OutType>& operator=(Expr<SubMatrixExprRow<OutType*,OutType>,OutType> e);
			//@}

			/** @name SubMatrix extraction
			*/
			//@{	
			/*! \brief SubMatrix - Range.  
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline __forceinline__ Expr<SubMatrixExprRow<OutType*,OutType>,OutType> operator()(Range range);
			///*! \brief SubMatrix - int Range.  
			// *
			// *	Example:
			// *
			// *      Under construction
			//*/
			__forceinline __forceinline__ Expr<SubMatrixExprRow<OutType*,OutType>,OutType> operator()(int index, Range range2);
			/*! \brief SubMatrix - int RangeStep. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline __forceinline__ Expr<SubMatrixExprRowStep<OutType*,OutType>,OutType> operator()(int index, RangeStep range2);
			/*! \brief SubMatrix - int Span (EXPR SOVRACCARICO). 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline __forceinline__ Expr<SubMatrixExprRow<OutType*,OutType>,OutType> operator()(int index, SpanClass span2);
			/*! \brief SubMatrix - Range int. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline __forceinline__ Expr<SubMatrixExprColumn<OutType*,OutType>,OutType> operator()(Range range1, int index);
			/*! \brief SubMatrix - RangeStep int . 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline __forceinline__ Expr<SubMatrixExprColumnStep<OutType*,OutType>,OutType> operator()(RangeStep range1, int index);
			/*! \brief SubMatrix - Span int. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline __forceinline__ Expr<SubMatrixExprColumn<OutType*,OutType>,OutType> operator()(SpanClass span1, int index);

			/*! \brief SubMatrix - Range Range. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline __forceinline__ Expr<SubMatrixExprColumnRow<OutType*,OutType>,OutType> operator()(Range range1, Range range2);
			/*! \brief SubMatrix - RangeStep Range. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline __forceinline__ Expr<SubMatrixExprRowStepColumn<OutType*,OutType>,OutType> operator()(RangeStep range1, Range range2);
			/*! \brief SubMatrix - RangeStep RangeStep. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline __forceinline__ Expr<SubMatrixExpr<OutType*,OutType>,OutType> operator()(RangeStep range1, RangeStep range2);
			/*! \brief SubMatrix - Range RangeStep.
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline Expr<SubMatrixExprRowColumnStep<OutType*,OutType>,OutType> operator()(Range range1, RangeStep range2);

			//// --- SubExpressions - RangeStep Range (da fare) (EXPR SOVRACCARICO)
			////Expr<CudaSubMatrixExprColumnRow<OutType*,OutType>,OutType> operator()(Range range1, Range range2);

			/*! \brief SubMatrix - Range Span. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline __forceinline__ Expr<SubMatrixExprColumnRow<OutType*,OutType>,OutType> operator()(Range range1, SpanClass span2);
			/*! \brief SubMatrix - Span Range. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline __forceinline__ Expr<SubMatrixExprColumnRow<OutType*,OutType>,OutType> operator()(SpanClass span1, Range range2);
			/*! \brief SubMatrix - Span RangeStep. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline __forceinline__ Expr<SubMatrixExprRowColumnStep<OutType*,OutType>,OutType> operator()(SpanClass span1, RangeStep range2);
			/*! \brief SubMatrix - RangeStep Span. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline __forceinline__ Expr<SubMatrixExprRowStepColumn<OutType*,OutType>,OutType> operator()(RangeStep range1, SpanClass range2);
			///*! \brief SubMatrix - int Hmatrix. 
			// *
			// *	Example:
			// *
			// *      Under construction
			//*/
			//Expr<CudaSubMatrixExprRowPerm<OutType*,OutType>,OutType> operator()(int index, Hmatrix<int> indices2);
			////@}

			//////void CopyToHost(BB::Hmatrix<OutType> &m) const
			//////{
			//////	if((m.GetRows() == Rows_) && (m.GetColumns() == Columns_))
			//////		CudaSafeCall(cudaMemcpy(m.GetDataPointer(),data_,GetNumElements()*sizeof(OutType),cudaMemcpyDeviceToHost));
			//////}

			/******************/
			/* MOVE SEMANTICS */
			/******************/

			/** @name Move semantics
			 */
			//@{	
			/*! \brief Move constructor. */
			Dmatrix(Dmatrix&&);
			/*! \brief Move assignment. */
			Dmatrix & operator=(Dmatrix&&) throw();
			//@}

	}; // class

} // namespace

#endif