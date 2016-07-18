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


#ifndef __MATRIX_H__
#define __MATRIX_H__

#include "BB.h"

//#include <ostream>						// Needed for ostream
#include <iostream>

#include "ComplexTypes.cuh"
#include "Dmatrix.cuh"

namespace BB
{
	template <typename OutType=DefaultType>
	class Hmatrix
	{
		//template <typename Type>			friend std::ostream & operator << (std::ostream &, const Hmatrix<Type> &);

		private:
			int Rows_;			// Number of rows
			int Columns_;		// Number of columns
			int IsPinned_;	
			OutType *data_;		// Row-major order allocation

		public:

			/** @name Constructors and Destructor
			 */
			//@{	
			/*! \brief Constructor. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int> foo(3,4);
			 *
			 *		defines an integer, 3x4, non-pinned host matrix. 
			 *
			 *      Hmatrix<int> foo(3,4,PINNED);
			 *
			 *		defines an integer, 3x4, pinned host matrix. 
			*/
			Hmatrix(const int Rows, const int Columns, const int IsPinned = 0);
			/*! \brief Constructor and inizializer on Hmatrix. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int> foo1(3,4);
			 *      Hmatrix<int> foo2(foo1);
			 *
			 *		defines and initializes an integer, non-pinned host matrix foo2 according to size and values of foo1. 
			 *
			 *      Hmatrix<int> foo1(3,4);
			 *      Hmatrix<int> foo2(foo1,PINNED);
			 *
			 *		defines and initializes an integer, pinned host matrix foo2 according to size and values of foo1. 
			 *
			*/
			Hmatrix(const Hmatrix &ob, const int IsPinned = 0);
			/*! \brief Constructor and inizializer on Dmatrix. 
			 *
			 *	Example:
			 *
			 *      Dmatrix<int> foo1(3,4);
			 *      Hmatrix<int> foo2(foo1);
			 *
			 *		defines and initializes an integer, non-pinned host matrix foo2 according to size and values of foo1. The operation inclues
			 *      data memory transfers from device to host.
			 *
			 *      Dmatrix<int> foo1(3,4);
			 *      Hmatrix<int> foo2(foo1,PINNED);
			 *
			 *		defines and initializes an integer, pinned host matrix foo2 according to size and values of foo1. The operation inclues
			 *      data memory transfers from device to host.
			 *
			*/
			Hmatrix(const Dmatrix<OutType> &ob, const int IsPinned = 0);
			/*! \brief Constructor and inizializer on expression. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int> foo1(3,4), foo2(3,4), foo3(3,4);
			 *      Hmatrix<int> foo4(foo1*foo2+sin(foo3));
			 *
			 *		defines and initializes an integer, non-pinned host matrix foo4 according to size and values of the input expression. 
			 *      The expression must be a host expression.
			 *
			 *      Hmatrix<int> foo1(3,4), foo2(3,4), foo3(3,4);
			 *      Hmatrix<int> foo4(foo1*foo2+sin(foo3),PINNED);
			 *
			 *		defines and initializes an integer, pinned host matrix foo4 according to size and values of the input expression. 
			 *      The expression must be a host expression.
			 *
			*/
			template<class A, class B>
			Hmatrix(const Expr<A,B>&e, const int IsPinned = 0): Rows_(e.GetRows()), Columns_(e.GetColumns()), IsPinned_(IsPinned)
			{
				if (IsPinned_)	{ CudaSafeCall(cudaHostAlloc((void**)&data_,GetNumElements()*sizeof(OutType),cudaHostAllocDefault)); } 
				else			{ data_ = new OutType[GetNumElements()]; }
				for (int i=0; i<GetNumElements(); i++) data_[i] = e[i];
			}
			/*! \brief Destructor. */ 
			~Hmatrix();
			//@}

			/** @name Utility methods
			 */
			//@{	
			/*! \brief Gets the number of matrix rows. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int> foo(3,4);
			 *      
			 *      foo.GetRows() 
			 *
			 *		returns 3. 
			*/
			int GetRows() const;
			/*! \brief Gets the number of matrix columns. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int> foo(3,4);
			 *      
			 *      foo.GetColumns() 
			 *
			 *		returns 4. 
			*/
			int GetColumns() const;
			/*! \brief Gets the number of matrix elements. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int> foo(3,4);
			 *      
			 *      foo.GetNumElements() 
			 *
			 *		returns 12. 
			*/
			int GetNumElements() const;
			/*! \brief Returns if the memory has been allocated as PINNED memory. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int> foo(3,4,PINNED);
			 *      
			 *      foo.IsPinned() 
			 *
			 *		returns 1 (true). 
			*/
			bool IsPinned()	const;
			/*! \brief Checks if matrix is a vector. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int> foo(3,4);
			 *      
			 *      foo.IsVector() 
			 *
			 *		returns 0 (false). 
			*/
			bool IsVector() const;
 			/*! \brief Get data pointer. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int> foo(3,4);
			 *      
			 *      foo.GetDataPointer()[0] 
			 *
			 *		returns foo(0,0). 
			*/
			OutType* GetDataPointer();
 			/*! \brief Get data pointer. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int> foo(3,4);
			 *      
			 *      foo.GetDataPointer()[0] 
			 *
			 *		returns foo(0,0). 
			*/
			OutType const* GetDataPointer() const;
			/*! \brief Reshape matrix (simply changes row and column size, no memory movement). 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int> foo(3,4);
			 *      
			 *      foo.Resize(2,6)
			 *
			 *		reshapes foo from a 3x4 matrix to a 2x6 matrix by taking the elements row-wise. 
			*/
			void Resize(int NewRows, int NewColumns);
			//@}

			/** @name Access operators
			 */
			//@{	
			/*! \brief One-index access operator - assignment. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int> foo(3,4);
			 *      
			 *      foo(3)=25;
			 *
			 *		assigns 25 to foo(3). 
			*/
			OutType & operator()(const int);
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
			OutType   operator()(const int)	const;
			/*! \brief Two-indices access operator - assignment. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int> foo(3,4);
			 *      
			 *      foo(2,1)=25;
			 *
			 *		assigns 25 to foo(2,1). 
			*/
			OutType & operator()(const int,const int);
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
			OutType   operator()(const int,const int) const;
			/*! \brief One-index access operator - assignment. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int> foo(3,4);
			 *      
			 *      foo[3]=25;
			 *
			 *		assigns 25 to foo(3). 
			*/
			OutType & operator[](const int);
			/*! \brief One-index access operator - return value only. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int> foo(3,4);
			 *      
			 *      cout << foo[3] << endl;
			 *
			 *		uses the value returned by foo(3). 
			*/
			OutType   operator[](const int) const;
			//@}

			/** @name Hmatrix assignments
			 */
			//@{	
			/*! \brief CPU integer Hmatrix to CPU Hmatrix assignment. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int2_> foo1(3,4);
			 *      Hmatrix<int> foo2(3,4);
			 *      
			 *      foo1 = foo2;
			 *
			 *		assigns the content of foo2 to foo1.  
			*/
			const Hmatrix& operator=(const Hmatrix<int>&);
			/*! \brief CPU float Hmatrix to CPU Hmatrix assignment. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int2_> foo1(3,4);
			 *      Hmatrix<float> foo2(3,4);
			 *      
			 *      foo1 = foo2;
			 *
			 *		assigns the content of foo2 to foo1.  
			*/
			const Hmatrix& operator=(const Hmatrix<float>&);
			/*! \brief CPU double Hmatrix to CPU Hmatrix assignment. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int2_> foo1(3,4);
			 *      Hmatrix<double> foo2(3,4);
			 *      
			 *      foo1 = foo2;
			 *
			 *		assigns the content of foo2 to foo1. 
			*/
			const Hmatrix& operator=(const Hmatrix<double>&);
			/*! \brief CPU int2_ Hmatrix to CPU Hmatrix assignment. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int2_> foo1(3,4);
			 *      Hmatrix<int2_> foo2(3,4);
			 *      
			 *      foo1 = foo2;
			 *
			 *		assigns the content of foo2 to foo1. foo1 cannot be a real Hmatrix.
			*/
			const Hmatrix& operator=(const Hmatrix<int2_>&);
			/*! \brief CPU float2_ Hmatrix to CPU Hmatrix assignment. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int2_> foo1(3,4);
			 *      Hmatrix<float2_> foo2(3,4);
			 *      
			 *      foo1 = foo2;
			 *
			 *		assigns the content of foo2 to foo1. foo1 cannot be a real Hmatrix.
			*/
			const Hmatrix& operator=(const Hmatrix<float2_>&);
			/*! \brief CPU double2_ Hmatrix to CPU Hmatrix assignment. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int2_> foo1(3,4);
			 *      Hmatrix<double2_> foo2(3,4);
			 *      
			 *      foo1 = foo2;
			 *
			 *		assigns the content of foo2 to foo1. foo1 cannot be a real Hmatrix.
			*/
			const Hmatrix& operator=(const Hmatrix<double2_>&);
			/*! \brief CPU Expression to CPU Hmatrix assignment. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int2_> foo1(3,4);
			 *      Hmatrix<int2_> foo2(3,4);
			 *      Hmatrix<double2_> foo3(3,4);
			 *      
			 *      foo3 = foo1+3*sin(foo2);
			 *
			 *		assigns the content of expression foo1+3*sin(foo2) to foo3. Hmatrix types must be such that the operations between heterogeneous
			 *      types and type castings are defined.
			*/
			template <typename A, typename B>
			__forceinline const Hmatrix<OutType>& operator=(Expr<A,B> e)
			{   
				if((e.GetRows() == Rows_) && (e.GetColumns() == Columns_)) for (int i=0; i<GetNumElements(); i++) data_[i] = e[i]; 
				else 
				{  
					char* str0 = "**********************************************\n";
					char* str1 = "* Size mismatch in CPU=CPU matrix assignment *\n";
					char* str2 = "Left operand size: ";
					char* str3 = "Right operand size: ";
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str1)+strlen(str2)+strlen(str3)+40*sizeof(char));
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i",str0,str1,str0,str2,Rows_,Columns_,str3,e.GetRows(),e.GetColumns());
					throw  GenericError(catString,__FILE__,__LINE__);
				} 
				return *this;
			}
			/*! \brief Constant to CPU Hmatrix assignment. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int2_> foo(3,4);
			 *      int2_ constant(2,3);
			 *      
			 *      foo = constant;
			 *
			 *		assigns the content of constant (real part 2 and imaginary part 3) to foo. Constant and foo must have the same type.
			*/
			const Hmatrix<OutType> & operator=(const OutType);
			/*! \brief CPU SubMatrix to CPU Hmatrix assignment. 
			 *
			 *	Example:
			 *
			 *      Hmatrix<int2_> foo1(3,4);
			 *      Hmatrix<int2_> foo2(5,7);
			 *      
			 *      foo1 = foo2(Range(0,2),Range(0,3));
			 *
			 *		assigns the SubMatrix foo2(Range(0,2),Range(0,3)) to foo1.
			*/
			const Hmatrix<OutType>& operator=(Expr<SubMatrixExprRow<OutType*,OutType>,OutType> e);
			//@}

			// --- Add documentation
			const Hmatrix& operator=(const Dmatrix<OutType>&);

			/** @name SubMatrix extraction
			 */
			//@{	
			/*! \brief SubMatrix - Range (EXPR OVERLOAD). 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline Expr<SubMatrixExprRow<OutType*,OutType>,OutType> operator()(Range range);
			/*! \brief SubMatrix - int Range. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline Expr<SubMatrixExprRow<OutType*,OutType>,OutType> operator()(int index, Range range2);
			/*! \brief SubMatrix - int RangeStep. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline Expr<SubMatrixExprRowStep<OutType*,OutType>,OutType> operator()(int index, RangeStep range2);
			/*! \brief SubMatrix - int Span. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline Expr<SubMatrixExprRow<OutType*,OutType>,OutType> operator()(int index, SpanClass span2);
			/*! \brief SubMatrix - Range int. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline Expr<SubMatrixExprColumn<OutType*,OutType>,OutType> operator()(Range range1, int index);
			/*! \brief SubMatrix - RangeStep int. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline Expr<SubMatrixExprColumnStep<OutType*,OutType>,OutType> operator()(RangeStep range1, int index);
			/*! \brief SubMatrix - Span int. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline Expr<SubMatrixExprColumn<OutType*,OutType>,OutType> operator()(SpanClass span1, int index);

			// --- SubExpressions - Range Range 
			/*! \brief SubMatrix - RangeStep RangeStep. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline Expr<SubMatrixExprColumnRow<OutType*,OutType>,OutType> operator()(Range range1, Range range2);
			/*! \brief SubMatrix - Range RangeStep. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline Expr<SubMatrixExprRowColumnStep<OutType*,OutType>,OutType> operator()(Range range1, RangeStep range2);
			/*! \brief SubMatrix - RangeStep Range. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline Expr<SubMatrixExprRowStepColumn<OutType*,OutType>,OutType> operator()(RangeStep range1, Range range2);
			/*! \brief SubMatrix - RangeStep RangeStep. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline Expr<SubMatrixExpr<OutType*,OutType>,OutType> operator()(RangeStep range1, RangeStep range2);
			/*! \brief SubMatrix - Range Span. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline Expr<SubMatrixExprColumnRow<OutType*,OutType>,OutType> operator()(Range range1, SpanClass span2);
			/*! \brief SubMatrix - Span Range. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline Expr<SubMatrixExprColumnRow<OutType*,OutType>,OutType> operator()(SpanClass span1, Range range2);
			/*! \brief SubMatrix - Span RangeStep. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline Expr<SubMatrixExprRowColumnStep<OutType*,OutType>,OutType> operator()(SpanClass span1, RangeStep range2);
			/*! \brief SubMatrix - RangeStep Span. 
			 *
			 *	Example:
			 *
			 *      Under construction
			*/
			__forceinline Expr<SubMatrixExprRowStepColumn<OutType*,OutType>,OutType> operator()(RangeStep range1, SpanClass range2);
			///*! \brief SubMatrix - int Hmatrix. 
			// *
			// *	Example:
			// *
			// *      Under construction
			//*/
			//Expr<CudaSubMatrixExprRowPerm<OutType*,OutType>,OutType> operator()(int index, Hmatrix<int> indices2);
			////@}

			/******************/
			/* MOVE SEMANTICS */
			/******************/

			/** @name Move semantics
			 */
			//@{	
			/*! \brief Move constructor. */
			Hmatrix(Hmatrix&&) throw();
			/*! \brief Move assignment. */
			Hmatrix & operator=(Hmatrix&&) throw();
			//@}

	};

}

#endif
