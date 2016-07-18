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


#ifndef __OOL_H__
#define __OOL_H__

#include <ostream>				// Needed for ostream

namespace BB
{

	// ComplexTypes
	class int2_;
	class float2_;
	class double2_;

	// Exceptions
	class LibraryException;
	class GenericError;
	struct PrivateLibraryException;

	// Expressions
	template <class A, class B> class Expr;
	template <class A, class B, class op, class OutType> class BinExpr;

	// Expression kernels
	template<class A, class T1, class T2> void evaluation_matrix_function_expr(T1*,const Expr<A,T2>,int);

	// Hmatrix
	template <class OutType> class Hmatrix;

	// Dmatrix
	template <class OutType> class Dmatrix;

	// MatrixExpr
	template <class A, class B> class Expr;

	// SubMatrixExpr
	template <class A, class Type> class CudaSubMatrixExpr;
	template <class A, class Type> class SubMatrixExprRow;
	template <class A, class Type> class CudaSubMatrixExprRowStep;
	template <class A, class Type> class CudaSubMatrixExprRowPerm;
	template <class A, class Type> class CudaSubMatrixExprColumn;
	template <class A, class Type> class SubMatrixExprColumnRow;
	template <class A, class Type> class CudaSubMatrixExprColumnStep;

	// Scalar
	template <class Type> class Scalar;

	// SubIndicesAccessHandling
	class Range;
	class RangeStep;
	class SpanClass;
	struct PrivateRange; 
	struct PrivateRangeStep; 

	// Streams
	class DStreams;
	struct PrivateStreams; 

	// Operations
	class Sum;
	class Sub;
	class Mul;
	class Div;
	
	// Grids
	template <typename A, typename OutType> class GridXExpr;

	// Timing
	class TimingCPU;
	class TimingGPU;
	struct PrivateTimingCPU; 
	struct PrivateTimingGPU; 

}

#endif