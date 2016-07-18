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


#ifndef __CUDAEXPRESSIONMACROS_CUH__
#define __CUDAEXPRESSIONMACROS_CUH__

#include <typeinfo>
#include <iostream>
#include <tuple>

// Scalar-Scalar Binary Operators -- TESTED
#define Scalar_Scalar(T1,T2,OpClass,overloaded_operator) template <class T1,class T2> \
	static inline typename Promotion<T1,T2>::strongest overloaded_operator(const T1 a,const T2 b) {	return OpClass::eval(a,b); }

// Hmatrix-Scalar Hmatrix Binary Operators -- TESTED
#define Matrix_Scalar_Matrix(T1,T2,OpClass,overloaded_operator) template <class T1,class T2> \
	Expr<BinExpr<const T1*,const Scalar<T2>,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const Hmatrix<T1> &v1,const T2 v2) \
	{ \
		Scalar<T2> c(v2); \
		typedef BinExpr<const T1*,const Scalar<T2>,OpClass,typename Promotion<T1,T2>::strongest> BExpr; \
		return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(v1.GetDataPointer(),c),v1.GetRows(),v1.GetColumns(),ISHOST); \
	}

// Hmatrix-Scalar Dmatrix Binary Operators -- TESTED
#define Matrix_Scalar_CudaMatrix(T1,T2,OpClass,overloaded_operator) template <class T1,class T2> \
	Expr<BinExpr<const T1*,const Scalar<T2>,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const Dmatrix<T1> &v1,const T2 v2) \
	{ \
		Scalar<T2> c(v2); \
		typedef BinExpr<const T1*,const Scalar<T2>,OpClass,typename Promotion<T1,T2>::strongest> BExpr; \
		return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(v1.GetDataPointer(),c),v1.GetRows(),v1.GetColumns(),ISDEVICE); \
	}

// Expression-Scalar Binary Operators -- TESTED
#define Expression_Scalar(Q,T1,T2,OpClass,overloaded_operator) template <class Q,class T1,class T2> \
	Expr<BinExpr<const Expr<Q,T1>,const Scalar<T2>,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const Expr<Q,T1> &v1,const T2 v2) \
	{ \
		Scalar<T2> c(v2); \
		typedef BinExpr<const Expr<Q,T1>,const Scalar <T2>,OpClass,typename Promotion<T1,T2>::strongest> BExpr; \
		return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(v1,c),v1.GetRows(),v1.GetColumns(),v1.IsDevice()); \
	}
		//Scalar<T2> c(v2,v1.GetRows(),v1.GetColumns()); \
		//return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(v1,c,v1.GetRows(),v1.GetColumns()),v1.GetRows(),v1.GetColumns(),v1.IsDevice()); \

// Scalar-Hmatrix Hmatrix Binary Operators -- TESTED
#define Scalar_Matrix_Matrix(T1,T2,OpClass,overloaded_operator) template <class T1,class T2> \
	Expr<BinExpr<const Scalar<T2> ,const T1*,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const T2 v1,const Hmatrix<T1> &v2) \
	{ \
		Scalar<T2> c(v1); \
		typedef BinExpr<const Scalar<T2>,const T1*,OpClass,typename Promotion<T1,T2>::strongest> BExpr; \
		return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(c,v2.GetDataPointer()),v2.GetRows(),v2.GetColumns(),ISHOST); \
	}

// Scalar-Hmatrix Dmatrix Binary Operators -- TESTED
#define Scalar_Matrix_CudaMatrix(T1,T2,OpClass,overloaded_operator) template <class T1,class T2> \
	Expr<BinExpr<const Scalar<T2>,const T1*,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const T2 v1,const Dmatrix<T1> &v2) \
	{ \
		Scalar<T2> c(v1,v2.GetRows(),v2.GetColumns()); \
		typedef BinExpr<const Scalar<T2>,const T1*,OpClass,typename Promotion<T1,T2>::strongest> BExpr; \
		return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(c,v2.GetDataPointer(),v2.GetRows(),v2.GetColumns()),v2.GetRows(),v2.GetColumns(),ISDEVICE); \
	}

// Scalar-Expression Binary Operators -- TESTED
#define Scalar_Expression(Q,T1,T2,OpClass,overloaded_operator) template <class Q,class T1,class T2> \
	Expr<BinExpr<const Scalar<T2>,const Expr<Q,T1>,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const T2 v2,const Expr<Q,T1> &v1) \
	{ \
		Scalar<T2> c(v2); \
		typedef BinExpr<const Scalar<T2>,const Expr<Q,T1>,OpClass,typename Promotion<T1,T2>::strongest> BExpr; \
		return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(c,v1),v1.GetRows(),v1.GetColumns(),v1.IsDevice()); \
	}

// Hmatrix-Hmatrix Hmatrix Binary Operators -- TESTED
#define Matrix_Matrix_Matrix(T1,T2,OpClass,overloaded_operator) template <class T1,class T2> \
	Expr<BinExpr<const T1*,const T2*,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const Hmatrix<T1> &v1,const Hmatrix<T2> &v2) \
	{ \
		if((v1.GetRows() == v2.GetRows()) && (v1.GetColumns() == v2.GetColumns())) \
		{ \
			typedef BinExpr<const T1*,const T2*,OpClass,typename Promotion<T1,T2>::strongest> BExpr; \
			return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(v1.GetDataPointer(),v2.GetDataPointer()),v1.GetRows(),v1.GetColumns(),ISHOST); \
		} else { char* str0 = "****************************************************************\n"; \
				 char* str1 = "* Size mismatch in binary CPU matrix operation (matrix-matrix) *\n"; \
			char* str2 = "Left operand size: "; \
			char* str3 = "Right operand size: "; \
			char* str4 = "Operation: "; \
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+50*sizeof(char)); \
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n%s%s",str0,str1,str0,str2,v1.GetRows(),v1.GetColumns(),str3,v2.GetRows(),v2.GetColumns(),str4,typeid(OpClass).name()); \
			throw  GenericError(catString,__FILE__,__LINE__); \
		} \
	}
			//return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(v1.GetDataPointer(),v2.GetDataPointer(),v1.GetRows(),v1.GetColumns()),v1.GetRows(),v1.GetColumns(),ISHOST); \

// Hmatrix-Hmatrix Dmatrix Binary Operators -- TESTED
#define Matrix_Matrix_CudaMatrix(T1,T2,OpClass,overloaded_operator) template <class T1,class T2> \
	Expr<BinExpr<const T1*,const T2*,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const Dmatrix<T1> &v1,const Dmatrix<T2> &v2) \
	{ \
		if((v1.GetRows() == v2.GetRows()) && (v1.GetColumns() == v2.GetColumns())) \
		{ \
			typedef BinExpr<const T1*,const T2*,OpClass,typename Promotion<T1,T2>::strongest> BExpr; \
			return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(v1.GetDataPointer(),v2.GetDataPointer()),v1.GetRows(),v1.GetColumns(),ISDEVICE); \
		} else { char* str0 = "****************************************************************\n"; \
				 char* str1 = "* Size mismatch in binary GPU matrix operation (matrix-matrix) *\n"; \
			char* str2 = "Left operand size: "; \
			char* str3 = "Right operand size: "; \
			char* str4 = "Operation: "; \
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+50*sizeof(char)); \
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n%s%s",str0,str1,str0,str2,v1.GetRows(),v1.GetColumns(),str3,v2.GetRows(),v2.GetColumns(),str4,typeid(OpClass).name()); \
			throw  GenericError(catString,__FILE__,__LINE__); \
		} \
	}

// Expression-Expression Binary Operators -- TESTED
#define Expression_Expression(Q1,T1,Q2,T2,OpClass,overloaded_operator) template <class Q1,class T1,class Q2,class T2> \
	Expr<BinExpr<const Expr<Q1,T1>,const Expr<Q2,T2>,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const Expr<Q1,T1> &v1,const Expr<Q2,T2> &v2) \
	{	\
		if((v1.GetRows() == v2.GetRows()) && (v1.GetColumns() == v2.GetColumns())) \
		{ \
			typedef BinExpr<const Expr<Q1,T1>,const Expr<Q2,T2>,OpClass,typename Promotion<T1,T2>::strongest> BExpr; \
			return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(v1,v2),v1.GetRows(),v1.GetColumns(),v1.IsDevice()); \
		} else { char* str0 = "********************************************************************\n"; \
				 char* str1 = "* Size mismatch in binary matrix operation (expression-expression) *\n"; \
			char* str2 = "Left operand size: "; \
			char* str3 = "Right operand size: "; \
			char* str4 = "Operation: "; \
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+50*sizeof(char)); \
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n%s%s",str0,str1,str0,str2,v1.GetRows(),v1.GetColumns(),str3,v2.GetRows(),v2.GetColumns(),str4,typeid(OpClass).name()); \
			throw  GenericError(catString,__FILE__,__LINE__); \
		} \
	}
			//return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(v1,v2,v1.GetRows(),v1.GetColumns()),v1.GetRows(),v1.GetColumns(),v1.IsDevice()); \

//// SubExpression-SubExpression Binary Operators -- TESTED
//#define SubExpression_SubExpression(T1,T2,OpClass,overloaded_operator) template <class T1,class T2> \
//	Expr<BinExpr<const Expr<CudaSubMatrixExpr<T1*,T1>,T1>,const Expr<CudaSubMatrixExpr<T2*,T2>,T2>,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const Expr<CudaSubMatrixExpr<T1*,T1>,T1> &v1,const Expr<CudaSubMatrixExpr<T2*,T2>,T2> &v2) \
//	{	\
//		if((v1.GetRows() == v2.GetRows()) && (v1.GetColumns() == v2.GetColumns())) \
//		{ \
//			typedef BinExpr<const Expr<CudaSubMatrixExpr<T1*,T1>,T1>,const Expr<CudaSubMatrixExpr<T2*,T2>,T2>,OpClass,typename Promotion<T1,T2>::strongest> BExpr; \
//			return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(v1,v2,v1.GetRows(),v1.GetColumns()),v1.GetRows(),v1.GetColumns(),v1.IsDevice()); \
//		} else { char* str0 = "********************************************************************\n"; \
//				 char* str1 = "* Size mismatch in binary matrix operation (expression-expression) *\n"; \
//			char* str2 = "Left operand size: "; \
//			char* str3 = "Right operand size: "; \
//			char* str4 = "Operation: "; \
//			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+50*sizeof(char)); \
//			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n%s%s",str0,str1,str0,str2,v1.GetRows(),v1.GetColumns(),str3,v2.GetRows(),v2.GetColumns(),str4,typeid(OpClass).name()); \
//			throw  GenericError(catString,__FILE__,__LINE__); \
//		} \
//	}

// Expression-Hmatrix Hmatrix Binary Operators --- TESTED
#define Expression_Matrix_Matrix(Q,T1,T2,OpClass,overloaded_operator) template <class Q,class T1,class T2> \
	Expr<BinExpr<const Expr<Q,T1>,const T2*,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const Expr<Q,T1> &v1,const Hmatrix<T2> &v2) \
	{	if((v1.GetRows() == v2.GetRows()) && (v1.GetColumns() == v2.GetColumns())) \
		{ \
			typedef BinExpr<const Expr<Q,T1>,const T2*,OpClass,typename Promotion<T1,T2>::strongest> BExpr; \
			return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(v1,v2.GetDataPointer()),v1.GetRows(),v1.GetColumns(),ISHOST); \
		} else { char* str0 = "********************************************************************\n"; \
				 char* str1 = "* Size mismatch in binary CPU matrix operation (expression-matrix) *\n"; \
			char* str2 = "Left operand size: "; \
			char* str3 = "Right operand size: "; \
			char* str4 = "Operation: "; \
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+50*sizeof(char)); \
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n%s%s",str0,str1,str0,str2,v1.GetRows(),v1.GetColumns(),str3,v2.GetRows(),v2.GetColumns(),str4,typeid(OpClass).name()); \
			throw  GenericError(catString,__FILE__,__LINE__); \
		} \
	}
			//return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(v1,v2.GetDataPointer(),v1.GetRows(),v1.GetColumns()),v1.GetRows(),v1.GetColumns(),ISHOST); \

// Expression-Hmatrix Dmatrix Binary Operators --- TESTED
#define Expression_Matrix_CudaMatrix(Q,T1,T2,OpClass,overloaded_operator) template <class Q,class T1,class T2> \
	Expr<BinExpr<const Expr<Q,T1>,const T2*,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const Expr<Q,T1> &v1,const Dmatrix<T2> &v2) \
	{	if((v1.GetRows() == v2.GetRows()) && (v1.GetColumns() == v2.GetColumns())) \
		{ \
			typedef BinExpr<const Expr<Q,T1>,const T2*,OpClass,typename Promotion<T1,T2>::strongest> BExpr; \
			return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(v1,v2.GetDataPointer()),v1.GetRows(),v1.GetColumns(),ISDEVICE); \
		} else { char* str0 = "********************************************************************\n"; \
				 char* str1 = "* Size mismatch in binary GPU matrix operation (expression-matrix) *\n"; \
			char* str2 = "Left operand size: "; \
			char* str3 = "Right operand size: "; \
			char* str4 = "Operation: "; \
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+50*sizeof(char)); \
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n%s%s",str0,str1,str0,str2,v1.GetRows(),v1.GetColumns(),str3,v2.GetRows(),v2.GetColumns(),str4,typeid(OpClass).name()); \
			throw  GenericError(catString,__FILE__,__LINE__); \
		} \
	}

// Hmatrix-Expression Hmatrix Binary Operators -- TESTED
#define Matrix_Expression_Matrix(Q,T1,T2,OpClass,overloaded_operator) template <class Q,class T1,class T2> \
	Expr<BinExpr<const T2*,const Expr<Q,T1>,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const Hmatrix<T2> &v1,const Expr<Q,T1> &v2) \
	{	if((v1.GetRows() == v2.GetRows()) && (v1.GetColumns() == v2.GetColumns())) \
		{ \
			typedef BinExpr<const T2*,const Expr<Q,T1>,OpClass,typename Promotion<T1,T2>::strongest> BExpr; \
			return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(v1.GetDataPointer(),v2),v1.GetRows(),v1.GetColumns(),ISHOST); \
		} else { char* str0 = "********************************************************************\n"; \
				 char* str1 = "* Size mismatch in binary CPU matrix operation (matrix-expression) *\n"; \
			char* str2 = "Left operand size: "; \
			char* str3 = "Right operand size: "; \
			char* str4 = "Operation: "; \
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+50*sizeof(char)); \
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n%s%s",str0,str1,str0,str2,v1.GetRows(),v1.GetColumns(),str3,v2.GetRows(),v2.GetColumns(),str4,typeid(OpClass).name()); \
			throw  GenericError(catString,__FILE__,__LINE__); \
		} \
	}

// Hmatrix-Expression Dmatrix Binary Operators -- TESETD
#define Matrix_Expression_CudaMatrix(Q,T1,T2,OpClass,overloaded_operator) template <class Q,class T1,class T2> \
	Expr<BinExpr<const T2*,const Expr<Q,T1>,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const Dmatrix<T2> &v1,const Expr<Q,T1> &v2) \
	{	if((v1.GetRows() == v2.GetRows()) && (v1.GetColumns() == v2.GetColumns())) \
		{ \
			typedef BinExpr<const T2*,const Expr<Q,T1>,OpClass,typename Promotion<T1,T2>::strongest> BExpr; \
			return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(v1.GetDataPointer(),v2),v1.GetRows(),v1.GetColumns(),ISDEVICE); \
		} else { char* str0 = "********************************************************************\n"; \
				 char* str1 = "* Size mismatch in binary GPU matrix operation (matrix-expression) *\n"; \
			char* str2 = "Left operand size: "; \
			char* str3 = "Right operand size: "; \
			char* str4 = "Operation: "; \
			char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+50*sizeof(char)); \
			sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n%s%s",str0,str1,str0,str2,v1.GetRows(),v1.GetColumns(),str3,v2.GetRows(),v2.GetColumns(),str4,typeid(OpClass).name()); \
			throw  GenericError(catString,__FILE__,__LINE__); \
		} \
	}

// Function on Scalar Promotion -- TESTED
#define Function_on_Scalar_Promotion(T,FunClass,overloaded_function) template <class T> \
	__host__ __device__ inline typename Promotion<double,T>::strongest overloaded_function(T a) \
	{ \
		return FunClass::eval(a); \
	}

// Function on Matrix Promotion Hmatrix -- TESTED
#define Function_on_Matrix_Promotion_Matrix(T,FunClass,overloaded_function) template<class T> \
	Expr<FunExpr<const T*,FunClass,typename Promotion<double,T>::strongest>,typename Promotion<double,T>::strongest> overloaded_function(const Hmatrix<T> &v) \
	{ \
		typedef FunExpr<const T*,FunClass,typename Promotion<double,T>::strongest> FExpr; \
		return Expr<FExpr,typename Promotion<double,T>::strongest>(FExpr(v.GetDataPointer(),v.GetRows(),v.GetColumns()),v.GetRows(),v.GetColumns(),ISHOST); \
	}

// Function on Matrix Promotion Dmatrix -- TESTED
#define Function_on_Matrix_Promotion_CudaMatrix(T,FunClass,overloaded_function) template<class T> \
	Expr<FunExpr<const T*,FunClass,typename Promotion<double,T>::strongest>,typename Promotion<double,T>::strongest> overloaded_function(const Dmatrix<T> &v) \
	{ \
		typedef FunExpr<const T*,FunClass,typename Promotion<double,T>::strongest> FExpr; \
		return Expr<FExpr,typename Promotion<double,T>::strongest>(FExpr(v.GetDataPointer(),v.GetRows(),v.GetColumns()),v.GetRows(),v.GetColumns(),ISDEVICE); \
	}

// Function on Expression Promotion -- TESTED
#define Function_on_Expression_Promotion(T,Q1,FunClass,overloaded_function) template<class T,class Q1> \
	Expr<FunExpr<const Expr<Q1,T>,FunClass,typename Promotion<double,T>::strongest>,typename Promotion<double,T>::strongest> overloaded_function(const Expr<Q1,T> &v) \
	{ \
		typedef FunExpr<const Expr<Q1,T>,FunClass,typename Promotion<double,T>::strongest> FExpr; \
		return Expr<FExpr,typename Promotion<double,T>::strongest>(FExpr(v,v.GetRows(),v.GetColumns()),v.GetRows(),v.GetColumns(),v.IsDevice()); \
	}

// Function on Scalar -- TESTED
#define Function_on_Scalar(T,FunClass,overloaded_function) template <class T> \
	inline T overloaded_function(T a) \
	{ \
		return FunClass::eval(a); \
	}

// Function on Hmatrix Hmatrix -- TESTED
#define Function_on_Matrix_Matrix(T,FunClass,overloaded_function) template<class T> \
	Expr<FunExpr<const T*,FunClass,T>,T> overloaded_function(const Hmatrix<T> &v) \
	{ \
		typedef FunExpr<const T*,FunClass,T> FExpr; \
		return Expr<FExpr,T>(FExpr(v.GetDataPointer(),v.GetRows(),v.GetColumns()),v.GetRows(),v.GetColumns(),ISHOST); \
	}

// Function on Hmatrix Dmatrix -- TESTED
#define Function_on_Matrix_CudaMatrix(T,FunClass,overloaded_function) template<class T> \
	Expr<FunExpr<const T*,FunClass,T>,T> overloaded_function(const Dmatrix<T> &v) \
	{ \
		typedef FunExpr<const T*,FunClass,T> FExpr; \
		return Expr<FExpr,T>(FExpr(v.GetDataPointer(),v.GetRows(),v.GetColumns()),v.GetRows(),v.GetColumns(),ISDEVICE); \
	}

// Function on Expression -- TESTED
#define Function_on_Expression(T,Q1,FunClass,overloaded_function) template<class T,class Q1> \
	Expr<FunExpr<const Expr<Q1,T>,FunClass,T>,T> overloaded_function(const Expr<Q1,T> &v) \
	{ \
		typedef FunExpr<const Expr<Q1,T>,FunClass,T> FExpr; \
		return Expr<FExpr,T>(FExpr(v,v.GetRows(),v.GetColumns()),v.GetRows(),v.GetColumns(),v.IsDevice()); \
	}

// Function (two arguments) on Scalar -- TESTED
#define Function_on_Scalar2(T1,T2,FunClass,overloaded_function) template <class T1, class T2> \
	inline T1 overloaded_function(T1 a, T2 b) \
	{ \
		return FunClass::eval(a,b); \
	}

// Function (two arguments) on Hmatrix Hmatrix -- TESTED
#define Function_on_Matrix_Matrix2(T1,T2,FunClass,overloaded_function) template<class T1, class T2> \
	Expr<FunExpr2<const T1*,const T2,FunClass,T1>,T1> overloaded_function(const Hmatrix<T1> &v, const T2 w) \
	{ \
		typedef FunExpr2<const T1*,const T2,FunClass,T1> FExpr; \
		return Expr<FExpr,T1>(FExpr(v.GetDataPointer(),w,v.GetRows(),v.GetColumns()),v.GetRows(),v.GetColumns(),ISHOST); \
	}

// Function (two arguments) on Hmatrix Dmatrix -- TESTED
#define Function_on_Matrix_CudaMatrix2(T1,T2,FunClass,overloaded_function) template<class T1, class T2> \
	Expr<FunExpr2<const T1*,const T2,FunClass,T1>,T1> overloaded_function(const Dmatrix<T1> &v, const T2 w) \
	{ \
		typedef FunExpr2<const T1*,const T2,FunClass,T1> FExpr; \
		return Expr<FExpr,T1>(FExpr(v.GetDataPointer(),w,v.GetRows(),v.GetColumns()),v.GetRows(),v.GetColumns(),ISDEVICE); \
	}

// Function (two arguments) on Expression -- 
#define Function_on_Expression2(T1,Q1,T2,FunClass,overloaded_function) template<class T1,class Q1,class T2> \
	Expr<FunExpr2<const Expr<Q1,T1>,const T2,FunClass,T1>,T1> overloaded_function(const Expr<Q1,T1> &v, const T2 w) \
	{ \
		typedef FunExpr2<const Expr<Q1,T1>,const T2,FunClass,T1> FExpr; \
		return Expr<FExpr,T1>(FExpr(v,w,v.GetRows(),v.GetColumns()),v.GetRows(),v.GetColumns(),v.IsDevice()); \
	}

// Function on Scalar Demotion -- TESTED
#define Function_on_Scalar_Demotion(T,FunClass,overloaded_function) template <class T> \
	inline typename Demotion<double,T>::weakest overloaded_function(T a) \
	{ \
		return FunClass::eval(a); \
	}

// Function on Hmatrix Demotion Hmatrix -- TESTED
#define Function_on_Matrix_Demotion_Matrix(T,FunClass,overloaded_function) template<class T> \
	Expr<FunExpr<const T*,FunClass,typename Demotion<double,T>::weakest>,typename Demotion<double,T>::weakest> overloaded_function(const Hmatrix<T> &v) \
	{ \
		typedef FunExpr<const T*,FunClass,typename Demotion<double,T>::weakest> FExpr; \
		return Expr<FExpr,typename Demotion<double,T>::weakest>(FExpr(v.GetDataPointer(),v.GetRows(),v.GetColumns()),v.GetRows(),v.GetColumns(),ISHOST); \
	}

// Function on Hmatrix Demotion Dmatrix -- TESTED
#define Function_on_Matrix_Demotion_CudaMatrix(T,FunClass,overloaded_function) template<class T> \
	Expr<FunExpr<const T*,FunClass,typename Demotion<double,T>::weakest>,typename Demotion<double,T>::weakest> overloaded_function(const Dmatrix<T> &v) \
	{ \
		typedef FunExpr<const T*,FunClass,typename Demotion<double,T>::weakest> FExpr; \
		return Expr<FExpr,typename Demotion<double,T>::weakest>(FExpr(v.GetDataPointer(),v.GetRows(),v.GetColumns()),v.GetRows(),v.GetColumns(),ISDEVICE); \
	}

// Function on Expression Demotion -- TESTED
#define Function_on_Expression_Demotion(T,Q1,FunClass,overloaded_function) template<class T,class Q1> \
	Expr<FunExpr<const Expr<Q1,T>,FunClass,typename Demotion<double,T>::weakest>,typename Demotion<double,T>::weakest> overloaded_function(const Expr<Q1,T> &v) \
	{ \
		typedef FunExpr<const Expr<Q1,T>,FunClass,typename Demotion<double,T>::weakest> FExpr; \
		return Expr<FExpr,typename Demotion<double,T>::weakest>(FExpr(v,v.GetRows(),v.GetColumns()),v.GetRows(),v.GetColumns(),v.IsDevice()); \
	}

/**************/
/* MESHGRID X */
/**************/
// GridX on Hmatrix-Hmatrix Hmatrix -- TESTED
#define MeshGridX_Matrix_Matrix_Matrix(T1,T2,MeshGridClass,MeshGridType) template<class T1,class T2> \
	Expr<MeshGridClass<const T1*,T1>,T1> MeshGridType(const Hmatrix<T1>&a,const Hmatrix<T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClass<const T1*,T1> MGExpr; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return Expr<MGExpr,T1>(MGExpr(a.GetDataPointer(),Rows,Columns),Rows,Columns,ISHOST); \
		}  else {	char* str0 = "*********************************************************************\n"; \
					char* str1 = "* CPU GridX of non-vector elements (matrix-matrix) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	}

// GridX on Hmatrix-Hmatrix Dmatrix -- TESTED
#define MeshGridX_Matrix_Matrix_CudaMatrix(T1,T2,MeshGridClass,MeshGridType) template<class T1,class T2> \
	Expr<MeshGridClass<const T1*,T1>,T1> MeshGridType(const Dmatrix<T1>&a,const Dmatrix<T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClass<const T1*,T1> MGExpr; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return Expr<MGExpr,T1>(MGExpr(a.GetDataPointer(),Rows,Columns),Rows,Columns,ISDEVICE); \
		}  else {	char* str0 = "*********************************************************************\n"; \
					char* str1 = "* GPU GridX of non-vector elements (matrix-matrix) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	} 

// GridX on Expression-Hmatrix Hmatrix -- TESTED
#define MeshGridX_Expression_Matrix_Matrix(T1,Q1,T2,MeshGridClass,MeshGridType) template<class T1,class Q1,class T2> \
	Expr<MeshGridClass<const Expr<Q1,T1>,T1>,T1> MeshGridType(const Expr<Q1,T1>&a,const Hmatrix<T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClass<const Expr<Q1,T1>,T1> MGExpr; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return Expr<MGExpr,T1>(MGExpr(a,Rows,Columns),Rows,Columns,ISHOST); \
		}  else {	char* str0 = "*************************************************************************\n"; \
					char* str1 = "* CPU GridX of non-vector elements (expression-matrix) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	}  

// GridX on Expression-Hmatrix Dmatrix -- TESTED
#define MeshGridX_Expression_Matrix_CudaMatrix(T1,Q1,T2,MeshGridClass,MeshGridType) template<class T1,class Q1,class T2> \
	Expr<MeshGridClass<const Expr<Q1,T1>,T1>,T1> MeshGridType(const Expr<Q1,T1>&a,const Dmatrix<T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClass<const Expr<Q1,T1>,T1> MGExpr; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return Expr<MGExpr,T1>(MGExpr(a,Rows,Columns),Rows,Columns,ISDEVICE); \
		}  else {	char* str0 = "*************************************************************************\n"; \
					char* str1 = "* GPU GridX of non-vector elements (expression-matrix) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	}   

// GridX on Hmatrix-Expression Hmatrix -- TESTED
#define MeshGridX_Matrix_Expression_Matrix(T1,T2,Q2,MeshGridClass,MeshGridType) template<class T1,class T2,class Q2> \
	Expr<MeshGridClass<const T1*,T1>,T1> MeshGridType(const Hmatrix<T1>&a,const Expr<Q2,T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClass<const T1*,T1> MGExpr; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return Expr<MGExpr,T1>(MGExpr(a.GetDataPointer(),Rows,Columns),Rows,Columns,ISHOST); \
		}  else {	char* str0 = "*************************************************************************\n"; \
					char* str1 = "* CPU GridX of non-vector elements (matrix-expression) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	}    

// GridX on Hmatrix-Expression Dmatrix -- TESTED
#define MeshGridX_Matrix_Expression_CudaMatrix(T1,T2,Q2,MeshGridClass,MeshGridType) template<class T1,class T2,class Q2> \
	Expr<MeshGridClass<const T1*,T1>,T1> MeshGridType(const Dmatrix<T1>&a,const Expr<Q2,T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClass<const T1*,T1> MGExpr; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return Expr<MGExpr,T1>(MGExpr(a.GetDataPointer(),Rows,Columns),Rows,Columns,ISDEVICE); \
		}  else {	char* str0 = "*************************************************************************\n"; \
					char* str1 = "* GPU GridX of non-vector elements (matrix-expression) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	}     

// GridX on Expression-Expression -- TESTED
#define MeshGridX_Expression_Expression(T1,Q1,T2,Q2,MeshGridClass,MeshGridType) template<class T1,class Q1,class T2,class Q2> \
	Expr<MeshGridClass<const Expr<Q1,T1>,T1>,T1> MeshGridType(const Expr<Q1,T1>&a,const Expr<Q2,T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClass<const Expr<Q1,T1>,T1> MGExpr; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return Expr<MGExpr,T1>(MGExpr(a,Rows,Columns),Rows,Columns,a.IsDevice()); \
		}  else {	char* str0 = "*************************************************************************\n"; \
					char* str1 = "* GridX of non-vector elements (expression-expression) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	}     

/**************/
/* MESHGRID Y */
/**************/
// GridY on Hmatrix-Hmatrix Hmatrix -- NONTESTED
#define MeshGridY_Matrix_Matrix_Matrix(T1,T2,MeshGridClass,MeshGridType) template<class T1,class T2> \
	Expr<MeshGridClass<const T2*,T2>,T2> MeshGridType(const Hmatrix<T1>&a,const Hmatrix<T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClass<const T2*,T2> MGExpr; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return Expr<MGExpr,T2>(MGExpr(b.GetDataPointer(),Rows,Columns),Rows,Columns,b.IsDevice()); \
		}  else {	char* str0 = "*********************************************************************\n"; \
					char* str1 = "* CPU GridY of non-vector elements (matrix-matrix) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	}     

// GridY on Hmatrix-Hmatrix Dmatrix -- NONTESTED
#define MeshGridY_Matrix_Matrix_CudaMatrix(T1,T2,MeshGridClass,MeshGridType) template<class T1,class T2> \
	Expr<MeshGridClass<const T2*,T2>,T2> MeshGridType(const Dmatrix<T1>&a,const Dmatrix<T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClass<const T2*,T2> MGExpr; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return Expr<MGExpr,T2>(MGExpr(b.GetDataPointer(),Rows,Columns),Rows,Columns,ISDEVICE); \
		}  else {	char* str0 = "*********************************************************************\n"; \
					char* str1 = "* GPU GridY of non-vector elements (matrix-matrix) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	}     
 

// GridY on Expression-Hmatrix Hmatrix -- NONTESTED
#define MeshGridY_Expression_Matrix_Matrix(T1,Q1,T2,MeshGridClass,MeshGridType) template<class T1,class Q1,class T2> \
	Expr<MeshGridClass<const T2*,T2>,T2> MeshGridType(const Expr<Q1,T1>&a,const Hmatrix<T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClass<const T2*,T2> MGExpr; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return Expr<MGExpr,T2>(MGExpr(b.GetDataPointer(),Rows,Columns),Rows,Columns,ISHOST); \
		}  else {	char* str0 = "*************************************************************************\n"; \
					char* str1 = "* CPU GridY of non-vector elements (expression-matrix) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	}     

// GridY on Expression-Hmatrix Dmatrix -- NONTESTED
#define MeshGridY_Expression_Matrix_CudaMatrix(T1,Q1,T2,MeshGridClass,MeshGridType) template<class T1,class Q1,class T2> \
	Expr<MeshGridClass<const T2*,T2>,T2> MeshGridType(const Expr<Q1,T1>&a,const Dmatrix<T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClass<const T2*,T2> MGExpr; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return Expr<MGExpr,T2>(MGExpr(b.GetDataPointer(),Rows,Columns),Rows,Columns,ISDEVICE); \
		}  else {	char* str0 = "*************************************************************************\n"; \
					char* str1 = "* GPU GridY of non-vector elements (expression-matrix) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	}     

// GridY on Hmatrix-Expression Hmatrix -- NONTESTED
#define MeshGridY_Matrix_Expression_Matrix(T1,T2,Q2,MeshGridClass,MeshGridType) template<class T1,class T2,class Q2> \
	Expr<MeshGridClass<const Expr<Q2,T2>,T2>,T2> MeshGridType(const Hmatrix<T1>&a,const Expr<Q2,T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClass<const Expr<Q2,T2>,T2> MGExpr; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return Expr<MGExpr,T2>(MGExpr(b,Rows,Columns),Rows,Columns,ISHOST); \
		} else {	char* str0 = "*************************************************************************\n"; \
					char* str1 = "* CPU GridY of non-vector elements (matrix-expression) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	}      

// GridY on Hmatrix-Expression Dmatrix -- NONTESTED
#define MeshGridY_Matrix_Expression_CudaMatrix(T1,T2,Q2,MeshGridClass,MeshGridType) template<class T1,class T2,class Q2> \
	Expr<MeshGridClass<const Expr<Q2,T2>,T2>,T2> MeshGridType(const Dmatrix<T1>&a,const Expr<Q2,T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClass<const Expr<Q2,T2>,T2> MGExpr; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return Expr<MGExpr,T2>(MGExpr(b,Rows,Columns),Rows,Columns,ISDEVICE); \
		} else {	char* str0 = "*************************************************************************\n"; \
					char* str1 = "* GPU GridY of non-vector elements (matrix-expression) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	}     

// GridY on Expression-Expression -- NONTESTED
#define MeshGridY_Expression_Expression(T1,Q1,T2,Q2,MeshGridClass,MeshGridType) template<class T1,class Q1,class T2,class Q2> \
	Expr<MeshGridClass<const Expr<Q2,T2>,T2>,T2> MeshGridType(const Expr<Q1,T1>&a,const Expr<Q2,T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClass<const Expr<Q2,T2>,T2> MGExpr; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return Expr<MGExpr,T2>(MGExpr(b,Rows,Columns),Rows,Columns,b.IsDevice()); \
		} else {	char* str0 = "*************************************************************************\n"; \
					char* str1 = "* GridY of non-vector elements (expression-expression) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	}     

/************/
/* MESHGRID */
/************/

// Grid on Hmatrix-Hmatrix Hmatrix -- TESTED
#define MeshGrid_Matrix_Matrix_Matrix(T1,T2,MeshGridClassX,MeshGridClassY) template<class T1,class T2> \
	std::tuple<Expr<MeshGridClassX<const T1*,T1>,T1>,Expr<MeshGridClassY<const T2*,T2>,T2>> Grid(const Hmatrix<T1>&a,const Hmatrix<T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClassX<const T1*,T1> MGExprX; \
			typedef MeshGridClassY<const T2*,T2> MGExprY; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return std::make_tuple(Expr<MGExprX,T1>(MGExprX(a.GetDataPointer(),Rows,Columns),Rows,Columns,ISHOST), \
                                   Expr<MGExprY,T2>(MGExprY(b.GetDataPointer(),Rows,Columns),Rows,Columns,ISHOST)); \
		}  else {	char* str0 = "********************************************************************\n"; \
					char* str1 = "* CPU Grid of non-vector elements (matrix-matrix) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	}     

// Grid on Hmatrix-Hmatrix Hmatrix -- TESTED
#define MeshGrid_Matrix_Matrix_CudaMatrix(T1,T2,MeshGridClassX,MeshGridClassY) template<class T1,class T2> \
	std::tuple<Expr<MeshGridClassX<const T1*,T1>,T1>,Expr<MeshGridClassY<const T2*,T2>,T2>> Grid(const Dmatrix<T1>&a,const Dmatrix<T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClassX<const T1*,T1> MGExprX; \
			typedef MeshGridClassY<const T2*,T2> MGExprY; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return std::make_tuple(Expr<MGExprX,T1>(MGExprX(a.GetDataPointer(),Rows,Columns),Rows,Columns,ISDEVICE), \
                                   Expr<MGExprY,T2>(MGExprY(b.GetDataPointer(),Rows,Columns),Rows,Columns,ISDEVICE)); \
		}  else {	char* str0 = "********************************************************************\n"; \
					char* str1 = "* GPU Grid of non-vector elements (matrix-matrix) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	}     
 
// Grid on Expression-Hmatrix Hmatrix -- TESTED
#define MeshGrid_Expression_Matrix_Matrix(T1,Q1,T2,MeshGridClassX,MeshGridClassY) template<class T1,class Q1,class T2> \
	std::tuple<Expr<MeshGridClassX<const Expr<Q1,T1>,T1>,T1>,Expr<MeshGridClassY<const T2*,T2>,T2>> Grid(const Expr<Q1,T1>&a,const Hmatrix<T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClassX<const Expr<Q1,T1>,T1> MGExprX; \
			typedef MeshGridClassY<const T2*,T2> MGExprY; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return std::make_tuple(Expr<MGExprX,T1>(MGExprX(a,Rows,Columns),Rows,Columns,ISHOST), \
			                       Expr<MGExprY,T2>(MGExprY(b.GetDataPointer(),Rows,Columns),Rows,Columns,ISHOST)); \
		}  else {	char* str0 = "************************************************************************\n"; \
					char* str1 = "* CPU Grid of non-vector elements (expression-matrix) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	}     

// Grid on Expression-Hmatrix Dmatrix -- TESTED
#define MeshGrid_Expression_Matrix_CudaMatrix(T1,Q1,T2,MeshGridClassX,MeshGridClassY) template<class T1,class Q1,class T2> \
	std::tuple<Expr<MeshGridClassX<const Expr<Q1,T1>,T1>,T1>,Expr<MeshGridClassY<const T2*,T2>,T2>> Grid(const Expr<Q1,T1>&a,const Dmatrix<T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClassX<const Expr<Q1,T1>,T1> MGExprX; \
			typedef MeshGridClassY<const T2*,T2> MGExprY; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return std::make_tuple(Expr<MGExprX,T1>(MGExprX(a,Rows,Columns),Rows,Columns,ISDEVICE), \
			                       Expr<MGExprY,T2>(MGExprY(b.GetDataPointer(),Rows,Columns),Rows,Columns,ISDEVICE)); \
		}  else {	char* str0 = "************************************************************************\n"; \
					char* str1 = "* GPU Grid of non-vector elements (expression-matrix) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	}     

// Grid on Hmatrix-Expression Hmatrix -- TESTED
#define MeshGrid_Matrix_Expression_Matrix(T1,T2,Q2,MeshGridClassX,MeshGridClassY) template<class T1,class T2,class Q2> \
	std::tuple<Expr<MeshGridClassX<const T1*,T1>,T1>,Expr<MeshGridClassY<const Expr<Q2,T2>,T2>,T2>> Grid(const Hmatrix<T1>&a,const Expr<Q2,T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClassX<const T1*,T1> MGExprX; \
			typedef MeshGridClassY<const Expr<Q2,T2>,T2> MGExprY; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return std::make_tuple(Expr<MGExprX,T1>(MGExprX(a.GetDataPointer(),Rows,Columns),Rows,Columns,ISHOST),Expr<MGExprY,T2>(MGExprY(b,Rows,Columns),Rows,Columns,ISHOST)); \
		} else {	char* str0 = "*************************************************************************\n"; \
					char* str1 = "* CPU GridY of non-vector elements (matrix-expression) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	}      

// Grid on Hmatrix-Expression Dmatrix -- TESTED
#define MeshGrid_Matrix_Expression_CudaMatrix(T1,T2,Q2,MeshGridClassX,MeshGridClassY) template<class T1,class T2,class Q2> \
	std::tuple<Expr<MeshGridClassX<const T1*,T1>,T1>,Expr<MeshGridClassY<const Expr<Q2,T2>,T2>,T2>> Grid(const Dmatrix<T1>&a,const Expr<Q2,T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClassX<const T1*,T1> MGExprX; \
			typedef MeshGridClassY<const Expr<Q2,T2>,T2> MGExprY; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return std::make_tuple(Expr<MGExprX,T1>(MGExprX(a.GetDataPointer(),Rows,Columns),Rows,Columns,ISDEVICE),Expr<MGExprY,T2>(MGExprY(b,Rows,Columns),Rows,Columns,ISDEVICE)); \
		} else {	char* str0 = "*************************************************************************\n"; \
					char* str1 = "* GPU GridY of non-vector elements (matrix-expression) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	}      

// Grid on Expression-Expression -- TESTED
			// Non dovrebbe essere Expr<MGExprY,T1>(MGExprY(b,Rows,Columns),Rows,Columns) con b invece di a ???
#define MeshGrid_Expression_Expression(T1,Q1,T2,Q2,MeshGridClassX,MeshGridClassY) template<class T1,class Q1,class T2,class Q2> \
	std::tuple<Expr<MeshGridClassX<const Expr<Q1,T1>,T1>,T1>,Expr<MeshGridClassY<const Expr<Q2,T2>,T2>,T2>> Grid(const Expr<Q1,T1>&a,const Expr<Q2,T2>&b) \
	{ if((a.IsVector()) && (b.IsVector())) \
		{ \
			typedef MeshGridClassX<const Expr<Q1,T1>,T1> MGExprX; \
			typedef MeshGridClassY<const Expr<Q2,T2>,T2> MGExprY; \
			int Rows	= (b.GetColumns()!=1)?b.GetColumns():b.GetRows(); \
			int Columns = (a.GetColumns()!=1)?a.GetColumns():a.GetRows(); \
			return std::make_tuple(Expr<MGExprX,T1>(MGExprX(a,Rows,Columns),Rows,Columns,a.IsDevice()),Expr<MGExprY,T1>(MGExprY(a,Rows,Columns),Rows,Columns,b.IsDevice())); \
		}  else {	char* str0 = "************************************************************************\n"; \
					char* str1 = "* Grid of non-vector elements (expression-expression) not possible *\n"; \
					char* str2 = "First operand size: "; \
					char* str3 = "Second operand size: "; \
					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
					sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n",str0,str1,str0,str2,a.GetRows(),a.GetColumns(),str3,b.GetRows(),b.GetColumns()); \
					throw  GenericError(catString,__FILE__,__LINE__); \
				} \
	} \

/*************/
/* SUBMATRIX */
/*************/

//// SubMatrix on Hmatrix -- LIGHTLY TESTED
//#define SubMatrix_Matrix(T) template<class T> \
//	Expr<CudaSubMatrixExpr<const T*,T>,T> SubMatrix(const Hmatrix<T>&v,const int a,const int b,const int c,const int d) \
//	{	if((a >= 0) && (a < v.GetRows()) && (a <= b) && (b >= 0) && (b < v.GetRows()) && \
//           (c >= 0) && (c < v.GetColumns()) && (c <= d) && (d >= 0) && (d < v.GetColumns())) \
//		{ \
//			typedef CudaSubMatrixExpr<const T*,T> SExpr; \
//			return Expr<SExpr,T>(SExpr(v.GetDataPointer(),v.GetRows(),v.GetColumns(),a,b,c,d),b-a+1,d-c+1); \
//		} else {	char* str0 = "************************************\n"; \
//					char* str1 = "* CPU SubMatrix indices must match *\n"; \
//					char* str2 = "Hmatrix size: "; \
//					char* str3 = "SubMatrix indices (a,b,c,d): "; \
//					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+10*sizeof(char)); \
//					sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n",str0,str1,str0,str2,v.GetRows(),v.GetColumns(),str3,a,b,c,d); \
//					throw  GenericError(catString,__FILE__,__LINE__); \
//				} \
//	}
//
//// SubMatrix on Dmatrix -- LIGHTLY TESTED
//#define SubMatrix_CudaMatrix(T) template<class T> \
//	Expr<CudaSubMatrixExpr<const T*,T>,T> SubMatrix(const Dmatrix<T>&v,const int a,const int b,const int c,const int d) \
//	{	if((a >= 0) && (a < v.GetRows()) && (a <= b) && (b >= 0) && (b < v.GetRows()) && \
//           (c >= 0) && (c < v.GetColumns()) && (c <= d) && (d >= 0) && (d < v.GetColumns())) \
//		{ \
//			typedef CudaSubMatrixExpr<const T*,T> SExpr; \
//			return Expr<SExpr,T>(SExpr(v.GetDataPointer(),v.GetRows(),v.GetColumns(),a,b,c,d),b-a+1,d-c+1); \
//		} else {	char* str0 = "************************************\n"; \
//					char* str1 = "* GPU SubMatrix indices must match *\n"; \
//					char* str2 = "Hmatrix size: "; \
//					char* str3 = "SubMatrix indices (a,b,c,d): "; \
//					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
//					sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n",str0,str1,str0,str2,v.GetRows(),v.GetColumns(),str3,a,b,c,d); \
//					throw  GenericError(catString,__FILE__,__LINE__); \
//				} \
//	}

//// SubMatrix on Expression -- LIGHTLY TESTED
//#define SubMatrix_Expression(T,Q1) template<class T,class Q1> \
//	Expr<CudaSubMatrixExpr<const Expr<Q1,T>,T>,T> SubMatrix(const Expr<Q1,T> &v,const int a,const int b,const int c,const int d) \
//	{	if((a >= 0) && (a < v.GetRows()) && (a <= b) && (b >= 0) && (b < v.GetRows()) && \
//           (c >= 0) && (c < v.GetColumns()) && (c <= d) && (d >= 0) && (d < v.GetColumns())) \
//		{ \
//			typedef CudaSubMatrixExpr<const Expr<Q1,T>,T> SExpr; \
//			return Expr<SExpr,T>(SExpr(v,v.GetRows(),v.GetColumns(),a,b,c,d),b-a+1,d-c+1); \
//		} else {	char* str0 = "*******************************************\n"; \
//					char* str1 = "* Expression SubMatrix indices must match *\n"; \
//					char* str2 = "Hmatrix size: "; \
//					char* str3 = "SubMatrix indices (a,b,c,d): "; \
//					char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+50*sizeof(char)); \
//					sprintf(catString, "%s%s%s\n%s%i x %i\n%s(%i,%i,%i,%i)\n",str0,str1,str0,str2,v.GetRows(),v.GetColumns(),str3,a,b,c,d); \
//					throw  GenericError(catString,__FILE__,__LINE__); \
//				} \
//	}

//// Function on Expression Promotion -- TESTED
//#define Function_on_Expression_Promotion(T,Q1,FunClass,overloaded_function) template<class T,class Q1> \
//	Expr<FunExpr<const Expr<Q1,T>,FunClass,typename Promotion<double,T>::strongest>,typename Promotion<double,T>::strongest> overloaded_function(const Expr<Q1,T> &v) \
//	{ \
//		typedef FunExpr<const Expr<Q1,T>,FunClass,typename Promotion<double,T>::strongest> FExpr; \
//		return Expr<FExpr,typename Promotion<double,T>::strongest>(FExpr(v,v.GetRows(),v.GetColumns()),v.GetRows(),v.GetColumns()); \
//	}
//
#endif
