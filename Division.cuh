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


#ifndef __CUDADIVOP_CUH__
#define __CUDADIVOP_CUH__

#include "BB.h"
#include "Promotion.cuh"

namespace BB
{

	class Div 
	{
		public:
			// --- Real-Real
			__host__ __device__ __forceinline__ static Promotion<int,int>::strongest			eval(const int a, const int b)				{ return a/b; }
			__host__ __device__ __forceinline__ static Promotion<int,float>::strongest			eval(const int a, const float b)			{ return a/b; }
			__host__ __device__ __forceinline__ static Promotion<float,int>::strongest			eval(const float a, const int b)			{ return a/b; }
			__host__ __device__ __forceinline__ static Promotion<int,double>::strongest			eval(const int a, const double b)			{ return a/b; }
			__host__ __device__ __forceinline__ static Promotion<double,int>::strongest			eval(const double a, const int b)			{ return a/b; }
			__host__ __device__ __forceinline__ static Promotion<float,float>::strongest		eval(const float a, const float b)			{ return a/b; }
			__host__ __device__ __forceinline__ static Promotion<float,double>::strongest		eval(const float a, const double b)			{ return a/b; }
			__host__ __device__ __forceinline__ static Promotion<double,float>::strongest		eval(const double a, const float b)			{ return a/b; }
			__host__ __device__ __forceinline__ static Promotion<double,double>::strongest		eval(const double a, const double b)		{ return a/b; }
			// --- Complex-Real
			__host__ __device__ __forceinline__ static Promotion<int2_,int>::strongest			eval(const int2_ a, const int b)			{ Promotion<int2_,int>::strongest				c; c.c.x = a.c.x/b; c.c.y = a.c.y/b; return c; }
			__host__ __device__ __forceinline__ static Promotion<int2_,float>::strongest		eval(const int2_ a, const float b)			{ Promotion<int2_,float>::strongest				c; c.c.x = a.c.x/b; c.c.y = a.c.y/b; return c; }
			__host__ __device__ __forceinline__ static Promotion<int2_,double>::strongest		eval(const int2_ a, const double b)			{ Promotion<int2_,double>::strongest			c; c.c.x = a.c.x/b; c.c.y = a.c.y/b; return c; }
			__host__ __device__ __forceinline__ static Promotion<float2_,int>::strongest		eval(const float2_ a, const int b)			{ Promotion<float2_,int>::strongest				c; c.c.x = a.c.x/b; c.c.y = a.c.y/b; return c; }
			__host__ __device__ __forceinline__ static Promotion<float2_,float>::strongest		eval(const float2_ a, const float b)		{ Promotion<float2_,float>::strongest			c; c.c.x = a.c.x/b; c.c.y = a.c.y/b; return c; }
			__host__ __device__ __forceinline__ static Promotion<float2_,double>::strongest		eval(const float2_ a, const double b)		{ Promotion<float2_,double>::strongest			c; c.c.x = a.c.x/b; c.c.y = a.c.y/b; return c; }
			__host__ __device__ __forceinline__ static Promotion<double2_,int>::strongest		eval(const double2_ a, const int b)			{ Promotion<double2_,int>::strongest			c; c.c.x = a.c.x/b; c.c.y = a.c.y/b; return c; }
			__host__ __device__ __forceinline__ static Promotion<double2_,float>::strongest		eval(const double2_ a, const float b)		{ Promotion<double2_,float>::strongest			c; c.c.x = a.c.x/b; c.c.y = a.c.y/b; return c; }
			__host__ __device__ __forceinline__ static Promotion<double2_,double>::strongest	eval(const double2_ a, const double b)		{ Promotion<double2_,double>::strongest			c; c.c.x = a.c.x/b; c.c.y = a.c.y/b; return c; }

			// --- Real-Complex
			__host__ __device__ __forceinline__ static Promotion<int,int2_>::strongest			eval(const int a, const int2_ b)			{ Promotion<int,int2_>::strongest				c; Root<int,int2_>::root den; den = b.c.x*b.c.x+b.c.y*b.c.y; c.c.x = a*b.c.x/den; c.c.y = -a*b.c.y/den; return c; }
			__host__ __device__ __forceinline__ static Promotion<float,int2_>::strongest		eval(const float a, const int2_ b)			{ Promotion<float,int2_>::strongest				c; Root<float,int2_>::root den; den = b.c.x*b.c.x+b.c.y*b.c.y; c.c.x = a*b.c.x/den; c.c.y = -a*b.c.y/den; return c; }
			__host__ __device__ __forceinline__ static Promotion<double,int2_>::strongest		eval(const double a, const int2_ b)			{ Promotion<double,int2_>::strongest			c; Root<double,int2_>::root den; den = b.c.x*b.c.x+b.c.y*b.c.y; c.c.x = a*b.c.x/den; c.c.y = -a*b.c.y/den; return c; }
			__host__ __device__ __forceinline__ static Promotion<int,float2_>::strongest		eval(const int a, const float2_ b)			{ Promotion<int,float2_>::strongest				c; Root<int,float2_>::root den; den = b.c.x*b.c.x+b.c.y*b.c.y; c.c.x = a*b.c.x/den; c.c.y = -a*b.c.y/den; return c; }
			__host__ __device__ __forceinline__ static Promotion<float,float2_>::strongest		eval(const float a, const float2_ b)		{ Promotion<float,float2_>::strongest			c; Root<float,float2_>::root den; den = b.c.x*b.c.x+b.c.y*b.c.y; c.c.x = a*b.c.x/den; c.c.y = -a*b.c.y/den; return c; }
			__host__ __device__ __forceinline__ static Promotion<double,float2_>::strongest		eval(const double a, const float2_ b)		{ Promotion<double,float2_>::strongest			c; Root<double,float2_>::root den; den = b.c.x*b.c.x+b.c.y*b.c.y; c.c.x = a*b.c.x/den; c.c.y = -a*b.c.y/den; return c; }
			__host__ __device__ __forceinline__ static Promotion<int,double2_>::strongest		eval(const int a, const double2_ b)			{ Promotion<int,double2_>::strongest			c; Root<int,double2_>::root den; den = b.c.x*b.c.x+b.c.y*b.c.y; c.c.x = a*b.c.x/den; c.c.y = -a*b.c.y/den; return c; }
			__host__ __device__ __forceinline__ static Promotion<float,double2_>::strongest		eval(const float a, const double2_ b)		{ Promotion<float,double2_>::strongest			c; Root<float,double2_>::root den; den = b.c.x*b.c.x+b.c.y*b.c.y; c.c.x = a*b.c.x/den; c.c.y = -a*b.c.y/den; return c; }
			__host__ __device__ __forceinline__ static Promotion<double,double2_>::strongest	eval(const double a, const double2_ b)		{ Promotion<double,double2_>::strongest			c; Root<double,double2_>::root den; den = b.c.x*b.c.x+b.c.y*b.c.y; c.c.x = a*b.c.x/den; c.c.y = -a*b.c.y/den; return c; }
			// --- Complex-Complex
			__host__ __device__ __forceinline__ static Promotion<int2_,int2_>::strongest		eval(const int2_ a, const int2_ b)			{ Promotion<int2_,int2_>::strongest				c; Root<int2_,int2_>::root den;			den = (b.c.x*b.c.x)+(b.c.y*b.c.y); c.c.x = ((a.c.x*b.c.x)+(a.c.y*b.c.y))/den; c.c.y = ((a.c.y*b.c.x)-(a.c.x*b.c.y))/den; return c; }
			__host__ __device__ __forceinline__ static Promotion<int2_,float2_>::strongest		eval(const int2_ a, const float2_ b)		{ Promotion<int2_,float2_>::strongest			c; Root<int2_,float2_>::root den;		den = (b.c.x*b.c.x)+(b.c.y*b.c.y); c.c.x = ((a.c.x*b.c.x)+(a.c.y*b.c.y))/den; c.c.y = ((a.c.y*b.c.x)-(a.c.x*b.c.y))/den; return c; }
			__host__ __device__ __forceinline__ static Promotion<float2_,int2_>::strongest		eval(const float2_ a, const int2_ b)		{ Promotion<float2_,int2_>::strongest			c; Root<float2_,int2_>::root den;		den = (b.c.x*b.c.x)+(b.c.y*b.c.y); c.c.x = ((a.c.x*b.c.x)+(a.c.y*b.c.y))/den; c.c.y = ((a.c.y*b.c.x)-(a.c.x*b.c.y))/den; return c; }
			__host__ __device__ __forceinline__ static Promotion<int2_,double2_>::strongest		eval(const int2_ a, const double2_ b)		{ Promotion<int2_,double2_>::strongest			c; Root<int2_,double2_>::root den;		den = (b.c.x*b.c.x)+(b.c.y*b.c.y); c.c.x = ((a.c.x*b.c.x)+(a.c.y*b.c.y))/den; c.c.y = ((a.c.y*b.c.x)-(a.c.x*b.c.y))/den; return c; }
			__host__ __device__ __forceinline__ static Promotion<double2_,int2_>::strongest		eval(const double2_ a, const int2_ b)		{ Promotion<double2_,int2_>::strongest			c; Root<double2_,int2_>::root den;		den = (b.c.x*b.c.x)+(b.c.y*b.c.y); c.c.x = ((a.c.x*b.c.x)+(a.c.y*b.c.y))/den; c.c.y = ((a.c.y*b.c.x)-(a.c.x*b.c.y))/den; return c; }
			__host__ __device__ __forceinline__ static Promotion<float2_,float2_>::strongest	eval(const float2_ a, const float2_ b)		{ Promotion<float2_,float2_>::strongest			c; Root<float2_,float2_>::root den;		den = (b.c.x*b.c.x)+(b.c.y*b.c.y); c.c.x = ((a.c.x*b.c.x)+(a.c.y*b.c.y))/den; c.c.y = ((a.c.y*b.c.x)-(a.c.x*b.c.y))/den; return c; }
			__host__ __device__ __forceinline__ static Promotion<float2_,double2_>::strongest	eval(const float2_ a, const double2_ b)		{ Promotion<float2_,double2_>::strongest		c; Root<float2_,double2_>::root den;	den = (b.c.x*b.c.x)+(b.c.y*b.c.y); c.c.x = ((a.c.x*b.c.x)+(a.c.y*b.c.y))/den; c.c.y = ((a.c.y*b.c.x)-(a.c.x*b.c.y))/den; return c; }
			__host__ __device__ __forceinline__ static Promotion<double2_,float2_>::strongest	eval(const double2_ a, const float2_ b)		{ Promotion<double2_,float2_>::strongest		c; Root<double2_,float2_>::root den;	den = (b.c.x*b.c.x)+(b.c.y*b.c.y); c.c.x = ((a.c.x*b.c.x)+(a.c.y*b.c.y))/den; c.c.y = ((a.c.y*b.c.x)-(a.c.x*b.c.y))/den; return c; }
			__host__ __device__ __forceinline__ static Promotion<double2_,double2_>::strongest	eval(const double2_ a, const double2_ b)	{ Promotion<double2_,double2_>::strongest		c; Root<double2_,double2_>::root den;	den = (b.c.x*b.c.x)+(b.c.y*b.c.y); c.c.x = ((a.c.x*b.c.x)+(a.c.y*b.c.y))/den; c.c.y = ((a.c.y*b.c.x)-(a.c.x*b.c.y))/den; return c; }

	};

	// Scalar-Scalar -- TESTED -- OK
	#define Scalar_Scalar(T1,T2,OpClass,overloaded_operator) template <class T1,class T2> \
	typename Promotion<T1,T2>::strongest overloaded_operator(const T1,const T2);

	// Hmatrix-Scalar Hmatrix -- TESTED -- OK
	#define Matrix_Scalar_Matrix(T1,T2,OpClass,overloaded_operator) template <class T1,class T2> \
	Expr<BinExpr<const T1*,const Scalar<T2>,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const Hmatrix<T1>&,const T2);

	// Hmatrix-Scalar Dmatrix -- TESTED -- OK
	#define Matrix_Scalar_CudaMatrix(T1,T2,OpClass,overloaded_operator) template <class T1,class T2> \
	Expr<BinExpr<const T1*,const Scalar<T2>,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const Dmatrix<T1>&,const T2);

	// Expression-Scalar -- TESTED
	#define Expression_Scalar(Q,T1,T2,OpClass,overloaded_operator) template <class Q,class T1,class T2> \
	Expr<BinExpr<const Expr<Q,T1>,const Scalar<T2>,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const Expr<Q,T1> &v1,const T2 v2) \
	{ \
		Scalar<T2> c(v2); \
		typedef BinExpr<const Expr<Q,T1>,const Scalar <T2>,OpClass,typename Promotion<T1,T2>::strongest> BExpr; \
		return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(v1,c),v1.GetRows(),v1.GetColumns(),v1.IsDevice()); \
	}

	// Scalar-Hmatrix Hmatrix -- TESTED -- OK
	#define Scalar_Matrix_Matrix(T1,T2,OpClass,overloaded_operator) template <class T1,class T2> \
	Expr<BinExpr<const Scalar<T2> ,const T1*,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const T2,const Hmatrix<T1>&);

	// Scalar-Hmatrix Dmatrix -- TESTED -- OK
	#define Scalar_Matrix_CudaMatrix(T1,T2,OpClass,overloaded_operator) template <class T1,class T2> \
	Expr<BinExpr<const Scalar<T2>,const T1*,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const T2,const Dmatrix<T1>&);

	// Scalar-Expression -- TESTED
	#define Scalar_Expression(Q,T1,T2,OpClass,overloaded_operator) template <class Q,class T1,class T2> \
	Expr<BinExpr<const Scalar<T2>,const Expr<Q,T1>,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const T2 v2,const Expr<Q,T1> &v1) \
	{ \
		Scalar<T2> c(v2); \
		typedef BinExpr<const Scalar<T2>,const Expr<Q,T1>,OpClass,typename Promotion<T1,T2>::strongest> BExpr; \
		return Expr<BExpr,typename Promotion<T1,T2>::strongest>(BExpr(c,v1),v1.GetRows(),v1.GetColumns(),v1.IsDevice()); \
	}

	// Hmatrix-Hmatrix Hmatrix -- TESTED -- OK
	#define Matrix_Matrix_Matrix(T1,T2,OpClass,overloaded_operator) template<class T1,class T2> \
	Expr<BinExpr<const T1*,const T2*,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const Hmatrix<T1>&,const Hmatrix<T2>&);

	// Hmatrix-Hmatrix Dmatrix -- TESTED -- OK
	#define Matrix_Matrix_CudaMatrix(T1,T2,OpClass,overloaded_operator) template <class T1,class T2> \
	Expr<BinExpr<const T1*,const T2*,OpClass,typename Promotion<T1,T2>::strongest>,typename Promotion<T1,T2>::strongest> overloaded_operator(const Dmatrix<T1>&,const Dmatrix<T2>&);

	// Expression-Expression -- TESTED
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

	// Expression-Hmatrix Hmatrix --- TESTED
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

	// Expression-Hmatrix Dmatrix --- TESTED
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

	// Hmatrix-Expression Hmatrix -- TESTED
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

	// Hmatrix-Expression Dmatrix -- TESTED
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

	//Scalar_Scalar(T1,T2,Div,operator/) 	
	Matrix_Scalar_Matrix(T1,T2,Div,operator/)
	Matrix_Scalar_CudaMatrix(T1,T2,Div,operator/)
	Scalar_Matrix_Matrix(T1,T2,Div,operator/)
	Scalar_Matrix_CudaMatrix(T1,T2,Div,operator/)
	Expression_Scalar(Q,T1,T2,Div,operator/)
	Scalar_Expression(Q,T1,T2,Div,operator/)
	Matrix_Matrix_Matrix(T1,T2,Div,operator/)
	Matrix_Matrix_CudaMatrix(T1,T2,Div,operator/)
	Expression_Expression(Q1,T1,Q2,T2,Div,operator/)
	Expression_Matrix_Matrix(Q,T1,T2,Div,operator/)
	Expression_Matrix_CudaMatrix(Q,T1,T2,Div,operator/)
	Matrix_Expression_Matrix(Q,T1,T2,Div,operator/)
	Matrix_Expression_CudaMatrix(Q,T1,T2,Div,operator/)

}



#endif
