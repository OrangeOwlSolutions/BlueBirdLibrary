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
#include "ComplexTypes.cuh"
#include "Expression.cuh"
#include "Functions.cuh"
#include "Hmatrix.h"
#include "Dmatrix.cuh"
#include <math.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

// Function on Scalar Promotion -- TESTED
#define Function_on_Scalar_Promotion(T,FunClass,overloaded_function) template <> \
	__host__ __device__ typename BB::Promotion<float,T>::strongest BB::overloaded_function(T a) \
	{ \
		return BB::FunClass::eval(a); \
	}

// Function on Hmatrix Promotion Hmatrix -- TESTED
#define Function_on_Matrix_Promotion_Matrix(T,FunClass,overloaded_function) template <> \
	BB::Expr<BB::FunExpr<const T*,BB::FunClass,typename BB::Promotion<float,T>::strongest>,typename BB::Promotion<float,T>::strongest> BB::overloaded_function(const BB::Hmatrix<T> &v) \
	{ \
		typedef BB::FunExpr<const T*,BB::FunClass,typename BB::Promotion<float,T>::strongest> FExpr; \
		return BB::Expr<FExpr,typename BB::Promotion<float,T>::strongest>(FExpr(v.GetDataPointer(),v.GetRows(),v.GetColumns()),v.GetRows(),v.GetColumns(),ISHOST); \
	}

// Function on Hmatrix Promotion Dmatrix -- TESTED
#define Function_on_Matrix_Promotion_CudaMatrix(T,FunClass,overloaded_function) template<> \
	BB::Expr<BB::FunExpr<const T*,BB::FunClass,typename BB::Promotion<float,T>::strongest>,typename BB::Promotion<float,T>::strongest> BB::overloaded_function(const BB::Dmatrix<T> &v) \
	{ \
		typedef BB::FunExpr<const T*,BB::FunClass,typename BB::Promotion<float,T>::strongest> FExpr; \
		return BB::Expr<FExpr,typename BB::Promotion<float,T>::strongest>(FExpr(v.GetDataPointer(),v.GetRows(),v.GetColumns()),v.GetRows(),v.GetColumns(),ISDEVICE); \
	}

//__host__ __device__ float						BB::Sin::eval(int a)						{ return std::sin((float)a); }
//__host__ __device__ float						BB::Sin::eval(float a)						{ return std::sin(a); }
//__host__ __device__ double						BB::Sin::eval(double a)						{ return std::sin(a); }
//__host__ __device__ BB::float2_	BB::Sin::eval(BB::int2_ a)	{ BB::float2_ c; BB::float2_ b; b.x = (float)a.x; b.y = (float)a.y; c.x = std::sin(b.x)*std::cosh(b.y); c.y = std::cos(b.x)*std::sinh(b.y); return c; }
//__host__ __device__ BB::float2_	BB::Sin::eval(BB::float2_ a)	{ BB::float2_ c;  c.x = std::sin(a.x)*std::cosh(a.y); c.y = std::cos(a.x)*std::sinh(a.y); return c; }
//__host__ __device__ BB::double2_	BB::Sin::eval(BB::double2_ a)	{ BB::double2_ c; c.x = std::sin(a.x)*std::cosh(a.y); c.y = std::cos(a.x)*std::sinh(a.y); return c; }

Function_on_Scalar_Promotion(int,Sin,sin);
Function_on_Scalar_Promotion(float,Sin,sin);
Function_on_Scalar_Promotion(double,Sin,sin);
Function_on_Scalar_Promotion(BB::int2_,Sin,sin);
Function_on_Scalar_Promotion(BB::float2_,Sin,sin);
Function_on_Scalar_Promotion(BB::double2_,Sin,sin);

Function_on_Matrix_Promotion_Matrix(int,Sin,sin);
Function_on_Matrix_Promotion_Matrix(float,Sin,sin);
Function_on_Matrix_Promotion_Matrix(double,Sin,sin);
Function_on_Matrix_Promotion_Matrix(BB::int2_,Sin,sin);
Function_on_Matrix_Promotion_Matrix(BB::float2_,Sin,sin);
Function_on_Matrix_Promotion_Matrix(BB::double2_,Sin,sin);

Function_on_Matrix_Promotion_CudaMatrix(int,Sin,sin);
Function_on_Matrix_Promotion_CudaMatrix(float,Sin,sin);
Function_on_Matrix_Promotion_CudaMatrix(double,Sin,sin);
Function_on_Matrix_Promotion_CudaMatrix(BB::int2_,Sin,sin);
Function_on_Matrix_Promotion_CudaMatrix(BB::float2_,Sin,sin);
Function_on_Matrix_Promotion_CudaMatrix(BB::double2_,Sin,sin);

//__host__ __device__ float						BB::Cos::eval(int a)						{ return std::cos((float)a); }
//__host__ __device__ float						BB::Cos::eval(float a)						{ return std::cos(a); }
//__host__ __device__ double						BB::Cos::eval(double a)						{ return std::cos(a); }
//__host__ __device__ BB::float2_	BB::Cos::eval(BB::int2_ a)	{ BB::float2_ c; BB::float2_ b; b.x = (float)a.x; b.y = (float)a.y; c.x = std::cos(b.x)*std::cosh(b.y); c.y = -std::sin(b.x)*std::sinh(b.y); return c; }
//__host__ __device__ BB::float2_	BB::Cos::eval(BB::float2_ a)	{ BB::float2_ c;  c.x = std::cos(a.x)*std::cosh(a.y); c.y = -std::sin(a.x)*std::sinh(a.y); return c; }
//__host__ __device__ BB::double2_	BB::Cos::eval(BB::double2_ a)	{ BB::double2_ c; c.x = std::cos(a.x)*std::cosh(a.y); c.y = -std::sin(a.x)*std::sinh(a.y); return c; }

Function_on_Scalar_Promotion(int,Cos,cos);
Function_on_Scalar_Promotion(float,Cos,cos);
Function_on_Scalar_Promotion(double,Cos,cos);
Function_on_Scalar_Promotion(BB::int2_,Cos,cos);
Function_on_Scalar_Promotion(BB::float2_,Cos,cos);
Function_on_Scalar_Promotion(BB::double2_,Cos,cos);

Function_on_Matrix_Promotion_Matrix(int,Cos,cos);
Function_on_Matrix_Promotion_Matrix(float,Cos,cos);
Function_on_Matrix_Promotion_Matrix(double,Cos,cos);
Function_on_Matrix_Promotion_Matrix(BB::int2_,Cos,cos);
Function_on_Matrix_Promotion_Matrix(BB::float2_,Cos,cos);
Function_on_Matrix_Promotion_Matrix(BB::double2_,Cos,cos);

Function_on_Matrix_Promotion_CudaMatrix(int,Cos,cos);
Function_on_Matrix_Promotion_CudaMatrix(float,Cos,cos);
Function_on_Matrix_Promotion_CudaMatrix(double,Cos,cos);
Function_on_Matrix_Promotion_CudaMatrix(BB::int2_,Cos,cos);
Function_on_Matrix_Promotion_CudaMatrix(BB::float2_,Cos,cos);
Function_on_Matrix_Promotion_CudaMatrix(BB::double2_,Cos,cos);

///****************/
	///* COSINE CLASS */
	///****************/
	//class Cos
	//{
	//	public:
	//		__host__ __device__ static inline float		eval(int a)			{ return cos((float)a); }
	//		__host__ __device__ static inline float		eval(float a)		{ return cos(a); }
	//		__host__ __device__ static inline double	eval(double a)		{ return cos(a); }
	//		__host__ __device__ static inline float2_	eval(int2_ a)		{ float2_ c; float2_ b; b.x = (float)a.x; b.y = (float)a.y; c.x = cos(b.x)*cosh(b.y); c.y = -sin(b.x)*sinh(b.y); return c; }
	//		__host__ __device__ static inline float2_	eval(float2_ a)		{ float2_ c;  c.x = cos(a.x)*cosh(a.y); c.y = -sin(a.x)*sinh(a.y); return c; }
	//		__host__ __device__ static inline double2_	eval(double2_ a)	{ double2_ c; c.x = cos(a.x)*cosh(a.y); c.y = -sin(a.x)*sinh(a.y); return c; }
	//};

	//Function_on_Scalar_Promotion(T,Cos,cos);
	//Function_on_Matrix_Promotion_Matrix(T,Cos,cos);
	//Function_on_Matrix_Promotion_CudaMatrix(T,Cos,cos);
	//Function_on_Expression_Promotion(T,Q1,Cos,cos);

	///*********************/
	///* EXPONENTIAL CLASS */
	///*********************/
	//class CudaOpExp
	//{
	//	public:
	//		__host__ __device__ static inline float		eval(int a)			{ return exp((float)a); }
	//		__host__ __device__ static inline float		eval(float a)		{ return exp(a); }
	//		__host__ __device__ static inline double	eval(double a)		{ return exp(a); }
	//		__host__ __device__ static inline float2_	eval(int2_ a)		{ float2_ c;		float2_ b; b.x = (float)a.x; b.y = (float)a.y; c.x = exp(b.x)*(cos(b.y)); c.y = exp(b.x)*(sin(b.y)); return b; }
	//		__host__ __device__ static inline float2_	eval(float2_ a)		{ float2_ c;		c.x = exp(a.x)*(cos(a.y)); c.y = exp(a.x)*(sin(a.y)); return c; }
	//		__host__ __device__ static inline double2_	eval(double2_ a)	{ double2_ c;	c.x = exp(a.x)*(cos(a.y)); c.y = exp(a.x)*(sin(a.y)); return c; }
	//};

	//Function_on_Scalar_Promotion(T,CudaOpExp,exp);
	//Function_on_Matrix_Promotion_Matrix(T,CudaOpExp,exp);
	//Function_on_Matrix_Promotion_CudaMatrix(T,CudaOpExp,exp);
	//Function_on_Expression_Promotion(T,Q1,CudaOpExp,exp);

Function_on_Scalar_Promotion(int,Exp,exp);
Function_on_Scalar_Promotion(float,Exp,exp);
Function_on_Scalar_Promotion(double,Exp,exp);
Function_on_Scalar_Promotion(BB::int2_,Exp,exp);
Function_on_Scalar_Promotion(BB::float2_,Exp,exp);
Function_on_Scalar_Promotion(BB::double2_,Exp,exp);

Function_on_Matrix_Promotion_Matrix(int,Exp,exp);
Function_on_Matrix_Promotion_Matrix(float,Exp,exp);
Function_on_Matrix_Promotion_Matrix(double,Exp,exp);
Function_on_Matrix_Promotion_Matrix(BB::int2_,Exp,exp);
Function_on_Matrix_Promotion_Matrix(BB::float2_,Exp,exp);
Function_on_Matrix_Promotion_Matrix(BB::double2_,Exp,exp);

Function_on_Matrix_Promotion_CudaMatrix(int,Exp,exp);
Function_on_Matrix_Promotion_CudaMatrix(float,Exp,exp);
Function_on_Matrix_Promotion_CudaMatrix(double,Exp,exp);
Function_on_Matrix_Promotion_CudaMatrix(BB::int2_,Exp,exp);
Function_on_Matrix_Promotion_CudaMatrix(BB::float2_,Exp,exp);
Function_on_Matrix_Promotion_CudaMatrix(BB::double2_,Exp,exp);

///*********************/
	///* CONJUGATION CLASS */
	///*********************/
	//class CudaOpConj
	//{
	//	public:
	//		__host__ __device__ static inline int		eval(int a)			{ return a; }
	//		__host__ __device__ static inline float		eval(float a)		{ return a; }
	//		__host__ __device__ static inline double	eval(double a)		{ return a; }
	//		__host__ __device__ static inline int2_		eval(int2_ a)		{ int2_ c;	 c.x =  a.x; c.y = -a.y; return c; }
	//		__host__ __device__ static inline float2_	eval(float2_ a)		{ float2_ c;	 c.x =  a.x; c.y = -a.y; return c; }
	//		__host__ __device__ static inline double2_	eval(double2_ a)	{ double2_ c; c.x =  a.x; c.y = -a.y; return c; }

	//};

	//Function_on_Scalar(T,CudaOpConj,conj);
	//Function_on_Matrix_Matrix(T,CudaOpConj,conj);
	//Function_on_Matrix_CudaMatrix(T,CudaOpConj,conj);
	//Function_on_Expression(T,Q1,CudaOpConj,conj);

	///*******************/
	///* REAL PART CLASS */
	///*******************/
	//class CudaOpReal
	//{
	//	public:
	//		__host__ __device__ static inline int		eval(int a)			{ return a; }
	//		__host__ __device__ static inline float		eval(float a)		{ return a; }
	//		__host__ __device__ static inline double	eval(double a)		{ return a; }
	//		__host__ __device__ static inline int		eval(int2_ a)		{ return a.x; }
	//		__host__ __device__ static inline float		eval(float2_ a)		{ return a.x; }
	//		__host__ __device__ static inline double	eval(double2_ a)	{ return a.x; }
	//};

	//Function_on_Scalar_Demotion(T,CudaOpReal,real);
	//Function_on_Matrix_Demotion_Matrix(T,CudaOpReal,real);
	//Function_on_Matrix_Demotion_CudaMatrix(T,CudaOpReal,real);
	//Function_on_Expression_Demotion(T,Q1,CudaOpReal,real);

	///************************/
	///* IMAGINARY PART CLASS */
	///************************/
	//class CudaOpImag
	//{
	//	public:
	//		__host__ __device__ static inline int		eval(int a)			{ int		c = 0;	return c; }
	//		__host__ __device__ static inline float		eval(float a)		{ float		c = 0.; return c; }
	//		__host__ __device__ static inline double	eval(double a)		{ double	c = 0.; return c; }
	//		__host__ __device__ static inline int		eval(int2_ a)		{ return a.y; }
	//		__host__ __device__ static inline float		eval(float2_ a)		{ return a.y; }
	//		__host__ __device__ static inline double	eval(double2_ a)	{ return a.y; }
	//};

	//Function_on_Scalar_Demotion(T,CudaOpImag,imag);
	//Function_on_Matrix_Demotion_Matrix(T,CudaOpImag,imag);
	//Function_on_Matrix_Demotion_CudaMatrix(T,CudaOpImag,imag);
	//Function_on_Expression_Demotion(T,Q1,CudaOpImag,imag);

	///************************/
	///* ABSOLUTE VALUE CLASS */
	///************************/
	//class CudaOpAbs	
	//{
	//	public:
	//		__host__ __device__ static inline int		eval(int a)			{ return abs(a); }
	//		__host__ __device__ static inline float		eval(float a)		{ return abs(a); }
	//		__host__ __device__ static inline double	eval(double a)		{ return abs(a); }
	//		__host__ __device__ static inline float		eval(int2_ a)		{ return sqrt((float)(a.x*a.x+a.y*a.y)); }
	//		__host__ __device__ static inline float		eval(float2_ a)		{ return sqrt(a.x*a.x+a.y*a.y); }
	//		__host__ __device__ static inline double	eval(double2_ a)	{ return sqrt(a.x*a.x+a.y*a.y); }

	//};

	//Function_on_Scalar_Demotion(T,CudaOpAbs,abs);
	//Function_on_Matrix_Demotion_Matrix(T,CudaOpAbs,abs);
	//Function_on_Matrix_Demotion_CudaMatrix(T,CudaOpAbs,abs);
	//Function_on_Expression_Demotion(T,Q1,CudaOpAbs,abs);

	///*********************/
	///* ANGLE VALUE CLASS */
	///*********************/
	//class CudaOpAngle	
	//{
	//	public:
	//		__host__ __device__ static inline float		eval(int a)			{ return (a>0) ? 0 : pi_f; }
	//		__host__ __device__ static inline float		eval(float a)		{ return (a>0) ? 0 : pi_f; }
	//		__host__ __device__ static inline double	eval(double a)		{ return (a>0) ? 0 : pi_d; }
	//		__host__ __device__ static inline float		eval(int2_ a)		{ return atan2((float)a.y,(float)a.x); }
	//		__host__ __device__ static inline float		eval(float2_ a)		{ return atan2(a.y,a.x); }
	//		__host__ __device__ static inline double	eval(double2_ a)	{ return atan2(a.y,a.x); }

	//};

	//Function_on_Scalar(T,CudaOpAngle,angle);
	//Function_on_Matrix_Matrix(T,CudaOpAngle,angle);
	//Function_on_Matrix_CudaMatrix(T,CudaOpAngle,angle);
	//Function_on_Expression(T,Q1,CudaOpAngle,angle);

	///**************/
	///* SQRT CLASS */
	///**************/
	//class CudaOpSqrt
	//{
	//	public:
	//		__host__ __device__ static inline float		eval(int a)			{ return sqrt((float)a); }
	//		__host__ __device__ static inline float		eval(float a)		{ return sqrt(a); }
	//		__host__ __device__ static inline double	eval(double a)		{ return sqrt(a); }
	//};

	//Function_on_Scalar_Promotion(T,CudaOpSqrt,sqrt);
	//Function_on_Matrix_Promotion_Matrix(T,CudaOpSqrt,sqrt);
	//Function_on_Matrix_Promotion_CudaMatrix(T,CudaOpSqrt,sqrt);
	//Function_on_Expression_Promotion(T,Q1,CudaOpSqrt,sqrt);

	///***************/
	///* ROUND CLASS */
	///***************/

	//class CudaOpRound
	//{
	//	public:
	//		__host__ __device__ static inline float		eval(int a)			{ return round((float)a); }
	//		__host__ __device__ static inline float		eval(float a)		{ return round(a); }
	//		__host__ __device__ static inline double	eval(double a)		{ return round(a); }
	//		__host__ __device__ static inline float2_	eval(int2_ a)		{ float2_ c;	c.x = round((float)a.x); c.y = round((float)a.y); return c; }
	//		__host__ __device__ static inline float2_	eval(float2_ a)		{ float2_ c;	c.x = round(a.x); c.y = round(a.y); return c; }
	//		__host__ __device__ static inline double2_	eval(double2_ a)	{ double2_ c;	c.x = round(a.x); c.y = round(a.y); return c; }
	//};

	//Function_on_Scalar(T,CudaOpRound,round);
	//Function_on_Matrix_Matrix(T,CudaOpRound,round);
	//Function_on_Matrix_CudaMatrix(T,CudaOpRound,round);
	//Function_on_Expression(T,Q1,CudaOpRound,round);

	///**************/
	///* SINH CLASS */
	///**************/

	//class CudaOpSinh
	//{
	//	public:
	//		__host__ __device__ static inline float		eval(int a)			{ return sinh((float)a); }
	//		__host__ __device__ static inline float		eval(float a)		{ return sinh(a); }
	//		__host__ __device__ static inline double	eval(double a)		{ return sinh(a); }
	//		__host__ __device__ static inline float2_	eval(int2_ a)		{ float2_ c; float2_ b; b.x = (float)a.x; b.y = (float)a.y; c.x = sinh(b.x)*cos(b.y); c.y = cosh(b.x)*sin(b.y); return c; }
	//		__host__ __device__ static inline float2_	eval(float2_ a)		{ float2_ c;  c.x = sinh(a.x)*cos(a.y); c.y = cosh(a.x)*sin(a.y); return c; }
	//		__host__ __device__ static inline double2_	eval(double2_ a)	{ double2_ c; c.x = sinh(a.x)*cos(a.y); c.y = cosh(a.x)*sin(a.y); return c; }
	//};

	//Function_on_Scalar_Promotion(T,CudaOpSinh,sinh);
	//Function_on_Matrix_Promotion_Matrix(T,CudaOpSinh,sinh);
	//Function_on_Matrix_Promotion_CudaMatrix(T,CudaOpSinh,sinh);
	//Function_on_Expression_Promotion(T,Q1,CudaOpSinh,sinh);
	//
	///**************/
	///* COSH CLASS */
	///**************/

	//class CudaOpCosh
	//{
	//	public:
	//		__host__ __device__ static inline float		eval(int a)			{ return cosh((float)a); }
	//		__host__ __device__ static inline float		eval(float a)		{ return cosh(a); }
	//		__host__ __device__ static inline double	eval(double a)		{ return cosh(a); }
	//		__host__ __device__ static inline float2_	eval(int2_ a)		{ float2_ c; float2_ b; b.x = (float)a.x; b.y = (float)a.y; c.x = cosh(b.x)*cos(b.y); c.y = sinh(b.x)*sin(b.y); return c; }
	//		__host__ __device__ static inline float2_	eval(float2_ a)		{ float2_ c;  c.x = cosh(a.x)*cos(a.y); c.y = sinh(a.x)*sin(a.y); return c; }
	//		__host__ __device__ static inline double2_	eval(double2_ a)	{ double2_ c; c.x = cosh(a.x)*cos(a.y); c.y = sinh(a.x)*sin(a.y); return c; }
	//};

	//Function_on_Scalar_Promotion(T,CudaOpCosh,cosh);
	//Function_on_Matrix_Promotion_Matrix(T,CudaOpCosh,cosh);
	//Function_on_Matrix_Promotion_CudaMatrix(T,CudaOpCosh,cosh);
	//Function_on_Expression_Promotion(T,Q1,CudaOpCosh,cosh);

	///***************/
	///* LOG10 CLASS */
	///***************/

	//class CudaOpLog10
	//{
	//	public:
	//		__device__ __host__  static inline float	eval(int a)			{ return log10((float)a); }
	//		__device__ __host__  static inline float	eval(float a)		{ return log10(a); }
	//		__device__ __host__  static inline double	eval(double a)		{ return log10(a); }
	//		__device__ __host__  static inline float2_	eval(int2_ a)		{ float2_ b, c; b.x = (float)a.x; b.y = (float)a.y; c.x = log10(CudaOpAbs::eval(b)); c.y = atan2(b.y,b.x); return c; }
	//		__device__ __host__  static inline float2_	eval(float2_ a)		{ float2_	c; c.x = log10(CudaOpAbs::eval(a)); c.y = atan2(a.y,a.x); return c; }
	//		__device__ __host__  static inline double2_	eval(double2_ a)	{ double2_	c; c.x = log10(CudaOpAbs::eval(a)); c.y = atan2(a.y,a.x); return c; }
	//};
	//
	//Function_on_Scalar_Promotion(T,CudaOpLog10,log10);
	//Function_on_Matrix_Promotion_Matrix(T,CudaOpLog10,log10);
	//Function_on_Matrix_Promotion_CudaMatrix(T,CudaOpLog10,log10);
	//Function_on_Expression_Promotion(T,Q1,CudaOpLog10,log10);

	///*************/
	///* POW CLASS */
	///*************/

	//class CudaPow
	//{
	//	public:
	//		//__device__ __host__  static inline float	eval(int a)			{ return log10((float)a); }
	//		__device__ __host__  static inline float	eval(float a, int b)		{ return pow(a,b); }
	//		__device__ __host__  static inline double	eval(double a, int b)		{ return pow(a,b); }
	//		//__device__ __host__  static inline float2_	eval(int2_ a)		{ float2_ b, c; b.x = (float)a.x; b.y = (float)a.y; c.x = log10(CudaOpAbs::eval(b)); c.y = atan2(b.y,b.x); return c; }
	//		//__device__ __host__  static inline float2_	eval(float2_ a)		{ float2_	c; c.x = log10(CudaOpAbs::eval(a)); c.y = atan2(a.y,a.x); return c; }
	//		//__device__ __host__  static inline double2_	eval(double2_ a)	{ double2_	c; c.x = log10(CudaOpAbs::eval(a)); c.y = atan2(a.y,a.x); return c; }
	//};
	//
	//Function_on_Scalar2(T1,T2,CudaPow,pow);
	//Function_on_Matrix_Matrix2(T1,T2,CudaPow,pow);
	//Function_on_Matrix_CudaMatrix2(T1,T2,CudaPow,pow);
	//Function_on_Expression2(T1,Q1,T2,CudaPow,pow);



//////ABS
////class AbsOp
////{
////public:
////    static inline double eval(double a)
////    {
////        return abs(a);
////    }
////    static inline double eval(double2_ a)
////    {
////        
////        return sqrt(a.x*a.x+a.y*a.y);
////    }
////    static inline float eval(float2_ a)
////    {
////        
////        return sqrt(a.x*a.x+a.y*a.y);
////    }
////    
////};
////
//////for constant
////template <class T>
////inline typename Demotion<double,T>::weakest Abs(T a)
////{
////    return AbsOp::eval(a);
////}
////
//////For Hmatrix 
////template<template<class> class A,class T>
////Expr<FunctionExpr<const A<T> &,AbsOp,
////typename Demotion<double,T>::weakest> ,
////typename Demotion<double,T>::weakest > Abs(const A<T> &v)
////{
////    typedef FunctionExpr<const A<T> &,AbsOp,typename Demotion<double,T>::weakest> FExpr;
////    return Expr<FExpr,typename Demotion<double,T>::weakest>(FExpr(v));
////}
//////For Expr 
////template<template<class,class> class A,class T,class Q1>
////Expr<FunctionExpr<const A<Q1,T> &,AbsOp,
////typename Demotion<double,T>::weakest> ,
////typename Demotion<double,T>::weakest> Abs(const A<Q1,T> &v)
////{
////    typedef FunctionExpr<const A<Q1,T> & ,AbsOp,typename Demotion<double,T>::weakest> FExpr;
////    return Expr<FExpr,typename Demotion<double,T>::weakest>(FExpr(v));
////}
////
//////Angle
////class AngleOp
////{
////public:
////    static inline double eval(double a)
////    {
////        return (a>0)?0:3.141592653589793;
////    }
////    static inline double eval(double2_ a)
////    {
////        
////        return atan2(a.y,a.x);
////    }
////    static inline float eval(float2_ a)
////    {
////        
////        return atan2(a.y,a.x);
////    }
////};
//////for constant
////template <class T>
////inline double Angle(T a)
////{
////    return AngleOp::eval(a);
////}
////
////
////
//////For Hmatrix 
////template<template<class> class A,class T>
////Expr<FunctionExpr<const A<T> &,AngleOp,double> ,double> Angle(const A<T> &v)
////{
////    typedef FunctionExpr<const A<T> &,AngleOp,double> FExpr;
////    return Expr<FExpr,double>(FExpr(v));
////}
//////For Expr 
////template<template<class,class> class A,class T,class Q1>
////Expr<FunctionExpr<const A<Q1,T> &,AngleOp,double> ,double> Angle(const A<Q1,T> &v)
////{
////    typedef FunctionExpr<const A<Q1,T> & ,AngleOp,double> FExpr;
////    return Expr<FExpr,double>(FExpr(v));
////}
////
////
////
////////bessel function
//////const double p_Constant_Bessel0[17]	= {-0.27288446572737951578789523409*1e10, 
//////				   -0.6768549084673824894340380223*1e9, 
//////				   -0.4130296432630476829274339869*1e8, 
//////				   -0.11016595146164611763171787004*1e7,
//////				   -0.1624100026427837007503320319*1e5,
//////				   -0.1503841142335444405893518061*1e3,
//////				   -0.947449149975326604416967031*1e0,
//////				   -0.4287350374762007105516581810*1e-2,
//////				   -0.1447896113298369009581404138*1e-4,
//////			           -0.375114023744978945259642850*1e-7,
//////				   -0.760147559624348256501094832*1e-10,
//////				   -0.121992831543841162565677055*1e-12,
//////				   -0.15587387207852991014838679*1e-15,
//////				   -0.15795544211478823152992269*1e-18,
//////				   -0.1247819710175804058844059*1e-21,
//////				   -0.72585406935875957424755*1e-25,
//////				   -0.28840544803647313855232*1e-28};
//////
//////const double q_Constant_Bessel0[4]	= {-0.2728844657273795156746641315*1e10,
//////				    0.5356255851066290475987259*1e7,
//////				   -0.38305191682802536272760*1e4,
////// 				    0.1*1e1};
//////
//////const double cp_Constant_Bessel0[26]	= {0.8013085461969871612106457692*1e0/2, 
//////				   0.17290977661213460475126976*1e-2, 
//////				   0.174344303731276665678151*1e-4, 
//////				   0.3391002544196612353807*1e-6,
//////				   0.101386676244446816514*1e-7,
//////				   0.4212987673401844259*1e-9,
//////				   0.230587178626748963*1e-10,
//////				   0.16147708215256485*1e-11,
//////				   0.1430354372503558*1e-12,
//////			           0.160493342152869*1e-13,
//////				   0.22704188628639*1e-14,
//////				   0.3839318408111*1e-15,
//////				   0.675456929962*1e-16,
//////				   0.91151381855*1e-17,
//////				   -0.3729687231*1e-18,
//////				   -0.8619437387*1e-18,
//////				   -0.3466400954*1e-18,
//////				   -0.706240530*1e-19,
//////				   0.30556730*1e-20,
//////    				   0.76603496*1e-20,
//////				   0.23745789*1e-20,
//////				   0.119838*1e-22,
//////				   -0.2438946*1e-21,
//////				   -0.720868*1e-22,
//////				   0.69870*1e-23,
//////				   0.96880*1e-23};
//////
//////
//////__host__ double bessi0(double x)
//////{
//////// Valid only for |x|<=15.0 -- See paper
//////// J.M. Blair, "Rational Chebyshev approximations for the modified Bessel functions I_0(x) and I_1(x)", Math. Comput., vol. 28, n. 126, pp. 581-583, Apr. 1974.   
//////
//////   double num, den, x2, y, tn_2, tn_1, tn, tn_1_old;
//////
//////   x2 = fabs(x*x);
//////
//////   if (fabs(x) <= 15.0) 
//////   {
//////      num=p_Constant_Bessel0[0]+x2*(p_Constant_Bessel0[1]+x2*(p_Constant_Bessel0[2]+x2*(p_Constant_Bessel0[3]+x2*(p_Constant_Bessel0[4]+x2*(p_Constant_Bessel0[5]+x2*(p_Constant_Bessel0[6]+x2*(p_Constant_Bessel0[7]+x2*(p_Constant_Bessel0[8]+x2*(p_Constant_Bessel0[9]+x2*(p_Constant_Bessel0[10]+x2*(p_Constant_Bessel0[11]+x2*(p_Constant_Bessel0[12]+x2*(p_Constant_Bessel0[13]+x2*(p_Constant_Bessel0[14]+x2*(p_Constant_Bessel0[15]+p_Constant_Bessel0[16]*x2)))))))))))))));
//////      den=q_Constant_Bessel0[0]+x2*(q_Constant_Bessel0[1]+x2*(q_Constant_Bessel0[2]+q_Constant_Bessel0[3]*x2));
//////      return num/den;
//////   }
//////   else
//////   {
//////      y=30./x-1.;
//////      num=0.;
//////      tn_2=1.;
//////      num=cp_Constant_Bessel0[0]*tn_2;
//////      tn_1=y;
//////      num=num+cp_Constant_Bessel0[1]*tn_1;
//////      for (int k=2; k<=25; k++)
//////      {
//////         tn=2.*y*tn_1-tn_2;
//////         num=num+cp_Constant_Bessel0[k]*tn;
//////         tn_1_old=tn_1;
//////         tn_1=tn;
//////         tn_2=tn_1_old;
//////      }
//////      //return num*exp(x)/sqrt(x);
//////      return num*exp(x)/sqrt(x);
//////   }
//////
//////}
//////
//////class BessI0Op
//////{
//////public:
//////    static inline double eval(double a)
//////    {
//////        return bessi0(a);
//////    }
//////};
//////
////////for constant
//////template<class T>
//////inline  typename Promotion<double,T>::strongest BessI0(T a)
//////{
//////    return BessI0Op::eval(a);
//////}
//////
////////BessI0
////////For Hmatrix 
//////template<template<class> class A,class T>
//////Expr<FunctionExpr<const A<T> &,BessI0Op,
//////typename Promotion<double,T>::strongest> ,
//////typename Promotion<double,T>::strongest> BessI0(const A<T> &v)
//////{
//////    typedef FunctionExpr<const A<T> &,BessI0Op,typename Promotion<double,T>::strongest> FExpr;
//////    return Expr<FExpr,typename Promotion<double,T>::strongest>(FExpr(v));
//////}
////////For Expr 
//////template<template<class,class> class A,class T,class Q1>
//////Expr<FunctionExpr<const A<Q1,T> &,BessI0Op,
//////typename Promotion<double,T>::strongest> ,
//////typename Promotion<double,T>::strongest> BessI0(const A<Q1,T> &v)
//////{
//////    typedef FunctionExpr<const A<Q1,T> &,BessI0Op,typename Promotion<double,T>::strongest> FExpr;
//////    return Expr<FExpr,typename Promotion<double,T>::strongest>(FExpr(v));
//////}
////
//////sqrt
////class SqrtOp
////{
////public:
////    static inline double eval(double a)
////    {
////        return sqrt(a);
////    }
////};
////
//////for constant
////template<class T>
////inline typename Promotion<double,T>::strongest Sqrt(T a)
////{
////    return SqrtOp::eval(a);
////}
////
////template<template<class> class A,class T>
////Expr<FunctionExpr<const A<T> &,SqrtOp,
////typename Promotion<double,T>::strongest> ,
////typename Promotion<double,T>::strongest> Sqrt(const A<T> &v)
////{
////    typedef FunctionExpr<const A<T> &,SqrtOp,typename Promotion<double,T>::strongest> FExpr;
////    return Expr<FExpr,typename Promotion<double,T>::strongest>(FExpr(v));
////}
//////For Expr 
////template<template<class,class> class A,class T,class Q1>
////Expr<FunctionExpr<const A<Q1,T> &,SqrtOp,
////typename Promotion<double,T>::strongest> ,
////typename Promotion<double,T>::strongest> Sqrt(const A<Q1,T> &v)
////{
////    typedef FunctionExpr<const A<Q1,T> &,SqrtOp,typename Promotion<double,T>::strongest> FExpr;
////    return Expr<FExpr,typename Promotion<double,T>::strongest>(FExpr(v));
////}
////
//////round
//////class RoundOp
//////{
//////public:
//////    static inline double eval(double a)
//////    {
//////        return round(a);
//////    }
//////    static inline double2_ eval(double2_ a)
//////    {
//////        double2_ c;
//////        c.x = round(a.x);
//////        c.y = round(a.y);
//////        
//////        return c;
//////    }
//////    static inline float2_ eval(float2_ a)
//////    {
//////        float2_ c;
//////        c.x = round(a.x);
//////        c.y = round(a.y);
//////        
//////        return c;
//////    }
//////    
//////};
//////
//////
////////for constant
//////template <class T>
//////inline T Round(T a)
//////{
//////    return RoundOp::eval(a);
//////}
//////
//////template<template<class> class A,class T>
//////Expr<FunctionExpr<const A<T> &,RoundOp,T> ,T> Round(const A<T> &v)
//////{
//////    typedef FunctionExpr<const A<T> &,RoundOp,T> FExpr;
//////    return Expr<FExpr,T>(FExpr(v));
//////}
////////For Expr 
//////template<template<class,class> class A,class T,class Q1>
//////Expr<FunctionExpr<const A<Q1,T> &,RoundOp,T> ,T> Round(const A<Q1,T> &v)
//////{
//////    typedef FunctionExpr<const A<Q1,T> &,RoundOp,T> FExpr;
//////    return Expr<FExpr,T>(FExpr(v));
//////}
//////
////
//////Sinh
////class SinhOp
////{
////public:
////    static inline double eval(double a)
////    {
////        return sinh(a);
////    }
////};
////
////
//////for constant
////template <class T>
////inline typename Promotion<double,T>::strongest Sinh(T a)
////{
////    return SinhOp::eval(a);
////}
////
////template<template<class> class A,class T>
////Expr<FunctionExpr<const A<T> &,SinhOp,
////typename Promotion<double,T>::strongest> ,
////typename Promotion<double,T>::strongest> Sinh(const A<T> &v)
////{
////    typedef FunctionExpr<const A<T> &,SinhOp,typename Promotion<double,T>::strongest> FExpr;
////    return Expr<FExpr,typename Promotion<double,T>::strongest>(FExpr(v));
////}
//////For Expr 
////template<template<class,class> class A,class T,class Q1>
////Expr<FunctionExpr<const A<Q1,T> &,SinhOp,
////typename Promotion<double,T>::strongest> ,
////typename Promotion<double,T>::strongest> Sinh(const A<Q1,T> &v)
////{
////    typedef FunctionExpr<const A<Q1,T> &,SinhOp,typename Promotion<double,T>::strongest> FExpr;
////    return Expr<FExpr,typename Promotion<double,T>::strongest>(FExpr(v));
////}
////
////
//////Log10
////class Log10Op
////{
////public:
////    static inline double eval(double a)
////    {
////        return log10(a);
////    }
////};
////
////
//////for constant
////template <class T>
////inline typename Promotion<double,T>::strongest Log10(T a)
////{
////    return Log10Op::eval(a);
////}
////
////template<template<class> class A,class T>
////Expr<FunctionExpr<const A<T> &,Log10Op,
////typename Promotion<double,T>::strongest> ,
////typename Promotion<double,T>::strongest> Log10(const A<T> &v)
////{
////    typedef FunctionExpr<const A<T> &,Log10Op,typename Promotion<double,T>::strongest> FExpr;
////    return Expr<FExpr,typename Promotion<double,T>::strongest>(FExpr(v));
////}
//////For Expr 
////template<template<class,class> class A,class T,class Q1>
////Expr<FunctionExpr<const A<Q1,T> &,Log10Op,
////typename Promotion<double,T>::strongest> ,
////typename Promotion<double,T>::strongest> Log10(const A<Q1,T> &v)
////{
////    typedef FunctionExpr<const A<Q1,T> &,Log10Op,typename Promotion<double,T>::strongest> FExpr;
////    return Expr<FExpr,typename Promotion<double,T>::strongest>(FExpr(v));
////}
////
////
////}
////
