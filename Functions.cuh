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


#ifndef __FUNCTIONS_H__
#define __FUNCTIONS_H__

#include "Promotion.cuh"

// Function on Scalar Promotion -- TESTED
#define Function_on_Scalar_Promotion(T,FunClass,overloaded_function) template <class T> \
__host__ __device__ typename Promotion<float,T>::strongest overloaded_function(T a);

// Function on Hmatrix Promotion Hmatrix -- TESTED
#define Function_on_Matrix_Promotion_Matrix(T,FunClass,overloaded_function) template<class T> \
Expr<FunExpr<const T*,FunClass,typename Promotion<float,T>::strongest>,typename Promotion<float,T>::strongest> overloaded_function(const Hmatrix<T> &v);
//#define Function_on_Matrix_Promotion_Matrix(T,FunClass,overloaded_function) Expr<FunExpr<const T*,FunClass,typename Promotion<double,T>::strongest>,typename Promotion<double,T>::strongest> overloaded_function(const Hmatrix<T> &v);

// Function on Hmatrix Promotion Dmatrix -- TESTED
#define Function_on_Matrix_Promotion_CudaMatrix(T,FunClass,overloaded_function) template<class T> \
Expr<FunExpr<const T*,FunClass,typename Promotion<float,T>::strongest>,typename Promotion<float,T>::strongest> overloaded_function(const Dmatrix<T> &v);

// Function on Expression Promotion -- TESTED
//#define Function_on_Expression_Promotion(T,Q1,FunClass,overloaded_function) template<class T,class Q1> \
//Expr<FunExpr<const Expr<Q1,T>,FunClass,typename Promotion<double,T>::strongest>,typename Promotion<double,T>::strongest> overloaded_function(const Expr<Q1,T> &v);

// Function on Expression Promotion -- TESTED
#define Function_on_Expression_Promotion(T,Q1,FunClass,overloaded_function) template<class T,class Q1> \
	Expr<FunExpr<const Expr<Q1,T>,FunClass,typename Promotion<float,T>::strongest>,typename Promotion<float,T>::strongest> overloaded_function(const Expr<Q1,T> &v) \
	{ \
		typedef FunExpr<const Expr<Q1,T>,FunClass,typename Promotion<float,T>::strongest> FExpr; \
		return Expr<FExpr,typename Promotion<float,T>::strongest>(FExpr(v,v.GetRows(),v.GetColumns()),v.GetRows(),v.GetColumns(),v.IsDevice()); \
	}

namespace BB
{
	/**************/
	/* SINE CLASS */
	/**************/
	class Sin
	{
		public:
			//__host__ __device__ static float	eval(int);
			//__host__ __device__ static float	eval(float);
			////__host__ __device__ static inline double	eval(double a) { return sin(a); };
			//__host__ __device__ static double	eval(double);
			//__host__ __device__ static float2_	eval(int2_);
			//__host__ __device__ static float2_	eval(float2_);
			//__host__ __device__ static double2_	eval(double2_);
			__host__ __device__ __forceinline __forceinline__ static float		eval(int a)				{ return std::sin((float)a); }
			__host__ __device__ __forceinline __forceinline__ static float		eval(float a)			{ return std::sin(a); }
			__host__ __device__ __forceinline __forceinline__ static double		eval(double a)			{ return std::sin(a); }
			__host__ __device__ __forceinline __forceinline__ static float2_		eval(int2_ a)			{ float2_ c; float2_ b; b.c.x = (float)a.c.x; b.c.y = (float)a.c.y; c.c.x = std::sin(b.c.x)*std::cosh(b.c.y); c.c.y = std::cos(b.c.x)*std::sinh(b.c.y); return c; }
			__host__ __device__ __forceinline __forceinline__ static float2_		eval(float2_ a)			{ float2_ c;  c.c.x = std::sin(a.c.x)*std::cosh(a.c.y); c.c.y = std::cos(a.c.x)*std::sinh(a.c.y); return c; }
			__host__ __device__ __forceinline __forceinline__ static double2_		eval(double2_ a)		{ double2_ c; c.c.x = std::sin(a.c.x)*std::cosh(a.c.y); c.c.y = std::cos(a.c.x)*std::sinh(a.c.y); return c; }
	};

	Function_on_Scalar_Promotion(T,Sin,sin);
	Function_on_Matrix_Promotion_Matrix(T,Sin,sin);
	Function_on_Matrix_Promotion_CudaMatrix(T,Sin,sin);
	Function_on_Expression_Promotion(T,Q1,Sin,sin);

	/****************/
	/* COSINE CLASS */
	/****************/
	class Cos
	{
		public:
			//__host__ __device__ static float	eval(int);
			//__host__ __device__ static float	eval(float);
			//__host__ __device__ static double	eval(double);
			//__host__ __device__ static float2_	eval(int2_);
			//__host__ __device__ static float2_	eval(float2_);
			//__host__ __device__ static double2_	eval(double2_);
			__host__ __device__ __forceinline __forceinline__ static float		eval(int a)				{ return std::cos((float)a); }
			__host__ __device__ __forceinline __forceinline__ static float		eval(float a)			{ return std::cos(a); }
			__host__ __device__ __forceinline __forceinline__ static double		eval(double a)			{ return std::cos(a); }
			__host__ __device__ __forceinline __forceinline__ static float2_	eval(int2_ a)			{ float2_ c; float2_ b; b.c.x = (float)a.c.x; b.c.y = (float)a.c.y; c.c.x = std::cos(b.c.x)*std::cosh(b.c.y); c.c.y = -std::sin(b.c.x)*std::sinh(b.c.y); return c; }
			__host__ __device__ __forceinline __forceinline__ static float2_	eval(float2_ a)			{ float2_ c;  c.c.x = std::cos(a.c.x)*std::cosh(a.c.y); c.c.y = -std::sin(a.c.x)*std::sinh(a.c.y); return c; }
			__host__ __device__ __forceinline __forceinline__ static double2_	eval(double2_ a)		{ double2_ c; c.c.x = std::cos(a.c.x)*std::cosh(a.c.y); c.c.y = -std::sin(a.c.x)*std::sinh(a.c.y); return c; }
	};

	Function_on_Scalar_Promotion(T,Cos,cos);
	Function_on_Matrix_Promotion_Matrix(T,Cos,cos);
	Function_on_Matrix_Promotion_CudaMatrix(T,Cos,cos);
	Function_on_Expression_Promotion(T,Q1,Cos,cos);

	/*********************/
	/* EXPONENTIAL CLASS */
	/*********************/
	class Exp
	{
		public:
			__host__ __device__ __forceinline __forceinline__ static float		eval(int a)			{ return std::exp((float)a); }
			__host__ __device__ __forceinline __forceinline__ static float		eval(float a)		{ return std::exp(a); }
			__host__ __device__ __forceinline __forceinline__ static double		eval(double a)		{ return std::exp(a); }
			__host__ __device__ __forceinline __forceinline__ static float2_	eval(int2_ a)		{ float2_ c;		float2_ b;	b.c.x = (float)a.c.x; b.c.y = (float)a.c.y; c.c.x = std::exp(b.c.x)*(std::cos(b.c.y)); c.c.y = std::exp(b.c.x)*(std::sin(b.c.y)); return b; }
			__host__ __device__ __forceinline __forceinline__ static float2_	eval(float2_ a)		{ float2_ c;					c.c.x = std::exp(a.c.x)*(std::cos(a.c.y)); c.c.y = std::exp(a.c.x)*(std::sin(a.c.y)); return c; }
			__host__ __device__ __forceinline __forceinline__ static double2_	eval(double2_ a)	{ double2_ c;					c.c.x = std::exp(a.c.x)*(std::cos(a.c.y)); c.c.y = std::exp(a.c.x)*(std::sin(a.c.y)); return c; }
	};

	Function_on_Scalar_Promotion(T,Exp,exp);
	Function_on_Matrix_Promotion_Matrix(T,Exp,exp);
	Function_on_Matrix_Promotion_CudaMatrix(T,Exp,exp);
	Function_on_Expression_Promotion(T,Q1,Exp,exp);

	///*********************/
	///* CONJUGATION CLASS */
	///*********************/
	//class CudaOpConj
	//{
	//	public:
	//		__host__ __device__ static inline int		eval(int a)			{ return a; }
	//		__host__ __device__ static inline float		eval(float a)		{ return a; }
	//		__host__ __device__ static inline double	eval(double a)		{ return a; }
	//		__host__ __device__ static inline int2_		eval(int2_ a)		{ int2_ c;	 c.c.x =  a.c.x; c.c.y = -a.c.y; return c; }
	//		__host__ __device__ static inline float2_	eval(float2_ a)		{ float2_ c;	 c.c.x =  a.c.x; c.c.y = -a.c.y; return c; }
	//		__host__ __device__ static inline double2_	eval(double2_ a)	{ double2_ c; c.c.x =  a.c.x; c.c.y = -a.c.y; return c; }

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
	//		__host__ __device__ static inline int		eval(int2_ a)		{ return a.c.x; }
	//		__host__ __device__ static inline float		eval(float2_ a)		{ return a.c.x; }
	//		__host__ __device__ static inline double	eval(double2_ a)	{ return a.c.x; }
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
	//		__host__ __device__ static inline int		eval(int2_ a)		{ return a.c.y; }
	//		__host__ __device__ static inline float		eval(float2_ a)		{ return a.c.y; }
	//		__host__ __device__ static inline double	eval(double2_ a)	{ return a.c.y; }
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
	//		__host__ __device__ static inline float		eval(int2_ a)		{ return sqrt((float)(a.c.x*a.c.x+a.c.y*a.c.y)); }
	//		__host__ __device__ static inline float		eval(float2_ a)		{ return sqrt(a.c.x*a.c.x+a.c.y*a.c.y); }
	//		__host__ __device__ static inline double	eval(double2_ a)	{ return sqrt(a.c.x*a.c.x+a.c.y*a.c.y); }

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
	//		__host__ __device__ static inline float		eval(int2_ a)		{ return atan2((float)a.c.y,(float)a.c.x); }
	//		__host__ __device__ static inline float		eval(float2_ a)		{ return atan2(a.c.y,a.c.x); }
	//		__host__ __device__ static inline double	eval(double2_ a)	{ return atan2(a.c.y,a.c.x); }

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
	//		__host__ __device__ static inline float2_	eval(int2_ a)		{ float2_ c;	c.c.x = round((float)a.c.x); c.c.y = round((float)a.c.y); return c; }
	//		__host__ __device__ static inline float2_	eval(float2_ a)		{ float2_ c;	c.c.x = round(a.c.x); c.c.y = round(a.c.y); return c; }
	//		__host__ __device__ static inline double2_	eval(double2_ a)	{ double2_ c;	c.c.x = round(a.c.x); c.c.y = round(a.c.y); return c; }
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
	//		__host__ __device__ static inline float2_	eval(int2_ a)		{ float2_ c; float2_ b; b.c.x = (float)a.c.x; b.c.y = (float)a.c.y; c.c.x = sinh(b.c.x)*cos(b.c.y); c.c.y = cosh(b.c.x)*sin(b.c.y); return c; }
	//		__host__ __device__ static inline float2_	eval(float2_ a)		{ float2_ c;  c.c.x = sinh(a.c.x)*cos(a.c.y); c.c.y = cosh(a.c.x)*sin(a.c.y); return c; }
	//		__host__ __device__ static inline double2_	eval(double2_ a)	{ double2_ c; c.c.x = sinh(a.c.x)*cos(a.c.y); c.c.y = cosh(a.c.x)*sin(a.c.y); return c; }
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
	//		__host__ __device__ static inline float2_	eval(int2_ a)		{ float2_ c; float2_ b; b.c.x = (float)a.c.x; b.c.y = (float)a.c.y; c.c.x = cosh(b.c.x)*cos(b.c.y); c.c.y = sinh(b.c.x)*sin(b.c.y); return c; }
	//		__host__ __device__ static inline float2_	eval(float2_ a)		{ float2_ c;  c.c.x = cosh(a.c.x)*cos(a.c.y); c.c.y = sinh(a.c.x)*sin(a.c.y); return c; }
	//		__host__ __device__ static inline double2_	eval(double2_ a)	{ double2_ c; c.c.x = cosh(a.c.x)*cos(a.c.y); c.c.y = sinh(a.c.x)*sin(a.c.y); return c; }
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
	//		__device__ __host__  static inline float2_	eval(int2_ a)		{ float2_ b, c; b.c.x = (float)a.c.x; b.c.y = (float)a.c.y; c.c.x = log10(CudaOpAbs::eval(b)); c.c.y = atan2(b.c.y,b.c.x); return c; }
	//		__device__ __host__  static inline float2_	eval(float2_ a)		{ float2_	c; c.c.x = log10(CudaOpAbs::eval(a)); c.c.y = atan2(a.c.y,a.c.x); return c; }
	//		__device__ __host__  static inline double2_	eval(double2_ a)	{ double2_	c; c.c.x = log10(CudaOpAbs::eval(a)); c.c.y = atan2(a.c.y,a.c.x); return c; }
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
	//		//__device__ __host__  static inline float2_	eval(int2_ a)		{ float2_ b, c; b.c.x = (float)a.c.x; b.c.y = (float)a.c.y; c.c.x = log10(CudaOpAbs::eval(b)); c.c.y = atan2(b.c.y,b.c.x); return c; }
	//		//__device__ __host__  static inline float2_	eval(float2_ a)		{ float2_	c; c.c.x = log10(CudaOpAbs::eval(a)); c.c.y = atan2(a.c.y,a.c.x); return c; }
	//		//__device__ __host__  static inline double2_	eval(double2_ a)	{ double2_	c; c.c.x = log10(CudaOpAbs::eval(a)); c.c.y = atan2(a.c.y,a.c.x); return c; }
	//};
	//
	//Function_on_Scalar2(T1,T2,CudaPow,pow);
	//Function_on_Matrix_Matrix2(T1,T2,CudaPow,pow);
	//Function_on_Matrix_CudaMatrix2(T1,T2,CudaPow,pow);
	//Function_on_Expression2(T1,Q1,T2,CudaPow,pow);

}	

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
////        return sqrt(a.c.x*a.c.x+a.c.y*a.c.y);
////    }
////    static inline float eval(float2_ a)
////    {
////        
////        return sqrt(a.c.x*a.c.x+a.c.y*a.c.y);
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
////        return atan2(a.c.y,a.c.x);
////    }
////    static inline float eval(float2_ a)
////    {
////        
////        return atan2(a.c.y,a.c.x);
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
//////        c.c.x = round(a.c.x);
//////        c.c.y = round(a.c.y);
//////        
//////        return c;
//////    }
//////    static inline float2_ eval(float2_ a)
//////    {
//////        float2_ c;
//////        c.c.x = round(a.c.x);
//////        c.c.y = round(a.c.y);
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
#endif