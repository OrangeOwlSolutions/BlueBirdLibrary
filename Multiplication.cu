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
#include "HExceptions.h"
#include "Hmatrix.h"
#include "Dmatrix.cuh"
#include "Multiplication.cuh"
#include "Expression.cuh"
#include "Promotion.cuh"
#include "Scalar.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

#define Scalar_Scalar(T1,T2,OpClass,overloaded_operator) template <> typename BB::Promotion<T1,T2>::strongest BB::overloaded_operator(const T1 a,const T2 b) { return BB::OpClass::eval(a,b); }

Promotion<int,int2_>::strongest operator*(const int a,const int2_ b)							{ return BB::Mul::eval(a,b); }
Promotion<int,float2_>::strongest operator*(const int a,const float2_ b)						{ return BB::Mul::eval(a,b); }
Promotion<int,double2_>::strongest operator*(const int a,const double2_ b)						{ return BB::Mul::eval(a,b); }
Promotion<float,int2_>::strongest operator*(const float a,const int2_ b)						{ return BB::Mul::eval(a,b); }
Promotion<float,float2_>::strongest operator*(const float a,const float2_ b)					{ return BB::Mul::eval(a,b); }
Promotion<float,double2_>::strongest operator*(const float a,const double2_ b)					{ return BB::Mul::eval(a,b); }
Promotion<double,int2_>::strongest operator*(const double a,const int2_ b)						{ return BB::Mul::eval(a,b); }
Promotion<double,float2_>::strongest operator*(const double a,const float2_ b)					{ return BB::Mul::eval(a,b); }
Promotion<double,double2_>::strongest operator*(const double a,const double2_ b)				{ return BB::Mul::eval(a,b); }
Promotion<int2_,int>::strongest operator*(const int2_ a,const int b)							{ return BB::Mul::eval(a,b); }
Promotion<int2_,float>::strongest operator*(const int2_ a,const float b)						{ return BB::Mul::eval(a,b); }
Promotion<int2_,double>::strongest operator*(const int2_ a,const double b)						{ return BB::Mul::eval(a,b); }
Promotion<int2_,int2_>::strongest operator*(const int2_ a,const int2_ b)						{ return BB::Mul::eval(a,b); }
Promotion<int2_,float2_>::strongest operator*(const int2_ a,const float2_ b)					{ return BB::Mul::eval(a,b); }
Promotion<int2_,double2_>::strongest operator*(const int2_ a,const double2_ b)					{ return BB::Mul::eval(a,b); }
Promotion<float2_,int>::strongest operator*(const float2_ a,const int b)						{ return BB::Mul::eval(a,b); }
Promotion<float2_,float>::strongest operator*(const float2_ a,const float b)					{ return BB::Mul::eval(a,b); }
Promotion<float2_,double>::strongest operator*(const float2_ a,const double b)					{ return BB::Mul::eval(a,b); }
Promotion<float2_,int2_>::strongest operator*(const float2_ a,const int2_ b)					{ return BB::Mul::eval(a,b); }
Promotion<float2_,float2_>::strongest operator*(const float2_ a,const float2_ b)				{ return BB::Mul::eval(a,b); }
Promotion<float2_,double2_>::strongest operator*(const float2_ a,const double2_ b)				{ return BB::Mul::eval(a,b); }
Promotion<double2_,int>::strongest operator*(const double2_ a,const int b)						{ return BB::Mul::eval(a,b); }
Promotion<double2_,float>::strongest operator*(const double2_ a,const float b)					{ return BB::Mul::eval(a,b); }
Promotion<double2_,double>::strongest operator*(const double2_ a,const double b)				{ return BB::Mul::eval(a,b); }
Promotion<double2_,int2_>::strongest operator*(const double2_ a,const int2_ b)					{ return BB::Mul::eval(a,b); }
Promotion<double2_,float2_>::strongest operator*(const double2_ a,const float2_ b)				{ return BB::Mul::eval(a,b); }
Promotion<double2_,double2_>::strongest operator*(const double2_ a,const double2_ b)			{ return BB::Mul::eval(a,b); }

#define Matrix_Scalar_Matrix(T1,T2,OpClass,overloaded_operator) template <> \
	BB::Expr<BB::BinExpr<const T1*,const BB::Scalar<T2>,BB::OpClass,typename BB::Promotion<T1,T2>::strongest>,typename BB::Promotion<T1,T2>::strongest> BB::overloaded_operator(const BB::Hmatrix<T1> &v1,const T2 v2) \
	{ \
		BB::Scalar<T2> c(v2); \
		typedef BB::BinExpr<const T1*,const BB::Scalar<T2>,BB::OpClass,typename BB::Promotion<T1,T2>::strongest> BExpr; \
		return BB::Expr<BExpr,typename BB::Promotion<T1,T2>::strongest>(BExpr(v1.GetDataPointer(),c),v1.GetRows(),v1.GetColumns(),ISHOST); \
	}

#define Matrix_Scalar_CudaMatrix(T1,T2,OpClass,overloaded_operator) template <> \
	BB::Expr<BB::BinExpr<const T1*,const BB::Scalar<T2>,BB::OpClass,typename BB::Promotion<T1,T2>::strongest>,typename BB::Promotion<T1,T2>::strongest> BB::overloaded_operator(const BB::Dmatrix<T1> &v1,const T2 v2) \
	{ \
		BB::Scalar<T2> c(v2); \
		typedef BB::BinExpr<const T1*,const BB::Scalar<T2>,BB::OpClass,typename BB::Promotion<T1,T2>::strongest> BExpr; \
		return BB::Expr<BExpr,typename BB::Promotion<T1,T2>::strongest>(BExpr(v1.GetDataPointer(),c),v1.GetRows(),v1.GetColumns(),ISDEVICE); \
	}

#define Scalar_Matrix_Matrix(T1,T2,OpClass,overloaded_operator) template <> BB::Expr<BB::BinExpr<const BB::Scalar<T2>,const T1*,BB::OpClass,typename BB::Promotion<T1,T2>::strongest>,typename BB::Promotion<T1,T2>::strongest> BB::overloaded_operator(const T2 v1,const BB::Hmatrix<T1> &v2) \
	{ \
		BB::Scalar<T2> c(v1); \
		typedef BB::BinExpr<const BB::Scalar<T2>,const T1*,BB::OpClass,typename BB::Promotion<T1,T2>::strongest> BExpr; \
		return BB::Expr<BExpr,typename BB::Promotion<T1,T2>::strongest>(BExpr(c,v2.GetDataPointer()),v2.GetRows(),v2.GetColumns(),ISHOST); \
	}

#define Scalar_Matrix_CudaMatrix(T1,T2,OpClass,overloaded_operator) template <> BB::Expr<BB::BinExpr<const BB::Scalar<T2>,const T1*,BB::OpClass,typename BB::Promotion<T1,T2>::strongest>,typename BB::Promotion<T1,T2>::strongest> BB::overloaded_operator(const T2 v1,const BB::Dmatrix<T1> &v2) \
	{ \
		BB::Scalar<T2> c(v1); \
		typedef BB::BinExpr<const BB::Scalar<T2>,const T1*,BB::OpClass,typename BB::Promotion<T1,T2>::strongest> BExpr; \
		return BB::Expr<BExpr,typename BB::Promotion<T1,T2>::strongest>(BExpr(c,v2.GetDataPointer()),v2.GetRows(),v2.GetColumns(),ISDEVICE); \
	}

// Hmatrix-Hmatrix Hmatrix -- TESTED
#define Matrix_Matrix_Matrix(T1,T2,OpClass,overloaded_operator) template <class T1,class T2> \
BB::Expr<BB::BinExpr<const T1*,const T2*,BB::OpClass,typename BB::Promotion<T1,T2>::strongest>,typename BB::Promotion<T1,T2>::strongest> BB::overloaded_operator(const BB::Hmatrix<T1> &v1,const BB::Hmatrix<T2> &v2) \
	{ \
		if((v1.GetRows() == v2.GetRows()) && (v1.GetColumns() == v2.GetColumns())) \
		{ \
			typedef BB::BinExpr<const T1*,const T2*,BB::OpClass,typename BB::Promotion<T1,T2>::strongest> BExpr; \
			return BB::Expr<BExpr,typename BB::Promotion<T1,T2>::strongest>(BExpr(v1.GetDataPointer(),v2.GetDataPointer()),v1.GetRows(),v1.GetColumns(),ISHOST); \
		} else { char* str0 = "****************************************************************\n"; \
				 char* str1 = "* Size mismatch in binary CPU matrix operation (matrix-matrix) *\n"; \
				 char* str2 = "Left operand size: "; \
				 char* str3 = "Right operand size: "; \
				 char* str4 = "Operation: "; \
				 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+50*sizeof(char)); \
				 sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n%s%s",str0,str1,str0,str2,v1.GetRows(),v1.GetColumns(),str3,v2.GetRows(),v2.GetColumns(),str4,typeid(BB::OpClass).name()); \
				 throw  BB::GenericError(catString,__FILE__,__LINE__); \
		} \
	}

// Hmatrix-Hmatrix Dmatrix -- TESTED
#define Matrix_Matrix_CudaMatrix(T1,T2,OpClass,overloaded_operator) template <class T1,class T2> \
	BB::Expr<BB::BinExpr<const T1*,const T2*,BB::OpClass,typename BB::Promotion<T1,T2>::strongest>,typename BB::Promotion<T1,T2>::strongest> BB::overloaded_operator(const BB::Dmatrix<T1> &v1,const BB::Dmatrix<T2> &v2) \
	{ \
		if((v1.GetRows() == v2.GetRows()) && (v1.GetColumns() == v2.GetColumns())) \
		{ \
			typedef BB::BinExpr<const T1*,const T2*,BB::OpClass,typename BB::Promotion<T1,T2>::strongest> BExpr; \
			return BB::Expr<BExpr,typename BB::Promotion<T1,T2>::strongest>(BExpr(v1.GetDataPointer(),v2.GetDataPointer()),v1.GetRows(),v1.GetColumns(),ISDEVICE); \
		} else { char* str0 = "****************************************************************\n"; \
				 char* str1 = "* Size mismatch in binary GPU matrix operation (matrix-matrix) *\n"; \
				 char* str2 = "Left operand size: "; \
				 char* str3 = "Right operand size: "; \
				 char* str4 = "Operation: "; \
				 char* catString = (char*) malloc(2*strlen(str0)+strlen(str1)+strlen(str2)+strlen(str3)+strlen(str4)+50*sizeof(char)); \
				 sprintf(catString, "%s%s%s\n%s%i x %i\n%s%i x %i\n%s%s",str0,str1,str0,str2,v1.GetRows(),v1.GetColumns(),str3,v2.GetRows(),v2.GetColumns(),str4,typeid(BB::OpClass).name()); \
				 throw  BB::GenericError(catString,__FILE__,__LINE__); \
		} \
	}

Matrix_Matrix_Matrix(T1,T2,Mul,operator*)
Matrix_Matrix_CudaMatrix(T1,T2,Mul,operator*)

// Hmatrix-Hmatrix Hmatrix -- TESTED
#define Matrix_Matrix_Matrix_Prototype(T1,T2,OpClass,overloaded_operator) template BB::Expr<BB::BinExpr<const T1*,const T2*,BB::OpClass,typename BB::Promotion<T1,T2>::strongest>,typename BB::Promotion<T1,T2>::strongest> BB::overloaded_operator(const BB::Hmatrix<T1>&,const BB::Hmatrix<T2>&);

// Hmatrix-Hmatrix Dmatrix -- TESTED
#define Matrix_Matrix_CudaMatrix_Prototype(T1,T2,OpClass,overloaded_operator) template BB::Expr<BB::BinExpr<const T1*,const T2*,BB::OpClass,typename BB::Promotion<T1,T2>::strongest>,typename BB::Promotion<T1,T2>::strongest> BB::overloaded_operator(const BB::Dmatrix<T1>&,const BB::Dmatrix<T2>&);

Matrix_Scalar_Matrix(int,int,Mul,operator*)
Matrix_Scalar_Matrix(int,float,Mul,operator*)
Matrix_Scalar_Matrix(int,double,Mul,operator*)
Matrix_Scalar_Matrix(int,BB::int2_,Mul,operator*)
Matrix_Scalar_Matrix(int,BB::float2_,Mul,operator*)
Matrix_Scalar_Matrix(int,BB::double2_,Mul,operator*)
Matrix_Scalar_Matrix(float,int,Mul,operator*)
Matrix_Scalar_Matrix(float,float,Mul,operator*)
Matrix_Scalar_Matrix(float,double,Mul,operator*)
Matrix_Scalar_Matrix(float,BB::int2_,Mul,operator*)
Matrix_Scalar_Matrix(float,BB::float2_,Mul,operator*)
Matrix_Scalar_Matrix(float,BB::double2_,Mul,operator*)
Matrix_Scalar_Matrix(double,int,Mul,operator*)
Matrix_Scalar_Matrix(double,float,Mul,operator*)
Matrix_Scalar_Matrix(double,double,Mul,operator*)
Matrix_Scalar_Matrix(double,BB::int2_,Mul,operator*)
Matrix_Scalar_Matrix(double,BB::float2_,Mul,operator*)
Matrix_Scalar_Matrix(double,BB::double2_,Mul,operator*)
Matrix_Scalar_Matrix(BB::int2_,int,Mul,operator*)
Matrix_Scalar_Matrix(BB::int2_,float,Mul,operator*)
Matrix_Scalar_Matrix(BB::int2_,double,Mul,operator*)
Matrix_Scalar_Matrix(BB::int2_,BB::int2_,Mul,operator*)
Matrix_Scalar_Matrix(BB::int2_,BB::float2_,Mul,operator*)
Matrix_Scalar_Matrix(BB::int2_,BB::double2_,Mul,operator*)
Matrix_Scalar_Matrix(BB::float2_,int,Mul,operator*)
Matrix_Scalar_Matrix(BB::float2_,float,Mul,operator*)
Matrix_Scalar_Matrix(BB::float2_,double,Mul,operator*)
Matrix_Scalar_Matrix(BB::float2_,BB::int2_,Mul,operator*)
Matrix_Scalar_Matrix(BB::float2_,BB::float2_,Mul,operator*)
Matrix_Scalar_Matrix(BB::float2_,BB::double2_,Mul,operator*)
Matrix_Scalar_Matrix(BB::double2_,int,Mul,operator*)
Matrix_Scalar_Matrix(BB::double2_,float,Mul,operator*)
Matrix_Scalar_Matrix(BB::double2_,double,Mul,operator*)
Matrix_Scalar_Matrix(BB::double2_,BB::int2_,Mul,operator*)
Matrix_Scalar_Matrix(BB::double2_,BB::float2_,Mul,operator*)
Matrix_Scalar_Matrix(BB::double2_,BB::double2_,Mul,operator*)

Matrix_Scalar_CudaMatrix(int,int,Mul,operator*)
Matrix_Scalar_CudaMatrix(int,float,Mul,operator*)
Matrix_Scalar_CudaMatrix(int,double,Mul,operator*)
Matrix_Scalar_CudaMatrix(int,BB::int2_,Mul,operator*)
Matrix_Scalar_CudaMatrix(int,BB::float2_,Mul,operator*)
Matrix_Scalar_CudaMatrix(int,BB::double2_,Mul,operator*)
Matrix_Scalar_CudaMatrix(float,int,Mul,operator*)
Matrix_Scalar_CudaMatrix(float,float,Mul,operator*)
Matrix_Scalar_CudaMatrix(float,double,Mul,operator*)
Matrix_Scalar_CudaMatrix(float,BB::int2_,Mul,operator*)
Matrix_Scalar_CudaMatrix(float,BB::float2_,Mul,operator*)
Matrix_Scalar_CudaMatrix(float,BB::double2_,Mul,operator*)
Matrix_Scalar_CudaMatrix(double,int,Mul,operator*)
Matrix_Scalar_CudaMatrix(double,float,Mul,operator*)
Matrix_Scalar_CudaMatrix(double,double,Mul,operator*)
Matrix_Scalar_CudaMatrix(double,BB::int2_,Mul,operator*)
Matrix_Scalar_CudaMatrix(double,BB::float2_,Mul,operator*)
Matrix_Scalar_CudaMatrix(double,BB::double2_,Mul,operator*)
Matrix_Scalar_CudaMatrix(BB::int2_,int,Mul,operator*)
Matrix_Scalar_CudaMatrix(BB::int2_,float,Mul,operator*)
Matrix_Scalar_CudaMatrix(BB::int2_,double,Mul,operator*)
Matrix_Scalar_CudaMatrix(BB::int2_,BB::int2_,Mul,operator*)
Matrix_Scalar_CudaMatrix(BB::int2_,BB::float2_,Mul,operator*)
Matrix_Scalar_CudaMatrix(BB::int2_,BB::double2_,Mul,operator*)
Matrix_Scalar_CudaMatrix(BB::float2_,int,Mul,operator*)
Matrix_Scalar_CudaMatrix(BB::float2_,float,Mul,operator*)
Matrix_Scalar_CudaMatrix(BB::float2_,double,Mul,operator*)
Matrix_Scalar_CudaMatrix(BB::float2_,BB::int2_,Mul,operator*)
Matrix_Scalar_CudaMatrix(BB::float2_,BB::float2_,Mul,operator*)
Matrix_Scalar_CudaMatrix(BB::float2_,BB::double2_,Mul,operator*)
Matrix_Scalar_CudaMatrix(BB::double2_,int,Mul,operator*)
Matrix_Scalar_CudaMatrix(BB::double2_,float,Mul,operator*)
Matrix_Scalar_CudaMatrix(BB::double2_,double,Mul,operator*)
Matrix_Scalar_CudaMatrix(BB::double2_,BB::int2_,Mul,operator*)
Matrix_Scalar_CudaMatrix(BB::double2_,BB::float2_,Mul,operator*)
Matrix_Scalar_CudaMatrix(BB::double2_,BB::double2_,Mul,operator*)

Scalar_Matrix_Matrix(int,int,Mul,operator*)
Scalar_Matrix_Matrix(int,float,Mul,operator*)
Scalar_Matrix_Matrix(int,double,Mul,operator*)
Scalar_Matrix_Matrix(int,BB::int2_,Mul,operator*)
Scalar_Matrix_Matrix(int,BB::float2_,Mul,operator*)
Scalar_Matrix_Matrix(int,BB::double2_,Mul,operator*)
Scalar_Matrix_Matrix(float,int,Mul,operator*)
Scalar_Matrix_Matrix(float,float,Mul,operator*)
Scalar_Matrix_Matrix(float,double,Mul,operator*)
Scalar_Matrix_Matrix(float,BB::int2_,Mul,operator*)
Scalar_Matrix_Matrix(float,BB::float2_,Mul,operator*)
Scalar_Matrix_Matrix(float,BB::double2_,Mul,operator*)
Scalar_Matrix_Matrix(double,int,Mul,operator*)
Scalar_Matrix_Matrix(double,float,Mul,operator*)
Scalar_Matrix_Matrix(double,double,Mul,operator*)
Scalar_Matrix_Matrix(double,BB::int2_,Mul,operator*)
Scalar_Matrix_Matrix(double,BB::float2_,Mul,operator*)
Scalar_Matrix_Matrix(double,BB::double2_,Mul,operator*)
Scalar_Matrix_Matrix(BB::int2_,int,Mul,operator*)
Scalar_Matrix_Matrix(BB::int2_,float,Mul,operator*)
Scalar_Matrix_Matrix(BB::int2_,double,Mul,operator*)
Scalar_Matrix_Matrix(BB::int2_,BB::int2_,Mul,operator*)
Scalar_Matrix_Matrix(BB::int2_,BB::float2_,Mul,operator*)
Scalar_Matrix_Matrix(BB::int2_,BB::double2_,Mul,operator*)
Scalar_Matrix_Matrix(BB::float2_,int,Mul,operator*)
Scalar_Matrix_Matrix(BB::float2_,float,Mul,operator*)
Scalar_Matrix_Matrix(BB::float2_,double,Mul,operator*)
Scalar_Matrix_Matrix(BB::float2_,BB::int2_,Mul,operator*)
Scalar_Matrix_Matrix(BB::float2_,BB::float2_,Mul,operator*)
Scalar_Matrix_Matrix(BB::float2_,BB::double2_,Mul,operator*)
Scalar_Matrix_Matrix(BB::double2_,int,Mul,operator*)
Scalar_Matrix_Matrix(BB::double2_,float,Mul,operator*)
Scalar_Matrix_Matrix(BB::double2_,double,Mul,operator*)
Scalar_Matrix_Matrix(BB::double2_,BB::int2_,Mul,operator*)
Scalar_Matrix_Matrix(BB::double2_,BB::float2_,Mul,operator*)
Scalar_Matrix_Matrix(BB::double2_,BB::double2_,Mul,operator*)

Scalar_Matrix_CudaMatrix(int,int,Mul,operator*)
Scalar_Matrix_CudaMatrix(int,float,Mul,operator*)
Scalar_Matrix_CudaMatrix(int,double,Mul,operator*)
Scalar_Matrix_CudaMatrix(int,BB::int2_,Mul,operator*)
Scalar_Matrix_CudaMatrix(int,BB::float2_,Mul,operator*)
Scalar_Matrix_CudaMatrix(int,BB::double2_,Mul,operator*)
Scalar_Matrix_CudaMatrix(float,int,Mul,operator*)
Scalar_Matrix_CudaMatrix(float,float,Mul,operator*)
Scalar_Matrix_CudaMatrix(float,double,Mul,operator*)
Scalar_Matrix_CudaMatrix(float,BB::int2_,Mul,operator*)
Scalar_Matrix_CudaMatrix(float,BB::float2_,Mul,operator*)
Scalar_Matrix_CudaMatrix(float,BB::double2_,Mul,operator*)
Scalar_Matrix_CudaMatrix(double,int,Mul,operator*)
Scalar_Matrix_CudaMatrix(double,float,Mul,operator*)
Scalar_Matrix_CudaMatrix(double,double,Mul,operator*)
Scalar_Matrix_CudaMatrix(double,BB::int2_,Mul,operator*)
Scalar_Matrix_CudaMatrix(double,BB::float2_,Mul,operator*)
Scalar_Matrix_CudaMatrix(double,BB::double2_,Mul,operator*)
Scalar_Matrix_CudaMatrix(BB::int2_,int,Mul,operator*)
Scalar_Matrix_CudaMatrix(BB::int2_,float,Mul,operator*)
Scalar_Matrix_CudaMatrix(BB::int2_,double,Mul,operator*)
Scalar_Matrix_CudaMatrix(BB::int2_,BB::int2_,Mul,operator*)
Scalar_Matrix_CudaMatrix(BB::int2_,BB::float2_,Mul,operator*)
Scalar_Matrix_CudaMatrix(BB::int2_,BB::double2_,Mul,operator*)
Scalar_Matrix_CudaMatrix(BB::float2_,int,Mul,operator*)
Scalar_Matrix_CudaMatrix(BB::float2_,float,Mul,operator*)
Scalar_Matrix_CudaMatrix(BB::float2_,double,Mul,operator*)
Scalar_Matrix_CudaMatrix(BB::float2_,BB::int2_,Mul,operator*)
Scalar_Matrix_CudaMatrix(BB::float2_,BB::float2_,Mul,operator*)
Scalar_Matrix_CudaMatrix(BB::float2_,BB::double2_,Mul,operator*)
Scalar_Matrix_CudaMatrix(BB::double2_,int,Mul,operator*)
Scalar_Matrix_CudaMatrix(BB::double2_,float,Mul,operator*)
Scalar_Matrix_CudaMatrix(BB::double2_,double,Mul,operator*)
Scalar_Matrix_CudaMatrix(BB::double2_,BB::int2_,Mul,operator*)
Scalar_Matrix_CudaMatrix(BB::double2_,BB::float2_,Mul,operator*)
Scalar_Matrix_CudaMatrix(BB::double2_,BB::double2_,Mul,operator*)

Matrix_Matrix_Matrix_Prototype(int,int,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(int,float,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(int,double,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(int,BB::int2_,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(int,BB::float2_,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(int,BB::double2_,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(float,int,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(float,float,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(float,double,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(float,BB::int2_,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(float,BB::float2_,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(float,BB::double2_,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(double,int,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(double,float,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(double,double,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(double,BB::int2_,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(double,BB::float2_,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(double,BB::double2_,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(BB::int2_,int,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(BB::int2_,float,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(BB::int2_,double,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(BB::int2_,BB::int2_,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(BB::int2_,BB::float2_,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(BB::int2_,BB::double2_,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(BB::float2_,int,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(BB::float2_,float,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(BB::float2_,double,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(BB::float2_,BB::int2_,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(BB::float2_,BB::float2_,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(BB::float2_,BB::double2_,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(BB::double2_,int,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(BB::double2_,float,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(BB::double2_,double,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(BB::double2_,BB::int2_,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(BB::double2_,BB::float2_,Mul,operator*)
Matrix_Matrix_Matrix_Prototype(BB::double2_,BB::double2_,Mul,operator*)

Matrix_Matrix_CudaMatrix_Prototype(int,int,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(int,float,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(int,double,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(int,BB::int2_,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(int,BB::float2_,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(int,BB::double2_,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(float,int,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(float,float,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(float,double,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(float,BB::int2_,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(float,BB::float2_,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(float,BB::double2_,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(double,int,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(double,float,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(double,double,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(double,BB::int2_,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(double,BB::float2_,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(double,BB::double2_,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(BB::int2_,int,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(BB::int2_,float,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(BB::int2_,double,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(BB::int2_,BB::int2_,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(BB::int2_,BB::float2_,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(BB::int2_,BB::double2_,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(BB::float2_,int,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(BB::float2_,float,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(BB::float2_,double,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(BB::float2_,BB::int2_,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(BB::float2_,BB::float2_,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(BB::float2_,BB::double2_,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(BB::double2_,int,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(BB::double2_,float,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(BB::double2_,double,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(BB::double2_,BB::int2_,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(BB::double2_,BB::float2_,Mul,operator*)
Matrix_Matrix_CudaMatrix_Prototype(BB::double2_,BB::double2_,Mul,operator*)

