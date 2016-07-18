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


#ifndef __COMPLEXTYPES_H__
#define __COMPLEXTYPES_H__

#include "BB.h"

#include <ostream>							// Needed for ostream
#include <iomanip>							// Needed for setw

#include <cuda.h>
#include <cuda_runtime.h>

namespace BB
{

	class int2_ {
	
		public:
/*! \brief Complex number.
 *
 *  Example:
 *
 *      Assign the real part of an int2_ object
 *
 *		foo.c.x = 3; 
 *
 *      Display the imaginary part of a double2_ object
 *
 *		coud << foo.c.y << endl; 
 *
 */
			int2 c;
/** @name Constructors and Destructor
 */
//@{	
/*! \brief Constructor with no input argument.
 *
 *  Example:
 *
 *      Create a non-initialized int2_ number
 *
 *		int2_ foo; 
 *
 */
			__host__ __device__ int2_() {};
/*! \brief Constructor with input arguments - real and imaginary parts.
 *
 *  Example:
 *
 *      Create an initialized int2_ number
 *
 *		int2_ foo(real_part,imaginary_part); 
 *
 */
			__host__ __device__ __forceinline __forceinline__ int2_(const int x_,const int y_) { c.x=x_; c.y=y_;};
/*! \brief Constructor with input arguments - integer real part - imaginary part set to zero.
 *
 *  Example:
 *
 *      Create an initialized int2_ number
 *
 *		int2_ foo(real_part); 
 *
 */
			__host__ __device__ __forceinline __forceinline__ int2_(const int a) { c.x = a; c.y=0; };
/*! \brief Constructor with input arguments - float real part - imaginary part set to zero.
 *
 *  Example:
 *
 *      Create an initialized int2_ number
 *
 *		int2_ foo(real_part); 
 *
 */
			__host__ __device__ __forceinline __forceinline__ int2_(const float a) { c.x = (int)a; c.y=0; };
/*! \brief Constructor with input arguments - double real part - imaginary part set to zero.
 *
 *  Example:
 *
 *      Create an initialized int2_ number
 *
 *		int2_ foo(real_part); 
 *
 */
			__host__ __device__ __forceinline __forceinline__ int2_(const double a) { c.x = (int)a; c.y=0; };
/*! \brief Constructor with float2_ complex input arguments.
 *
 *  Example:
 *
 *      Create an initialized int2_ number
 *
 *		int2_ foo(complex_number); 
 *
 */
			__host__ __device__ __forceinline __forceinline__ int2_(const float2_ a);
/*! \brief Constructor with double2_ complex input arguments.
 *
 *  Example:
 *
 *      Create an initialized int2_ number
 *
 *		int2_ foo(complex_number); 
 *
 */
			__host__ __device__ __forceinline __forceinline__ int2_(const double2_ a);
//@}


/** @name Assignment operators
 */
//@{	
/*! \brief Assignment from an int.
 *
 *  Example:
 *
 *      Assign an int to an int2_ object.
 *
 *      int a; int2_ foo;
 *      
 *      foo = a;
 *
 */
			__host__ __device__ __forceinline __forceinline__ const int2_& operator=(const int a) { c.x = a; c.y = 0.; return *this; };
/*! \brief Assignment from a float.
 *
 *  Example:
 *
 *      Assign a float to an int2_ object.
 *
 *      float a; int2_ foo;
 *      
 *      foo = a;
 *
 */
			__host__ __device__ __forceinline __forceinline__ const int2_& operator=(const float a) { c.x = (int)a;	c.y = 0.; return *this; };
/*! \brief Assignment from a double.
 *
 *  Example:
 *
 *      Assign a double to an int2_ object.
 *
 *      double a; int2_ foo;
 *      
 *      foo = a;
 *
 */
			__host__ __device__ __forceinline __forceinline__ const int2_& operator=(const double a) { c.x = (int)a; c.y = 0.; return *this; };
/*! \brief Assignment from a int2_.
 *
 *  Example:
 *
 *      Assign an int2_ to an int2_ object.
 *
 *      int2_ a; int2_ foo;
 *
 */
			__host__ __device__ __forceinline __forceinline__ const int2_& operator=(const int2_ a) { c.x = a.c.x; c.y = a.c.y;	return *this; };
/*! \brief Assignment from a float2_.
 *
 *  Example:
 *
 *      Assign a float2_ to an int2_ object.
 *
 *      float2_ a; int2_ foo;
 *      
 */
			__host__ __device__ __forceinline __forceinline__ const int2_& operator=(const float2_ a);
/*! \brief Assignment from a double2_.
 *
 *  Example:
 *
 *      Assign a double2_ to an int2_ object.
 *
 *      double2_ a; int2_ foo;
 */
			__host__ __device__ __forceinline __forceinline__ const int2_& operator=(const double2_ a);

	};

	class float2_ {
		
		public:
/*! \brief Complex number.
 *
 *  Example:
 *
 *      Assign the real part of a float2_ object
 *
 *		foo.c.x = 3; 
 *
 *      Display the imaginary part of a double2_ object
 *
 *		coud << foo.c.y << endl; 
 *
 */
			float2 c;
/** @name Constructors and Destructor
 */
//@{	
/*! \brief Constructor with no input argument.
 *
 *  Example:
 *
 *      Create a non-initialized float2_ number
 *
 *		float2_ foo; 
 *
 */
			__host__ __device__ __forceinline __forceinline__ float2_(const float x_,const float y_) { c.x=x_; c.y=y_; };
/*! \brief Constructor with input arguments - real and imaginary parts.
 *
 *  Example:
 *
 *      Create an initialized float2_ number
 *
 *		float2_ foo(real_part,imaginary_part); 
 *
 */
			__host__ __device__ __forceinline __forceinline__ float2_() {};
/*! \brief Constructor with input arguments - integer real part - imaginary part set to zero.
 *
 *  Example:
 *
 *      Create an initialized float2_ number
 *
 *		float2_ foo(real_part); 
 *
 */
			__host__ __device__ __forceinline __forceinline__ float2_(const int a) { c.x = (float)a; c.y=0.; };
/*! \brief Constructor with input arguments - float real part - imaginary part set to zero.
 *
 *  Example:
 *
 *      Create an initialized float2_ number
 *
 *		float2_ foo(real_part); 
 *
 */
			__host__ __device__ __forceinline __forceinline__ float2_(const float a) { c.x = a; c.y=0.; };
/*! \brief Constructor with input arguments - double real part - imaginary part set to zero.
 *
 *  Example:
 *
 *      Create an initialized float2_ number
 *
 *		float2_ foo(real_part); 
 *
 */
			__host__ __device__ __forceinline __forceinline__ float2_(const double a) { c.x = (float)a; c.y=0.; };
/*! \brief Constructor with int2_ complex input arguments.
 *
 *  Example:
 *
 *      Create an initialized float2_ number
 *
 *		float2_ foo(complex_number); 
 *
 */
			__host__ __device__ __forceinline __forceinline__ float2_(const int2_ a) { c.x = (float)a.c.x; c.y=(float)a.c.y; };;
/*! \brief Constructor with double2_ complex input arguments.
 *
 *  Example:
 *
 *      Create an initialized float2_ number
 *
 *		float2_ foo(complex_number); 
 *
 */
			__host__ __device__ __forceinline __forceinline__ float2_(const double2_ a);
//@}


/** @name Assignment operators
 */
//@{	
/*! \brief Assignment from an int.
 *
 *  Example:
 *
 *      Assign an int to a float2_ object.
 *
 *      int a; float2_ foo;
 *      
 *      foo = a;
 *
 */
			__host__ __device__ __forceinline __forceinline__ const float2_& operator=(const int a) { c.x = (float)a; c.y = 0.;	return *this; };
/*! \brief Assignment from a float.
 *
 *  Example:
 *
 *      Assign a float to a float2_ object.
 *
 *      float a; float2_ foo;
 *      
 *      foo = a;
 *
 */			__host__ __device__ __forceinline __forceinline__ const float2_& operator=(const float a) { c.x = a; c.y = 0.; return *this; };
/*! \brief Assignment from a double.
 *
 *  Example:
 *
 *      Assign a double to a float2_ object.
 *
 *      double a; float2_ foo;
 *      
 *      foo = a;
 *
 */			__host__ __device__ __forceinline __forceinline__ const float2_& operator=(const double a) { c.x = (float)a; c.y = 0.; return *this; };
/*! \brief Assignment from an int2_.
 *
 *  Example:
 *
 *      Assign a int2_ to a float2_ object.
 *
 *      int2_ a; float2_ foo;
 *      
 *      foo = a;
 *
 */			__host__ __device__ __forceinline __forceinline__ const float2_& operator=(const int2_ a) { c.x = (float)a.c.x;	c.y = (float)a.c.y;	return *this; };
/*! \brief Assignment from a float2_.
 *
 *  Example:
 *
 *      Assign a float2_ to a float2_ object.
 *
 *      float2_ a; float2_ foo;
 *      
 *      foo = a;
 *
 */			__host__ __device__ __forceinline __forceinline__ const float2_& operator=(const float2_ a) { c.x = a.c.x; c.y = a.c.y;	return *this; };
/*! \brief Assignment from a double2_.
 *
 *  Example:
 *
 *      Assign a double2_ to a float2_ object.
 *
 *      double2_ a; float2_ foo;
 *      
 *      foo = a;
 *
 */			
		__host__ __device__ __forceinline __forceinline__ const float2_& operator=(const double2_ a);

	};

	class double2_ {
	
		public:
/*! \brief Complex number.
 *
 *  Example:
 *
 *      Assign the real part of a double2_ object
 *
 *		foo.c.x = 3; 
 *
 *      Display the imaginary part of a double2_ object
 *
 *		coud << foo.c.y << endl; 
 *
 */
			double2 c;
/** @name Constructors and Destructor
 */
//@{	
/*! \brief Constructor with no input argument.
 *
 *  Example:
 *
 *      Create a non-initialized float2_ number
 *
 *		float2_ foo; 
 *
 */
			//__host__ __device__ __forceinline__ double2_(const double x_,const double y_) { printf("double2_ constructor\n"); x=x_; y=y_; };
			__host__ __device__ __forceinline __forceinline__ double2_(const double x_,const double y_) { c.x=x_; c.y=y_; };
/*! \brief Constructor with input arguments - real and imaginary parts.
 *
 *  Example:
 *
 *      Create an initialized double2_ number
 *
 *		double2_ foo(real_part,imaginary_part); 
 *
 */
			__host__ __device__ __forceinline __forceinline__ double2_() {};
/*! \brief Constructor with input arguments - integer real part - imaginary part set to zero.
 *
 *  Example:
 *
 *      Create an initialized double2_ number
 *
 *		double2_ foo(real_part); 
 *
 */
			__host__ __device__ __forceinline __forceinline__ double2_(const int a) { c.x = (double)a; c.y=0.; };
/*! \brief Constructor with input arguments - float real part - imaginary part set to zero.
 *
 *  Example:
 *
 *      Create an initialized double2_ number
 *
 *		double2_ foo(real_part); 
 *
 */
			__host__ __device__ __forceinline __forceinline__ double2_(const float a) { c.x = (double)a; c.y=0.; };
/*! \brief Constructor with input arguments - double real part - imaginary part set to zero.
 *
 *  Example:
 *
 *      Create an initialized double2_ number
 *
 *		double2_ foo(real_part); 
 *
 */
			__host__ __device__ __forceinline __forceinline__ double2_(const double a) { c.x = a; c.y=0.; };
/*! \brief Constructor with int2_ complex input arguments.
 *
 *  Example:
 *
 *      Create an initialized double2_ number
 *
 *		double2_ foo(complex_number); 
 *
 */
			__host__ __device__ __forceinline __forceinline__ double2_(const int2_ a) { c.x = (double)a.c.x; c.y=(double)a.c.y; };
/*! \brief Constructor with float2_ complex input arguments.
 *
 *  Example:
 *
 *      Create an initialized double2_ number
 *
 *		double2_ foo(complex_number); 
 *
 */
			__host__ __device__ __forceinline __forceinline__ double2_(const float2_ a) { c.x = (double)a.c.x; c.y=(double)a.c.y; };
//@}


/** @name Assignment operators
 */
//@{	
/*! \brief Assignment from an int.
 *
 *  Example:
 *
 *      Assign an int to a double2_ object.
 *
 *      int a; double2_ foo;
 *      
 *      foo = a;
 *
 */
			__host__ __device__ __forceinline __forceinline__ const double2_& operator=(const int a) { c.x = (double)a;	c.y = 0.; return *this; };
/*! \brief Assignment from a float.
 *
 *  Example:
 *
 *      Assign a float to a double2_ object.
 *
 *      float a; double2_ foo;
 *      
 *      foo = a;
 *
 */			
			__host__ __device__ __forceinline __forceinline__ const double2_& operator=(const float a) { c.x = (double)a; c.y = 0.;	return *this; };
/*! \brief Assignment from a double.
 *
 *  Example:
 *
 *      Assign a double to a double2_ object.
 *
 *      double a; double2_ foo;
 *      
 *      foo = a;
 *
 */				
			__host__ __device__ __forceinline __forceinline__ const double2_& operator=(const double a) { c.x = a; c.y = 0.; return *this; };
/*! \brief Assignment from an int2_.
 *
 *  Example:
 *
 *      Assign an int2_ to a double2_ object.
 *
 *      int2_ a; double2_ foo;
 *      
 *      foo = a;
 *
 */	
			__host__ __device__ __forceinline __forceinline__ const double2_& operator=(const int2_ a) { c.x = (double)a.c.x; c.y = (double)a.c.y; return *this; };
/*! \brief Assignment from a float2_.
 *
 *  Example:
 *
 *      Assign a float2_ to a double2_ object.
 *
 *      float2_ a; double2_ foo;
 *      
 *      foo = a;
 *
 */				
			__host__ __device__ __forceinline __forceinline__ const double2_& operator=(const float2_ a) { c.x = (double)a.c.x;	c.y = (double)a.c.y; return *this; };
/*! \brief Assignment from a double2_.
 *
 *  Example:
 *
 *      Assign a double2_ to a double2_ object.
 *
 *      double2_ a; double2_ foo;
 *      
 *      foo = a;
 *
 */				
			__host__ __device__ __forceinline __forceinline__ const double2_& operator=(const double2_ a) { c.x = a.c.x; c.y = a.c.y; return *this; };

	};

		__host__ __device__ __forceinline __forceinline__ int2_::int2_(const float2_ a) { c.x = (int)a.c.x; c.y=(int)a.c.y; };
		__host__ __device__ __forceinline __forceinline__ int2_::int2_(const double2_ a) { c.x = (int)a.c.x; c.y=(int)a.c.y; };
		__host__ __device__ __forceinline __forceinline__ const int2_& int2_::operator=(const float2_ a) { c.x = (int)a.c.x; c.y = (int)a.c.y;	return *this; };
		__host__ __device__ __forceinline __forceinline__ const int2_& int2_::operator=(const double2_ a) { c.x = (int)a.c.x; c.y = (int)a.c.y; return *this; };
		__host__ __device__ __forceinline __forceinline__ float2_::float2_(const double2_ a) { c.x = (float)a.c.x; c.y=(float)a.c.y; };;
		__host__ __device__ __forceinline __forceinline__ const float2_& float2_::operator=(const double2_ a) { c.x = (float)a.c.x; c.y = (float)a.c.y; return *this; };
}



#endif