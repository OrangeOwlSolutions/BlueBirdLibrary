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



#ifndef __PROMOTION_CUH__
#define __PROMOTION_CUH__

namespace BB
{
	/******************/
	/* CUDA PROMOTION */
	/******************/

	// int-int, float-float, double-double, int2_-int2_, float2_-float2_-, double2_-double2_
	template<class T1,class T2> struct Promotion { typedef T1 strongest; };

	// --- From int to others
	template<> struct Promotion< float,		int >			{ typedef float strongest; };
	template<> struct Promotion< int,		float >			{ typedef float strongest; };

	template<> struct Promotion< double,	int >			{ typedef double strongest; };
	template<> struct Promotion< int,		double >		{ typedef double strongest; };

	template<> struct Promotion< int2_,		int >			{ typedef int2_ strongest; };
	template<> struct Promotion< int,		int2_ >			{ typedef int2_ strongest; };

	template<> struct Promotion< float2_,	int >			{ typedef float2_ strongest; };
	template<> struct Promotion< int,		float2_ >		{ typedef float2_ strongest; };

	template<> struct Promotion< double2_,	int >			{ typedef double2_ strongest; };
	template<> struct Promotion< int,		double2_ >		{ typedef double2_ strongest; };

	// --- From float to others
	template<> struct Promotion< double,	float >			{ typedef double strongest; };
	template<> struct Promotion< float,		double >		{ typedef double strongest; };

	template<> struct Promotion< int2_,		float >			{ typedef float2_ strongest; };
	template<> struct Promotion< float,		int2_ >			{ typedef float2_ strongest; };

	template<> struct Promotion< float2_,	float >			{ typedef float2_ strongest; };
	template<> struct Promotion< float,		float2_ >		{ typedef float2_ strongest; };

	template<> struct Promotion< double2_,	float >			{ typedef double2_ strongest; };
	template<> struct Promotion< float,		double2_ >		{ typedef double2_ strongest; };

	// --- From double to others
	template<> struct Promotion< int2_,		double >		{ typedef double2_ strongest; };
	template<> struct Promotion< double,	int2_ >			{ typedef double2_ strongest; };

	template<> struct Promotion< float2_,	double >		{ typedef double2_ strongest; };
	template<> struct Promotion< double,	float2_ >		{ typedef double2_ strongest; };

	template<> struct Promotion< double2_,	double >		{ typedef double2_ strongest; };
	template<> struct Promotion< double,	double2_ >		{ typedef double2_ strongest; };

	// --- From int2_ to others
	template<> struct Promotion< float2_,	int2_ >			{ typedef float2_ strongest; };
	template<> struct Promotion< int2_,		float2_ >		{ typedef float2_ strongest; };

	template<> struct Promotion< double2_,	int2_ >			{ typedef double2_ strongest; };
	template<> struct Promotion< int2_,		double2_ >		{ typedef double2_ strongest; };

	// --- From float2_ to others
	template<> struct Promotion< double2_,	float2_ >		{ typedef double2_ strongest; };
	template<> struct Promotion< float2_,	double2_ >		{ typedef double2_ strongest; };

	/*****************/
	/* CUDA DEMOTION */
	/*****************/

	// int-int, float-float, double-double, int2_-int2_, float2_-float2_-, double2_-double2_
	template <class T1,class T2> struct Demotion {typedef T1 weakest; };

	// --- From int to others
	template<> struct Demotion< float,		int >			{ typedef int weakest; };
	template<> struct Demotion< int,		float >			{ typedef int weakest; };

	template<> struct Demotion< double,		int >			{ typedef int weakest; };
	template<> struct Demotion< int,		double >		{ typedef int weakest; };

	// --- Check!!
	template<> struct Demotion< int2_,		int >			{ typedef int weakest; };
	template<> struct Demotion< int,		int2_ >			{ typedef int weakest; };

	// --- Check!!
	template<> struct Demotion< float2_,		int >			{ typedef int weakest; };
	template<> struct Demotion< int,		float2_ >		{ typedef int weakest; };

	// --- Check!!
	template<> struct Demotion< double2_,	int >			{ typedef int weakest; };
	template<> struct Demotion< int,		double2_ >		{ typedef int weakest; };

	// --- From float to others
	template<> struct Demotion< double,		float >			{ typedef float weakest; };
	template<> struct Demotion< float,		double >		{ typedef float weakest; };

	// --- Check!!
	template<> struct Demotion< int2_,		float >			{ typedef int weakest; };
	template<> struct Demotion< float,		int2_ >			{ typedef int weakest; };

	template<> struct Demotion< float2_,		float >			{ typedef float weakest; };
	template<> struct Demotion< float,		float2_ >		{ typedef float weakest; };

	template<> struct Demotion< double2_,	float >			{ typedef float weakest; };
	template<> struct Demotion< float,		double2_ >		{ typedef float weakest; };

	// --- From double to others
	// --- Check !!
	template<> struct Demotion< int2_,		double >		{ typedef int weakest; };
	template<> struct Demotion< double,		int2_ >			{ typedef int weakest; };

	// --- Check !!
	template<> struct Demotion< float2_,		double >		{ typedef float weakest; };
	template<> struct Demotion< double,		float2_ >		{ typedef float weakest; };
	//template<> struct Demotion< float2_,		double >		{ typedef double weakest; };
	//template<> struct Demotion< double,		float2_ >		{ typedef double weakest; };

	// --- Check !!
	template<> struct Demotion< double2_,	double >		{ typedef double weakest; };
	template<> struct Demotion< double,		double2_ >		{ typedef double weakest; };

	// --- From int2_ to others
	template<> struct Demotion< float2_,		int2_ >			{ typedef int2_ weakest; };
	template<> struct Demotion< int2_,		float2_ >		{ typedef int2_ weakest; };

	template<> struct Demotion< double2_,	int2_ >			{ typedef int2_ weakest; };
	template<> struct Demotion< int2_,		double2_ >		{ typedef int2_ weakest; };

	// --- From float2_ to others
	template<> struct Demotion< double2_,	float2_ >		{ typedef float2_ weakest; };
	template<> struct Demotion< float2_,		double2_ >		{ typedef float2_ weakest; };


	/*************/
	/* CUDA ROOT */
	/*************/

	// int-int, float-float, double-double
	template<class T1,class T2> struct Root { typedef T1 root; };

	// --- From int to others
	template<> struct Root< float,		int >			{ typedef float root; };
	template<> struct Root< int,		float >			{ typedef float root; };

	template<> struct Root< double,		int >			{ typedef double root; };
	template<> struct Root< int,		double >		{ typedef double root; };

	template<> struct Root< int2_,		int >			{ typedef int	root; };
	template<> struct Root< int,		int2_ >			{ typedef int	root; };

	template<> struct Root< float2_,	int >			{ typedef float root; };
	template<> struct Root< int,		float2_ >		{ typedef float root; };

	template<> struct Root< double2_,	int >			{ typedef double root; };
	template<> struct Root< int,		double2_ >		{ typedef double root; };

	// --- From float to others
	template<> struct Root< double,		float >			{ typedef double root; };
	template<> struct Root< float,		double >		{ typedef double root; };

	template<> struct Root< int2_,		float >			{ typedef float root; };
	template<> struct Root< float,		int2_ >			{ typedef float root; };

	template<> struct Root< float2_,	float >			{ typedef float root; };
	template<> struct Root< float,		float2_ >		{ typedef float root; };

	template<> struct Root< double2_,	float >			{ typedef double root; };
	template<> struct Root< float,		double2_ >		{ typedef double root; };

	// --- From double to others
	template<> struct Root< int2_,		double >		{ typedef double root; };
	template<> struct Root< double,		int2_ >			{ typedef double root; };

	template<> struct Root< float2_,	double >		{ typedef double root; };
	template<> struct Root< double,		float2_ >		{ typedef double root; };

	template<> struct Root< double2_,	double >		{ typedef double root; };
	template<> struct Root< double,		double2_ >		{ typedef double root; };

	// --- From int2_ to others
	template<> struct Root< int2_,		int2_ >			{ typedef int root; };

	template<> struct Root< float2_,	int2_ >			{ typedef float root; };
	template<> struct Root< int2_,		float2_ >		{ typedef float root; };

	template<> struct Root< double2_,	int2_ >			{ typedef double root; };
	template<> struct Root< int2_,		double2_ >		{ typedef double root; };

	// --- From float2_ to others
	template<> struct Root< float2_,	float2_ >		{ typedef float root; };

	template<> struct Root< double2_,	float2_ >		{ typedef double root; };
	template<> struct Root< float2_,	double2_ >		{ typedef double root; };
	
	// --- From double2_ to others
	template<> struct Root< double2_,	double2_ >		{ typedef double root; };

	//template <class T1> struct NumbersOnly 
//{
//    
//        typedef T1 Type;
//};
//
//template <> struct NumbersOnly<int> 
//{
//        typedef int Type;
//};
//template <> struct NumbersOnly<float> 
//{
//        typedef float Type;
//};
//template <> struct NumbersOnly<double> 
//{
//        typedef double Type;
//};
//
//template <> struct NumbersOnly<float2_> 
//{
//        typedef float2_ Type;
//};
//template <> struct NumbersOnly<double2_> 
//{
//        typedef double2_ Type;
//};



}
#endif
