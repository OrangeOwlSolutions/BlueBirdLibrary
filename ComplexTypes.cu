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


#include "ComplexTypes.cuh"
#include "Constants.h"

using namespace BB;

/*********/
/* INT2_ */
/*********/
//__host__ __device__			BB::int2_::int2_(const int x_,const int y_)			{ c.x=x_; c.y=y_;}
//__host__ __device__			BB::int2_::int2_() {}
//__host__ __device__			BB::int2_::int2_(const int a)							{ c.x = a; c.y=0; }
//__host__ __device__			BB::int2_::int2_(const float a)						{ c.x = (int)a; c.y=0; }
//__host__ __device__			BB::int2_::int2_(const double a)						{ c.x = (int)a; c.y=0; }
//__host__ __device__			BB::int2_::int2_(const BB::float2_ a)	{ c.x = (int)a.c.x; c.y=(int)a.c.y; }
//__host__ __device__			BB::int2_::int2_(const BB::double2_ a)	{ c.x = (int)a.c.x; c.y=(int)a.c.y; }

//__host__ __device__ const	BB::int2_& BB::int2_::operator=(const int a)						{ c.x = a;			c.y = 0.;				return *this; }
//__host__ __device__ const	BB::int2_& BB::int2_::operator=(const float a)						{ c.x = (int)a;		c.y = 0.;				return *this; }
//__host__ __device__ const	BB::int2_& BB::int2_::operator=(const double a)						{ c.x = (int)a;		c.y = 0.;				return *this; }
//__host__ __device__ const	BB::int2_& BB::int2_::operator=(const BB::int2_ a)	{ c.x = a.c.x;			c.y = a.c.y;			return *this; }
//__host__ __device__ const	BB::int2_& BB::int2_::operator=(const BB::float2_ a)	{ c.x = (int)a.c.x;		c.y = (int)a.c.y;		return *this; }
//__host__ __device__ const	BB::int2_& BB::int2_::operator=(const BB::double2_ a)	{ c.x = (int)a.c.x;		c.y = (int)a.c.y;		return *this; }

/***********/
/* FLOAT2_ */
/***********/

//__host__ __device__			BB::float2_::float2_(const float x_, const float y_)		{ c.x=x_; c.y=y_; }
//__host__ __device__			BB::float2_::float2_()									{}
//__host__ __device__			BB::float2_::float2_(const int a)							{ c.x = (float)a; c.y=0.; };
//__host__ __device__			BB::float2_::float2_(const float a)						{ c.x = a; c.y=0.; };
//__host__ __device__			BB::float2_::float2_(const double a)						{ c.x = (float)a; c.y=0.; };
//__host__ __device__			BB::float2_::float2_(const BB::int2_ a)		{ c.x = (float)a.c.x; c.y=(float)a.c.y; };
//__host__ __device__			BB::float2_::float2_(const BB::double2_ a)	{ c.x = (float)a.c.x; c.y=(float)a.c.y; };

//__host__ __device__ const	BB::float2_& BB::float2_::operator=(const int a)						{ c.x = (float)a;		c.y = 0.;				return *this; }
//__host__ __device__ const	BB::float2_& BB::float2_::operator=(const float a)						{ c.x = a;			c.y = 0.;				return *this; }
//__host__ __device__ const	BB::float2_& BB::float2_::operator=(const double a)						{ c.x = (float)a;		c.y = 0.;				return *this; }
//__host__ __device__ const	BB::float2_& BB::float2_::operator=(const int2_ a)						{ c.x = (float)a.c.x;	c.y = (float)a.c.y;		return *this; }
//__host__ __device__ const	BB::float2_& BB::float2_::operator=(const BB::float2_ a)	{ c.x = a.c.x;			c.y = a.c.y;			return *this; }
//__host__ __device__ const	BB::float2_& BB::float2_::operator=(const BB::double2_ a)	{ c.x = (float)a.c.x;	c.y = (float)a.c.y;		return *this; }

/************/
/* DOUBLE2_ */
/************/

//__host__ __device__			BB::double2_::double2_(const double x_,const double y_)	{ c.x=x_; c.y=y_; }
//__host__ __device__			BB::double2_::double2_()									{}
//__host__ __device__			BB::double2_::double2_(const int a)						{ c.x = (double)a; c.y=0.; }
//__host__ __device__			BB::double2_::double2_(const float a)						{ c.x = (double)a; c.y=0.; }
//__host__ __device__			BB::double2_::double2_(const double a)					{ c.x = a; c.y=0.; }
//__host__ __device__			BB::double2_::double2_(const BB::int2_ a)	{ c.x = (double)a.c.x; c.y=(double)a.c.y; }
//__host__ __device__			BB::double2_::double2_(const BB::float2_ a)	{ c.x = (double)a.c.x; c.y=(double)a.c.y; }

//__host__ __device__ const	BB::double2_& BB::double2_::operator=(const int a)							{ c.x = (double)a;	c.y = 0.;				return *this; }
//__host__ __device__ const	BB::double2_& BB::double2_::operator=(const float a)						{ c.x = (double)a;	c.y = 0.;				return *this; }
//__host__ __device__ const	BB::double2_& BB::double2_::operator=(const double a)						{ c.x = a;			c.y = 0.;				return *this; }
//__host__ __device__ const	BB::double2_& BB::double2_::operator=(const BB::int2_ a)		{ c.x = (double)a.c.x;	c.y = (double)a.c.y;	return *this; }
//__host__ __device__ const	BB::double2_& BB::double2_::operator=(const BB::float2_ a)	{ c.x = (double)a.c.x;	c.y = (double)a.c.y;	return *this; }
//__host__ __device__ const	BB::double2_& BB::double2_::operator=(const BB::double2_ a)	{ c.x = a.c.x;			c.y = a.c.y;			return *this; }

// --- Overload of << for int2_
std::ostream& operator<<(std::ostream& output, const BB::int2_& v)
{
	output << std::setw(prec_cout) << "(" << v.c.x << "," << v.c.y << ")\t";
	return output;
}

// --- Overload of << for float2_
std::ostream& operator<<(std::ostream& output, const BB::float2_& v)
{
	output << std::setw(prec_cout) << "(" << v.c.x << "," << v.c.y << ")\t";
	return output;
}
// --- Overload of << for double2_
std::ostream& operator<<(std::ostream& output, const BB::double2_& v)
{
	output << std::setw(prec_cout) << "(" << v.c.x << "," << v.c.y << ")\t";
	return output;
}