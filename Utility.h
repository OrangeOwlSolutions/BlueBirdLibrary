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


#ifndef __UTILITY_H__
#define __UTILITY_H__

namespace BB
{
	/*******************/
	/* RANDOM MATRICES */
	/*******************/
	Hmatrix<int> Rand(const int r, const int s, const int seed)
	{
		Hmatrix<int> out(r,s);

		for (int i=0; i<out.GetNumElements(); i++)
			out(i) = (int)(rand() & 0xFF);
		
		return out;
	}

	Hmatrix<float> Rand(const int r, const int s, const float seed)
	{
		Hmatrix<float> out(r,s);

		for (int i=0; i<out.GetNumElements(); i++)
			out(i) = (rand() & 0xFF) / (float)RAND_MAX;
		
		return out;
	}

	Hmatrix<double> Rand(const int r, const int s, const double seed)
	{
		Hmatrix<double> out(r,s);

		for (int i=0; i<out.GetNumElements(); i++)
			out(i) = (rand() & 0xFF) / (double)RAND_MAX;
		
		return out;
	}

}
#endif
