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


#ifndef __MACROS_H__
#define __MACROS_H__

/*! \brief Macro to access a 2D array stored in row-major order. 
           i is the row index, j is the column index,  N is the number of columns. */
#ifndef IDX2R
#define IDX2R(i,j,N) (((i)*(N))+(j))
#endif

/*! \brief Returns the minimum between two numbers. */
#ifndef MIN
#define MIN(a,b) ((a < b) ? a : b)
#endif

/*! \brief Returns the maximum between two numbers. */
#ifndef MAX
#define MAX(a,b) ((a > b) ? a : b)
#endif

#endif