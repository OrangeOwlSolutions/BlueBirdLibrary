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


#ifndef __CONSTANTS_H__
#define __CONSTANTS_H__

#include "BB.h"

/*! \brief Host matrix allocation is non-pinned. */
#define NONPINNED 0
/*! \brief Host matrix allocation is pinned. */
#define PINNED 1

/*! \brief Expression is a host expression. */
#define ISHOST 0
/*! \brief Expression is a device expression. */
#define ISDEVICE 1
/*! \brief Expression is neither a host nor a device expression. */
#define DEVICEAGNOSTIC 3

/*! \brief Greek pi in single precision. */
#define pi_f  3.14159265358979f
/*! \brief Greek pi in double precision. */
#define pi_d  3.141592653589793238463

/*! \brief Precision used to save .txt files. */
#define prec_save 10			// Precision in saving txt files

/*! \brief Precision used to cout matrices. */
#define prec_cout 10			// Precision in couting matrices

/*! \brief Maximum number of available GPUs. */
#define MAX_NUM_GPUs 64			

namespace BB {
	/*! \brief Imaginary unit in single precision. */
	const float2_ J_f(0.,1.);
	/*! \brief Imaginary unit in double precision. */
	const double2_ J_d(0.,1.);
	/*! \brief Default data type. */
	typedef double2_ DefaultType;
}

/*! \brief Block size for the kernel calls. */
#define BLOCKSIZE 512

/*! \brief Shared memory size for the kernel calls. */
#define SHAREDSIZE 512

#endif