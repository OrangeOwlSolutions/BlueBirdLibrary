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

#ifndef __DSTREAMS_CUH__
#define __DSTREAMS_CUH__

namespace BB
{
	class DStreams
	{
		private:
			PrivateStreams *privateStreams;

		public:
			
			/*! \brief Default constructor. */
 			DStreams();
	
			/*! \brief Sets the number of streams and initializes them. 
			 *
			 *	Example:
			 *
			 *      InitStreams(3);
			 *
			 *		defines and initializes 3 streams. 
			 *
			 *      InitStreams();
			 *
			 *		defines and initializes 1 stream. 
			*/
			void InitStreams(const int nStreams = 1);

			/*! \brief Destroys the current streams, sets the number of streams to 1 and initializes it. */
			void DestroyStreams();

			/*! \brief Sets the active stream.
			 *
			 *	Example:
			 *
			 *      SetStreams(3);
			 *
			 *		sets stream #3 as the active stream.
			*/
			void SetStream(const int i);

			/*! \brief Synchronizes all the streams. */
			void SynchronizeAll();

			/*! \brief Synchronizes the i-th stream. 
			 *
			 *	Example:
			 *
			 *      Synchronize(3);
			 *
			 *		synchronizes stream #3.
			 *
			 *      Synchronize();
			 *   
			 *      synchronizes stream #0.
			*/
			void Synchronize(const int i = 0);

			/*! \brief Default destructor. */
			~DStreams();

			/*! \brief Returns the number of active streams. */
			int GetNumStreams()	const;

			/*! \brief Returns the active stream. */
			cudaStream_t GetActiveStream() const;

	}; 

} 

#endif 
