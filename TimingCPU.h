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


// 1 micro-second accuracy
// Returns the time in seconds

#ifndef __TIMINGCPU_H__
#define __TIMINGCPU_H__

namespace BB
{
	class TimingCPU
	{
		private:
			PrivateTimingCPU *privateTimingCPU;

		public:
			
			/*! \brief Default constructor. */ 
			TimingCPU();

			/*! \brief Default destructor. */
			~TimingCPU();

			/*! \brief Starts the timing counter. */
			void StartCounter();

			/*! \brief Stops the timing counter and returns the timing in [ms]. */
			double GetCounter();

	}; // TimingCPU class

} // namespace

#endif 

// 1 micro-second accuracy

//#ifndef __TIMINGCPU_H__
//#define __TIMINGCPU_H__
//
//namespace BB
//{
//	#include <windows.h>
//
//	class TimingCPU
//	{
//		private:
//			double	PCFreq;
//			__int64 CounterStart;
//
//		public:
//			
//			// default constructor
//			TimingCPU()
//			{
//				PCFreq = 0.0;
//				CounterStart = 0;
//			}
//
//			// default destructor
//			~TimingCPU() { }
//
//			void StartCounter()
//			{
//				LARGE_INTEGER li;
//				if(!QueryPerformanceFrequency(&li)) std::cout << "QueryPerformanceFrequency failed!\n";
//
//				PCFreq = double(li.QuadPart)/1000.0;
//
//				QueryPerformanceCounter(&li);
//				CounterStart = li.QuadPart;
//			}
//
//			// Gets the counter in ms
//			double GetCounter()
//			{
//				LARGE_INTEGER li;
//				QueryPerformanceCounter(&li);
//				return double(li.QuadPart-CounterStart)/PCFreq;
//			}
//
//	}; // TimingCPU class
//
//} // namespace
//
//#endif 

