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

#ifndef __P2P_CUH__
#define __P2P_CUH__

namespace BB
{
	class P2P
	{
		private:

			// checks if the OS is 64-bit and the build is for a 64-bit target
			inline bool IsAppBuiltAs64()
			{
				#if defined(__x86_64) || defined(AMD64) || defined(_M_AMD64)
				return 1;
				#else
				return 0;
				#endif
			}

		public:
			bool has_uva[MAX_NUM_GPUs];
			int active_GPU;
			int gpu_n;
			cudaDeviceProp prop[MAX_NUM_GPUs];
			int gpu_count;						// Number of GPUs supporting P2P
			int gpuid[MAX_NUM_GPUs];			// IDs of GPUs supporting P2P
			int can_access_peer[MAX_NUM_GPUs][MAX_NUM_GPUs];
			
			P2P() { }
	
			void InitP2P() {
				if (!IsAppBuiltAs64()) {
					cout << "Peer-to-peer is only supported with on 64-bit OSs and the application must be built as a 64-bit target. Exiting...\n";
					getch();
					exit(EXIT_SUCCESS); }

				CudaSafeCall(cudaGetDeviceCount(&gpu_n));

				if (gpu_n < 2)
				{
					cout << "Two or more GPUs with compute capability 2.0 or higher are required for peer-to-peer\n";
					getch();
					exit(EXIT_SUCCESS);
				}

				gpu_count = 0;
				for (int i=0; i < gpu_n; i++)
				{
					CudaSafeCall(cudaGetDeviceProperties(&prop[i],i));

//					// Only boards based on Fermi can support P2P. 
					if ((prop[i].major >= 2)
#ifdef _WIN32
					// On Windows (64-bit), the Tesla Compute Cluster driver for windows must be enabled
											&& prop[i].tccDriver
#endif
																)
						// This is an array of P2P capable GPUs
						gpuid[gpu_count++] = i;
				}

				// Check for TCC for Windows
				if (gpu_count < 2)
				{
					printf("\nTwo GPUs with compute capability 2.0 or higher are needed to use P2P/UVA functionality.\n");
#ifdef _WIN32
					printf("\nFor Windows Vista/Win7, a TCC driver must be installed and enabled to use P2P/UVA functionality.\n");
#endif
					exit(EXIT_SUCCESS);
				}

#if CUDART_VERSION >= 4000
				for (int i=0; i < gpu_count; i++) {
					has_uva[i] = prop[gpuid[0]].unifiedAddressing;
					for (int j=0; j < gpu_count; j++) {
						can_access_peer[i][j] = 0;
						if (i!=j) {
							CudaSafeCall(cudaDeviceCanAccessPeer(&can_access_peer[i][j], gpuid[i], gpuid[j]));
							if (can_access_peer[i][j]) {
								CudaSafeCall(cudaSetDevice(gpuid[i]));
								CudaSafeCall(cudaDeviceEnablePeerAccess(gpuid[j],0));
							}	
						}
					}
				}
#else // Using CUDA 3.2 or older
				printf("P2P requires CUDA 4.0 to build and run.\n");
				exit(EXIT_SUCCESS);
#endif
			}

			void ResetGPUs() { 
				for (int i=0; i<gpu_n; i++)
				{
					CudaSafeCall(cudaSetDevice(i));
					cudaDeviceReset();
				}

			}

			int GetNumGPUs() { return gpu_n; }

			cudaDeviceProp* GetDeviceProperties() { return &prop[0]; }

			int* GetP2PDeviceIDs() { return &gpuid[0]; }

			int GetNumP2PGPUs() { return gpu_count; }

			int* GetP2PCanAccessPeer() { return &can_access_peer[0][0]; }

			void SetDevice(int GPUID) { active_GPU = GPUID; CudaSafeCall(cudaSetDevice(GPUID)); }
	}; 

} 

#endif 
