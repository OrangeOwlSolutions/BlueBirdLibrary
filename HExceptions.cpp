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

#include <exception>
#include <stdio.h>

#define Error_msg_1 "Error in file"
#define Double_new_line "\n\n"
#define Error_msg_2 "on line"

using namespace std;
using namespace BB;

struct BB::PrivateLibraryException {
	const char *message_;
	const char *file_;
	int line_;
};

LibraryException::LibraryException() {}
LibraryException::LibraryException(const char *message, const char* file, int line) {
	(*privateLibraryException).message_	= message;
	(*privateLibraryException).file_		= file;
	(*privateLibraryException).line_		= line;
}

const char* LibraryException::what() const throw() 
{
	char buf[20];
	sprintf(buf, "%d", (*privateLibraryException).line_);

	char catString[2048];
	sprintf(catString, "%s \n\n%s\n\n%s %s\n\n%s", Error_msg_1,(*privateLibraryException).file_,Error_msg_2,buf,(*privateLibraryException).message_);
	return &catString[0]; 
}

GenericError::GenericError(const char *message, const char* file, int line) { LibraryException except(message,file,line); };
