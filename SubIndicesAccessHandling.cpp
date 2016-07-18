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

#include "SubIndicesAccessHandling.h"
#include <stdio.h>

/***************/
/* RANGE CLASS */
/***************/

struct BB::PrivateRange {
		int		start_;
		int		end_;
		int		numelements_; 
        };

BB::Range::Range(const int start, const int end) { privateRange = new BB::PrivateRange;
	(*privateRange).start_=start; (*privateRange).end_=end; (*privateRange).numelements_=end-start+1; }

int BB::Range::GetStart()	const			{ return (*privateRange).start_; };
int BB::Range::GetEnd() const				{ return (*privateRange).end_; };
int BB::Range::GetNumElements() const		{ return (*privateRange).numelements_; };
void BB::Range::SetStart(const int i)		{ (*privateRange).start_ = i; (*privateRange).numelements_=(*privateRange).end_-(*privateRange).start_+1; };
void BB::Range::SetEnd(const int i)			{ (*privateRange).end_ = i;   (*privateRange).numelements_=(*privateRange).end_-(*privateRange).start_+1; };

/*******************/
/* RANGESTEP CLASS */
/*******************/

struct BB::PrivateRangeStep {
		int		start_;
		int		end_;
		int		step_;
		int		numelements_; 
        };

BB::RangeStep::RangeStep(const int start, const int step, const int end) { 
	privateRangeStep = new BB::PrivateRangeStep;
	(*privateRangeStep).start_=start; (*privateRangeStep).end_=end; (*privateRangeStep).step_=step; (*privateRangeStep).numelements_=(end-start+1)/step+(((end-start+1)-step*(int)((end-start+1)/step))%step==0?0:1); }

int BB::RangeStep::GetStart()	const			{ return (*privateRangeStep).start_; };
int BB::RangeStep::GetStep() const			{ return (*privateRangeStep).step_; };
int BB::RangeStep::GetEnd() const				{ return (*privateRangeStep).end_; };
int BB::RangeStep::GetNumElements() const		{ return (*privateRangeStep).numelements_; };
void BB::RangeStep::SetStart(const int i)		{ (*privateRangeStep).start_ = i;	(*privateRangeStep).numelements_=((*privateRangeStep).end_-(*privateRangeStep).start_+1)/(*privateRangeStep).step_+((((*privateRangeStep).end_-(*privateRangeStep).start_+1)-(*privateRangeStep).step_*(int)(((*privateRangeStep).end_-(*privateRangeStep).start_+1)/(*privateRangeStep).step_))%(*privateRangeStep).step_==0?0:1); };
void BB::RangeStep::SetStep(const int i)		{ (*privateRangeStep).step_ = i;	(*privateRangeStep).numelements_=((*privateRangeStep).end_-(*privateRangeStep).start_+1)/(*privateRangeStep).step_+((((*privateRangeStep).end_-(*privateRangeStep).start_+1)-(*privateRangeStep).step_*(int)(((*privateRangeStep).end_-(*privateRangeStep).start_+1)/(*privateRangeStep).step_))%(*privateRangeStep).step_==0?0:1); };
void BB::RangeStep::SetEnd(const int i)		{ (*privateRangeStep).end_ = i;		(*privateRangeStep).numelements_=((*privateRangeStep).end_-(*privateRangeStep).start_+1)/(*privateRangeStep).step_+((((*privateRangeStep).end_-(*privateRangeStep).start_+1)-(*privateRangeStep).step_*(int)(((*privateRangeStep).end_-(*privateRangeStep).start_+1)/(*privateRangeStep).step_))%(*privateRangeStep).step_==0?0:1); };

