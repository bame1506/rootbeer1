#include "Stopwatch.h"

void stopwatchStart( struct stopwatch * watch )
{
    #if defined linux || defined __linux || defined __APPLE_CC__ || defined __APPLE__ || defined __MACOSX__
        gettimeofday(&(watch->startTime), 0);
    #elif defined _WIN32 || defined __WIN32__
        QueryPerformanceCounter((LARGE_INTEGER*)&(watch->startTime));
    #else
    #   pragma error "Unknown Operating System!"
    #endif
}

void stopwatchStop( struct stopwatch * watch )
{
    #if defined linux || defined __linux || defined __APPLE_CC__ || defined __APPLE__ || defined __MACOSX__
        struct timeval endTime;
        long seconds, useconds;

        gettimeofday(&endTime, 0);

        seconds  = endTime.tv_sec  - (watch->startTime).tv_sec;
        useconds = endTime.tv_usec - (watch->startTime).tv_usec;

        watch->time = (seconds * 1000) + (useconds / 1000);
    #elif defined _WIN32 || defined __WIN32__
        QueryPerformanceCounter((LARGE_INTEGER*)&(watch->stopTime));
    #else
    #   pragma error "Unknown Operating System!"
    #endif
}

/**
 * Calculates measured time between start and stop event in milliseconds
 *
 * This is used to set the member variables in StatsRow.java
 */
long long stopwatchTimeMS ( struct stopwatch * watch )
{
    #if defined linux || defined __linux || defined __APPLE_CC__ || defined __APPLE__ || defined __MACOSX__
        return watch->time;
    #elif defined _WIN32 || defined __WIN32__
        long long freq;
        long long d;
        QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
        d = watch->stopTime - watch->startTime;
        return (d * 1000UL) / freq;
    #else
    #   pragma error "Unknown Operating System!"
    #endif
}
