#ifndef ROOTBEER_STOPWATCH_H
#define ROOTBEER_STOPWATCH_H

#if defined linux || defined __linux || defined __APPLE_CC__ || defined __APPLE__ || defined __MACOSX__
#   include <sys/time.h>
#elif defined _WIN32 || defined __WIN32__
#   include <Windows.h>
#else
#   pragma error "Unknown Operating System!"
#endif

struct stopwatch
{
#if defined linux || defined __linux || defined __APPLE_CC__ || defined __APPLE__ || defined __MACOSX__
    struct timeval startTime;
    long long time;
#elif defined _WIN32 || defined __WIN32__
    long long startTime;
    long long stopTime;
#else
#   pragma error "Unknown Operating System!"
#endif
};

void      stopwatchStart ( struct stopwatch * watch );
void      stopwatchStop  ( struct stopwatch * watch );
long long stopwatchTimeMS( struct stopwatch * watch );

#endif
