#pragma once

#include <assert.h>
#include <stddef.h>     // NULL
#include <stdint.h>     // uintptr_t


/**
 * They work asserts. I.e. if the condition given is untrue, then
 * a compile error will occur: "error: size of unnamed array is negative"
 * https://scaryreasoner.wordpress.com/2009/02/28/checking-sizeof-at-compile-time/
 */
#define COMPILER_ASSERT(condition)((void)sizeof(char[1 - 2*!(condition)]));

/**
 * Notes: In order to avoid -Wint-to-pointer-cast and -Wpointer-to-int-cast
 *        errors uintptr_t is to be used.
 */
inline void * jlongToPointer( jlong const p )
{
    COMPILER_ASSERT( sizeof(jlong) == 8 )
    COMPILER_ASSERT( sizeof(unsigned long long) == 8 )

    assert( (unsigned long long) p <= UINTPTR_MAX && "Got a non 32-bit pointer on a 32-bit system!!" );
    return (unsigned long long) p <= UINTPTR_MAX ? (void*) (uintptr_t) p : NULL;
}

inline jlong pointerToJlong( void const * const p )
{
    COMPILER_ASSERT( sizeof(jlong)     == sizeof(uint64_t) )
    COMPILER_ASSERT( sizeof(void *)    <= sizeof(uint64_t) )
    COMPILER_ASSERT( sizeof(uintptr_t) <= sizeof(jlong   ) )
    return (uintptr_t) p <= INT64_MAX ? (jlong) (uintptr_t) p : 0; /* ignore the -Wtype-limits warning thrown here */
}

#undef COMPILER_ASSERT
