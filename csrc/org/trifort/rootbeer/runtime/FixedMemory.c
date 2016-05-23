#include "FixedMemory.h"
#include <cuda.h>
#include <stdlib.h>     // calloc, free
#include <assert.h>

#define __FIXED_MEMORY_WRAPPER( NAME, JTYPE, TYPE )                     \
                                                                        \
JNIEXPORT JTYPE JNICALL                                                 \
Java_org_trifort_rootbeer_runtime_FixedMemory_doRead##NAME              \
(                                                                       \
    JNIEnv * env     ,                                                  \
    jobject  this_obj,                                                  \
    jlong    ptr     ,                                                  \
    jlong    cpu_base                                                   \
)                                                                       \
{                                                                       \
    assert( ptr % sizeof( TYPE ) == 0 );                                \
    assert( sizeof(JTYPE) == sizeof(TYPE) );                            \
    return ( (TYPE*) cpu_base )[ ptr / sizeof(TYPE) ];                  \
    /* first go ptr bytes onward, starting from cpu_base, then cast     \
     * to wanted type pointer and read / write to it. jlong is          \
     * basically tp be handled like uintptr_t */                        \
    /*return ( (TYPE*)( cpu_base + ptr ) )[0]; */                           \
}                                                                       \
                                                                        \
JNIEXPORT void JNICALL                                                  \
Java_org_trifort_rootbeer_runtime_FixedMemory_doWrite##NAME             \
(                                                                       \
    JNIEnv * env     ,                                                  \
    jobject  this_obj,                                                  \
    jlong    ptr     ,                                                  \
    JTYPE    value   ,                                                  \
    jlong    cpu_base                                                   \
)                                                                       \
{                                                                       \
    /* This fails !!! !!! There are two reasons: ptr should be aligned, meaning we have to adjust the call somewhere. Or we need to implement non-aligned write, see uncommented line down below */ \
    /* assert( ptr % sizeof( TYPE ) == 0 ); */                          \
    ( (TYPE*) cpu_base )[ ptr / sizeof(TYPE) ] = value;                 \
    /* ( (TYPE*) ( cpu_base + ptr ) )[0] = value; */                          \
}                                                                       \
                                                                        \
JNIEXPORT void JNICALL                                                  \
Java_org_trifort_rootbeer_runtime_FixedMemory_doRead##NAME##Array       \
(                                                                       \
    JNIEnv *     env     ,                                              \
    jobject      this_obj,                                              \
    JTYPE##Array array   ,                                              \
    jlong        ref     ,                                              \
    jint         start   ,                                              \
    jint         len                                                    \
)                                                                       \
{                                                                       \
    JTYPE* dest = (JTYPE *)( ref + start );                             \
    (*env)->Set##NAME##ArrayRegion( env, array, start, len, dest );     \
}                                                                       \
                                                                        \
JNIEXPORT void JNICALL                                                  \
Java_org_trifort_rootbeer_runtime_FixedMemory_doWrite##NAME##Array      \
(                                                                       \
    JNIEnv *     env     ,                                              \
    jobject      this_obj,                                              \
    JTYPE##Array array   ,                                              \
    jlong        ref     ,                                              \
    jint         start   ,                                              \
    jint         len                                                    \
)                                                                       \
{                                                                       \
    JTYPE * dest = (JTYPE *)( ref + start );                            \
    (*env)->Get##NAME##ArrayRegion( env, array, start, len, dest );     \
}                                                                       \

__FIXED_MEMORY_WRAPPER( Byte   , jbyte   , char   )  /* Signature: ()B */
__FIXED_MEMORY_WRAPPER( Boolean, jboolean, char   )  /* Signature: ()Z */
__FIXED_MEMORY_WRAPPER( Short  , jshort  , short  )  /* Signature: ()S */
__FIXED_MEMORY_WRAPPER( Int    , jint    , int    )  /* Signature: ()I */
__FIXED_MEMORY_WRAPPER( Float  , jfloat  , float  )  /* Signature: ()F */
__FIXED_MEMORY_WRAPPER( Double , jdouble , double )  /* Signature: ()D */
__FIXED_MEMORY_WRAPPER( Long   , jlong   , jlong  )  /* Signature: ()J */
/* The doWrite variants have signature e.g. (B)V instead of ()B */

#undef __FIXED_MEMORY_WRAPPER

JNIEXPORT jlong JNICALL Java_org_trifort_rootbeer_runtime_FixedMemory_malloc
( JNIEnv *env, jobject this_obj, jlong size )
{
    return (jlong) calloc(size, 1);
}

JNIEXPORT void JNICALL Java_org_trifort_rootbeer_runtime_FixedMemory_free
( JNIEnv *env, jobject this_obj, jlong address )
{
    free( (void *) address );
}
