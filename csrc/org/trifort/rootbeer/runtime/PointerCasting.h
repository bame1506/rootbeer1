#pragma once

#include <jni.h>

void * jlongToPointer( jlong const p );
jlong pointerToJlong( void const * const p );
