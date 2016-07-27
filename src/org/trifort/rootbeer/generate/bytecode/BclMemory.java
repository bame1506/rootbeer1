/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.bytecode;


import soot.Local      ;
import soot.Type       ;
import soot.Value      ;
import soot.LongType   ;
import soot.IntType    ;
import soot.VoidType   ;
import soot.ByteType   ;
import soot.BooleanType;
import soot.ShortType  ;
import soot.CharType   ;
import soot.FloatType  ;
import soot.DoubleType ;

import soot.jimple.IntConstant;
import soot.jimple.LongConstant;


/**
 * This class simplifies generating Jimple code for accesses to
 * runtime/Memory.java class. Could basically be generated using preprocessors
 * Uses BytecodeLanguage.pushMethod and .invokeMethodRet
 */
public class BclMemory
{
    private final BytecodeLanguage mBcl;
    private final Local mMem;

    public BclMemory( BytecodeLanguage bcl, Local mem )
    {
        mBcl = bcl;
        mMem = mem;
    }

    public void writeByte(byte value){ writeByte(IntConstant.v(value)); }

    /**
     * Same as @see callVoidMethod2, but takes takes no arguments
     */
    private void voidCall0( final String methodName )
    {
        mBcl.pushMethod( mBcl.getTypeString(mMem), methodName, VoidType.v() );
        mBcl.invokeMethodNoRet( mMem );
    }

    /**
     * Calls the method with name methodName in Memory.java with argument value
     * @param type e.g. ByteType. The value it hols is ignored
     */
    private void voidCall1
    (
        final String methodName,
        Type         type,
        Value        value
    )
    {
        mBcl.pushMethod(
            mBcl.getTypeString(mMem), methodName, VoidType.v(), type
        );
        mBcl.invokeMethodNoRet( mMem, value );
    }

    /* why do they all have different scope specifiers ? (no specifier
     * means package-private. */
            void pushAddress       (){ voidCall0( "pushAddress"        ); }
            void popAddress        (){ voidCall0( "popAddress"         ); }
    public  void useInstancePointer(){ voidCall0( "useInstancePointer" ); }
    public  void useStaticPointer  (){ voidCall0( "useStaticPointer"   ); }
            void startIntegerList  (){ voidCall0( "startIntegerList"   ); }
            void endIntegerList    (){ voidCall0( "endIntegerList"     ); }
            void finishReading     (){ voidCall0( "finishReading"      ); }
    public  void align             (){ voidCall0( "align"              ); }
    /* In the end all these calls will finally after n indirections
     * end up in their corresponding functions in FixedMemory.c */
    public  void writeByte       (Value v){ voidCall1( "writeByte   ",    ByteType.v(), v ); }
    public  void writeBoolean    (Value v){ voidCall1( "writeBoolean", BooleanType.v(), v ); }
            void writeShort      (Value v){ voidCall1( "writeShort  ",   ShortType.v(), v ); }
    private void writeChar       (Value v){ voidCall1( "writeChar   ",    CharType.v(), v ); }
            void writeInt        (Value v){ voidCall1( "writeInt    ",     IntType.v(), v ); }
            void writeFloat      (Value v){ voidCall1( "writeFloat  ",   FloatType.v(), v ); }
            void writeDouble     (Value v){ voidCall1( "writeDouble ",  DoubleType.v(), v ); }
            void writeLong       (Value v){ voidCall1( "writeLong   ",    LongType.v(), v ); }
            void incrementAddress(Value v){ voidCall1( "incrementAddress", IntType.v(), v ); }
            void setAddress      (Value v){ voidCall1( "setAddress"  ,    LongType.v(), v ); }
     public void writeRef        (Value v){ voidCall1( "writeRef"    ,    LongType.v(), v ); }

    public void mallocWithSize(Value size) {
      mBcl.pushMethod(mBcl.getTypeString(mMem), "mallocWithSize", LongType.v(), IntType.v());
      mBcl.invokeMethodNoRet(mMem, size);
    }

    void addIntegerToList(Local array_elements) {
      mBcl.pushMethod(mMem, "addIntegerToList", VoidType.v(), LongType.v());
      mBcl.invokeMethodNoRet(mMem, array_elements);
    }

    public void readIntArray(Value ret, Value size) {
      mBcl.pushMethod(mMem, "readIntArray", VoidType.v(), ret.getType(), size.getType());
      mBcl.invokeMethodNoRet(mMem, ret, size);
    }

    void writeInt(int size) { writeInt( IntConstant.v(size) ); }
    void incrementAddress(int size) { incrementAddress(IntConstant.v(size)); }

    void writeVar(Local curr)
    {
        final Type type = curr.getType();
        if      ( type instanceof ByteType    ) writeByte   (curr);
        else if ( type instanceof BooleanType ) writeBoolean(curr);
        else if ( type instanceof ShortType   ) writeShort  (curr);
        else if ( type instanceof CharType    ) writeChar   (curr);
        else if ( type instanceof IntType     ) writeInt    (curr);
        else if ( type instanceof FloatType   ) writeFloat  (curr);
        else if ( type instanceof DoubleType  ) writeDouble (curr);
        else if ( type instanceof LongType    ) writeLong   (curr);
    }

    private Local typedCall0
    (
        final String methodName,
        Type         type
    )
    {
        mBcl.pushMethod( mBcl.getTypeString(mMem), methodName, type );
        return mBcl.invokeMethodRet( mMem );
    }

    Local readByte   (){ return typedCall0( "readByte"   ,    ByteType.v() ); }
    Local readBoolean(){ return typedCall0( "readBoolean", BooleanType.v() ); }
    Local readShort  (){ return typedCall0( "readShort"  ,   ShortType.v() ); }
    Local readChar   (){ return typedCall0( "readChar"   ,    CharType.v() ); }
    Local readInt    (){ return typedCall0( "readInt"    ,     IntType.v() ); }
    Local readFloat  (){ return typedCall0( "readFloat"  ,   FloatType.v() ); }
    Local readDouble (){ return typedCall0( "readDouble" ,  DoubleType.v() ); }
    Local readLong   (){ return typedCall0( "readLong"   ,    LongType.v() ); }
    Local getPointer (){ return typedCall0( "getPointer" ,    LongType.v() ); }
    Local malloc     (){ return typedCall0( "malloc"     ,    LongType.v() ); }
    public Local readRef(){ return typedCall0( "readRef", LongType.v() ); }

    Local readVar( final Type type )
    {
        if      ( type instanceof ByteType    ) return readByte   ();
        else if ( type instanceof BooleanType ) return readBoolean();
        else if ( type instanceof ShortType   ) return readShort  ();
        else if ( type instanceof CharType    ) return readChar   ();
        else if ( type instanceof IntType     ) return readInt    ();
        else if ( type instanceof FloatType   ) return readFloat  ();
        else if ( type instanceof DoubleType  ) return readDouble ();
        else if ( type instanceof LongType    ) return readLong   ();
        else throw new RuntimeException("How do we handle this case?");
    }

}
