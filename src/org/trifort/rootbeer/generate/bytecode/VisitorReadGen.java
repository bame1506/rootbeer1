/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.bytecode;


import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.trifort.rootbeer.generate.opencl.OpenCLScene;
import org.trifort.rootbeer.generate.opencl.OpenCLType;
import org.trifort.rootbeer.generate.opencl.fields.OpenCLField;
import org.trifort.rootbeer.util.Stack;

import soot.Local;
import soot.Scene;
import soot.SootClass;
import soot.VoidType;
import soot.BooleanType;
import soot.LongType;
import soot.RefLikeType;
import soot.ArrayType;
import soot.RefType;
import soot.IntType;
import soot.PrimType;
import soot.CharType;
import soot.Type;

import soot.jimple.IntConstant;
import soot.jimple.LongConstant;
import soot.jimple.NullConstant;
import soot.rbclassload.ClassHierarchy;
import soot.rbclassload.RootbeerClassLoader;


public class VisitorReadGen extends AbstractVisitorGen
{
    private static final boolean debugging = true;

    private final Stack<Local> m_currObj;

    private Local m_param0    ;
    private Local m_param1    ;
    private Local m_refParam  ;
    private Local m_mem       ;
    private Local m_textureMem;

    private final Map<Type, Local> m_readFromHeapMethodsMade    ;
    private final Map<Type, Local> m_ctorReadFromHeapMethodsMade;
    private final List<Type>       m_orderedHistory             ;
    private final Set<String>      m_visitedReader              ;

    public VisitorReadGen
    (
        final List<Type>       ordered_history,
        final String           class_name     , /* unused ??? */
        final BytecodeLanguage bcl
    )
    {
        /* parent no-arg constructor initializes m_bcl, m_gcObjVisitor,
         * m_objSerializing, m_currMem with empty Stacks */
        m_readFromHeapMethodsMade     = new HashMap<Type, Local>();
        m_ctorReadFromHeapMethodsMade = new HashMap<Type, Local>();
        m_orderedHistory              = ordered_history;
        m_visitedReader               = new HashSet<String>();
        m_currObj                     = new Stack<Local>();
        /* Inherited members from AbstractVisitorGen */
        m_bcl.push(bcl);
    }

    /**
     * doReadFromHeap( Object o, boolean readData, long ref )
     * {
     *     if ( ref != -1 )
     *     {
     *         System.out.println( "returning null" );
     *         return null;
     *     }
     *     Memory mem        = mMem;
     *     Memory textureMem = mTextureMem;
     *     mem.setAddress( ref ):
     *     mem.incrementAddress( 3 ):
     *     byte ctor_used    = mem.readByte(); // constructor used ???
     *     int  class_number = mem.readInt();
     *
     *     if ( class_number == RootbeerClassLoader.v().getClassHierarchy().
     *                          getNumberForType( "java.lang.String" ) )
     *         goto string_label;
     *     if ( ctor_used == 1 ) goto ctors_label;
     *     if ( o == null      ) goto ctors_label;
     *
     *     string_label:
     *         return /* makeReadFromHeapBodyForString * /
     *
     *
     * }
     *
     * Note that 'debugging' can be set to true in BytecodeLanguage.java
     * to write out the finished method at the call of
     * BytecodeLanguage.endMethod
     *
     * @todo It seems like 'goto' statements have the first instruction after
     *       after the label instead of the label!
     *
     * Example Output:
     *
     *   doReadFromHeap
     *   {
     *       this0 := @this: CountKernelSerializer
     *       parameter0 := @parameter0: java.lang.Object
     *       parameter1 := @parameter1: boolean
     *       parameter2 := @parameter2: long
     *
     *       // Memory mem        = mMem;
     *       // Memory textureMem = mTextureMem;
     *       rbreg110 = this0.<org.trifort.rootbeer.runtime.Serializer: org.trifort.rootbeer.runtime.Memory mMem>
     *       rbreg111 = this0.<org.trifort.rootbeer.runtime.Serializer: org.trifort.rootbeer.runtime.Memory mTextureMem>
     *
     *       if parameter2 != -1L  // I think this is null
     *          goto dont_return_null_label
     *
     *       rbreg112 = <java.lang.System: java.io.PrintStream out>
     *       virtualinvoke rbreg112.<java.io.PrintStream: void println(java.lang.String)>("returning null")
     *       return null
     *
     *   dont_return_null_label:
     *       rbreg113 = rbreg110
     *       interfaceinvoke rbreg113.<org.trifort.rootbeer.runtime.Memory: void setAddress(long)>(parameter2)
     *       interfaceinvoke rbreg113.<org.trifort.rootbeer.runtime.Memory: void incrementAddress(int)>(3)
     *       rbreg114 = interfaceinvoke rbreg113.<org.trifort.rootbeer.runtime.Memory: byte readByte()>()
     *       rbreg115 = interfaceinvoke rbreg113.<org.trifort.rootbeer.runtime.Memory: int readInt()>()
     *       if rbreg115 == 3224
     *           goto rblabel3
     *       if rbreg114 == 1
     *           goto (branch) ???
     *       if parameter0 == null
     *           goto (branch)
     *       goto [?= interfaceinvoke rbreg113.<org.trifort.rootbeer.runtime.Memory: void incrementAddress(int)>(8)] ???
     *
     *   rblabel3:
     *       interfaceinvoke rbreg113.<org.trifort.rootbeer.runtime.Memory: void incrementAddress(int)>(24)
     *       rbreg116 = interfaceinvoke rbreg113.<org.trifort.rootbeer.runtime.Memory: long readRef()>()
     *       interfaceinvoke rbreg113.<org.trifort.rootbeer.runtime.Memory: void setAddress(long)>(rbreg116)
     *       interfaceinvoke rbreg113.<org.trifort.rootbeer.runtime.Memory: void incrementAddress(int)>(12)
     *       rbreg117 = interfaceinvoke rbreg113.<org.trifort.rootbeer.runtime.Memory: int readInt()>()
     *       interfaceinvoke rbreg113.<org.trifort.rootbeer.runtime.Memory: void incrementAddress(int)>(16)
     *       rbreg118 = newarray (char)[rbreg117]
     *       rbreg119 = 0
     *
     *   rblabel6:
     *       if rbreg119 == rbreg117
     *           goto rblabel5
     *       rbreg120 = interfaceinvoke rbreg113.<org.trifort.rootbeer.runtime.Memory: char readChar()>()
     *       rbreg118[rbreg119] = rbreg120
     *       rbreg119 = rbreg119 + 1
     *       goto [?= (branch)]
     *
     *   rblabel5:
     *       rbreg121 = new java.lang.String
     *       specialinvoke rbreg121.<java.lang.String: void <init>(char[])>(rbreg118)
     *       rbreg122 = rbreg121
     *       interfaceinvoke rbreg113.<org.trifort.rootbeer.runtime.Memory: void finishReading()>()
     *       return rbreg122
     *
     *   [...]
     */
    public void makeReadFromHeapMethod()
    {
        final BytecodeLanguage bcl = m_bcl.top();
        final SootClass obj_cls = Scene.v().getSootClass( "java.lang.Object" );

        /* create `Object doReadFromHeap( Object o, boolean readData, long ref )`*/
        bcl.startMethod( "doReadFromHeap", obj_cls.getType(),
                         obj_cls.getType(), BooleanType.v(), LongType.v() );

        m_thisRef    = bcl.refThis();
        m_gcObjVisitor.push( m_thisRef );
        m_param0     = bcl.refParameter(0); // Object o
        m_objSerializing.push( m_param0 );
        m_param1     = bcl.refParameter(1); // boolean readData
        m_refParam   = bcl.refParameter(2); // long ref
        /* get member variables of Serializer.java class */
        m_mem        = bcl.refInstanceField( m_thisRef, "mMem"        );
        m_textureMem = bcl.refInstanceField( m_thisRef, "mTextureMem" );

        final String dont_return_null_label = getNextLabel(); // AbstractVisitorGen.getNextLabel
        bcl.ifStmt( m_refParam, "!=", LongConstant.v(-1), dont_return_null_label );
        bcl.println( "returning null" );
        bcl.returnValue( NullConstant.v() );
        bcl.label( dont_return_null_label );
        /* does this mean the label which was already used for the statement
         * box at the if-statement signals the end of that statement block ? */

        final Local mem = bcl.local( m_mem.getType() );
        bcl.assign( mem, m_mem );
        m_currMem.push( mem );
        final BclMemory bcl_mem = new BclMemory( bcl, mem );
        /* setAddress( m_refParam + 3 ) ist not easily possible using Soot,
         * that's why this is split into setAddress followed by incrementAddress
         * @todo Why increment by 3 bytes in the first place ? */
        bcl_mem.setAddress( m_refParam );
        bcl_mem.incrementAddress(3);
        final Local ctor_used    = bcl_mem.readByte();
        final Local class_number = bcl_mem.readInt();

        final long string_number = RootbeerClassLoader.v().getClassHierarchy().
                                   getNumberForType( "java.lang.String" );

        /* Create readers for String and char[] */

        final String after_ctors_label    = getNextLabel();
        final String ctors_label          = getNextLabel();
        final String string_label         = getNextLabel();
        final String increment_addr_label = getNextLabel();
        bcl.ifStmt( class_number, "==", IntConstant.v((int) string_number), string_label);
        bcl.ifStmt( ctor_used   , "==", IntConstant.v(1), ctors_label);
        bcl.ifStmt( m_param0    , "==", NullConstant.v(), ctors_label);
        bcl.gotoLabel(increment_addr_label);

        bcl.label(string_label);
        final RefType string_type = RefType.v( "java.lang.String" );
        final Local ret = makeReadFromHeapBodyForString( bcl, bcl_mem, string_type );
        m_readFromHeapMethodsMade.put( string_type, ret );
        bcl.returnValue(ret);

        /* @todo I don't understand why string_label comes before ctors_label ? */
        bcl.label(ctors_label);

        for ( final Type type : m_orderedHistory )
        {
            makeCtorReadFromHeapMethodForType(type, false, class_number, ctor_used, after_ctors_label);
        }

        RefType obj = RefType.v("java.lang.Object");
        makeCtorReadFromHeapMethodForType(obj, true, class_number, ctor_used, after_ctors_label);

        bcl.label(increment_addr_label);
        bcl_mem.incrementAddress(8);

        bcl.label(after_ctors_label);
        bcl_mem.incrementAddress(16);

        for ( Type type : m_orderedHistory )
            makeReadFromHeapMethodForType( type, false, ctor_used );

        makeReadFromHeapMethodForType( obj, true, ctor_used );

        bcl.returnValue( NullConstant.v() );
        bcl.endMethod();
    }

    private void makeReadFromHeapMethodForType
    (
        final Type    type        ,
        final boolean doing_object,
        final Local   ctor_used
    )
    {
        if ( type instanceof ArrayType == false &&
             type instanceof RefType   == false  )
        {
            return;
        }

        if ( m_readFromHeapMethodsMade.containsKey(type) )
            return;

        if ( type instanceof RefType )
        {
            RefType ref_type = (RefType) type;
            if ( ref_type.getClassName().equals("java.lang.Object") )
            {
                if ( ! doing_object )
                    return;
            }
            SootClass soot_class = ref_type.getSootClass();
            if ( soot_class.isInterface() )
                return;
            if ( differentPackageAndPrivate(ref_type) )
                return;
        }

        String label = getNextLabel();
        BytecodeLanguage bcl = m_bcl.top();

        bcl.ifInstanceOfStmt(m_param0, type, label);

        //bcl.println("reading: "+type.toString());
        //BclMemory bcl_mem = new BclMemory(bcl, m_mem);
        //Local ptr = bcl_mem.getPointer();
        //bcl.println(ptr);

        Local ret;
        if(type instanceof ArrayType){
            ret = makeReadFromHeapBodyForArrayType((ArrayType) type);
        }
        else {
            ret = makeReadFromHeapBodyForSootClass((RefType) type);
        }
        m_readFromHeapMethodsMade.put(type, ret);
        bcl.returnValue(ret);
        bcl.label(label);
    }

    private void makeCtorReadFromHeapMethodForType(Type type, boolean doing_object,
            Local class_number, Local ctor_used, String after_ctors_label)
    {
        if(type instanceof ArrayType == false &&
             type instanceof RefType == false){
            return;
        }

        if(m_ctorReadFromHeapMethodsMade.containsKey(type))
            return;

        if(type instanceof RefType){
            RefType ref_type = (RefType) type;
            if(ref_type.getClassName().equals("java.lang.Object")){
                if(!doing_object){
                    return;
                }
            }
            SootClass soot_class = ref_type.getSootClass();
            if(soot_class.isInterface()){
                return;
            }
            if(differentPackageAndPrivate(ref_type)){
                return;
            }
        }

        String label = getNextLabel();
        BytecodeLanguage bcl = m_bcl.top();

        ClassHierarchy class_hierarchy = RootbeerClassLoader.v().getClassHierarchy();
        long number = class_hierarchy.getNumberForType(type.toString());

        bcl.ifStmt(class_number, "!=", IntConstant.v((int) number), label);

        Local ret;
        if(type instanceof ArrayType){
            ret = makeCtorReadFromHeapBodyForArrayType((ArrayType) type, ctor_used);
            m_ctorReadFromHeapMethodsMade.put(type, ret);
            bcl.assign(m_param0, ret);
            bcl.gotoLabel(after_ctors_label);
        }
        else {
            ret = makeCtorReadFromHeapBodyForSootClass((RefType) type, ctor_used, class_number);
            m_ctorReadFromHeapMethodsMade.put(type, ret);
            bcl.assign(m_param0, ret);
            bcl.gotoLabel(after_ctors_label);
        }
        bcl.label(label);
    }

    private Local makeCtorReadFromHeapBodyForArrayType(ArrayType type,
            Local ctor_used) {

        BytecodeLanguage bcl = m_bcl.top();
        BclMemory bcl_mem = new BclMemory(bcl, m_currMem.top());
        bcl_mem.incrementAddress(4);

        Local array_length = bcl_mem.readInt();
        //bcl.println("reading size: ");
        //bcl.println(size);
        Local ret = bcl.local(type);

        bcl.assign(ret, bcl.newArray(type, array_length));

        return ret;
    }

    private Local makeReadFromHeapBodyForArrayType(ArrayType type) {
        BytecodeLanguage bcl = m_bcl.top();

        BclMemory bcl_mem = new BclMemory(bcl, m_currMem.top());

        Local ret = bcl.local(type);
        ret = bcl.cast(type, m_param0);
        Local size = bcl.lengthof(ret);

        //optimization for single-dimensional arrays of primitive types.
        //doesn't work for chars yet because they are still stored as ints on the gpu
        if(type.baseType instanceof PrimType && type.numDimensions == 1 &&
             type.baseType.equals(CharType.v()) == false){

            bcl.pushMethod(m_currMem.top(), "readArray", VoidType.v(), type);
            bcl.invokeMethodNoRet(m_currMem.top(), ret);
            OpenCLType ocl_type = new OpenCLType(type.baseType);
            Local element_size = bcl.local(IntType.v());
            bcl.assign(element_size, IntConstant.v(ocl_type.getSize()));
            bcl.mult(element_size, size);
            bcl_mem.incrementAddress(element_size);

            return ret;
        }

        Local i = bcl.local(IntType.v());
        bcl.assign(i, IntConstant.v(0));

        String end_for_label = getNextLabel();
        String before_if_label = getNextLabel();
        bcl.label(before_if_label);
        bcl.ifStmt(i, "==", size, end_for_label);

        Local new_curr;

        if(type.numDimensions != 1){
            new_curr = readFromHeapArray(ret, i, size);
        } else if(type.baseType instanceof RefType){
            Local temp = readFromHeapArray(ret, i, size);
            new_curr = bcl.cast(type.baseType, temp);
        } else {
            new_curr = bcl_mem.readVar(type.baseType);
        }

        bcl.assignElementToArray(ret, new_curr, i);

        bcl.plus(i, 1);
        bcl.gotoLabel(before_if_label);
        bcl.label(end_for_label);

        bcl_mem.finishReading();

        return ret;
    }

    private Local makeReadFromHeapBodyForString
    (
        final BytecodeLanguage bcl    ,
        final BclMemory        bcl_mem,
        final RefType          type
    )
    {
        bcl_mem.incrementAddress(4+4+16);
        Local ref = bcl_mem.readRef();

        bcl_mem.setAddress(ref);
        bcl_mem.incrementAddress(12);

        Local array_length = bcl_mem.readInt();

        bcl_mem.incrementAddress(16);

        ArrayType array_type = ArrayType.v(CharType.v(), 1);

        Local ret = bcl.local(array_type);
        bcl.assign(ret, bcl.newArray(array_type, array_length));

        Local i = bcl.local(IntType.v());
        bcl.assign(i, IntConstant.v(0));

        String end_for_label = getNextLabel();
        String before_if_label = getNextLabel();
        bcl.label(before_if_label);
        bcl.ifStmt(i, "==", array_length, end_for_label);

        Local new_curr = bcl_mem.readChar();
        bcl.assignElementToArray(ret, new_curr, i);

        bcl.plus(i, 1);
        bcl.gotoLabel(before_if_label);
        bcl.label(end_for_label);

        final SootClass string_class = Scene.v().getSootClass( "java.lang.String" );
        Local new_string = bcl.newInstance( string_class.getName(), ret );
        bcl_mem.finishReading();

        m_ctorReadFromHeapMethodsMade.put( type, new_string );
        m_readFromHeapMethodsMade.    put( type, new_string );

        return new_string;
    }

    private Local makeCtorReadFromHeapBodyForSootClass(RefType type,
            Local ctor_used, Local class_number)
    {
        BytecodeLanguage bcl = m_bcl.top();
        BclMemory bcl_mem = new BclMemory(bcl, m_currMem.top());
        bcl_mem.incrementAddress(8);

        SootClass soot_class = type.getSootClass();

        String name = soot_class.getName();
        if(soot_class.isApplicationClass() == false){
            if(soot_class.declaresMethod("void <init>()")){
                Local object_to_write_to = bcl.local(type);
                Local new_object = bcl.newInstance(name);
                bcl.assign(object_to_write_to, new_object);
                return object_to_write_to;
            } else {
                JavaNumberTypes number_types = new JavaNumberTypes();
                String type_string = type.toString();
                if(number_types.get().contains(type_string)){

                    bcl_mem.incrementAddress(Constants.MallocAlignBytes);

                    Local value;

                    if(type_string.equals("java.lang.Byte")){
                        value = bcl_mem.readByte();
                    } else if(type_string.equals("java.lang.Boolean")){
                        value = bcl_mem.readBoolean();
                    } else if(type_string.equals("java.lang.Character")){
                        value = bcl_mem.readChar();
                    } else if(type_string.equals("java.lang.Short")){
                        value = bcl_mem.readShort();
                    } else if(type_string.equals("java.lang.Integer")){
                        value = bcl_mem.readInt();
                    } else if(type_string.equals("java.lang.Long")){
                        value = bcl_mem.readLong();
                    } else if(type_string.equals("java.lang.Float")){
                        value = bcl_mem.readFloat();
                    } else if(type_string.equals("java.lang.Double")){
                        value = bcl_mem.readDouble();
                    } else {
                        throw new UnsupportedOperationException("cannot create type: "+type_string);
                    }

                    Local object_to_write_to = bcl.local(type);
                    Local new_object = bcl.newInstance(name, value);
                    bcl.assign(object_to_write_to, new_object);

                    bcl_mem.finishReading();

                    bcl.returnValue(object_to_write_to);
                    return object_to_write_to;
                } else {
                    Local object_to_write_to = bcl.local(type);
                    bcl.assign(object_to_write_to, NullConstant.v());
                    return object_to_write_to;
                }
            }
        } else {
            Local object_to_write_to = bcl.local(type);
            Local sentinal = bcl.newInstance("org.trifort.rootbeer.runtime.Sentinal");
            Local new_object = bcl.newInstance(name, sentinal);
            bcl.assign(object_to_write_to, new_object);
            return object_to_write_to;
        }
    }

    private Local makeReadFromHeapBodyForSootClass(RefType type){

        BytecodeLanguage bcl = m_bcl.top();
        SootClass soot_class = type.getSootClass();

        Local object_to_write_to = bcl.cast(type, m_param0);
        BclMemory bcl_mem = new BclMemory(bcl, m_currMem.top());

        SootClass obj_class = Scene.v().getSootClass("java.lang.Object");
        bcl.pushMethod(m_thisRef, "checkCache",obj_class.getType(), LongType.v(), obj_class.getType());
        object_to_write_to = bcl.invokeMethodRet(m_thisRef, m_refParam, object_to_write_to);
        object_to_write_to = bcl.cast(type, object_to_write_to);

        m_currObj.push(object_to_write_to);
        m_objSerializing.push(object_to_write_to);
        readFields(soot_class, true);
        readFields(soot_class, false);
        m_currObj.pop();
        m_objSerializing.pop();

        bcl_mem.finishReading();

        return object_to_write_to;
    }

    private void readFields(SootClass curr_class, boolean ref_types){
        if(curr_class.isApplicationClass()){
            attachReader(curr_class.getName(), ref_types);
            callBaseClassReader(curr_class.getName(), ref_types);
        } else {
            insertReader(curr_class.getName(), ref_types);
        }
    }

    private Local readFromHeapArray(Local object_to_read_from, Local i, Local size) {
        BytecodeLanguage bcl = m_bcl.top();
        BclMemory bcl_mem = new BclMemory(bcl, m_currMem.top());

        Local curr;

        String after_read = getNextLabel();
        String before_read_int = getNextLabel();

        bcl.ifStmt(i, ">=", size, after_read);
        curr = bcl.indexArray(object_to_read_from, i);
        bcl.gotoLabel(before_read_int);
        bcl.label(after_read);
        if(object_to_read_from.getType() instanceof RefLikeType){
            bcl.assign(curr, NullConstant.v());
        } else {
            bcl.assign(curr, IntConstant.v(0));
        }
        bcl.label(before_read_int);
        Local curr_phi = bcl.local(object_to_read_from.getType());
        bcl.assign(curr_phi, curr);
        Local object_addr = bcl_mem.readRef();
        bcl_mem.pushAddress();
        SootClass obj_cls = Scene.v().getSootClass("java.lang.Object");
        bcl.pushMethod(m_thisRef, "readFromHeap", obj_cls.getType(), obj_cls.getType(), BooleanType.v(), LongType.v());
        Local ret = bcl.invokeMethodRet(m_thisRef, curr_phi, m_param1, object_addr);
        bcl_mem.popAddress();

        return ret;
    }

    public void attachReader( final String class_name, final boolean ref_fields )
    {
        String specialization = ref_fields ? "RefFields" : "NonRefFields";
        specialization += JavaNameToOpenCL.convert(class_name);
        specialization += OpenCLScene.v().getIdent();
        String visited_name = class_name + specialization;

        if(m_visitedReader.contains(visited_name))
            return;
        m_visitedReader.add(visited_name);

        SootClass curr_class = Scene.v().getSootClass(class_name);
        SootClass parent = curr_class.getSuperclass();
        parent = Scene.v().getSootClass(parent.getName());
        if(parent.isApplicationClass()){
            attachReader(parent.getName(), ref_fields);
        }

        BytecodeLanguage bcl = new BytecodeLanguage();
        Local gc_obj_visit = m_gcObjVisitor.top();
        m_bcl.push(bcl);
        bcl.openClass(class_name);
        SootClass mem = Scene.v().getSootClass("org.trifort.rootbeer.runtime.Memory");
        /* this is where that weird ClassCastException occurs, i.e. for
         * specialization == RefFields */
        bcl.startMethod( "org_trifort_readFromHeap" + specialization,
            VoidType.v() /*return*/, mem.getType(), gc_obj_visit.getType() );
        m_objSerializing.push( bcl.refThis() );
        m_currMem.push( bcl.refParameter(0) );
        m_gcObjVisitor.push( bcl.refParameter(1) );

        doReader( class_name, ref_fields );

        if ( ! parent.getName().equals( "java.lang.Object" ) )
        {
            if ( parent.isApplicationClass() )
                callBaseClassReader( parent.getName(), ref_fields );
            else
                insertReader(parent.getName(), ref_fields);
        }

        bcl.returnVoid();
        bcl.endMethod();

        m_objSerializing.pop();
        m_currMem.pop();
        m_gcObjVisitor.pop();
        m_bcl.pop();
    }

    public void insertReader(String class_name, boolean ref_fields)
    {
        doReader(class_name, ref_fields);

        SootClass curr_class = Scene.v().getSootClass(class_name);
        if(curr_class.hasSuperclass() == false)
            return;

        SootClass parent = curr_class.getSuperclass();
        parent = Scene.v().getSootClass(parent.getName());
        if(parent.getName().equals("java.lang.Object") == false){
            if(parent.isApplicationClass()){
                attachReader(parent.getName(), ref_fields);
                callBaseClassReader(parent.getName(), ref_fields);
            } else {
                insertReader(parent.getName(), ref_fields);
            }
        }
    }

    public void doReader( final String class_name, final boolean do_ref_fields )
    {
        final BytecodeLanguage bcl = m_bcl.top();
        final BclMemory bcl_mem    = new BclMemory( bcl, m_currMem.top() );
        final SootClass soot_class = Scene.v().getSootClass( class_name );

        //read all the ref fields
        int inc_size = 0;
        if ( do_ref_fields )
        {
            List<OpenCLField> ref_fields = getRefFields(soot_class);
            for ( final OpenCLField ref_field : ref_fields )
            {
                //increment the address to get to this location
                bcl_mem.incrementAddress( inc_size );
                inc_size = 0;
                readRefField( bcl, m_gcObjVisitor.top(), m_currMem.top(), m_objSerializing.top(), ref_field );
            }

            if ( inc_size > 0 )
                bcl_mem.incrementAddress(inc_size);
        }
        else
        {
            List<OpenCLField> non_ref_fields = getNonRefFields(soot_class);
            for ( OpenCLField non_ref_field : non_ref_fields )
            {
                //increment the address to get to this location
                if ( inc_size > 0 )
                {
                    bcl_mem.incrementAddress(inc_size);
                    inc_size = 0;
                }
                //read the field
                readNonRefField(non_ref_field);
            }
            if ( inc_size > 0 )
                bcl_mem.incrementAddress(inc_size);
        }
        bcl_mem.align();
    }

    public void callBaseClassReader( String class_name, boolean ref_types )
    {
         String specialization = ref_types ? "RefFields" : "NonRefFields";
        specialization += JavaNameToOpenCL.convert(class_name);
        specialization += OpenCLScene.v().getIdent();
        final SootClass        mem = Scene.v().getSootClass( "org.trifort.rootbeer.runtime.Memory" );
        final BytecodeLanguage bcl = m_bcl.top();
        final Local   gc_obj_visit = m_gcObjVisitor.top();
        /* this is where that weird ClassCastException occurs, i.e. for
         * specialization == RefFields */
        bcl.pushMethod( class_name, "org_trifort_readFromHeap" + specialization, VoidType.v(), mem.getType(), gc_obj_visit.getType() );
        bcl.invokeMethodNoRet( m_objSerializing.top(), m_currMem.top(), gc_obj_visit );
    }

}
