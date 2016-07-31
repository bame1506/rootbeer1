/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.bytecode;


import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.trifort.rootbeer.generate.opencl.OpenCLClass;
import org.trifort.rootbeer.generate.opencl.OpenCLScene;
import org.trifort.rootbeer.generate.opencl.OpenCLType;

import soot.Scene;
import soot.SootClass;
import soot.IntType;
import soot.VoidType;
import soot.RefType;
import soot.ArrayType;
import soot.Type;
import soot.Local;
import soot.jimple.IntConstant;
import soot.jimple.NullConstant;
import soot.rbclassload.NumberedType;
import soot.rbclassload.RootbeerClassLoader;
import soot.rbclassload.StringToType;


public final class VisitorGen extends AbstractVisitorGen
{
    private       String      m_className           ;
            final SootClass   m_runtimeBasicBlock   ;
    private final Set<Type>   m_getSizeMethodsMade  ;
    private final Set<String> m_sentinalCtorsCreated;

    //Locals from code generation
    private Local m_param0;

    public VisitorGen( SootClass runtime_basic_block )
    {
        m_runtimeBasicBlock    = runtime_basic_block;
        m_getSizeMethodsMade   = new HashSet<Type>();
        m_sentinalCtorsCreated = new HashSet<String>();
    }

    public void generate()
    {
        m_bcl.push( new BytecodeLanguage() );
        makeSentinalCtors( m_sentinalCtorsCreated );
        makeSerializer();
        addGetSerializerMethod( m_bcl.peek(), m_runtimeBasicBlock, m_className );
    }

    private void makeSerializer()
    {
        makeGcObjectClass();
        makeCtor();
        makeWriteStaticsToHeapMethod ( m_bcl.peek()              );
        makeReadStaticsFromHeapMethod( m_bcl.peek()              );
        makeGetSizeMethod            ( m_bcl.peek()              );
        makeGetLengthMethod          ( m_bcl.peek()              );
        makeWriteToHeapMethod        ( m_bcl.peek(), m_className );
        makeReadFromHeapMethod       ( m_bcl.peek(), m_className );
    }

    private void makeGcObjectClass()
    {
        String base_name = m_runtimeBasicBlock.getName();
        m_className = base_name+"Serializer";
        m_bcl.peek().makeClass(m_className, "org.trifort.rootbeer.runtime.Serializer");
    }

    /**
     * Make Method returning size for a given object
     * int doGetSize( Object o )
     * {
     *     if ( type instanceof short[] )
     * }
     */
    private void makeGetLengthMethod( final BytecodeLanguage bcl )
    {
        SootClass object_soot_class = Scene.v().getSootClass( "java.lang.Object" );
        bcl.startMethod( "doGetSize", IntType.v(), object_soot_class.getType() );
        m_thisRef = bcl.refThis();
        m_param0 = bcl.refParameter(0);

        List<Type> types = RootbeerClassLoader.v().getDfsInfo().getOrderedRefLikeTypes();
        for ( Type type : types )
        {
            if ( ! ( type instanceof ArrayType ) &&
                 ! ( type instanceof RefType   )   )
            {
                continue;
            }

            if ( m_getSizeMethodsMade.contains( type ) )
                continue;
            m_getSizeMethodsMade.add( type );

            /* Ignore size methods for reference to objects, interfaces and
             * private types */
            if ( type instanceof RefType )
            {
                final RefType ref_type = (RefType) type;
                final SootClass soot_class = ref_type.getSootClass();
                if ( soot_class.getName().equals("java.lang.Object") )
                    continue;
                if ( soot_class.isInterface() )
                    continue;
                if ( differentPackageAndPrivate( m_thisRef, ref_type ) )
                    continue;
            }
            if ( ! typeIsPublic( type ) )
                continue;

            final String label = getNextLabel(); // non-static
            /* if argument object is not of this type, skip the next code
             * code block (and test next type) */
            bcl.ifInstanceOfStmt( m_param0, type, label );

            if ( type instanceof ArrayType )
            {
                final ArrayType atype = (ArrayType) type;
                final Local size = bcl.local( IntType.v() );
                bcl.assign(size, IntConstant.v(Constants.ArrayOffsetSize));
                Local element_size = bcl.local(IntType.v());
                OpenCLType ocl_type = new OpenCLType(atype.baseType);
                if ( atype.numDimensions == 1 )
                    bcl.assign(element_size, IntConstant.v(ocl_type.getSize()));
                else
                    bcl.assign(element_size, IntConstant.v(4));
                Local object_to_write_from = bcl.cast(type, m_param0);
                Local length = bcl.lengthof(object_to_write_from);
                bcl.mult(element_size, length);
                bcl.plus(size, element_size);
                bcl.returnValue(size);
            }
            /* hardcode return size of type given by Soot
             * @todo asser that they are the same as C sizeof types ??? */
            else if ( type instanceof RefType )
            {
                final RefType rtype = (RefType) type;
                bcl.returnValue( IntConstant.v( OpenCLScene.v().
                        getOpenCLClass( rtype.getSootClass() ).getSize() ) );
            }
            bcl.label( label ); // set label (skip to here)
        }
        /* default return type 0. I don't think this should ever happen!
         * Better somehow exception or assert? */
        bcl.returnValue( IntConstant.v(0) );
        bcl.endMethod();
    }

    private void makeGetSizeMethod( final BytecodeLanguage bcl )
    {
        SootClass object_soot_class = Scene.v().getSootClass("java.lang.Object");
        bcl.startMethod("getArrayLength", IntType.v(), object_soot_class.getType());
        m_thisRef = bcl.refThis();
        m_param0 = bcl.refParameter(0);

        List<Type> types = RootbeerClassLoader.v().getDfsInfo().getOrderedRefLikeTypes();
        for ( Type type : types )
        {
            if ( ! ( type instanceof ArrayType ) )
                continue;

            String label = getNextLabel(); // non-static
            bcl.ifInstanceOfStmt( m_param0, type, label );
            bcl.returnValue( bcl.lengthof( bcl.cast( type, m_param0 ) ) );
            bcl.label( label );
        }

        bcl.returnValue(IntConstant.v(0));
        bcl.endMethod();
    }

    private static void makeWriteToHeapMethod
    (
        final BytecodeLanguage bcl,
        final String className
    )
    {
        List<Type> types = RootbeerClassLoader.v().getDfsInfo().getOrderedRefLikeTypes();
        VisitorWriteGen write_gen = new VisitorWriteGen(types,
            className, bcl );
        write_gen.makeWriteToHeapMethod();
    }

    private static void makeReadFromHeapMethod
    (
        final BytecodeLanguage bcl,
        final String className
    )
    {
        List<Type> types = RootbeerClassLoader.v().getDfsInfo().getOrderedRefLikeTypes();
        VisitorReadGen read_gen = new VisitorReadGen(types,
            className, bcl );
        read_gen.makeReadFromHeapMethod();
    }


    private static void makeWriteStaticsToHeapMethod( final BytecodeLanguage bcl )
    {
        new VisitorWriteGenStatic( bcl ).makeMethod();
    }

    private static void makeReadStaticsFromHeapMethod( final BytecodeLanguage bcl )
    {
        new VisitorReadGenStatic( bcl ).makeMethod();
    }

    private static void addGetSerializerMethod
    (
        final BytecodeLanguage bcl,
        final SootClass runtimeBasicBlock,
        final String className
    )
    {
        bcl.openClass( runtimeBasicBlock );
        SootClass gc_object_visitor_soot_class = Scene.v().getSootClass("org.trifort.rootbeer.runtime.Serializer");
        SootClass mem_cls = Scene.v().getSootClass("org.trifort.rootbeer.runtime.Memory");
        bcl.startMethod("getSerializer", gc_object_visitor_soot_class.getType(), mem_cls.getType(), mem_cls.getType());
        Local thisref = bcl.refThis();
        Local param0 = bcl.refParameter(0);
        Local param1 = bcl.refParameter(1);
        Local ret = bcl.newInstance( className, param0, param1 );
        bcl.returnValue(ret);
        bcl.endMethod();
    }

    private void makeCtor() {
        SootClass mem_cls = Scene.v().getSootClass("org.trifort.rootbeer.runtime.Memory");

        m_bcl.peek().startMethod("<init>", VoidType.v(), mem_cls.getType(), mem_cls.getType());
        Local this_ref = m_bcl.peek().refThis();
        Local param0 = m_bcl.peek().refParameter(0);
        Local param1 = m_bcl.peek().refParameter(1);
        m_bcl.peek().pushMethod("org.trifort.rootbeer.runtime.Serializer", "<init>", VoidType.v(), mem_cls.getType(), mem_cls.getType());
        m_bcl.peek().invokeMethodNoRet(this_ref, param0, param1);
        m_bcl.peek().returnVoid();
        m_bcl.peek().endMethod();
    }

    private static void makeSentinalCtors( final Set<String> sentinalCtorsCreated )
    {
        List<RefType> types = RootbeerClassLoader.v().getDfsInfo().getOrderedRefTypes();
        //types are ordered from largest type number to smallest
        //reverse the order for this computation because the sentinal ctors
        //need the parent to first have the sential ctor made.
        Collections.reverse( types );

        for ( final RefType ref_type : types )
        {
            AcceptableGpuTypes accept = new AcceptableGpuTypes();
            if ( ! accept.shouldGenerateCtor( ref_type.getClassName() ) )
                continue;

            final String refClassName = ref_type.getSootClass().getName();
            if ( sentinalCtorsCreated.contains( refClassName ) )
                continue;
            sentinalCtorsCreated.add( refClassName );

            final SootClass soot_class = Scene.v().getSootClass( refClassName );
            if ( ! soot_class.isApplicationClass() )
                continue;

            if ( soot_class.declaresMethod( "void <init>(org.trifort.rootbeer.runtime.Sentinal)" ) )
                continue;

            SootClass parent_class = soot_class.getSuperclass();
            parent_class = Scene.v().getSootClass(parent_class.getName());

            BytecodeLanguage bcl = new BytecodeLanguage();
            bcl.openClass(soot_class);
            bcl.startMethod("<init>", VoidType.v(), RefType.v("org.trifort.rootbeer.runtime.Sentinal"));
            Local thisref = bcl.refThis();

            String parent_name = parent_class.getName();
            if ( ! parent_class.isApplicationClass() )
            {
                if ( parent_class.declaresMethod("void <init>()") )
                {
                    bcl.pushMethod(parent_name, "<init>", VoidType.v());
                    bcl.invokeMethodNoRet(thisref);
                } else {
                    System.out.println("Library class "+parent_name+" on the GPU does not have a void constructor");
                    System.exit(-1);
                }
            } else {
                bcl.pushMethod(parent_name, "<init>", VoidType.v(), RefType.v("org.trifort.rootbeer.runtime.Sentinal"));
                bcl.invokeMethodNoRet(thisref, NullConstant.v());
            }
            bcl.returnVoid();
            bcl.endMethod();

        }
    }

    String getClassName() { return m_className; }
}
