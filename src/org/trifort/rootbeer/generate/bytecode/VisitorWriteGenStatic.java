/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.bytecode;


import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.trifort.rootbeer.generate.bytecode.permissiongraph.PermissionGraph;
import org.trifort.rootbeer.generate.bytecode.permissiongraph.PermissionGraphNode;
import org.trifort.rootbeer.generate.opencl.ClassConstantNumbers;
import org.trifort.rootbeer.generate.opencl.OpenCLScene;
import org.trifort.rootbeer.generate.opencl.fields.OpenCLField;

import soot.SootClass  ;
import soot.LongType   ;
import soot.IntType    ;
import soot.VoidType   ;
import soot.Local      ;
import soot.RefType    ;
import soot.ArrayType  ;
import soot.BooleanType;
import soot.Type       ;
import soot.Scene      ;

import soot.jimple.ClassConstant;
import soot.jimple.IntConstant;
import soot.jimple.LongConstant;
import soot.jimple.StringConstant;
import soot.rbclassload.RootbeerClassLoader;


/**
 * Analyzes Kernel code using soot and defines methods for serialization
 */
public final class VisitorWriteGenStatic extends AbstractVisitorGen
{
    private       Local         m_Mem;
    private final StaticOffsets m_StaticOffsets;
    private final Set<String>   m_AttachedWriters;

    public VisitorWriteGenStatic( final BytecodeLanguage bcl )
    {
        m_bcl.push( bcl );
        m_StaticOffsets   = new StaticOffsets();
        m_AttachedWriters = new HashSet<String>();
    }

    /**
     * Creates the 'doWriteStaticsToHeap' method which is declared in
     * Serializer.java
     */
    public void makeMethod()
    {
        final BytecodeLanguage bcl = m_bcl.peek();
        bcl.startMethod( "doWriteStaticsToHeap", VoidType.v() );

        /* get 'mMem' variable from Serializer which is of type Memory.java.
         * Then start writing the members of the Kenrel class to it */
        m_thisRef = bcl.refThis();
        m_currThisRef .push( m_thisRef );
        m_gcObjVisitor.push( m_thisRef );
        m_Mem = bcl.refInstanceField( m_thisRef, "mMem" );
        m_currMem     .push( m_Mem );
        /* this means BclMemory doesn't extend Memory, it is another boiler-
         * plate code wrapper needed to simplify accessing it using bytecode.
         * This means it's not a problem to change Memory.java if methods are
         * neither used by BclMemory nor FixedMemory */
        final BclMemory bcl_mem = new BclMemory( bcl, m_Mem );


        bcl_mem.useInstancePointer();
        bcl_mem.mallocWithSize( IntConstant.v( m_StaticOffsets.getEndIndex() ) );
        final PermissionGraph           graph = new PermissionGraph();
        final List<PermissionGraphNode> roots = graph.getRoots();
        for ( final PermissionGraphNode node : roots )
        {
            final SootClass soot_class = node.getSootClass();
            if ( soot_class.isApplicationClass() &&
                ! m_classesToIgnore.contains( soot_class.getName() ) )
            {
                attachWriter( soot_class, node.getChildren(), m_thisRef, m_AttachedWriters, m_StaticOffsets, m_classesToIgnore );
                /* call writer */
                bcl.pushMethod( soot_class, getWriterName( soot_class ), VoidType.v(),
                                Scene.v().getSootClass("org.trifort.rootbeer.runtime.Memory").getType(),
                                m_gcObjVisitor.peek().getType() );
                bcl.invokeStaticMethodNoRet( m_currMem.peek(), m_gcObjVisitor.peek() );
            }
            else {
                //doWriter(soot_class, node.getChildren());
                doWriter( bcl, m_Mem, m_thisRef, soot_class, new ArrayList<SootClass>(), m_StaticOffsets, m_classesToIgnore, m_AttachedWriters );
            }
        }

        //write .class's for array types
        Set<ArrayType> array_types = RootbeerClassLoader.v().getDfsInfo().getArrayTypes();
        for ( final ArrayType type : array_types )
            writeType( bcl, m_thisRef, type );

        bcl_mem.useStaticPointer();
        bcl_mem.setAddress(LongConstant.v(m_StaticOffsets.getLockStart()));
        //write the lock objects for all the classes
        int count = m_StaticOffsets.getClassSize();
        for ( int i = 0; i < count; ++i )
            bcl_mem.writeInt(-1);
        int zeros = m_StaticOffsets.getZerosSize();
        for ( int i = 0; i < zeros; ++i )
            bcl_mem.writeByte((byte) 0);
        bcl_mem.useInstancePointer();

        bcl.returnVoid();
        bcl.endMethod();

        m_gcObjVisitor.pop();
    }

    private static String getWriterName( final SootClass soot_class )
    {
        return "org_trifort_writeStaticsToHeap" +
               JavaNameToOpenCL.convert( soot_class.getName() ) +
               OpenCLScene.v().getIdent();
    }

    private static void attachWriter
    (
        final SootClass       soot_class     ,
        final List<SootClass> children       ,
        final Local           old_gc_visit   ,
        final Set<String>     attachedWriters,
        final StaticOffsets   staticOffsets  ,
        final List<String>    classesToIgnore
    )
    {
        final String method_name = getWriterName(soot_class);
        if ( attachedWriters.contains( method_name ) )
            return;
        attachedWriters.add( method_name );

        BytecodeLanguage bcl = new BytecodeLanguage();
        bcl.openClass(soot_class);
        SootClass mem = Scene.v().getSootClass("org.trifort.rootbeer.runtime.Memory");
        bcl.startStaticMethod( method_name, VoidType.v(), mem.getType(), old_gc_visit.getType() );

        final Local memory   = bcl.refParameter(0);
        final Local gc_visit = bcl.refParameter(1);
        doWriter( bcl, memory, gc_visit, soot_class, children, staticOffsets, classesToIgnore, attachedWriters );

        bcl.returnVoid();
        bcl.endMethod();
    }

    private static void doWriter
    (
        final BytecodeLanguage bcl            ,
        final Local            memory         ,
        final Local            gc_visit       ,
        final SootClass        soot_class     ,
        final List<SootClass>  children       ,
        final StaticOffsets    staticOffsets  ,
        final List<String>     classesToIgnore,
        final Set<String>      attachedWriters
    )
    {
        writeType( bcl, gc_visit, soot_class.getType() );

        List<OpenCLField> static_fields = staticOffsets.getStaticFields(soot_class);

        BclMemory bcl_mem = new BclMemory(bcl, memory);
        SootClass obj = Scene.v().getSootClass("java.lang.Object");
        for ( final OpenCLField field : static_fields )
        {
            Local field_value;
            if ( soot_class.isApplicationClass() )
                field_value = bcl.refStaticField(soot_class.getType(), field.getName());
            else
            {
                SootClass string = Scene.v().getSootClass("java.lang.String");
                SootClass cls = Scene.v().getSootClass("java.lang.Class");
                bcl.pushMethod(gc_visit, "readStaticField", obj.getType(), cls.getType(), string.getType());
                final Local obj_field_value = bcl.invokeMethodRet( gc_visit,
                    ClassConstant.v( toConstant( soot_class.getName() ) ), StringConstant.v( field.getName() ) );
                if ( field.getType().isRefType() )
                    field_value = obj_field_value;
                else
                {
                    Local capital_value = bcl.cast(field.getType().getCapitalType(), obj_field_value);
                    bcl.pushMethod(capital_value, field.getType().getName()+"Value", field.getType().getSootType());
                    field_value = bcl.invokeMethodRet(capital_value);
                }
            }
            if ( field.getType().isRefType() )
            {
                bcl.pushMethod(gc_visit, "writeToHeap", LongType.v(), obj.getType(), BooleanType.v());
                Local ref = bcl.invokeMethodRet(gc_visit, field_value, IntConstant.v(1));
                bcl_mem.useStaticPointer();
                bcl_mem.setAddress( LongConstant.v( staticOffsets.getIndex(field) ) );
                bcl_mem.writeRef( ref );
                bcl_mem.useInstancePointer();
            }
            else
            {
                bcl_mem.useStaticPointer();
                bcl_mem.setAddress( LongConstant.v( staticOffsets.getIndex(field) ) );
                bcl_mem.writeVar( field_value );
                bcl_mem.useInstancePointer();
            }
        }

        for ( final SootClass child : children )
        {
            if ( soot_class.isApplicationClass() &&
                ! classesToIgnore.contains( soot_class.getName() ) )
            {
                attachWriter( child, new ArrayList<SootClass>(), gc_visit, attachedWriters, staticOffsets, classesToIgnore );
                /* call Writer */
                bcl.pushMethod( child, getWriterName( child ), VoidType.v(),
                                Scene.v().getSootClass("org.trifort.rootbeer.runtime.Memory").getType(),
                                gc_visit.getType() );
                bcl.invokeStaticMethodNoRet( memory, gc_visit );
            }
            else
                doWriter( bcl, memory, gc_visit, child, new ArrayList<SootClass>(), staticOffsets, classesToIgnore, attachedWriters );
        }
    }

    private static boolean reachesJavaLangClass()
    {
        return RootbeerClassLoader.v().getDfsInfo().getOrderedRefTypes().
               contains( RefType.v( "java.lang.Class" ) );
    }

    private static void writeType( final BytecodeLanguage bcl, final Local gc_visit, final Type type )
    {
        if ( ! reachesJavaLangClass() )
            return;

        final int number = OpenCLScene.v().getClassConstantNumbers().get(type);
        final Local class_obj = bcl.classConstant( type );

        //getName has to be called to load the name variable
        bcl.pushMethod( class_obj, "getName", Scene.v().getSootClass("java.lang.String").getType() );
        bcl.invokeMethodRet( class_obj );

        bcl.pushMethod( gc_visit, "writeToHeap", LongType.v(),
            Scene.v().getSootClass( "java.lang.Object" ).getType(), BooleanType.v() );

        final Local ref = bcl.invokeMethodRet( gc_visit, class_obj, IntConstant.v(1) );
        bcl.pushMethod( gc_visit, "addClassRef", VoidType.v(), LongType.v(), IntType.v() );
        bcl.invokeMethodNoRet( gc_visit, ref, IntConstant.v( number ) );
    }
}
