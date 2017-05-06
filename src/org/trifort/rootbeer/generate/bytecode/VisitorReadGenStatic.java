/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.bytecode;


import java.util.ArrayList;
import java.util.HashSet  ;
import java.util.List     ;
import java.util.Set      ;
import java.util.Stack    ;

import org.trifort.rootbeer.generate.bytecode.permissiongraph.PermissionGraph    ;
import org.trifort.rootbeer.generate.bytecode.permissiongraph.PermissionGraphNode;
import org.trifort.rootbeer.generate.opencl.OpenCLScene                          ;
import org.trifort.rootbeer.generate.opencl.fields.OpenCLField                   ;

import soot.Scene      ;
import soot.SootClass  ;
import soot.Local      ;
import soot.VoidType   ;
import soot.BooleanType;
import soot.LongType   ;

import soot.jimple.ClassConstant           ;
import soot.jimple.IntConstant             ;
import soot.jimple.LongConstant            ;
import soot.jimple.StringConstant          ;
import soot.rbclassload.RootbeerClassLoader;


public final class VisitorReadGenStatic extends AbstractVisitorGen
{
    private       Local         m_mem            ;
    private final Set<String>   m_attachedReaders;
    private final StaticOffsets m_staticOffsets  ;

    /* uselessly complex argument stack like in assembler */
    private       Local                   m_thisRef        ;
    private final Stack<BytecodeLanguage> m_bcl            ;
    private final Stack<Local>            m_gcObjVisitor   ;
    private final Stack<Local>            m_currMem        ;
    private final Stack<Local>            m_objSerializing ;

    public VisitorReadGenStatic( final BytecodeLanguage bcl )
    {
        m_bcl             = new Stack<BytecodeLanguage>();
        m_gcObjVisitor    = new Stack<Local>();
        m_currMem         = new Stack<Local>();
        m_objSerializing  = new Stack<Local>();
        /* parent no-arg constructor initializes m_objSerializing,
         * m_bcl and m_currMem with empty Stacks */
        m_bcl.push(bcl);

        m_attachedReaders = new HashSet<String>();
        m_staticOffsets   = new StaticOffsets();
    }

    public void makeMethod()
    {
        final BytecodeLanguage bcl = m_bcl.peek();
        bcl.startMethod( "doReadStaticsFromHeap", VoidType.v() );

        m_thisRef = bcl.refThis();
        m_mem     = bcl.refInstanceField( m_thisRef, "mMem" );
        m_currMem     .push( m_mem     );
        m_gcObjVisitor.push( m_thisRef );

        final List<PermissionGraphNode> roots = new PermissionGraph().getRoots();
        for ( final PermissionGraphNode node : roots )
        {
            final SootClass soot_class = node.getSootClass();
            if ( soot_class.isApplicationClass() )
                attachAndCallReader( soot_class, node.getChildren() );
            else
                doReader( bcl, m_mem, m_thisRef, soot_class );
        }

        bcl.returnVoid();
        bcl.endMethod();

        m_currMem.pop();
        m_gcObjVisitor.pop();
    }

    private String getReaderName(SootClass soot_class){
        return "org_trifort_readStaticsFromHeap"+JavaNameToOpenCL.convert(soot_class.getName())+OpenCLScene.v().getIdent();
    }

    private void attachReader( final SootClass soot_class, final List<SootClass> children )
    {
        final String method_name = getReaderName( soot_class );
        if ( m_attachedReaders.contains( method_name ) )
            return;
        m_attachedReaders.add( method_name );

        List<OpenCLField> static_fields = m_staticOffsets.getStaticFields(soot_class);

        final BytecodeLanguage bcl = new BytecodeLanguage();
        m_bcl.push(bcl);
        bcl.openClass( soot_class );
        SootClass mem = Scene.v().getSootClass("org.trifort.rootbeer.runtime.Memory");
        bcl.startStaticMethod( method_name, VoidType.v(), mem.getType(), m_thisRef.getType() );

        final Local memory   = bcl.refParameter(0);
        final Local gc_visit = bcl.refParameter(1);
        m_gcObjVisitor.push( gc_visit );
        m_currMem     .push( memory   );

        BclMemory bcl_mem = new BclMemory( bcl, memory );
        for ( final OpenCLField field : static_fields )
        {
            final int index = m_staticOffsets.getIndex( field );
            bcl_mem.setAddress( LongConstant.v( index ) );
            if ( field.getType().isRefType() )
                readRefField( bcl, gc_visit, memory, null /* m_objSerializing.peek() @todo Where is this set !!! It seems like this wasn't effectively set in the original Rootbeer version either! */, field );
            else
                readNonRefField( bcl, memory, null, field );
        }

        for ( final SootClass child : children )
            attachAndCallReader( child, new ArrayList<SootClass>() );

        bcl.returnVoid();
        bcl.endMethod();

        m_gcObjVisitor.pop();
        m_currMem.pop();
        m_bcl.pop();
    }

    private void attachAndCallReader( final SootClass soot_class, final List<SootClass> children )
    {
        final String class_name = soot_class.getName();
        if ( m_classesToIgnore.contains( class_name ) )
            return;
        attachReader( soot_class, children );
        callReader  ( soot_class );
    }

    private void callReader( final SootClass soot_class )
    {
        BytecodeLanguage bcl = m_bcl.peek();
        String method_name = getReaderName(soot_class);
        SootClass mem = Scene.v().getSootClass("org.trifort.rootbeer.runtime.Memory");
        bcl.pushMethod(soot_class, method_name, VoidType.v(), mem.getType(), m_thisRef.getType());
        bcl.invokeStaticMethodNoRet(m_currMem.peek(), m_gcObjVisitor.peek());
    }

    private void doReader
    (
        final BytecodeLanguage bcl       ,
        final Local            memory    ,
        final Local            gc_visit  ,
        final SootClass        soot_class
    )
    {
        final List<OpenCLField> static_fields = m_staticOffsets.getStaticFields(soot_class);

        final BclMemory bcl_mem = new BclMemory( bcl, memory );
        final SootClass obj     = Scene.v().getSootClass("java.lang.Object");
        for ( final OpenCLField field : static_fields )
        {
            Local field_value;

            if ( field.getType().isRefType() )
            {
                bcl_mem.useStaticPointer();
                bcl_mem.setAddress(LongConstant.v(m_staticOffsets.getIndex(field)));
                Local ref = bcl_mem.readRef();
                bcl_mem.useInstancePointer();

                if ( soot_class.isApplicationClass() )
                {
                    bcl_mem.useStaticPointer();
                    bcl_mem.setAddress(LongConstant.v(m_staticOffsets.getIndex(field)));
                    field_value = bcl_mem.readVar(field.getType().getSootType());
                    bcl_mem.useInstancePointer();
                }
                else
                {
                    SootClass string = Scene.v().getSootClass("java.lang.String");
                    SootClass cls = Scene.v().getSootClass("java.lang.Class");
                    bcl.pushMethod(gc_visit, "readStaticField", obj.getType(), cls.getType(), string.getType());
                    Local obj_field_value = bcl.invokeMethodRet(gc_visit, ClassConstant.v(toConstant(soot_class.getName())), StringConstant.v(field.getName()));
                    if ( field.getType().isRefType() )
                        field_value = obj_field_value;
                    else
                    {
                        Local capital_value = bcl.cast(field.getType().getCapitalType(), obj_field_value);
                        bcl.pushMethod(capital_value, field.getType().getName()+"Value", field.getType().getSootType());
                        field_value = bcl.invokeMethodRet(capital_value);
                    }
                }

                bcl.pushMethod(m_thisRef, "readFromHeap", obj.getType(), obj.getType(), BooleanType.v(), LongType.v());
                field_value = bcl.invokeMethodRet(m_thisRef, field_value, IntConstant.v(0), ref);
            }
            else
            {
                bcl_mem.useStaticPointer();
                bcl_mem.setAddress(LongConstant.v(m_staticOffsets.getIndex(field)));
                field_value = bcl_mem.readVar(field.getType().getSootType());
                bcl_mem.useInstancePointer();
            }

            if ( field.isFinal() )
                continue;

            if ( soot_class.isApplicationClass() )
                bcl.setStaticField(field.getSootField(), field_value);
            else
            {
                final SootClass string = Scene.v().getSootClass( "java.lang.String" );
                final SootClass cls    = Scene.v().getSootClass( "java.lang.Class"  );
                if ( field.getType().isRefType() )
                {
                    bcl.pushMethod(gc_visit, "writeStaticField", VoidType.v(), cls.getType(), string.getType(), obj.getType());
                    bcl.invokeMethodNoRet(gc_visit, ClassConstant.v(toConstant(soot_class.getName())), StringConstant.v(field.getName()), field_value);
                }
                else
                {
                    bcl.pushMethod(gc_visit, "writeStatic"+field.getType().getCapitalName()+"Field", VoidType.v(), cls.getType(), string.getType(), field.getType().getSootType());
                    bcl.invokeMethodNoRet(gc_visit, ClassConstant.v(toConstant(soot_class.getName())), StringConstant.v(field.getName()), field_value);
                }
            }

        }
    }
}
