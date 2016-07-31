/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.bytecode;


import java.util.ArrayList;
import java.util.List;

import org.trifort.rootbeer.generate.opencl.OpenCLClass;
import org.trifort.rootbeer.generate.opencl.OpenCLScene;
import org.trifort.rootbeer.generate.opencl.fields.OpenCLField;
import org.trifort.rootbeer.util.Stack;

import soot.Scene      ;
import soot.SootClass  ;
import soot.Local      ;
import soot.ArrayType  ;
import soot.BooleanType;
import soot.LongType   ;
import soot.RefType    ;
import soot.SootField  ;
import soot.Type       ;
import soot.Value      ;
import soot.VoidType   ;

import soot.jimple.ClassConstant;
import soot.jimple.IntConstant;
import soot.jimple.StringConstant;


public class AbstractVisitorGen
{
    protected       Local                   m_thisRef        ;
    protected final Stack<Local>            m_currThisRef    ;
    private         int                     m_labelIndex     ;
    /* these complicated stacks are only for parameter passing.
     * @todo use simple method arguments instead !!! */
    protected final Stack<BytecodeLanguage> m_bcl            ;
    protected final Stack<Local>            m_gcObjVisitor   ;
    protected final Stack<Local>            m_currMem        ;
    protected final Stack<Local>            m_objSerializing ;
    protected final List<String>            m_classesToIgnore;

    public AbstractVisitorGen()
    {
        m_labelIndex      = 0;
        m_bcl             = new Stack<BytecodeLanguage>();
        m_gcObjVisitor    = new Stack<Local>();
        m_currMem         = new Stack<Local>();
        m_currThisRef     = new Stack<Local>();
        m_objSerializing  = new Stack<Local>();
        m_classesToIgnore = new ArrayList<String>();
        m_classesToIgnore.add( "org.trifort.rootbeer.runtime.RootbeerGpu"     );
        m_classesToIgnore.add( "org.trifort.rootbeer.runtime.Sentinal"        );
        m_classesToIgnore.add( "org.trifort.rootbeer.runtimegpu.GpuException" );
    }

    protected boolean differentPackageAndPrivate( RefType ref_inspecting )
    {
        RefType ref_type = (RefType) m_thisRef.getType();
        SootClass this_class = getClassForType(ref_type);
        SootClass class_inspecting = getClassForType(ref_inspecting);
        if(this_class.getPackageName().equals(class_inspecting.getPackageName()))
            return false;
        if(class_inspecting.isPublic() == false)
            return true;
        return false;
    }

    protected static SootClass getClassForType(RefType ref_type){
        SootClass soot_class = ref_type.getSootClass();
        soot_class = Scene.v().getSootClass(soot_class.getName());
        return soot_class;
    }

    protected static String getTypeString(SootField soot_field){
        Type type = soot_field.getType();
        String name = type.toString();
        char[] name_array = name.toCharArray();
        name_array[0] = Character.toUpperCase(name_array[0]);
        return new String(name_array);
    }

    protected static List<OpenCLField> getNonRefFields(SootClass soot_class){
        OpenCLClass ocl_class = OpenCLScene.v().getOpenCLClass(soot_class);
        return ocl_class.getInstanceNonRefFields();
    }

    protected static List<OpenCLField> getRefFields(SootClass soot_class){
        OpenCLClass ocl_class = OpenCLScene.v().getOpenCLClass(soot_class);
        if(ocl_class == null){
            System.out.println("ocl_class == null: "+soot_class.getName());
        }
        return ocl_class.getInstanceRefFields();
    }

    protected static SootClass getGcVisitorClass(Local visitor){
        RefType type = (RefType) visitor.getType();
        SootClass gc_visitor = Scene.v().getSootClass(type.getClassName());
        return gc_visitor;
    }

    /* only returns a unique string, no soot magic here */
    protected String getNextLabel(){ return "rblabel" + m_labelIndex++; }

    /**
     * Returns true if public.
     * All non reference types are primitive types and therefore public.
     * For arrays the array element type is analysed.
     * Private types are e.g. classes defined inside another classes with
     * private specifier.
     */
    protected static boolean typeIsPublic( final Type type )
    {
        Type poss_ref_type;
        if ( type instanceof ArrayType )
        {
            ArrayType array_type = (ArrayType) type;
            poss_ref_type = array_type.baseType;
        }
        else
            poss_ref_type = type;

        if ( poss_ref_type instanceof RefType )
        {
            final RefType ref_type = (RefType) poss_ref_type;
            return ref_type.getSootClass().isPublic();
        }
        else
            return true;
    }

    protected static void readRefField
    (
        final BytecodeLanguage bcl           ,
        final Local            gc_obj_visit  ,
        final Local            currMem       ,
        final Local            objSerializing,
        final OpenCLField      ref_field
    )
    {
        SootField soot_field = ref_field.getSootField();
        final SootClass soot_class = Scene.v().getSootClass( soot_field.getDeclaringClass().getName() );
        final BclMemory bcl_mem = new BclMemory( bcl, currMem );

        final Local ref = bcl_mem.readRef();
        bcl_mem.useInstancePointer();
        bcl_mem.pushAddress();
        bcl_mem.setAddress(ref);

        //bcl.println("reading field: "+ref_field.getName());

        SootClass obj_class = Scene.v().getSootClass("java.lang.Object");
        SootClass string = Scene.v().getSootClass("java.lang.String");
        SootClass class_class = Scene.v().getSootClass("java.lang.Class");
        Local original_field_value;
        if ( ! soot_class.isApplicationClass() )
        {
            if ( ref_field.isInstance() )
            {
                bcl.pushMethod( gc_obj_visit, "readField", obj_class.getType(), obj_class.getType(), string.getType() );
                original_field_value = bcl.invokeMethodRet( gc_obj_visit, objSerializing, StringConstant.v( soot_field.getName() ) );
            }
            else
            {
                bcl.pushMethod(gc_obj_visit, "readStaticField", obj_class.getType(), class_class.getType(), string.getType());
                Local cls = bcl.classConstant(soot_field.getDeclaringClass().getType());
                original_field_value = bcl.invokeMethodRet(gc_obj_visit, cls, StringConstant.v(soot_field.getName()));
            }
        }
        else
        {
            if ( ref_field.isInstance() )
                original_field_value = bcl.refInstanceField( objSerializing, ref_field.getName() );
            else
                original_field_value = bcl.refStaticField( soot_class.getType(), ref_field.getName() );
        }
        bcl.pushMethod(gc_obj_visit, "readFromHeap", obj_class.getType(), obj_class.getType(), BooleanType.v(), LongType.v());
        int should_read = 1;
        Local ret_obj = bcl.invokeMethodRet( gc_obj_visit, original_field_value, IntConstant.v(should_read), ref );

        Type type = soot_field.getType();
        /* @todo I think this could be the cast which throws the ClassCastException !!! */
        Local ret = bcl.cast( type, ret_obj );

        if ( ! soot_class.isApplicationClass() )
        {
            if ( ref_field.isInstance() )
            {
                bcl.pushMethod( gc_obj_visit, "writeField", VoidType.v(), obj_class.getType(), string.getType(), obj_class.getType() );
                bcl.invokeMethodNoRet( gc_obj_visit, objSerializing, StringConstant.v(soot_field.getName()), ret );
            }
            else
            {
                bcl.pushMethod(gc_obj_visit, "writeStaticField", VoidType.v(), class_class.getType(), string.getType(), obj_class.getType());
                Local cls = bcl.classConstant(soot_field.getDeclaringClass().getType());
                bcl.invokeMethodNoRet(gc_obj_visit, cls, StringConstant.v(soot_field.getName()), ret);
            }
        }
        else
        {
            if ( ref_field.isInstance() )
                bcl.setInstanceField(soot_field, objSerializing, ret);
            else
                bcl.setStaticField(soot_field, ret);
        }

        bcl_mem.popAddress();
    }

    protected static void readNonRefField
    (
        final BytecodeLanguage bcl           ,
        final Local            currMem       ,
        final Local            objSerializing,
        final OpenCLField      field
    )
    {
        if ( bcl     == null ) throw new IllegalArgumentException( "bcl may not be null!"     );
        if ( currMem == null ) throw new IllegalArgumentException( "currMem may not be null!" );
        if ( field   == null ) throw new IllegalArgumentException( "field may not be null!"   );
        if ( ! ( objSerializing != null || ! field.isInstance() ) ) throw new IllegalArgumentException( "objSerializing may not be null!" );

        final SootField soot_field = field.getSootField();
        final String function_name = "read" + getTypeString( soot_field );

        bcl.pushMethod( currMem, function_name, soot_field.getType() );
        final Local data = bcl.invokeMethodRet( currMem );

        final SootClass soot_class = Scene.v().getSootClass( soot_field.getDeclaringClass().getName() );
        if ( soot_class.isApplicationClass() )
        {
            if ( field.isInstance() )
                bcl.setInstanceField( soot_field, objSerializing, data );
            else
                bcl.setStaticField( soot_field, data );
        }
        else
        {
            SootClass string = Scene.v().getSootClass("java.lang.String");
            String static_str;
            SootClass first_param_type;
            Value first_param;
            if ( field.isInstance() )
            {
                static_str       = "";
                first_param_type = Scene.v().getSootClass( "java.lang.Object" );
                first_param      = objSerializing;
            }
            else
            {
                static_str       = "Static";
                first_param_type = Scene.v().getSootClass( "java.lang.Class" );
                first_param      = ClassConstant.v( soot_class.getName() );
            }
            final Local private_fields = bcl.newInstance( "org.trifort.rootbeer.runtime.PrivateFields" );
            bcl.pushMethod( private_fields, "write"+static_str+getTypeString( soot_field ),
                            VoidType.v(), first_param_type.getType(),
                            string.getType(), string.getType(), soot_field.getType() );
            bcl.invokeMethodNoRet(
                private_fields, first_param,
                StringConstant.v( soot_field.getName() ),
                StringConstant.v( soot_field.getDeclaringClass().getName() ),
                data
            );
        }
    }

    protected static String toConstant(String name) { return name.replace(".", "/"); }

}
