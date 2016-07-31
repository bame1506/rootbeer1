/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.bytecode;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Stack;

import soot.Scene        ;
import soot.Local        ;
import soot.SootClass    ;
import soot.SootField    ;
import soot.SootMethod   ;
import soot.Type         ;
import soot.Unit         ;
import soot.Value        ;
import soot.ArrayType    ;
import soot.BooleanType  ;
import soot.ByteType     ;
import soot.CharType     ;
import soot.DoubleType   ;
import soot.FloatType    ;
import soot.IntType      ;
import soot.LongType     ;
import soot.Modifier     ;
import soot.PrimType     ;
import soot.RefType      ;
import soot.ShortType    ;
import soot.SootMethodRef;
import soot.VoidType     ;

import soot.jimple.Jimple        ;
import soot.jimple.IntConstant   ;
import soot.jimple.StringConstant;
import soot.jimple.ClassConstant ;
import soot.jimple.JimpleBody    ;

import soot.rbclassload.ClassHierarchy     ;
import soot.rbclassload.HierarchyGraph     ;
import soot.rbclassload.MethodSignatureUtil;
import soot.rbclassload.RootbeerClassLoader;
import soot.rbclassload.TypeToString       ;


/**
 * Helper class to simplify writing dynamically produced code with Soot ???
 */
public class BytecodeLanguage
{
    private final static boolean debugging = false;

    /* the cached Jimple singleton object. It does not contain user-data */
    private final Jimple     m_jimple        ;
    private SootClass        m_currClass     ;

    //method fields
    private SootMethod       m_currMethod    ;
    private JimpleBody       m_currBody      ;
    private List<Type>       m_parameterTypes;
    private UnitAssembler    m_assembler     ;

    /* holds soot method to be called.
     * @todo I'm not sure what reason there is to make this as stack.
     *       In all the cases up till now invokeMethod is directly called
     *       after pushMethod, meaning this stack should never contain more
     *       than one element anyway.
     */
    private final Stack<SootMethod> m_methodStack;

    public BytecodeLanguage()
    {
        m_jimple      = Jimple.v();
        m_methodStack = new Stack<SootMethod>();
    }

    public SootClass makeClass( String name )
    {
        SootClass ret = new SootClass(name, Modifier.PUBLIC);

        SootClass object_soot_class = Scene.v().getSootClass("java.lang.Object");
        ret.setSuperclass(object_soot_class);
        Scene.v().addClass(ret);
        ret.setApplicationClass();

        m_currClass = ret;
        return ret;
    }

    public SootClass makeClass(String name, String parent)
    {
        SootClass ret = new SootClass(name, Modifier.PUBLIC);

        //set superclass
        SootClass parent_class = Scene.v().getSootClass(parent);
        ret.setSuperclass(parent_class);

        Scene.v().addClass(ret);
        ret.setApplicationClass();

        m_currClass = ret;
        return ret;
    }

    public void addFieldToClass(Local local)
    {
        SootField field = new SootField(local.getName(), local.getType(), Modifier.PUBLIC);
        m_currClass.addField(field);
    }

    public void addFieldToClass(Local local, String name)
    {
        SootField field = new SootField(name, local.getType(), Modifier.PUBLIC);
        m_currClass.addField(field);
    }

    public void openClass( final String name){ m_currClass = Scene.v().getSootClass(name); }
    public void openClass( final SootClass soot_class){ m_currClass = soot_class; }

    public void startMethod( String method_name, Type return_type, Type... arg_types )
    {
        doStartMethod( method_name, return_type, Modifier.PUBLIC, arg_types );
    }

    private void doStartMethod
    (
        final String method_name,
        final Type   return_type,
        final int    modifiers  ,
        Type... arg_types
    )
    {
        m_assembler      = new UnitAssembler();

        m_parameterTypes = convertTypeArrayToList(arg_types);
        m_currMethod     = new SootMethod(method_name, m_parameterTypes, return_type, modifiers);
        m_currMethod.setDeclaringClass(m_currClass);

        m_currBody       = m_jimple.newBody(m_currMethod);
        m_currMethod.setActiveBody(m_currBody);
        m_currClass.addMethod(m_currMethod);

        RootbeerClassLoader.v().addGeneratedMethod(m_currMethod.getSignature());
    }

    public void startStaticMethod(String method_name, Type return_type, Type... arg_types){
        doStartMethod(method_name, return_type, Modifier.PUBLIC | Modifier.STATIC, arg_types);
    }

    public void continueMethod(UnitAssembler assembler){ m_assembler = assembler; }

    /**
     * Creates a local variable 'this0' which stores a pointer to this object
     * In Java 'this' automatically exists, but that's not the case in Jimple.
     *
     * @todo This should only be called one time per method I think ?!
     *       I.e. exactly one time between 'startMethod' and 'endMethod'
     */
    public Local refThis()
    {
        final RefType type    = m_currClass.getType();
        final Local thislocal = m_jimple.newLocal( "this0", type );
        m_assembler.add( m_jimple.newIdentityStmt( thislocal, m_jimple.newThisRef( type ) ) );
        return thislocal;
    }

    /**
     * Copies the non-mutable 'index'-th parameter to a local variable
     */
    public Local refParameter( final int index )
    {
        final Type type        = m_parameterTypes.get(index);
        final Local parameterI = m_jimple.newLocal( "parameter" + Integer.toString(index), type );
        m_assembler.add( m_jimple.newIdentityStmt( parameterI, m_jimple.newParameterRef( type, index ) ) );
        return parameterI;
    }

    public Local binOp( final Value lhs, final String op, final Value rhs )
    {
        Value binop = null;
        if(op.equals("*")){
            binop = m_jimple.newMulExpr(lhs, rhs);
        }

        Local ret = m_jimple.newLocal(getLocalName(), lhs.getType());
        Unit u = m_jimple.newAssignStmt(ret, binop);
        m_assembler.add(u);
        return ret;
    }

    public void setInstanceField( final SootField field, final Local field_instance, final Value value )
    {
        Value lhs;
        if(field.isStatic() == false)
            lhs = m_jimple.newInstanceFieldRef(field_instance, field.makeRef());
        else
            lhs = m_jimple.newStaticFieldRef(field.makeRef());
        Unit u = m_jimple.newAssignStmt(lhs, value);
        m_assembler.add(u);
    }

    public void setInstanceField( final String field_name, final Local field_instance, final Value value ){
        Type type = field_instance.getType();
        if(type instanceof RefType == false)
            throw new RuntimeException("How do we handle this case?");
        RefType ref_type = (RefType) type;
        SootClass soot_class = ref_type.getSootClass();
        SootField soot_field = soot_class.getFieldByName(field_name);
        setInstanceField(soot_field, field_instance, value);
    }

    public void setStaticField( final SootField field, final Value value )
    {
        Value lhs;
        lhs = m_jimple.newStaticFieldRef(field.makeRef());
        Unit u = m_jimple.newAssignStmt(lhs, value);
        m_assembler.add(u);
    }

    public void endMethod()
    {
        m_assembler.assemble( m_currBody );
        if ( debugging )
        {
            System.out.println( "Ending method: " + m_currMethod.getName() );
            System.out.println( m_assembler.toString() );
        }
    }

    private List<Type> convertTypeArrayToList(Type[] type_array)
    {
        List<Type> ret = new ArrayList<Type>();
        for(int i = 0; i < type_array.length; ++i){
            ret.add(type_array[i]);
        }
        return ret;
    }

    public void pushMethod
    (
        final Local   class_instance,
        final String  method_name   ,
        final Type    return_type   ,
        final Type... arg_types
    )
    {
        pushMethod( getTypeString(class_instance), method_name, return_type, arg_types );
    }

    public void pushMethod
    (
        final SootClass soot_class ,
        final String    method_name,
        final Type      return_type,
        final Type...   arg_types
    )
    {
        pushMethod( soot_class.getName(), method_name, return_type, arg_types );
    }

    /**
     * Uses MethodSignatureUtil to convert the function signature to a string
     * and then find a loaded SootMethod in rbclassload that fits the signature
     * and then push it to m_methodStack
     */
    public void pushMethod
    (
        final String  class_name ,
        final String  method_name,
        final Type    return_type,
        final Type... arg_types
    )
    {
        final TypeToString converter   = new TypeToString();
        final MethodSignatureUtil util = new MethodSignatureUtil();

        util.setClassName ( class_name );
        util.setMethodName( method_name );
        util.setReturnType( converter.convert(return_type) );

        /* convert all Soot Types for parameters to Strings and use util to set */
        final List<String> parameter_types = new ArrayList<String>();
        for ( final Type arg_type : arg_types )
            parameter_types.add( converter.convert( arg_type ) );
        util.setParameterTypes( parameter_types );

        m_methodStack.push( util.getSootMethod() );
    }

    /**
     * Generate Jimple code in m_assembler using m_jimple for calling the
     * topmost method from m_methodStack from object 'base'
     * Normally base is 'this'
     */
    public void invokeMethodNoRet( final Local base, final Value... args )
    {
        final SootMethod method = m_methodStack.pop();
        final List<Value> args_list = Arrays.asList( args );
        Value invoke_expr;

        /* if method is constructor */
        if ( method.getName().equals( "<init>" ) )
            invoke_expr = m_jimple.newSpecialInvokeExpr( base, method.makeRef(), args_list );
        else
        {
            //I can't find any way to distinguish between an interface and non-interface
            //method.    let's just try both and use whatever works.
            try {
                invoke_expr = m_jimple.newVirtualInvokeExpr( base, method.makeRef(), args_list );
            } catch ( RuntimeException ex ){
                invoke_expr = m_jimple.newInterfaceInvokeExpr( base, method.makeRef(), args_list );
            }
        }
        m_assembler.add( m_jimple.newInvokeStmt(invoke_expr) );
    }

    public void invokeStaticMethodNoRet(Value... args)
    {
        Value invoke_expr = m_jimple.newStaticInvokeExpr( m_methodStack.pop().makeRef(), Arrays.asList(args) );
        m_assembler.add( m_jimple.newInvokeStmt(invoke_expr) );
    }

    public void invokeSpecialNoRet(Local base, Value... args)
    {
        Value invoke_expr = m_jimple.newSpecialInvokeExpr(base, m_methodStack.pop().makeRef(), Arrays.asList(args) );
        m_assembler.add( m_jimple.newInvokeStmt(invoke_expr) );
    }

    public Local invokeMethodRet(Local base, Value... args)
    {
        SootMethod method = m_methodStack.pop();
        List<Value> args_list = Arrays.asList( args );
        Value invoke_expr;
        if(method.getName().equals("<init>")){
            invoke_expr = m_jimple.newSpecialInvokeExpr(base, method.makeRef(), args_list);
        } else {
            //I can't find any way to distinguish between an interface and non-interface
            //method.    let's just try both and use whatever works.
            try {
                invoke_expr = m_jimple.newVirtualInvokeExpr(base, method.makeRef(), args_list);
            } catch(RuntimeException ex){
                invoke_expr = m_jimple.newInterfaceInvokeExpr(base, method.makeRef(), args_list);
            }
        }

        String name = getLocalName();
        Local ret = m_jimple.newLocal(name, method.getReturnType());
        Unit u = m_jimple.newAssignStmt(ret, invoke_expr);
        m_assembler.add(u);
        return ret;
    }

    /**
     * Return a unique ID/name for a local variable for Soot. Else it would
     * be possible to accidentally reuse some still needed local variable
     */
    private static String getLocalName(){ return RegisterNamer.v().getName(); }

    public Local local(Type type)
    {
        return m_jimple.newLocal( getLocalName(), type );
    }

    /**
     * If lhs <op> rhs then jump to target_label
     */
    public void ifStmt
    (
        final Value  lhs,
        final String op ,
        final Value  rhs,
        final String target_label
    )
    {
        Value condition;
        if ( op.length() == 2 && op.charAt(1) == '=' )
        {
            switch ( op.charAt(0) )
            {
                case '=': condition = m_jimple.newEqExpr( lhs, rhs ); break;
                case '!': condition = m_jimple.newNeExpr( lhs, rhs ); break;
                case '>': condition = m_jimple.newGeExpr( lhs, rhs ); break;
                case '<': condition = m_jimple.newLeExpr( lhs, rhs ); break;
                default: throw new UnsupportedOperationException();
            }
        } else
            throw new UnsupportedOperationException();

        m_assembler.addIf( condition, target_label );
    }

    /**
     * Adds Jimple code for 'if ( lhs instanceof rhs == 0 ) goto target;'
     * @todo the name is highly confusing, because it suggests the target is
     *       jumped to if it is an instance of rhs, but the oppossitve is the
     *       case !!! -> rename to ifNotInstanceOfStmt
     */
    public void ifInstanceOfStmt( final Value lhs, final Type rhs, final String target )
    {
        /* @todo why is this evaluated and hardcoded instanceof commented out ? */
        //Local lhs_instanceof_rhs_local = lhs instanceof rhs;
        final Value condValue = m_jimple.newInstanceOfExpr( lhs, rhs );
        final Local condVar   = m_jimple.newLocal( getLocalName(), BooleanType.v() );
        m_assembler.add  ( m_jimple.newAssignStmt( condVar, condValue ) );
        m_assembler.addIf( m_jimple.newEqExpr( condVar, IntConstant.v(0) ), target );
    }

    public void label( String label_string ){ m_assembler.addLabel(label_string); }
    public void returnVoid() { m_assembler.add(m_jimple.newReturnVoidStmt()); }

    public void returnValue(Value value){
        Unit u = m_jimple.newReturnStmt(value);
        m_assembler.add(u);
    }

    public Local refInstanceField(Local base, String field_name)
    {
        Type base_type = base.getType();
        SootClass base_class = Scene.v().getSootClass(base_type.toString());
        SootField field = getFieldByName(base_class, field_name);
        Local ret = m_jimple.newLocal(getLocalName(), field.getType());

        Value rhs = m_jimple.newInstanceFieldRef(base, field.makeRef());
        Unit u = m_jimple.newAssignStmt(ret, rhs);
        m_assembler.add(u);
        return ret;
    }

    Local refStaticField(Local base, String field_name)
    {
        Type base_type = base.getType();
        return refStaticField(base_type, field_name);
    }

    Local refStaticField(Type base_type, String field_name)
    {
        SootClass base_class = Scene.v().getSootClass(base_type.toString());
        SootField field = getFieldByName(base_class, field_name);
        Local ret = m_jimple.newLocal(getLocalName(), field.getType());

        Value rhs = m_jimple.newStaticFieldRef(field.makeRef());
        Unit u = m_jimple.newAssignStmt(ret, rhs);
        m_assembler.add(u);
        return ret;
    }

    public void refInstanceFieldToInput(Local base, String field_name, Local input)
    {
        Type base_type = base.getType();
        SootClass base_class = Scene.v().getSootClass(base_type.toString());
        SootField field = getFieldByName(base_class, field_name);

        Value rhs = m_jimple.newInstanceFieldRef(base, field.makeRef());
        Unit u = m_jimple.newAssignStmt(input, rhs);
        m_assembler.add(u);
    }

    public void refInstanceFieldFromInput(Local base, String field_name, Local input)
    {
        Type base_type = base.getType();
        SootClass base_class = Scene.v().getSootClass(base_type.toString());
        SootField field = getFieldByName(base_class, field_name);

        Value rhs = m_jimple.newInstanceFieldRef(base, field.makeRef());
        Unit u = m_jimple.newAssignStmt(rhs, input);
        m_assembler.add(u);
    }

    public String getTypeString(Local local)
    {
        Type type = local.getType();
        return type.toString();
    }

    public Local cast(Type type, Local rhs)
    {
        Local ret = m_jimple.newLocal(getLocalName(), type);
        Value rhs_value = m_jimple.newCastExpr(rhs, type);
        Unit u = m_jimple.newAssignStmt(ret, rhs_value);
        m_assembler.add(u);
        return ret;
    }

    public Local newInstance(String mClassName, Value... params)
    {
        SootClass soot_class = Scene.v().getSootClass(mClassName);
        Local u1_lhs = m_jimple.newLocal( getLocalName(), soot_class.getType() );
        Value u1_rhs = m_jimple.newNewExpr(soot_class.getType());
        Unit u1 = m_jimple.newAssignStmt(u1_lhs, u1_rhs);
        m_assembler.add(u1);

        Type[] arg_types = new Type[params.length];
        for(int i = 0; i < params.length; ++i){
            arg_types[i] = params[i].getType();
        }
        pushMethod(u1_lhs, "<init>", VoidType.v(), arg_types);
        invokeMethodNoRet( u1_lhs, params );

        Local u2_lhs = m_jimple.newLocal(getLocalName(), soot_class.getType());
        Unit u2 = m_jimple.newAssignStmt(u2_lhs, u1_lhs);
        m_assembler.add(u2);
        return u2_lhs;
    }

    public Local newInstanceValueOf(String mClassName, Value param)
    {
        // Generate mClassName.valueOf(param)
        SootClass soot_class = Scene.v().getSootClass(mClassName);
        Local l_res = m_jimple.newLocal(getLocalName(), soot_class.getType());
        SootMethodRef classForNameRef = soot.Scene.v().makeMethodRef(soot_class,
                "valueOf", Arrays.asList(param.getType()), soot_class.getType(), true);
        Unit u1 = m_jimple.newAssignStmt(l_res, m_jimple.newStaticInvokeExpr(classForNameRef, Arrays.asList(new Value[]{param})));
        m_assembler.add(u1);
        return l_res;
    }

    public Value newArray(Type type, Value size)
    {
        ArrayType atype = (ArrayType) type;
        if(atype.numDimensions == 1)
            return m_jimple.newNewArrayExpr(atype.baseType, size);
        else {
            ArrayType to_create = ArrayType.v(atype.baseType, atype.numDimensions-1);
            return m_jimple.newNewArrayExpr(to_create, size);
        }
    }

    public void assign(Value lhs, Value rhs)
    {
        Unit u = m_jimple.newAssignStmt(lhs, rhs);
        m_assembler.add(u);
    }

    public Unit getLastUnitCreated(){ return m_assembler.getLastUnitCreated(); }
    public void gotoLabel(String label2) { m_assembler.addGoto(label2); }

    public void makeVoidCtor()
    {
        startMethod("<init>", VoidType.v());
        SootClass super_soot_class = m_currClass.getSuperclass();
        Local thisref = refThis();
        pushMethod(super_soot_class.getName(), "<init>", VoidType.v());
        invokeMethodNoRet(thisref);
        returnVoid();
        endMethod();
    }

    public UnitAssembler getAssembler() { return m_assembler; }

    Local lengthof(Value array) {
        Value rhs = m_jimple.newLengthExpr(array);
        Local lhs = m_jimple.newLocal(getLocalName(), IntType.v());
        Unit u = m_jimple.newAssignStmt(lhs, rhs);
        m_assembler.add(u);
        return lhs;
    }

    Local indexArray(Local base, Value i) {
        Value rhs = m_jimple.newArrayRef(base, i);
        Type type = base.getType();
        if(type instanceof ArrayType == false)
            throw new RuntimeException("How do we handle this case?");
        ArrayType atype = (ArrayType) type;

        Local lhs;
        if(atype.numDimensions == 1){
            lhs = m_jimple.newLocal(getLocalName(), atype.baseType);
        } else {
            Type lhs_type = ArrayType.v(atype.baseType, atype.numDimensions-1);
            lhs = m_jimple.newLocal(getLocalName(), lhs_type);
        }
        m_assembler.add( m_jimple.newAssignStmt(lhs, rhs) );
        return lhs;
    }

    public void assignArray(Local base, Value i, Local value)
    {
        Value lhs = m_jimple.newArrayRef(base, i);
        m_assembler.add( m_jimple.newAssignStmt(lhs, value) );
    }

    void plus(Local i, int add_value)
    {
        Value rhs = m_jimple.newAddExpr(i, IntConstant.v(add_value));
        m_assembler.add( m_jimple.newAssignStmt(i, rhs) );
    }

    void plus(Local i, Value add_value)
    {
        Value rhs = m_jimple.newAddExpr(i, add_value);
        m_assembler.add( m_jimple.newAssignStmt(i, rhs) );
    }

    void mult(Local lhs, Value mult_value)
    {
        Value rhs = m_jimple.newMulExpr(lhs, mult_value);
        m_assembler.add( m_jimple.newAssignStmt(lhs, rhs) );
    }

    void noOp() { m_assembler.add( m_jimple.newNopStmt() ); }

    void assignElementToArray(Local base, Value rhs, Value i)
    {
        Value lhs = m_jimple.newArrayRef(base, i);
        m_assembler.add( m_jimple.newAssignStmt(lhs, rhs) );
    }

    public void println(String message)
    {
        Type system = RefType.v("java.lang.System");
        Local out = refStaticField(system, "out");
        Type string = RefType.v("java.lang.String");
        pushMethod(out, "println", VoidType.v(), string);
        invokeMethodNoRet(out, StringConstant.v(message));
    }

    void println(Value number)
    {
        Type system = RefType.v("java.lang.System");
        Local out = refStaticField(system, "out");
        pushMethod(out, "println", VoidType.v(), IntType.v());
        invokeMethodNoRet(out, number);
    }

    void printlnLong(Value number)
    {
        Type system = RefType.v("java.lang.System");
        Local out = refStaticField(system, "out");
        pushMethod(out, "println", VoidType.v(), LongType.v());
        invokeMethodNoRet(out, number);
    }

    SootClass getSootClass() { return m_currClass; }

    Local classConstant(Type type)
    {
        String class_name = convertToConstant(type);
        Value curr = ClassConstant.v(class_name);
        Local ret = m_jimple.newLocal(getLocalName(), curr.getType());
        Unit u = m_jimple.newAssignStmt(ret, curr);
        m_assembler.add(u);
        return ret;
    }

    private String convertToConstant(Type type)
    {
        if(type instanceof ArrayType){
            ArrayType array_type = (ArrayType) type;
            String prefix = "";
            int dims = array_type.numDimensions;
            for(int i = 0; i < dims; ++i){
                prefix += "[";
            }
            String base_string = convertToConstant(array_type.baseType);
            if(array_type.baseType instanceof RefType){
                return prefix + "L" + base_string + ";";
            } else {
                return prefix + base_string;
            }
        } else if(type instanceof RefType){
            RefType ref_type = (RefType) type;
            return ref_type.getSootClass().getName().replace(".", "/");
        } else if(type instanceof PrimType){
            if      ( type.equals(BooleanType.v () ) ) return "Z";
            else if ( type.equals(ByteType.v    () ) ) return "B";
            else if ( type.equals(ShortType.v   () ) ) return "S";
            else if ( type.equals(CharType.v    () ) ) return "C";
            else if ( type.equals(IntType.v     () ) ) return "I";
            else if ( type.equals(LongType.v    () ) ) return "J";
            else if ( type.equals(FloatType.v   () ) ) return "F";
            else if ( type.equals(DoubleType.v  () ) ) return "D";

        }

        throw new RuntimeException("please report bug in BytecodeLanguage.convertToConstant");
    }

    private SootField getFieldByName(SootClass base_class, String field_name)
    {
        List<String> queue = new LinkedList<String>();
        queue.add(base_class.getName());
        while(queue.isEmpty() == false){
            String curr_class = queue.get(0);
            queue.remove(0);
            SootClass soot_class = Scene.v().getSootClass(curr_class);
            if(soot_class.declaresFieldByName(field_name)){
                return soot_class.getFieldByName(field_name);
            }
            if(soot_class.hasSuperclass()){
                queue.add(soot_class.getSuperclass().getName());
            }
            if(soot_class.hasOuterClass()){
                queue.add(soot_class.getOuterClass().getName());
            }
        }
        throw new RuntimeException("cannot find field: "+field_name+" in "+base_class.getName());
    }
}
