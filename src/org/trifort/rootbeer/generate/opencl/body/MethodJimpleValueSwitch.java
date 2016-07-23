/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.opencl.body;


import org.trifort.rootbeer.generate.opencl.*;
import org.trifort.rootbeer.generate.opencl.fields.OpenCLField;

import soot.rbclassload.ClassConstantReader;
import soot.*;
import soot.jimple.AddExpr;
import soot.jimple.AndExpr;
import soot.jimple.ArrayRef;
import soot.jimple.BinopExpr;
import soot.jimple.CastExpr;
import soot.jimple.CaughtExceptionRef;
import soot.jimple.ClassConstant;
import soot.jimple.CmpExpr;
import soot.jimple.CmpgExpr;
import soot.jimple.CmplExpr;
import soot.jimple.DivExpr;
import soot.jimple.DoubleConstant;
import soot.jimple.DynamicInvokeExpr;
import soot.jimple.EqExpr;
import soot.jimple.FloatConstant;
import soot.jimple.GeExpr;
import soot.jimple.GtExpr;
import soot.jimple.InstanceFieldRef;
import soot.jimple.InstanceInvokeExpr;
import soot.jimple.InstanceOfExpr;
import soot.jimple.IntConstant;
import soot.jimple.InterfaceInvokeExpr;
import soot.jimple.JimpleValueSwitch;
import soot.jimple.LeExpr;
import soot.jimple.LengthExpr;
import soot.jimple.LongConstant;
import soot.jimple.LtExpr;
import soot.jimple.MulExpr;
import soot.jimple.NeExpr;
import soot.jimple.NegExpr;
import soot.jimple.NewArrayExpr;
import soot.jimple.NewExpr;
import soot.jimple.NewMultiArrayExpr;
import soot.jimple.NullConstant;
import soot.jimple.OrExpr;
import soot.jimple.ParameterRef;
import soot.jimple.RemExpr;
import soot.jimple.ShlExpr;
import soot.jimple.ShrExpr;
import soot.jimple.SpecialInvokeExpr;
import soot.jimple.StaticFieldRef;
import soot.jimple.StaticInvokeExpr;
import soot.jimple.StringConstant;
import soot.jimple.SubExpr;
import soot.jimple.ThisRef;
import soot.jimple.UshrExpr;
import soot.jimple.VirtualInvokeExpr;
import soot.jimple.XorExpr;
import soot.rbclassload.RootbeerClassLoader;


/**
 * implements JimpleValueSwitch which is used by Value.apply e.g. in
 * rootbeer/generate/opencl/OpenCLMethod.java to convert Jimple code to
 * C / CUDA statements.
 * The case* methods are called by Soot.
 */
public class MethodJimpleValueSwitch implements JimpleValueSwitch
{
    protected final StringBuilder       m_output             ;
    private boolean                     m_lhs                ;
    private boolean                     m_rhs                ;
    private boolean                     m_newCalled          ;
    private boolean                     m_caughtExceptionRef ;
    private String                      m_thisRef            ;
    private String                      m_previousLocal      ;
    private boolean                     m_checkException     ;
    private final ClassConstantReader   m_classConstantReader;

    public MethodJimpleValueSwitch( final StringBuilder output )
    {
        m_output              = output;
        m_newCalled           = false;
        m_classConstantReader = new ClassConstantReader();
        clearLhsRhs();
    }

    public boolean newHasBeenCalled(){ return m_newCalled; }
    public void resetNewCalled(){ m_newCalled = false; }

    void setLhs     (){ m_lhs = true ; m_rhs = false; }
    void setRhs     (){ m_lhs = false; m_rhs = true ; }
    void clearLhsRhs(){ m_lhs = false; m_rhs = false; }

    boolean isLhs()
    {
        if ( ! m_lhs && ! m_rhs )
            throw new IllegalStateException( "Lhs/Rhs in invalid state. e.g. after calling clearLhsRhs()" );
        return m_lhs;
    }

    /**
     * Parses a Jimple binary expression and converts it to C source code
     * calls appending the result to m_output StringBuilder.
     */
    private void writeBinOpExpr( final BinopExpr arg0 )
    {
        String symbol = arg0.getSymbol().trim();
        if ( needDoubleMod(arg0 ) )
        {
            m_output.append("org_trifort_modulus(");
            arg0.getOp1().apply(this);
            m_output.append(", ");
            arg0.getOp2().apply(this);
            m_output.append(")");
        }
        else if ( symbol.equals("cmp") )
        {
            m_output.append("org_trifort_cmp(");
            arg0.getOp1().apply(this);
            m_output.append(", ");
            arg0.getOp2().apply(this);
            m_output.append(")");
        }
        else if ( symbol.equals("cmpl") )
        {
            m_output.append("org_trifort_cmpl((double)");
            arg0.getOp1().apply(this);
            m_output.append(", (double)");
            arg0.getOp2().apply(this);
            m_output.append(")");
        }
        else if ( symbol.equals("cmpg") )
        {
            m_output.append("org_trifort_cmpg((double)");
            arg0.getOp1().apply(this);
            m_output.append(", (double)");
            arg0.getOp2().apply(this);
            m_output.append(")");
        }
        else
        {
            arg0.getOp1().apply(this);
            m_output.append(" "+symbol+" ");
            arg0.getOp2().apply(this);
            m_output.append(" ");
        }
    }

    /**
     * Tests if the Jimple binary expression is a float modulo operation
     * @todo It seems to me, that this returns false for things like
     *       3 % 3.1 or 5.4f % 2l i.e. those will be calculated using integer
     *       modulo. I don't think this should be !!! See these tests:
     *
     *         final double b1 = 4    % 3    ;  // 1.0
     *         final double b2 = 4    % 3.1  ;  // 0.8999999999999999
     *         final double b3 = 4.1  % 3    ;  // 1.0999999999999996
     *         final double b4 = 4.   % 3.1  ;  // 0.8999999999999999
     *         final double b5 = 4.   % 3.1f ;  // 0.9000000953674316
     *         final double b6 = 4.1f % 3    ;  // 1.0999999046325684
     *         final double b7 = 4.1  % 3.   ;  // 1.0999999999999996
     *         final double b8 = 4.1f % 3.   ;  // 1.0999999046325684
     */
    private boolean needDoubleMod
    (
        final BinopExpr arg0
    )
    {
        /* should be equivalent to this:
        return arg0.getSymbol().trim().equals( "%" ) &&
           ( arg0.getOp1().getType().equals( DoubleType.v() ) ||
             arg0.getOp1().getType().equals( FloatType.v()  )    ) &&
           ( arg0.getOp2().getType().equals( DoubleType.v() ) ||
             arg0.getOp2().getType().equals( FloatType.v()  )    );
        */
        if ( ! arg0.getSymbol().trim().equals( "%" ) )
            return false;

        if ( ! arg0.getOp1().getType().equals( DoubleType.v() ) &&
             ! arg0.getOp1().getType().equals( FloatType.v()  ) ) /* e.g. if int */
        {
            return false;
        }

        if ( ! arg0.getOp2().getType().equals( DoubleType.v() ) &&
             ! arg0.getOp2().getType().equals( FloatType.v()  ) )
        {
            return false;
        }

        return true;

        /* I think this is how it should be like however (i.e. upcasting
         * if one operand is a float:
        return arg0.getSymbol().trim().equals( "%" ) && (
            arg0.getOp1().getType().equals( DoubleType.v() ||
            arg0.getOp1().getType().equals( FloatType.v()  ||
            arg0.getOp2().getType().equals( DoubleType.v() ||
            arg0.getOp2().getType().equals( FloatType.v()     );
        */
    }
    public void caseAddExpr (AddExpr  arg0) { writeBinOpExpr(arg0); }
    public void caseAndExpr (AndExpr  arg0) { writeBinOpExpr(arg0); }
    public void caseCmpExpr (CmpExpr  arg0) { writeBinOpExpr(arg0); }
    public void caseCmpgExpr(CmpgExpr arg0) { writeBinOpExpr(arg0); }
    public void caseCmplExpr(CmplExpr arg0) { writeBinOpExpr(arg0); }
    public void caseDivExpr (DivExpr  arg0) { writeBinOpExpr(arg0); }
    public void caseEqExpr  (EqExpr   arg0) { writeBinOpExpr(arg0); }
    public void caseNeExpr  (NeExpr   arg0) { writeBinOpExpr(arg0); }
    public void caseGeExpr  (GeExpr   arg0) { writeBinOpExpr(arg0); }
    public void caseGtExpr  (GtExpr   arg0) { writeBinOpExpr(arg0); }
    public void caseLeExpr  (LeExpr   arg0) { writeBinOpExpr(arg0); }
    public void caseLtExpr  (LtExpr   arg0) { writeBinOpExpr(arg0); }
    public void caseMulExpr (MulExpr  arg0) { writeBinOpExpr(arg0); }
    public void caseOrExpr  (OrExpr   arg0) { writeBinOpExpr(arg0); }
    public void caseRemExpr (RemExpr  arg0) { writeBinOpExpr(arg0); }
    public void caseShlExpr (ShlExpr  arg0) { writeBinOpExpr(arg0); }
    public void caseShrExpr (ShrExpr  arg0) { writeBinOpExpr(arg0); }
    public void caseSubExpr( SubExpr arg0 ) { writeBinOpExpr(arg0); }
    public void caseXorExpr( XorExpr arg0 ) { writeBinOpExpr(arg0); }

    public void caseUshrExpr( UshrExpr arg0 )
    {
        m_output.append("(");
        arg0.getOp1().apply(this);
        m_output.append(" >> ");
        arg0.getOp2().apply(this);
        m_output.append(" ) & ");

        OpenCLType lhs_ocl_type = new OpenCLType(arg0.getOp1().getType());
        OpenCLType rhs_ocl_type = new OpenCLType(arg0.getOp2().getType());
        int max_size = lhs_ocl_type.getSize();
        if(rhs_ocl_type.getSize() > max_size){
            max_size = rhs_ocl_type.getSize();
        }

        String mask = "";
        switch(max_size)
        {
            case 1: mask = "0x7f"               ; break;
            case 2: mask = "0x7fff"             ; break;
            case 4: mask = "0x7fffffff"         ; break;
            case 8: mask = "0x7fffffffffffffffL"; break;
        }
        m_output.append(mask);
    }

    private void caseInstanceInvokeExpr( InstanceInvokeExpr arg0 )
    {
        SootMethod soot_method = arg0.getMethod();
        SootClass soot_class = soot_method.getDeclaringClass();
        OpenCLMethod ocl_method = new OpenCLMethod(soot_method, soot_class);

        m_output.append(ocl_method.getInstanceInvokeString(arg0));
        setCheckException();
    }

    public void caseInterfaceInvokeExpr( InterfaceInvokeExpr arg0 )
    {
        caseInstanceInvokeExpr(arg0);
        setCheckException();
    }

    public void caseSpecialInvokeExpr(SpecialInvokeExpr arg0)
    {
        SootMethod soot_method = arg0.getMethod();
        SootClass soot_class = soot_method.getDeclaringClass();
        OpenCLMethod ocl_method = new OpenCLMethod(soot_method, soot_class);
        m_output.append(ocl_method.getInstanceInvokeString(arg0));
        setCheckException();
    }

    public void caseStaticInvokeExpr(StaticInvokeExpr arg0)
    {
        SootMethod soot_method = arg0.getMethod();
        SootClass soot_class = soot_method.getDeclaringClass();
        OpenCLMethod ocl_method = new OpenCLMethod(soot_method, soot_class);
        m_output.append(ocl_method.getStaticInvokeString(arg0));
        setCheckException();
    }

    public void caseVirtualInvokeExpr(VirtualInvokeExpr arg0) { caseInstanceInvokeExpr(arg0); }

    public void caseCastExpr(CastExpr arg0)
    {
        Type cast_type = arg0.getCastType();
        OpenCLType ocl_type = new OpenCLType(cast_type);
        m_output.append("("+ocl_type.getCudaTypeString()+") ");
        Value rhs = arg0.getOp();
        rhs.apply(this);
    }

    public void caseInstanceOfExpr(InstanceOfExpr arg0)
    {
        OpenCLScene.v().addInstanceof(arg0.getCheckType());
        OpenCLInstanceof instance_of = new OpenCLInstanceof(arg0.getCheckType());
        m_output.append(instance_of.invokeExpr(arg0));
    }

    public void caseNewArrayExpr(NewArrayExpr arg0)
    {
        OpenCLScene.v().setUsingGarbageCollector();
        OpenCLArrayType array_type = new OpenCLArrayType((ArrayType) arg0.getType());
        m_output.append(array_type.invokeNewArrayExpr(arg0));
        m_newCalled = true;
    }

    public void caseNewMultiArrayExpr(NewMultiArrayExpr arg0)
    {
        OpenCLScene.v().setUsingGarbageCollector();
        OpenCLArrayType array_type = new OpenCLArrayType((ArrayType) arg0.getType());
        m_output.append(array_type.invokeNewMultiArrayExpr(arg0));
        m_newCalled = true;
    }

    public void caseNewExpr(NewExpr arg0)
    {
        OpenCLScene.v().setUsingGarbageCollector();
        m_output.append(" -1 ");
    }

    public void caseLengthExpr(LengthExpr arg0)
    {
        Value op = arg0.getOp();
        m_output.append("org_trifort_array_length(");
        op.apply(this);
        m_output.append(", exception)");
        setCheckException();
    }

    public void caseNegExpr(NegExpr arg0)
    {
        Value op = arg0.getOp();
        m_output.append("-");
        op.apply(this);
    }

    public void defaultCase(Object arg0)
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    public void caseLocal(Local arg0)
    {
        m_output.append(" "+arg0.getName()+" ");
        m_previousLocal = arg0.getName();
    }

    public void caseDoubleConstant  (DoubleConstant arg0) { m_output.append(" "+replaceNumber(arg0.toString())+" "); }
    public void caseFloatConstant   (FloatConstant  arg0) { m_output.append(" "+replaceNumber(arg0.toString())+" "); }
    public void caseIntConstant     (IntConstant    arg0) { m_output.append(" "+replaceNumber(arg0.toString())+" "); }
    public void caseLongConstant    (LongConstant   arg0) { m_output.append(" "+replaceNumber(arg0.toString())+" "); }
    public void caseNullConstant    (NullConstant   arg0) { m_output.append(" -1 "); }

    private String replaceNumber(String number)
    {
        if(number.equals("#Infinity"))
            return "INFINITY";
        if(number.equals("#-Infinity"))
            return "-INFINITY";
        if(number.equals("#NaN"))
            return "NAN";
        return number;
    }

    public void caseStringConstant(StringConstant arg0)
    {
        m_output.append(" org_trifort_string_constant((char *) "+arg0.toString()+", exception) ");
    }

    public void caseClassConstant(ClassConstant arg0)
    {
        String value = arg0.getValue();
        Type type = m_classConstantReader.stringToType(value);
        int num = OpenCLScene.v().getClassConstantNumbers().get(type);
        m_output.append("org_trifort_classConstant("+num+")");
    }

    public void caseArrayRef(ArrayRef arg0)
    {
        OpenCLArrayType array = new OpenCLArrayType((ArrayType) arg0.getBase().getType());
        if(isLhs()){
            m_output.append(array.getArrayRefSetter(arg0));
            setCheckException();
        } else {
            m_output.append(array.getArrayRefGetter(arg0));
            setCheckException();
        }
    }

    public void caseStaticFieldRef(StaticFieldRef arg0)
    {
        SootField field = arg0.getField();
        OpenCLField ocl_field = new OpenCLField(arg0.getField(), field.getDeclaringClass());
        if(isLhs()){
            m_output.append(ocl_field.getStaticSetterInvoke());
        } else {
            m_output.append(ocl_field.getStaticGetterInvoke());
        }
    }

    public void caseInstanceFieldRef(InstanceFieldRef arg0)
    {
        Value base = arg0.getBase();
        if(base instanceof Local == false)
            throw new UnsupportedOperationException("How do I handle base is not a local?");
        Local local = (Local) base;
        Type type = local.getType();
        if(type instanceof RefType == false)
            throw new UnsupportedOperationException("How do I handle type is not a ref type?");
        RefType ref = (RefType) type;
        OpenCLField ocl_field = new OpenCLField(arg0.getField(), ref.getSootClass());
        if(isLhs()){
            m_output.append(ocl_field.getInstanceSetterInvoke(arg0.getBase()));
        } else {
            m_output.append(ocl_field.getInstanceGetterInvoke(arg0.getBase()));
        }
        setCheckException();
    }

    public void caseParameterRef(ParameterRef arg0)
    {
        m_output.append(" parameter"+Integer.toString(arg0.getIndex())+" ");
    }

    public void caseCaughtExceptionRef(CaughtExceptionRef arg0)
    {
        m_output.append(" *exception ");
        m_caughtExceptionRef = true;
    }

    public void caseThisRef(ThisRef arg0)
    {
        m_output.append(" thisref ");
        m_thisRef = m_previousLocal;
    }

    public void caseDynamicInvokeExpr(DynamicInvokeExpr die)
    {
        throw new UnsupportedOperationException("Not supported yet.");
    }

    void reset()
    {
        m_caughtExceptionRef = false;
        m_checkException     = false;
    }

    boolean hasCaughtExceptionRef    () { return m_caughtExceptionRef; }
    public        String getThisRef  () { return m_thisRef           ; }
    public  boolean getCheckException() { return m_checkException    ; }
    private void  setCheckException  () { m_checkException = true    ; }
}
