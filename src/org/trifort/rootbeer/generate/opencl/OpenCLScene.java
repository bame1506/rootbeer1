/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.generate.opencl;


import soot.jimple.NewExpr                 ;
import soot.rbclassload.MethodSignatureUtil;
import java.io.BufferedReader              ;
import java.io.FileReader                  ;
import java.io.FileWriter                  ;
import java.io.IOException                 ;
import java.io.PrintWriter                 ;
import java.util.ArrayList                 ;
import java.util.HashMap                   ;
import java.util.HashSet                   ;
import java.util.Iterator                  ;
import java.util.LinkedHashMap             ;
import java.util.LinkedHashSet             ;
import java.util.List                      ;
import java.util.Map                       ;
import java.util.Set                       ;
import java.util.Arrays                    ;
                                                                        ;
import org.trifort.rootbeer.configuration.Configuration                 ;
import org.trifort.rootbeer.configuration.RootbeerPaths                 ;
import org.trifort.rootbeer.entry.ForcedFields                          ;
import org.trifort.rootbeer.entry.CompilerSetup                         ;
import org.trifort.rootbeer.generate.bytecode.Constants                 ;
import org.trifort.rootbeer.generate.bytecode.MethodCodeSegment         ;
import org.trifort.rootbeer.generate.bytecode.ReadOnlyTypes             ;
import org.trifort.rootbeer.generate.opencl.fields.CompositeField       ;
import org.trifort.rootbeer.generate.opencl.fields.CompositeFieldFactory;
import org.trifort.rootbeer.generate.opencl.fields.FieldCodeGeneration  ;
import org.trifort.rootbeer.generate.opencl.fields.FieldTypeSwitch      ;
import org.trifort.rootbeer.generate.opencl.fields.OffsetCalculator     ;
import org.trifort.rootbeer.generate.opencl.fields.OpenCLField          ;
import org.trifort.rootbeer.generate.opencl.tweaks.CompileResult        ;
import org.trifort.rootbeer.generate.opencl.tweaks.CudaTweaks           ;
import org.trifort.rootbeer.generate.opencl.tweaks.Tweaks               ;
import org.trifort.rootbeer.util.ReadFile                               ;
import org.trifort.rootbeer.util.ResourceReader                         ;

import soot.Scene                          ;
import soot.SootClass                      ;
import soot.SootMethod                     ;
import soot.Local                          ;
import soot.Type                           ;
import soot.ArrayType                      ;
import soot.SootField                      ;
import soot.VoidType                       ;
import soot.rbclassload.FieldSignatureUtil ;
import soot.rbclassload.NumberedType       ;
import soot.rbclassload.RootbeerClassLoader;


/**
 * Manual Singleton (need to set and release instance manually) which
 * increments an ID when reinitialized.
 * The instance is set in compiler/Transform2.java
 */
public class OpenCLScene
{
    private static OpenCLScene              m_instance            ;
    private static int                      m_curentIdent         ;
    private final Map<String, OpenCLClass>  m_classes             ;
    private final Set<OpenCLArrayType>      m_arrayTypes          ;
    private final MethodHierarchies         m_methodHierarchies   ;
    private boolean                         m_usesGarbageCollector; /* by default: false */
    private SootClass                       m_rootSootClass       ;
    private int                             m_endOfStatics        ;
    private ReadOnlyTypes                   m_readOnlyTypes       ;
    private final List<SootMethod>          m_methods             ;
    private final Set<OpenCLInstanceof>     m_instanceOfs         ;
    private List<CompositeField>            m_compositeFields     ;
    private final ClassConstantNumbers      m_constantNumbers     ;
    private final FieldCodeGeneration       m_fieldCodeGeneration ;
    private final Configuration             m_configuration       ;

    static { m_curentIdent = 0; }
    public OpenCLScene( final Configuration configuration )
    {
        m_configuration        = configuration;
        m_classes              = new LinkedHashMap<String, OpenCLClass>();
        m_arrayTypes           = new LinkedHashSet<OpenCLArrayType>    ();
        m_methodHierarchies    = new MethodHierarchies                 ();
        m_methods              = new ArrayList<SootMethod>             ();
        m_instanceOfs          = new HashSet<OpenCLInstanceof>         ();
        m_constantNumbers      = new ClassConstantNumbers              ();
        m_fieldCodeGeneration  = new FieldCodeGeneration               ();
        /* some dependent classes used by loadTypes need the singleton to be set already! */
        setInstance( this );
        loadTypes();
    }

    public static OpenCLScene v()
    {
        if ( m_instance == null )
            throw new NullPointerException( "setInstance needs to be called before first vall to v()!" );
        return m_instance;
    }
    public static void setInstance( OpenCLScene scene ){ m_instance = scene; }

    /* @todo why is this necessary ? */
    public static void releaseV(){
        m_instance = null;
        m_curentIdent++;
    }
    public String        getIdent(){ return "" + m_curentIdent                ; }
    public static String  getUuid(){ return "ab850b60f96d11de8a390800200c9a66"; }
    public int    getEndOfStatics(){ return m_endOfStatics                    ; }

    public static int getClassType( final SootClass soot_class ){
        return RootbeerClassLoader.v().getClassNumber(soot_class);
    }

    public void addMethod( final SootMethod soot_method )
    {
        SootClass soot_class = soot_method.getDeclaringClass();

        OpenCLClass ocl_class = getOpenCLClass(soot_class);
        ocl_class.addMethod(new OpenCLMethod(soot_method, soot_class));

        //add the method
        m_methodHierarchies.addMethod(soot_method);
        m_methods.add(soot_method);
    }

    public List<SootMethod> getMethods(){ return m_methods; }

    public void addInstanceof( final Type type )
    {
        OpenCLInstanceof to_add = new OpenCLInstanceof(type);
        if ( ! m_instanceOfs.contains(to_add) )
            m_instanceOfs.add( to_add );
    }

    public OpenCLClass getOpenCLClass( SootClass soot_class )
    {
        String class_name = soot_class.getName();
        if ( m_classes.containsKey(class_name) )
            return m_classes.get(class_name);
        else {
            OpenCLClass ocl_class = new OpenCLClass(soot_class);
            m_classes.put(class_name, ocl_class);
            return ocl_class;
        }
    }

    private String getRuntimeBasicBlockClassName()
    {
        SootClass soot_class = m_rootSootClass;
        OpenCLClass ocl_class = getOpenCLClass(soot_class);
        return ocl_class.getName();
    }

    private static String readCudaCodeFromFile()
    {
        try
        {
            BufferedReader reader = new BufferedReader( new FileReader("generated.cu") );
            String ret = "";
            String temp = null;
            do
            {
                temp = reader.readLine();
                ret += temp + "\n";
            } while ( temp != null );
            return ret;
        } catch ( Exception ex ) {
            throw new RuntimeException();
        }
    }

    public void setUsingGarbageCollector(){ m_usesGarbageCollector = true; }
    public boolean getUsingGarbageCollector(){ return m_usesGarbageCollector; }

    /**
     * Writes out the found and numbered types to ~/.rootbeer/types
     * e.g.
     *
     *   19658 java.lang.AssertionError
     *   13642 java.lang.Error
     *   11879 java.lang.StackTraceElement[]
     *   9292 org.trifort.rootbeer.runtime.Sentinal
     *   9259 org.trifort.rootbeer.runtime.RootbeerGpu
     *   9112 org.trifort.rootbeer.runtimegpu.GpuException
     *   3806 java.lang.StackTraceElement
     *   3219 java.lang.Throwable
     *   3213 long[]
     *   3210 CountKernel
     *   3186 org.trifort.rootbeer.runtime.Kernel
     *   2818 java.util.List
     *   2817 java.util.Collection
     *   2816 java.lang.Iterable
     *   2454 java.io.Serializable
     *   6 int
     *   2 boolean
     *   1 java.lang.Object
     *
     * The numbering is done in the RootbeerClassLoader fork of Soot seemingly
     * with a increasing counter, although it's weird that there are to be
     * 19658 and more different types.
     *
     * @todo Is it necessary to write out this file or is it just debug output?
     */
    private void writeTypesToFile( final List<NumberedType> types )
    {
        try
        {
            PrintWriter writer = new PrintWriter( RootbeerPaths.v().getTypeFile() );
            for ( NumberedType type : types )
                writer.println( type.getNumber() + " " + type.getType().toString() );
            writer.flush();
            writer.close();
        } catch(Exception ex){
            ex.printStackTrace();
        }
    }

    public static int getOutOfMemoryNumber()
    {
        SootClass soot_class = Scene.v().getSootClass( "java.lang.OutOfMemoryError" );
        return RootbeerClassLoader.v().getClassNumber( soot_class );
    }

    private void loadTypes()
    {
        Set<String> methods = RootbeerClassLoader.v().getDfsInfo().getMethods();
        MethodSignatureUtil util = new MethodSignatureUtil();
        for ( final String method_sig : methods )
        {
            util.parse(method_sig);
            SootMethod method = util.getSootMethod();
            addMethod(method);
        }
        CompilerSetup compiler_setup = new CompilerSetup();
        for( final String extra_method : compiler_setup.getExtraMethods() )
        {
            util.parse(extra_method);
            addMethod(util.getSootMethod());
        }

        Set<SootField> fields = RootbeerClassLoader.v().getDfsInfo().getFields();
        for ( final SootField field : fields )
        {
            final SootClass soot_class = field.getDeclaringClass();
            getOpenCLClass( soot_class ).addField( new OpenCLField( field, soot_class) );
        }

        FieldSignatureUtil field_util = new FieldSignatureUtil();
        ForcedFields forced_fields = new ForcedFields();
        for ( final String field_sig : forced_fields.get() )
        {
            field_util.parse(field_sig);
            final SootField field = field_util.getSootField();
            final SootClass soot_class = field.getDeclaringClass();
            getOpenCLClass( soot_class ).addField( new OpenCLField( field, soot_class) );
        }

        Set<ArrayType> array_types = RootbeerClassLoader.v().getDfsInfo().getArrayTypes();
        for ( final ArrayType array_type : array_types )
        {
            final OpenCLArrayType ocl_array_type = new OpenCLArrayType(array_type);
            if ( ! m_arrayTypes.contains( ocl_array_type ) )
                m_arrayTypes.add( ocl_array_type );
        }
        for ( final ArrayType array_type : compiler_setup.getExtraArrayTypes() )
        {
            final OpenCLArrayType ocl_array_type = new OpenCLArrayType( array_type );
            if ( ! m_arrayTypes.contains( ocl_array_type ) )
                m_arrayTypes.add( ocl_array_type );
        }

        Set<Type> instanceofs = RootbeerClassLoader.v().getDfsInfo().getInstanceOfs();
        for ( final Type type : instanceofs )
        {
            final OpenCLInstanceof to_add = new OpenCLInstanceof(type);
            if ( ! m_instanceOfs.contains(to_add) )
                m_instanceOfs.add( to_add );
        }

        m_compositeFields = CompositeFieldFactory.getCompositeFields( this, m_classes );
    }

    /**
     * For manually provided CUDA code it returns the file as a string.
     * If not it converts the java code to CUDA using Jimple as intermediary
     *
     * @return List of Strings. 1st is CUDA code for Linux, 2nd is for Windows
     */
    private String[] makeSourceCode() throws Exception
    {
        assert( m_configuration != null );
        if ( m_configuration.isManualCuda() )
        {
            final String cuda_code = readCode(
                 m_configuration.getManualCudaFilename()
            );
            final String[] ret = new String[2];
            ret[0] = cuda_code;
            ret[1] = cuda_code;
            return ret;
        }

        /* this gets switched to true again, if a new statement is found in
         * MethodJimpleValueSwitch e.g. when parsed in OpenCLMethod's or
         * OpenCLArrayType's Value.apply */
        m_usesGarbageCollector = false;

        List<NumberedType> types = RootbeerClassLoader.v().getDfsInfo().getNumberedTypes();
        writeTypesToFile(types);

        final StringBuilder unix_code    = new StringBuilder();
        final StringBuilder windows_code = new StringBuilder();

        final String method_protos       = methodPrototypesString();
        final String gc_string           = garbageCollectorString();
        final String bodies_string       = methodBodiesString();

        /* true/false arguments are used to differ between unix and windows */
        unix_code.append   ( headerString(true ) );
        unix_code.append   ( method_protos       );
        unix_code.append   ( gc_string           );
        unix_code.append   ( bodies_string       );
        unix_code.append   ( kernelString(true ) );

        windows_code.append( headerString(false) );
        windows_code.append( method_protos       );
        windows_code.append( gc_string           );
        windows_code.append( bodies_string       );
        windows_code.append( kernelString(false) );

        final String cuda_unix    = setupEntryPoint( unix_code    );
        final String cuda_windows = setupEntryPoint( windows_code );

        //print out code for debugging
        PrintWriter writer = new PrintWriter(new FileWriter(RootbeerPaths.v().getRootbeerHome()+"generated_unix.cu"));
        writer.println(cuda_unix);
        writer.flush();
        writer.close();

        //print out code for debugging
        writer = new PrintWriter(new FileWriter(RootbeerPaths.v().getRootbeerHome()+"generated_windows.cu"));
        writer.println(cuda_windows);
        writer.flush();
        writer.close();

        NameMangling.v().writeTypesToFile();

        String[] ret = new String[2];
        ret[0] = cuda_unix;
        ret[1] = cuda_windows;
        return ret;
    }

    private static String readCode( final String filename )
    {
        final ReadFile reader = new ReadFile( filename );
        try {
            return reader.read();
        } catch( Exception ex )
        {
            ex.printStackTrace( System.out );
            throw new RuntimeException( ex );
        }
    }

    /**
     * Replaces place holders in the form of %%placeholder_name%% with
     * valid values. E.g. %%shared_mem_size%%. These are used in the CUDA
     * templates provided by Rootbeer
     * @see rootbeer/generate/opencl/CudaKernel.c
     *
     * It also replaces %%java_lang_*_TypeNumber%% with their respective
     * type numbers as returned by the Soot RoobteerClassLoader fork.
     * @see rootbeer/generate/opencl/GarbageCollector.c
     */
    private String setupEntryPoint( StringBuilder builder )
    {
        String cuda_code   = builder.toString();
        String replacement = getRuntimeBasicBlockClassName() + "_gpuMethod" +
                             NameMangling.v().mangle( VoidType.v() );
        //class names can have $ in them, make them regex safe
        replacement = replacement.replace( "$", "\\$" );
        cuda_code   = cuda_code.replaceAll( "%%invoke_run%%", replacement );

        assert( m_configuration != null );
        int size = m_configuration.getSharedMemSize();
        String size_str = ""+size;
        cuda_code = cuda_code.replaceAll("%%shared_mem_size%%", size_str);

        cuda_code = cuda_code.replaceAll( "%%MallocAlignZeroBits%%", ""+Constants.MallocAlignZeroBits );
        cuda_code = cuda_code.replaceAll( "%%MallocAlignBytes%%"   , ""+Constants.MallocAlignBytes    );

        boolean exceptions = m_configuration.getExceptions();
        String exceptions_str;
        if ( exceptions )
            exceptions_str = "" + 1;
        else
            exceptions_str = "" + 0;
        cuda_code = cuda_code.replaceAll( "%%using_exceptions%%", exceptions_str );

        for ( String typeString : Arrays.asList(
              "StringBuilder", "NullPointerException", "OutOfMemoryError",
              "String", "Integer", "Long", "Float", "Double", "Boolean" ) )
        {
            cuda_code = cuda_code.replaceAll(
                "%%java_lang_" + typeString + "_TypeNumber%%",
                "" + RootbeerClassLoader.v().getClassNumber( "java.lang." + typeString )
            );
        }
        return cuda_code;
    }

    public String[] getOpenCLCode() throws Exception
    {
        String[] source_code = makeSourceCode();
        return source_code;
    }

    public CompileResult[] getCudaCode() throws Exception
    {
        assert( m_configuration != null );
        String[] source_code = makeSourceCode();
        return new CudaTweaks().compileProgram( source_code[0], m_configuration );
    }

    private String headerString(boolean unix) throws IOException
    {
        assert( m_configuration != null );
        String defines = "";
        if ( m_configuration.getArrayChecks() )
            defines += "#define ARRAY_CHECKS\n";

        String specific_path;
        if ( unix )
            specific_path = Tweaks.v().getUnixHeaderPath();
        else
            specific_path = Tweaks.v().getWindowsHeaderPath();

        if ( specific_path == null )
            return "";

        String both_path = Tweaks.v().getBothHeaderPath();
        String both_header = "";
        if ( both_path != null )
            both_header = ResourceReader.getResource(both_path);
        String specific_header = ResourceReader.getResource(specific_path);

        String barrier_path = Tweaks.v().getBarrierPath();
        String barrier_code = "";
        if ( barrier_path != null )
            barrier_code = ResourceReader.getResource(barrier_path);

        return defines + "\n" + specific_header + "\n" + both_header + "\n" + barrier_code;
    }

    /**
     * Returns the template kernels for Rootbeer added functionalites:
     * BothNativeKernel.c and ( CudaKernel.c xor UnixNativeKernel.c )
     * as a String
     */
    private static String kernelString( final boolean unix ) throws IOException
    {
        final String kernel_path = unix ? Tweaks.v().getUnixKernelPath() :
                                          Tweaks.v().getWindowsKernelPath();
        String specific_kernel_code = ResourceReader.getResource( kernel_path );

        String both_kernel_code = "";
        String both_kernel_path = Tweaks.v().getBothKernelPath();
        if ( both_kernel_path != null )
            both_kernel_code = ResourceReader.getResource( both_kernel_path );

        return both_kernel_code + "\n" + specific_kernel_code;
    }

    private static String garbageCollectorString() throws IOException
    {
        String path = Tweaks.v().getGarbageCollectorPath();
        String ret = ResourceReader.getResource(path);
        ret = ret.replace("$$__device__$$", Tweaks.v().getDeviceFunctionQualifier());
        ret = ret.replace("$$__global$$", Tweaks.v().getGlobalAddressSpaceQualifier());
        return ret;
    }

    private String methodPrototypesString()
    {
        //using a set so duplicates get filtered out.
        Set<String> protos = new HashSet<String>();
        StringBuilder ret = new StringBuilder();

        ArrayCopyGenerate arr_generate = new ArrayCopyGenerate();
        protos.add(arr_generate.getProto());

        List<OpenCLMethod> methods = m_methodHierarchies.getMethods();
        for ( OpenCLMethod method : methods )
            protos.add(method.getMethodPrototype());
        List<OpenCLPolymorphicMethod> poly_methods = m_methodHierarchies.getPolyMorphicMethods();
        for ( OpenCLPolymorphicMethod poly_method : poly_methods )
            protos.add(poly_method.getMethodPrototypes());
        protos.add(m_fieldCodeGeneration.prototypes(m_classes));
        for ( OpenCLArrayType array_type : m_arrayTypes )
            protos.add(array_type.getPrototypes());
        for ( OpenCLInstanceof type : m_instanceOfs )
            protos.add(type.getPrototype());
        Iterator<String> iter = protos.iterator();
        while ( iter.hasNext() )
            ret.append(iter.next());
        return ret.toString();
    }

    private String methodBodiesString() throws IOException
    {
        StringBuilder ret = new StringBuilder();
        if ( m_usesGarbageCollector )
            ret.append("#define USING_GARBAGE_COLLECTOR\n");

        /* a set is used so duplicates get filtered out */
        Set<String> bodies = new HashSet<String>();
        Set<OpenCLArrayType> new_types = new ArrayCopyTypeReduction().run( m_arrayTypes, m_methodHierarchies );
        bodies.add( new ArrayCopyGenerate().get( new_types ) );

        /* add all different kinds of bodies i.e. code */
        List<OpenCLMethod> methods = m_methodHierarchies.getMethods();
        for ( OpenCLMethod method : methods )
            if(method.getSootMethod().isConcrete())
                bodies.add( method.getMethodBody() );
        List<OpenCLPolymorphicMethod> poly_methods = m_methodHierarchies.getPolyMorphicMethods();
        for ( OpenCLPolymorphicMethod poly_method : poly_methods )
            bodies.addAll( poly_method.getMethodBodies() );
        /* FieldTypeSwitch does something with the offsets ??? */
        FieldTypeSwitch type_switch = new FieldTypeSwitch();
        String field_bodies = m_fieldCodeGeneration.bodies( m_classes, type_switch );
        bodies.add( field_bodies );
        for ( OpenCLArrayType array_type : m_arrayTypes )
            bodies.add( array_type.getBodies() );
        for ( OpenCLInstanceof type : m_instanceOfs )
            bodies.add( type.getBody() );

        /* join all bodies together to one string (why not do it above? */
        ret.append( type_switch.getFunctions() );
        final Iterator<String> iter = bodies.iterator();
        while ( iter.hasNext() )
            ret.append( iter.next() );

        return ret.toString();
    }

    public OffsetCalculator getOffsetCalculator(SootClass soot_class)
    {
        List<CompositeField> composites = getCompositeFields();
        for ( CompositeField composite : composites )
        {
            List<SootClass> classes = composite.getClasses();
            if(classes.contains(soot_class))
                return new OffsetCalculator(composite);
        }
        throw new RuntimeException("Cannot find composite field for soot_class");
    }

    public void addCodeSegment( MethodCodeSegment codeSegment )
    {
        m_rootSootClass = codeSegment.getRootSootClass();
        m_readOnlyTypes = new ReadOnlyTypes( codeSegment.getRootMethod() );
        getOpenCLClass(m_rootSootClass);
    }

    public boolean isRootClass(SootClass soot_class)
    {
        return soot_class.getName().equals(m_rootSootClass.getName());
    }

    public boolean                  isArrayLocalWrittenTo(Local local){ return true  ; }
    public ReadOnlyTypes            getReadOnlyTypes     (){ return m_readOnlyTypes  ; }
    public Map<String, OpenCLClass> getClassMap          (){ return m_classes        ; }
    public List<CompositeField>     getCompositeFields   (){ return m_compositeFields; }
    public ClassConstantNumbers   getClassConstantNumbers(){ return m_constantNumbers; }
}
