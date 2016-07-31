/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.entry;


import java.nio.file.Files;
import static java.nio.file.StandardCopyOption.REPLACE_EXISTING;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.zip.CRC32;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

import org.trifort.rootbeer.compiler.Transform2;
import org.trifort.rootbeer.configuration.Configuration;
import org.trifort.rootbeer.configuration.RootbeerPaths;
import org.trifort.rootbeer.generate.opencl.tweaks.CudaTweaks;
import org.trifort.rootbeer.generate.opencl.tweaks.NativeCpuTweaks;
import org.trifort.rootbeer.generate.opencl.tweaks.Tweaks;
import org.trifort.rootbeer.util.CurrJarName;
import org.trifort.rootbeer.util.DeleteFolder;
import org.trifort.rootbeer.util.JarEntryHelp;
import org.trifort.rootbeer.util.JimpleWriter;

import pack.Pack;
import soot.*;
import soot.options.Options;
import soot.rbclassload.DfsInfo;
import soot.rbclassload.ListClassTester;
import soot.rbclassload.ListMethodTester;
import soot.rbclassload.MethodTester;
import soot.rbclassload.RootbeerClassLoader;
import soot.util.JasminOutputStream;


public final class RootbeerCompiler
{
    private final static boolean debugging = false;  /* activates some debug output */
    private       String        m_provider            ;
    private       boolean       m_enableClassRemapping;
    /* functional returning true for classes which Rootbeer shall parse
     * i.e. Kernel implementations with gpuMethod */
    private       MethodTester  m_entryDetector       ;
    /* These will be included automatically for classes implementing the Kernel interface */
    private final Set<String>   m_runtimePackages     ;
    private       boolean       m_packFatJar          ;
    private final String        m_JarWithoutRootbeer  ;
    private final Configuration m_configuration       ;

    public RootbeerCompiler( final Configuration configuration )
    {
        clearOutputFolders();

        if ( configuration.getMode() == Configuration.MODE_GPU )
            Tweaks.setInstance( new CudaTweaks() );
        else // NEMU or JEMU
            Tweaks.setInstance( new NativeCpuTweaks() );

        m_configuration        = configuration;
        m_enableClassRemapping = true;
        m_packFatJar           = true;
        m_runtimePackages      = new HashSet<String>( Arrays.asList(
                                     "com.lmax.disruptor."                ,
                                     "org.trifort.rootbeer.compiler."     ,
                                     "org.trifort.rootbeer.configuration.",
                                     "org.trifort.rootbeer.entry."        ,
                                     "org.trifort.rootbeer.generate."     ,
                                     "org.trifort.rootbeer.runtime."      ,
                                     "org.trifort.rootbeer.runtime2."     ,
                                     "org.trifort.rootbeer.util."         ,
                                     "org.trifort.rootbeer.test."
                                 ) );
        final String outFolder = RootbeerPaths.v().getOutputJarFolder() + File.separator;
        JarEntryHelp.mkdir( outFolder );
        m_JarWithoutRootbeer   = outFolder + "partial-ret.jar";
    }

    public void disableClassRemapping(){ m_enableClassRemapping = false; }
    public void dontPackFatJar       (){ m_packFatJar           = false; }

    /**
     * Sets default options and loads selected classes and methods
     */
    private void setupSoot
    (
        final String jar_filename,
        final String rootbeer_jar,
        final boolean runtests
    )
    {
        /* RootbeerClassLoader is something added in the forked Soot.
         * It is also undocumented which makes everything a hassle */
        RootbeerClassLoader.v().setUserJar( jar_filename );
        //extractJar(jar_filename);

        final List<String> proc_dir = new ArrayList<String>();
        proc_dir.add( jar_filename );

        Options.v().set_allow_phantom_refs( true );
        Options.v().set_rbclassload( true );
        Options.v().set_prepend_classpath( true );
        Options.v().set_process_dir( proc_dir );
        if ( m_enableClassRemapping )
            Options.v().set_rbclassload_buildcg( true ); /* ??? */
        if ( ! rootbeer_jar.equals("") )
            Options.v().set_soot_classpath( rootbeer_jar );

        //Options.v().set_rbcl_remap_all(m_configuration.getRemapAll());
        Options.v().set_rbcl_remap_all(false);
        Options.v().set_rbcl_remap_prefix("org.trifort.rootbeer.runtime.remap.");

        RootbeerClassLoader.v().addEntryMethodTester( m_entryDetector );

        /* some known libraries which shouldn't get searched for gpuMethods
         * in Kernel implementations, because we know that they are libraries.
         * This saves some time */
        final ListClassTester ignore_packages = new ListClassTester();
        ignore_packages.addPackage( "com.lmax.disruptor."                 );
        ignore_packages.addPackage( "org.trifort.rootbeer.compressor."    );
        ignore_packages.addPackage( "org.trifort.rootbeer.deadmethods."   );
        ignore_packages.addPackage( "org.trifort.rootbeer.compiler."      );
        ignore_packages.addPackage( "org.trifort.rootbeer.configuration." );
        ignore_packages.addPackage( "org.trifort.rootbeer.entry."         );
        ignore_packages.addPackage( "org.trifort.rootbeer.generate."      );
        ignore_packages.addPackage( "org.trifort.rootbeer.test."          );
        if ( ! runtests )
            ignore_packages.addPackage("org.trifort.rootbeer.testcases.");
        ignore_packages.addPackage( "org.trifort.rootbeer.util." );
        ignore_packages.addPackage( "pack."                      );
        ignore_packages.addPackage( "jasmin."                    );
        ignore_packages.addPackage( "soot."                      );
        ignore_packages.addPackage( "beaver."                    );
        ignore_packages.addPackage( "polyglot."                  );
        ignore_packages.addPackage( "org.antlr."                 );
        ignore_packages.addPackage( "java_cup."                  );
        ignore_packages.addPackage( "ppg."                       );
        ignore_packages.addPackage( "antlr."                     );
        ignore_packages.addPackage( "jas."                       );
        ignore_packages.addPackage( "scm."                       );
        ignore_packages.addPackage( "org.xmlpull.v1."            );
        ignore_packages.addPackage( "android.util."              );
        ignore_packages.addPackage( "android.content.res."       );
        ignore_packages.addPackage( "org.apache.commons.codec."  );
        RootbeerClassLoader.v().addDontFollowClassTester( ignore_packages );

        final ListClassTester keep_packages = new ListClassTester();
        for ( final String runtime_class : m_runtimePackages )
            keep_packages.addPackage( runtime_class );
        RootbeerClassLoader.v().addToSignaturesClassTester(keep_packages);

        RootbeerClassLoader.v().addNewInvoke("java.lang.StringBuilder");

        final ListMethodTester follow_tester = new ListMethodTester();
        follow_tester.addSignature( "<java.lang.String: void <init>()>"                       );
        follow_tester.addSignature( "<java.lang.String: void <init>(char[])>"                 );
        follow_tester.addSignature( "<java.lang.StringBuilder: void <init>()>"                );
        follow_tester.addSignature( "<java.lang.Boolean: java.lang.String toString(boolean)>" );
        follow_tester.addSignature( "<java.lang.Character: java.lang.String toString(char)>"  );
        follow_tester.addSignature( "<java.lang.Double: java.lang.String toString(double)>"   );
        follow_tester.addSignature( "<java.lang.Float: java.lang.String toString(float)>"     );
        follow_tester.addSignature( "<java.lang.Integer: java.lang.String toString(int)>"     );
        follow_tester.addSignature( "<java.lang.Long: java.lang.String toString(long)>"       );
        follow_tester.addSignature( "<org.trifort.rootbeer.runtime.Sentinal: void <init>()>"  );
        follow_tester.addSignature( "<org.trifort.rootbeer.runtimegpu.GpuException: void <init>()>");
        follow_tester.addSignature( "<org.trifort.rootbeer.runtimegpu.GpuException: org.trifort.rootbeer.runtimegpu.GpuException arrayOutOfBounds(int,int,int)>");
        follow_tester.addSignature( "<org.trifort.rootbeer.runtime.Serializer: void <init>(org.trifort.rootbeer.runtime.Memory,org.trifort.rootbeer.runtime.Memory)>");
        follow_tester.addSignature("<org.trifort.rootbeer.testcases.rootbeertest.serialization.CovarientTest: void <init>()>");
        RootbeerClassLoader.v().addFollowMethodTester( follow_tester );

        if ( runtests )
            RootbeerClassLoader.v().addFollowClassTester( new TestCaseFollowTester() );

        if ( m_configuration.getKeepMains() )
        {
            MainTester main_tester = new MainTester();
            RootbeerClassLoader.v().addFollowMethodTester( main_tester );
        }

        final ListMethodTester dont_dfs_tester = new ListMethodTester();

        final CompilerSetup setup = new CompilerSetup();
        for ( String no_dfs : setup.getDontDfs() )
            dont_dfs_tester.addSignature(no_dfs);
        RootbeerClassLoader.v().addDontFollowMethodTester(dont_dfs_tester);

        final ForcedFields forced_fields = new ForcedFields();
        for ( String field_sig : forced_fields.get() )
            RootbeerClassLoader.v().loadField( field_sig );

        final ListMethodTester to_sig_methods = new ListMethodTester();
        to_sig_methods.addSignature( "<java.lang.Object: int hashCode()>"        );
        to_sig_methods.addSignature( "<java.io.PrintStream: void println(java.lang.String)>" );
        to_sig_methods.addSignature( "<java.io.PrintStream: void println(int)>"  );
        to_sig_methods.addSignature( "<java.io.PrintStream: void println(long)>" );
        RootbeerClassLoader.v().addToSignaturesMethodTester( to_sig_methods );

        RootbeerClassLoader.v().addClassRemapping( "java.util.concurrent.atomic.AtomicLong", "org.trifort.rootbeer.remap.GpuAtomicLong" );
        RootbeerClassLoader.v().addClassRemapping(
            "org.trifort.rootbeer.testcases.rootbeertest.remaptest.CallsPrivateMethod",
            "org.trifort.rootbeer.remap.DoesntCallPrivateMethod"
        );
        RootbeerClassLoader.v().loadNecessaryClasses();
    }

    public void compile
    (
        final String jar_filename,
        final String outname     ,
        final String test_case
    ) throws Exception
    {
        TestCaseEntryPointDetector detector = new TestCaseEntryPointDetector( test_case );
        m_entryDetector = detector; /* copy and convert to MethodTester */
        setupSoot( jar_filename, new CurrJarName().get(), true );
        m_provider = detector.getProvider();

        List<SootMethod> kernel_methods = RootbeerClassLoader.v().getEntryPoints();
        compileForKernels(outname, kernel_methods, jar_filename);
    }

    /**
     * Called when compiling with rootbeer (run_tests = false)
     */
    public void compile
    (
        final String  jar_filename,
        final String  outname     ,
        final boolean run_tests
    ) throws Exception
    {
        /* functional detecting implementations of Kernel interface with gpuMethod */
        m_entryDetector = new KernelEntryPointDetector( run_tests );
        /* CurrJarName.get should normally return 'Rootbeer.jar' */
        setupSoot( jar_filename, new CurrJarName().get(), run_tests );

        /* list of all found Kernel.gpuMethod implementations */
        final List<SootMethod> kernel_methods = RootbeerClassLoader.v().getEntryPoints();
        compileForKernels( outname, kernel_methods, jar_filename );
    }

    /**
     * default run_tests to false. This is normally called by the rootbeer
     * compiler
     */
    public void compile
    (
        final String jar_filename,
        final String outname
    ) throws Exception
    {
        compile( jar_filename, outname, false );
    }

    /**
     * The non-simple Rootbeer syntax
     */
    public void compile
    (
        final String       main_jar,
        final List<String> lib_jars,
        final List<String> dirs    ,
        final String       dest_jar
    )
    {
        throw new RuntimeException( "The non-simple syntax is not yet implemented, please use the simple syntax!" );
    }

    private void compileForKernels
    (
        final String           outname       ,
        final List<SootMethod> kernel_methods,
        final String           jar_filename
    ) throws Exception
    {
        if ( kernel_methods.isEmpty() )
        {
            System.out.println( "There are no kernel classes. Please implement the following interface to use rootbeer:" );
            System.out.println( "org.trifort.runtime.Kernel" );
            System.exit(0);
        }

        final Transform2 transform2 = new Transform2();
        for ( final SootMethod kernel_method : kernel_methods )
        {
            if ( debugging )
            {
                System.out.println(
                    "[RootbeerCompiler.compileForKernels] Found a gpuMethod in a Kernel implementation: " +
                    kernel_method.getSignature() +
                    ". Run transform2 on it now ..."
                );
            }
            RootbeerClassLoader.v().loadDfsInfo( kernel_method );
            final DfsInfo dfs_info = RootbeerClassLoader.v().getDfsInfo();

            new RootbeerDfs().run( dfs_info );

            dfs_info.expandArrayTypes();
            dfs_info.finalizeTypes();

            SootClass soot_class = kernel_method.getDeclaringClass();
            transform2.run( soot_class.getName(), m_configuration ); /* <- juicy part */
        }

        if ( debugging )
            System.out.println( "[RootbeerCompiler.compileForKernels] writing classes out..." );

        /* Write out Jimple and class files, but only if class name does not
         * start with one out of m_runtimePackages and also not if it does
         * implement org.trifort.rootbeer.test. */
        Iterator<SootClass> iter = Scene.v().getClasses().iterator();
        while ( iter.hasNext() )
        {
            final SootClass soot_class = iter.next();
            if ( soot_class.isLibraryClass() )
                continue;

            final String class_name = soot_class.getName();
            boolean write = true;
            for ( final String runtime_class : m_runtimePackages )
            {
                if ( class_name.startsWith( runtime_class ) )
                {
                    write = false;
                    break; /* not necessary, just an optimization */
                }
            }
            Iterator<SootClass> ifaces = soot_class.getInterfaces().iterator();
            while ( ifaces.hasNext() )
            {
                SootClass iface = ifaces.next();
                if ( iface.getName().startsWith( "org.trifort.rootbeer.test." ) )
                {
                    write = false;
                    break; /* not necessary, just an optimization */
                }
            }

            if ( write )
            {
                writeJimpleFile( class_name );
                writeClassFile( class_name );
            }
        }

        makeOutJar( jar_filename );
        pack( outname );
    }

    /**
     * Packs or copies the jar containing all dynamically compiled classes
     * E.g. zipinfo ~/.rootbeer/.../partial-ret.jar
     *   META-INF/
     *   META-INF/MANIFEST.MF
     *   sun/security/provider/PolicyParser.class
     *   java/lang/Integer$IntegerCache.class
     *   org/trifort/rootbeer/runtime/config.txt
     *   MonteCarloPiKernel.class
     *   MonteCarloPiKernelSerializer.class
     *   MonteCarloPiKernelSerializer-64.cubin
     */
    public void pack( final String outjar_name ) throws Exception
    {
        if ( m_packFatJar )
        {
            List<String> lib_jars = Arrays.asList( new CurrJarName().get() );
            final Pack p = new Pack();
            p.run( m_JarWithoutRootbeer, lib_jars, outjar_name );
        }
        else
        {
            /* don't pack Rootbeer.jar using the pack Routine. User can do
             * it himself e.g. with zipmerge, thereby saving time
             * Do not use renameTo as it does not work if src and destination
             * are in different hard drives. Note Files is Java 1.6+
             */
            System.out.println( "m_JarWithoutRootbeer = " + m_JarWithoutRootbeer );
            Files.copy( new File(m_JarWithoutRootbeer).toPath(),
                        new File(outjar_name).toPath(), REPLACE_EXISTING );
        }
    }

    /**
     * Packs the compiled classes
     */
    public void makeOutJar( final String jar_filename ) throws Exception
    {
        ZipOutputStream zos = new ZipOutputStream( new FileOutputStream( m_JarWithoutRootbeer ) );

        /* Copy the META-INF/ folder from jar_filename to the opened
         * jar stream in zos */
        final ZipInputStream jin = new ZipInputStream( new FileInputStream( jar_filename ) );
        ZipEntry jar_entry;
        while ( ( jar_entry = jin.getNextEntry() ) != null )
        {
            if ( jar_entry.getName().contains("META-INF") )
                writeFileToOutput( jin, jar_entry, zos );
        }
        jin.close();

        /* add the class files written out after the transform2 method.
         * Do this recursively for all subfolders */
        final List<File> output_class_files = getFiles( RootbeerPaths.v().getOutputClassFolder() );
        for ( File f : output_class_files )
            writeFileToOutput( f, zos, RootbeerPaths.v().getOutputClassFolder() );

        addConfigurationFile( zos, m_configuration );
        zos.flush();
        zos.close();
    }

    /**
     * Initializer for recursive getFiles method
     */
    private static List<File> getFiles( final String path )
    {
        File f = new File(path);
        List<File> ret = new ArrayList<File>();
        getFiles( ret, f );
        return ret;
    }

    /**
     * Get all files recursively for the given directory
     *
     * @param[out] total_files will get the found files appended
     * @param[in]  dir directory to traverse
     */
    private static void getFiles
    (
        final List<File> total_files,
        final File dir
    )
    {
        File[] files = dir.listFiles();
        for ( File f : files )
        {
            if ( f.isDirectory() )
                getFiles(total_files, f);
            else
                total_files.add(f);
        }
    }

    private static String makeJarFileName
    (
        final File f,
        String folder
    )
    {
        try
        {
            String abs_path = f.getAbsolutePath();
            if ( f.isDirectory() )
                abs_path += File.separator;
            folder += File.separator;
            folder = folder.replace("\\", "\\\\");
            String[] tokens = abs_path.split(folder);
            String ret = tokens[1];
            if ( File.separator.equals("\\") )
                ret = ret.replace("\\", "/");
            return ret;
        } catch(Exception ex){
            throw new RuntimeException(ex);
        }
    }

    /**
     * Adds config.txt which contains 2 bytes to jar and home config directory
     * The first specifies whether the GPU or an Emulator (JEMU, NEMU) is to be used ???
     * The second byte specifies whether to use and serialize kernel exceptions
     */
    private static void addConfigurationFile
    (
        final ZipOutputStream zos,
        final Configuration   configuration
    ) throws IOException
    {
        final String folderName = "org/trifort/rootbeer/runtime/";
        final String fileName   = folderName + "config.txt";
        final ZipEntry entry    = new ZipEntry( fileName );
        entry.setSize(1);

        final byte[] contents = new byte[2];
        contents[0] = (byte)   configuration.getMode();
        contents[1] = (byte) ( configuration.getExceptions() ? 1 : 0 );


        final CRC32 crc = new CRC32();
        crc.update( contents );
        entry.setCrc( crc.getValue() );
        zos.putNextEntry( entry );
        zos.write( contents );
        zos.flush();

        /* Also write out config.txt to ~/.rootbeer */
        final File file = new File( RootbeerPaths.v().getOutputClassFolder() +
                                    File.separator + folderName );
        if ( ! file.exists() )
            file.mkdirs();

        final FileOutputStream fout = new FileOutputStream(
            RootbeerPaths.v().getOutputClassFolder() + File.separator + fileName
        );
        fout.write( contents );
        // http://stackoverflow.com/questions/9272585/difference-between-flush-and-close-function-in-case-of-filewriter-in-java
        fout.flush();
        fout.close();
    }

    /**
     * Copies a given entry from the input jar to the output jar.
     */
    private static void writeFileToOutput
    (
        final ZipInputStream  jin      ,
        final ZipEntry        jar_entry,
        final ZipOutputStream zos
    ) throws Exception
    {
        if ( ! jar_entry.isDirectory() )
        {
            /* Because the ZipStream wants to know the file size buffer the
             * whole file and count the bytes, only then forwarding it to
             * the output zip archive */
            final List<byte[]> buffered = new ArrayList<byte[]>();
            int total_size = 0;
            while ( true )
            {
                final byte[] buffer = new byte[ 32*1024 ];

                final int len = jin.read(buffer);
                if ( len == -1 )
                    break;

                total_size += len;

                final byte[] truncated = new byte[len];
                for ( int i = 0; i < len; ++i )
                    truncated[i] = buffer[i];

                buffered.add( truncated );
            }

            /* create new entry and write buffer to it */
            final ZipEntry entry = new ZipEntry( jar_entry.getName() );
            entry.setSize( total_size );
            entry.setCrc( jar_entry.getCrc() );
            zos.putNextEntry( entry );

            for ( byte[] buffer : buffered )
                zos.write( buffer );
            zos.flush();
        } else {
            zos.putNextEntry( jar_entry );
        }
    }

    /**
     * Adds a file to a jar i.e. zipStream
     */
    private static void writeFileToOutput
    (
        final File f,
        final ZipOutputStream zos,
        final String folder
    ) throws Exception
    {
        final String   name  = makeJarFileName(f, folder);
        final ZipEntry entry = new ZipEntry(name);
        byte[] contents = readFile(f);
        entry.setSize( contents.length );

        final CRC32 crc = new CRC32();
        crc.update( contents );
        entry.setCrc( crc.getValue() );
        zos.putNextEntry( entry );

        int wrote_len = 0;
        final int total_len = contents.length;
        while(wrote_len < total_len)
        {
            int len = 4096;
            final int len_left = total_len - wrote_len;
            if( len > len_left )
                len = len_left;
            zos.write( contents, wrote_len, len );
            wrote_len += len;
        }
        zos.flush();
    }

    private static byte[] readFile( final File f ) throws Exception
    {
        List<Byte> contents = new ArrayList<Byte>();
        byte[] buffer = new byte[4096];
        FileInputStream fin = new FileInputStream(f);
        while ( true )
        {
            int len = fin.read(buffer);
            if ( len == -1 )
                break;
            for ( int i = 0; i < len; ++i )
                contents.add( buffer[i]) ;
        }
        fin.close();
        byte[] ret = new byte[contents.size()];
        for ( int i = 0; i < contents.size(); ++i )
            ret[i] = contents.get(i);
        return ret;
    }

    private static void writeJimpleFile( final String cls )
    {
        try
        {
            SootClass c = Scene.v().getSootClass(cls);
            JimpleWriter writer = new JimpleWriter();
            writer.write( classNameToFileName(cls, true), c );
        }
        catch ( Exception ex )
        {
            System.out.println( "Error writing .jimple: " + cls );
        }
    }

    private static List<String> getMethodSignatures( final SootClass c )
    {
        List<String> ret = new ArrayList<String>();
        List<SootMethod> methods = c.getMethods();
        for ( SootMethod method : methods )
            ret.add( method.getSignature() );
        return ret;
    }

    private static void writeClassFile( final String cls )
    {
        FileOutputStream fos    = null;
        OutputStream     out1   = null;
        PrintWriter      writer = null;
        SootClass        c      = Scene.v().getSootClass(cls);
        List<String> before_sigs = getMethodSignatures(c);
        try
        {
            fos    = new FileOutputStream( classNameToFileName( cls, false ) );
            out1   = new JasminOutputStream( fos );
            writer = new PrintWriter( new OutputStreamWriter(out1) );
            new soot.jimple.JasminClass(c).print(writer);
        }
        catch ( Exception ex )
        {
            System.out.println( "Error writing .class: " + cls );
            ex.printStackTrace( System.out );
            List<String> after_sigs = getMethodSignatures( c );

            System.out.println( "Before sigs: " );
            for( String sig : before_sigs )
                System.out.println( "  " + sig );

            System.out.println( "After sigs: " );
            for( String sig : after_sigs )
                System.out.println( "  " + sig );
        }
        finally
        {
            try
            {
                writer.flush();
                writer.close();
                out1.close();
                fos.close();
            }
            catch ( Exception ex )
            {
                ex.printStackTrace();
            }
        }
    }

    private static String classNameToFileName
    (
        String cls,
        final boolean jimple
    )
    {
        File f;
        if ( jimple )
            f = new File( RootbeerPaths.v().getOutputJimpleFolder() );
        else
            f = new File( RootbeerPaths.v().getOutputClassFolder() );

        cls = cls.replace(".", File.separator);

        if ( jimple )
            cls += ".jimple";
        else
            cls += ".class";

        cls = f.getAbsolutePath()+File.separator + cls;

        File f2 = new File(cls);
        String folder = f2.getParent();
        new File(folder).mkdirs();

        return cls;
    }

    private static void clearOutputFolders()
    {
        DeleteFolder deleter = new DeleteFolder();
        deleter.delete( RootbeerPaths.v().getOutputJarFolder    () );
        deleter.delete( RootbeerPaths.v().getOutputClassFolder  () );
        deleter.delete( RootbeerPaths.v().getOutputShimpleFolder() );
        deleter.delete( RootbeerPaths.v().getJarContentsFolder  () );
    }

    public String getProvider() { return m_provider; }
}
