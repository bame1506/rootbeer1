/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.entry;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.trifort.rootbeer.configuration.Configuration;
import org.trifort.rootbeer.generate.opencl.tweaks.GencodeOptions.CompileArchitecture;
import org.trifort.rootbeer.generate.opencl.tweaks.GencodeOptions.ComputeCapability;
/* needed for printDeviceInfo */
import org.trifort.rootbeer.runtime.GpuDevice;
import org.trifort.rootbeer.runtime.Rootbeer;

public class Main
{
    private int            m_num_args               ;
    private boolean        m_dontPackFatJar         ;
    private boolean        m_runTests               ;
    private boolean        m_runHardTests           ;
    private boolean        m_disableClassRemapping  ;
    private String         m_testCase               ;
    private boolean        m_simpleCompile          ;
    private boolean        m_printDeviceInfo        ;
    private boolean        m_largeMemTests          ;
    private String         m_mainJar                ;
    private List<String>   m_libJars                ;
    private List<String>   m_directories            ;
    private String         m_destJar                ;
    private Configuration  m_configuration          ;

    /* Constructor */
    public Main()
    {
        m_libJars           = new ArrayList<String>();
        m_directories       = new ArrayList<String>();
        m_simpleCompile     = false;
        m_runHardTests      = false;
        m_dontPackFatJar    = false;
        m_printDeviceInfo   = false;
        m_largeMemTests     = false;
    }

    /**
     * This sets many of the members of the singleton Configuration.java
     * and also private members of this Main class to be used by 'run'
     *
     * @todo rewrite e.g. using https://commons.apache.org/proper/commons-cli/usage.html
     * @todo add help message like in Readme.md
     */
    private void parseArgs( final String[] args )
    {
        m_num_args = args.length;

        boolean arch32bit   = false;
        boolean arch64bit   = false;
        boolean invalidArgs = false;

        /* default values */
        CompileArchitecture arch = CompileArchitecture.Arch32bit64bit;
        ComputeCapability comp            = null;
        Integer           mode            = null;
        Integer           nBytesSharedMem = null;


        for ( int i = 0; i < args.length; ++i )
        {
            final String arg = args[i];

            if ( arg.equals( "-nemu" ) )
            {
                if ( mode != null )
                {
                    System.out.println( "Only one mode -jemu xor -nemu may be specified." );
                    invalidArgs = true;
                }
                mode = new Integer( Configuration.MODE_NEMU );
            }
            else if ( arg.equals( "-jemu" ) )
            {
                if ( mode != null )
                {
                    System.out.println( "Only one mode -jemu xor -nemu may be specified." );
                    invalidArgs = true;
                }
                mode = new Integer( Configuration.MODE_JEMU );
            }
            else if ( arg.equals( "-remap-sparse" ) )
                Configuration.compilerInstance().setRemapSparse();
            else if ( arg.equals( "-disable-class-remapping" ) )
                m_disableClassRemapping = true;
            else if ( arg.equals( "-mainjar" ) )
            {
                if ( m_mainJar != null )
                {
                    System.out.println( "Only one -mainjar may be specified." );
                    invalidArgs = true;
                }
                m_mainJar = safeGet(args, ++i, "-mainjar" );

                final File file = new File( m_mainJar );
                if ( ! ( file.exists() && file.isFile() )  )
                {
                    System.out.println( "'" + m_mainJar + "' does not exist or is not a file." );
                    invalidArgs = true;
                }
            }
            else if ( arg.equals( "-libjar" ) )
            {
                final String libJar = safeGet(args, ++i, "-libjar" );
                m_libJars.add( libJar );

                final File file = new File( libJar );
                if ( ! ( file.exists() && file.isFile() )  )
                {
                    System.out.println( "'" + libJar + "' does not exist or is not a file." );
                    invalidArgs = true;
                }
            }
            else if ( arg.equals( "-directory" ))
            {
                final String dir = safeGet(args, ++i, "-directory");
                m_directories.add( dir );

                final File file = new File( dir );
                if ( ! ( file.exists() && file.isDirectory() )  )
                {
                    System.out.println( "'" + dir + "' does not exist or is not a directory." );
                    invalidArgs = true;
                }
            }
            else if ( arg.equals( "-destjar" ) )
            {
                if ( m_destJar != null )
                {
                    System.out.println( "Only one -destjar may be specified." );
                    invalidArgs = true;
                }

                m_destJar = safeGet(args, ++i, "-destjar");

                final File file = new File( m_destJar );
                if ( ! ( file.exists() && file.isFile() )  )
                {
                    System.out.println( "'" + m_destJar + "' does not exist or is not a file." );
                    invalidArgs = true;
                }
            }
            else if ( arg.equals( "-nofatjar" ) )
                m_dontPackFatJar = true;
            else if ( arg.equals( "-runtests" ) )
            {
                m_runTests     = true;
                m_runHardTests = true;
                if ( m_testCase != null )
                {
                    System.out.println( "-runtest may not be mixed with other test options e.g. -runtests." );
                    invalidArgs = true;
                }
            }
            else if ( arg.equals( "-runeasytests" ) )
            {
                m_runTests     = true;
                m_runHardTests = false;
                if ( m_testCase != null )
                {
                    System.out.println( "-runtest may not be mixed with other test options e.g. -runeasytests." );
                    invalidArgs = true;
                }
            }
            else if ( arg.equals( "-runtest" ) )
            {
                m_runTests     = true;
                m_testCase     = safeGet( args, ++i, "-runtest" );
                m_runHardTests = true;
            }
            else if ( arg.equals( "-large-mem-tests" ) )
                m_largeMemTests = true;
            else if ( arg.equals( "-printdeviceinfo" ) )
                m_printDeviceInfo = true;
            else if ( arg.equals( "-maxrregcount") )
            {
                final String count = safeGet( args, ++i, "-maxrregcount" );
                if ( Configuration.compilerInstance().isMaxRegCountSet() )
                {
                    System.out.println( "Only one -maxrregcount may be specified." );
                    invalidArgs = true;
                }
                Configuration.compilerInstance().setMaxRegCount( Integer.parseInt(count) );
            }
            else if ( arg.equals( "-noarraychecks" ) )
                Configuration.compilerInstance().setArrayChecks(false);
            else if ( arg.equals( "-nodoubles" ) )
                Configuration.compilerInstance().setDoubles(false);
            else if ( arg.equals( "-norecursion" ) )
                Configuration.compilerInstance().setRecursion(false);
            else if ( arg.equals( "-noexceptions" ) )
                Configuration.compilerInstance().setExceptions(false);
            else if ( arg.equals( "-keepmains" ) )
                Configuration.compilerInstance().setKeepMains(true);
            else if ( arg.equals( "-shared-mem-size" ) )
            {
                if ( nBytesSharedMem != null )
                {
                    System.out.println( "Only one -maxrregcount may be specified." );
                    invalidArgs = true;
                }
                nBytesSharedMem = new Integer( safeGet(args, ++i, "-shared-mem-size") );
                Configuration.compilerInstance().setSharedMemSize( nBytesSharedMem.intValue() );
            }
            else if ( arg.equals( "-32bit" ) )
                arch32bit = true;
            else if ( arg.equals( "-64bit" ) )
                arch64bit = true;
            else if ( arg.equals( "-manualcuda" ) )
            {
                String filename = safeGet(args, ++i, "-manualcuda");
                Configuration.compilerInstance().setManualCuda();
                Configuration.compilerInstance().setManualCudaFilename(filename);
            }
            else if ( arg.equals( "-computecapability" ) )
            {
                if ( comp != null )
                {
                    /* @todo allow multiple compute capabilities */
                    System.out.println( "Only one -computecapability may be specified (yet. Future feature may include compiling for all multiple specified architectures.)." );
                    invalidArgs = true;
                }

                final String computeString = safeGet( args, ++i, "-computecapability" );

                if (      computeString.equalsIgnoreCase( "sm_11" ) )
                    comp = ComputeCapability.SM_11;
                else if ( computeString.equalsIgnoreCase( "sm_12" ) )
                    comp = ComputeCapability.SM_12;
                else if ( computeString.equalsIgnoreCase( "sm_20" ) )
                    comp = ComputeCapability.SM_20;
                else if ( computeString.equalsIgnoreCase( "sm_21" ) )
                    comp = ComputeCapability.SM_21;
                else if ( computeString.equalsIgnoreCase( "sm_30" ) )
                    comp = ComputeCapability.SM_30;
                else if ( computeString.equalsIgnoreCase( "sm_35" ) )
                    comp = ComputeCapability.SM_35;
                else
                    System.out.println( "Unsupported compute capability: " + computeString );
            }
            /* if argument is no valid option, then trigger simple compile
             * syntax and interpret next two arguments as mainjar name and
             * destination jar name */
            else if ( m_simpleCompile == false )
            {
                if ( m_mainJar != null || m_directories.size() != 0 ||
                     m_destJar != null || m_libJars.size()     != 0   )
                {
                    System.out.println( "Simple syntax was triggered by unknown argument (" + arg + "). For simple syntax the options -mainjar, -destjar, -directory and -libjar are not allowed!" );
                    invalidArgs = true;
                }

                m_mainJar = arg;
                m_destJar = safeGet(args, ++i, arg);

                File main_jar = new File(m_mainJar);
                if ( ! main_jar.exists() )
                {
                    System.out.println( "Cannot find: " + m_mainJar );
                    System.exit(0);
                }
                m_simpleCompile = true;
            }
            else
            {
                System.out.println( "Invalid command line argument specified! (" + arg + ")" );
                invalidArgs = true;
            }
        }

        if ( invalidArgs )
            throw new IllegalArgumentException( "There were illegal arguments specified" );

        if ( Configuration.compilerInstance().getRecursion() && ! m_printDeviceInfo )
        {
            System.out.println("warning: sm_12 and sm_11 not supported with recursion. use -norecursion to enable.");
        }

        if ( Configuration.compilerInstance().getDoubles() && ! m_printDeviceInfo )
        {
            System.out.println("warning: sm_12 and sm_11 not supported with doubles. use -nodoubles to enable.");
        }

        if ( arch32bit == arch64bit ) /* if both are set or not specified */
            arch = CompileArchitecture.Arch32bit64bit;
        else if ( arch32bit )
            arch = CompileArchitecture.Arch32bit;
        else if( arch64bit )
            arch = CompileArchitecture.Arch64bit;

        if ( mode == null )
            mode = new Integer( Configuration.MODE_GPU );
        Configuration.compilerInstance().setMode( mode.intValue() );
        Configuration.compilerInstance().setCompileArchitecture( arch );
        if ( comp != null )
            Configuration.compilerInstance().setComputeCapability( comp );
    }

    /**
     * Checks if there actually is an argument like required after the option
     */
    private String safeGet
    (
        final String[] args,
        final int index,
        final String argname
    )
    {
        if ( index >= args.length )
        {
            System.out.println( argname + " needs another argument after it." );
            System.exit( -1 );
        }
        return args[index];
    }

    /**
     * Does what was specified in command arguments. Normally this means
     * compiling by calling RootbeerCompiler.compile
     */
    private void run()
    {
        if ( m_printDeviceInfo )
        {
            if ( m_num_args == 1 )
                printDeviceInfo();
            else
            {
                System.out.println( "-printdeviceinfo can only be used by itself. Remove other arguments." );
                System.out.flush();
                return;
            }
            return;
        }

        if ( m_runTests )
        {
            RootbeerTest test = new RootbeerTest();
            test.runTests( m_testCase, m_runHardTests, m_largeMemTests );
            return;
        }

        RootbeerCompiler compiler = new RootbeerCompiler();
        if ( m_disableClassRemapping )
            compiler.disableClassRemapping();

        if ( m_simpleCompile )
        {
            try {
                if ( m_dontPackFatJar )
                    compiler.dontPackFatJar();
                compiler.compile( m_mainJar, m_destJar );
            } catch( Exception ex ) {
                ex.printStackTrace();
            }
        }
        else
        {
            try {
                /* not yet implemented yet ! */
                compiler.compile( m_mainJar, m_libJars, m_directories, m_destJar );
            } catch( Exception ex ) {
                ex.printStackTrace();
            }
        }
    }

    /**
     * Prints e.g. CUDA device properties
     */
    private void printDeviceInfo()
    {
        Rootbeer rootbeer = new Rootbeer();
        List<GpuDevice> devices = rootbeer.getDevices();
        System.out.println( "Found " + devices.size() + " GPU devices" );

        for ( GpuDevice device : devices )
        {
            System.out.println( "device: "+device.getDeviceName());
            System.out.println( "  compute_capability            : " + device.getMajorVersion()+"."+device.getMinorVersion());
            System.out.println( "  total_global_memory           : " + device.getTotalGlobalMemoryBytes()+" bytes");
            System.out.println( "  max_shared_memory_per_block   : " + device.getMaxSharedMemoryPerBlock()+" bytes");
            System.out.println( "  num_multiprocessors           : " + device.getMultiProcessorCount());
            System.out.println( "  clock_rate                    : " + device.getClockRateHz()+" Hz");
            System.out.println( "  max_block_dim_x               : " + device.getMaxBlockDimX());
            System.out.println( "  max_block_dim_y               : " + device.getMaxBlockDimY());
            System.out.println( "  max_block_dim_z               : " + device.getMaxBlockDimZ());
            System.out.println( "  max_grid_dim_x                : " + device.getMaxGridDimX());
            System.out.println( "  max_grid_dim_x                : " + device.getMaxGridDimY());
            System.out.println( "  max_threads_per_multiprocessor: " + device.getMaxThreadsPerMultiprocessor());
        }
    }

    /**
     * Actal start entry when running the Rootbeer compiler e.g. with:
     *  java -jar Rootbeer.jar $@.tmp.jar gpu.jar -64bit -computecapability=sm_30
     */
    public static void main( String[] args )
    {
        Main main = new Main(); /* Constructor does nothing of note */
        main.parseArgs( args ); /* read options into members and Configuration instance */
        main.run();
    }
}
