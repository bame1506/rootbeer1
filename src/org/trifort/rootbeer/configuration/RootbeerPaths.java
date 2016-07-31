/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.configuration;


import java.io.File;
import java.lang.management.ManagementFactory; // for processID
import java.lang.Thread;
import java.net.InetAddress;
import java.net.UnknownHostException;


/**
 * Singleton which finds and or creates a working directory for rootbeer
 * I think it was made a singleton to first buffer the name creation and
 * also to optimized trying to create the folder every time and third to
 * not have to bother passing an object.
 */
public final class RootbeerPaths
{
    private static final ThreadLocal<RootbeerPaths> m_instance;
    private final String m_rootbeerhome;

    /**
     * The returned path may not change between subsequent calls!
     * I.e. including the time is not a good idea, but the process ID
     * should not change
     */
    private RootbeerPaths()
    {
        final String home = System.getProperty("user.home");
        final String path = home + File.separator + ".rootbeer" + File.separator
                          + getHostname() + File.separator
                          + getProcessId("pid") + "-"
                          + Thread.currentThread().getId() + "-"
                          + System.nanoTime()
                          + File.separator;
        final File folder = new File( path );
        if ( ! folder.exists() )
            folder.mkdirs();
        m_rootbeerhome = folder.getAbsolutePath() + File.separator;
    }

    static {
        m_instance = new ThreadLocal<RootbeerPaths>(){
            @Override protected RootbeerPaths initialValue() { return new RootbeerPaths(); }
        };
        /* using set would be equivalent to overriding the initial value,
         * but only because this method is synchronized. If not, then
         * a race condition could appear between creating and setting
         * m_instance */
        // m_instance.set( new RootbeerPaths() );
    }

    public static RootbeerPaths v(){ return m_instance.get(); }

    public static String getConfigFile         (){ return getRootbeerHome() + "config"        ; }
    public static String getJarContentsFolder  (){ return getRootbeerHome() + "jar-contents"  ; }
    public static String getOutputJarFolder    (){ return getRootbeerHome() + "output-jar"    ; }
    public static String getOutputClassFolder  (){ return getRootbeerHome() + "output-class"  ; }
    public static String getOutputShimpleFolder(){ return getRootbeerHome() + "output-shimple"; }
    public static String getOutputJimpleFolder (){ return getRootbeerHome() + "output-jimple" ; }
    public static String getTypeFile           (){ return getRootbeerHome() + "types"         ; }

    private static String getHostname()
    {
        // http://stackoverflow.com/questions/7348711/recommended-way-to-get-hostname-in-java
        // try environment properties.
        String host = System.getenv("COMPUTERNAME");
        if ( host != null )
            return host;
        host = System.getenv("HOSTNAME");
        if ( host != null )
            return host;

        // try InetAddress.LocalHost;
        //      NOTE -- InetAddress.getLocalHost().getHostName() will not work in certain environments.
        try {
            final String result = InetAddress.getLocalHost().getHostName();
            if ( result != null && ! result.isEmpty() )
                return result;
        } catch ( UnknownHostException e ) {
            // failed;  try alternate means.
        }

        return "UnknownHost";
    }

    private static String getProcessId( final String fallback )
    {
        // http://stackoverflow.com/questions/35842/how-can-a-java-program-get-its-own-process-id
        // Note: may fail in some JVM implementations
        // therefore fallback has to be provided

        // something like '<pid>@<hostname>', at least in SUN / Oracle JVMs
        final String jvmName = ManagementFactory.getRuntimeMXBean().getName();
        final int index = jvmName.indexOf('@');

        if (index < 1) {
            // part before '@' empty (index = 0) / '@' not found (index = -1)
            return fallback;
        }

        try {
            return Long.toString(Long.parseLong(jvmName.substring(0, index)));
        } catch (NumberFormatException e) {
            // ignore
        }
        return fallback;
    }

    public static String getRootbeerHome(){ return m_instance.get().m_rootbeerhome; }
}
