/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.configuration;

import java.io.File;
import java.lang.management.ManagementFactory;
import java.net.InetAddress;
import java.net.UnknownHostException;

/**
 * Singleton which finds and or creates a working directory for rootbeer
 */
public class RootbeerPaths
{
    private static RootbeerPaths m_instance;
    private static String        m_rootbeerhome;

    public static RootbeerPaths v()
    {
        if ( m_instance == null )
        {
            m_instance = new RootbeerPaths();
            m_rootbeerhome = new String("");
        }
        return m_instance;
    }

    public String getConfigFile         (){ return getRootbeerHome() + "config"        ; }
    public String getJarContentsFolder  (){ return getRootbeerHome() + "jar-contents"  ; }
    public String getOutputJarFolder    (){ return getRootbeerHome() + "output-jar"    ; }
    public String getOutputClassFolder  (){ return getRootbeerHome() + "output-class"  ; }
    public String getOutputShimpleFolder(){ return getRootbeerHome() + "output-shimple"; }
    public String getOutputJimpleFolder (){ return getRootbeerHome() + "output-jimple" ; }
    public String getTypeFile           (){ return getRootbeerHome() + "types"         ; }

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

    public String getRootbeerHome()
    {
        if ( m_rootbeerhome == null || m_rootbeerhome.isEmpty() )
        {
            final String home = System.getProperty("user.home");
            m_rootbeerhome = home + File.separator + ".rootbeer" + File.separator
                           + getHostname() + File.separator
                           + getProcessId("pid") + "-" + System.nanoTime()
                           + File.separator;
        }
        File folder = new File( m_rootbeerhome );
        if ( ! folder.exists() )
            folder.mkdirs();
        return folder.getAbsolutePath() + File.separator;
    }
}
