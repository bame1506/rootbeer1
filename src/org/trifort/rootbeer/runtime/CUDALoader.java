/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.runtime;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import org.trifort.rootbeer.configuration.RootbeerPaths;

/**
 * This class loads the CUDA binaries for the corresponding operating system
 * from the jar to the lcoal file system.
 */
public class CUDALoader
{
  private List<String> m_libCudas;
  private List<String> m_rootbeerRuntimes;
  private List<String> m_rootbeerCudas;

  /**
   * This constructor adds the CUDA shared library paths to the member
   * variables and extracts shared libraries using JNI to getRootbeerHome
   * after adding them too. No shared libraries are loaded yet.
   **/
  public CUDALoader()
  {
      m_libCudas         = new ArrayList<String>();
      m_rootbeerRuntimes = new ArrayList<String>();
      m_rootbeerCudas    = new ArrayList<String>();
      final String prefix = RootbeerPaths.v().getRootbeerHome();

      if ( "Mac OS X".equals(System.getProperty( "os.name" )) )
      {
          m_libCudas.add( "/usr/local/cuda/lib/libcuda.dylib" );
          m_rootbeerRuntimes.add( prefix + "rootbeer.dylib"      );
          m_rootbeerCudas.   add( prefix + "rootbeer_cuda.dylib" );
          extract( "rootbeer.dylib" );
          extract( "rootbeer_cuda.dylib" );
      }
      else if ( File.separator.equals( "/" ) )
      {
          if ( is32Bit() )
          {
              m_libCudas.add( "/usr/lib/libcuda.so" );
              m_libCudas.add( "/usr/lib/x86_64-linux-gnu/libcudart.so.5.0" );
              m_rootbeerRuntimes.add(prefix + "rootbeer_x86.so.1" );
              m_rootbeerCudas.add(prefix + "rootbeer_cuda_x86.so.1" );
              extract( "rootbeer_x86.so.1" );
              extract( "rootbeer_cuda_x86.so.1" );
          }
          else
          {
              m_libCudas.add( "/usr/lib64/libcuda.so" );
              m_libCudas.add( "/usr/lib/x86_64-linux-gnu/libcudart.so.5.0" );
              m_rootbeerRuntimes.add(prefix + "rootbeer_x64.so.1" );
              m_rootbeerCudas   .add(prefix + "rootbeer_cuda_x64.so.1" );
              extract               ( "rootbeer_x64.so.1" );
              extract               ( "rootbeer_cuda_x64.so.1" );
          }
      }
      else
      {
        if(is32Bit()){
          m_libCudas.add( "C:\\Windows\\System32\\nvcuda.dll" );
          m_rootbeerRuntimes.add(prefix + "rootbeer_x86.dll" );
          m_rootbeerCudas.add(prefix + "rootbeer_cuda_x86.dll" );
          extract( "rootbeer_x86.dll" );
          extract( "rootbeer_cuda_x86.dll" );
        } else {
          m_libCudas.add( "C:\\Windows\\System32\\nvcuda.dll" );
          m_libCudas.add( "C:\\Windows\\SysWow64\\nvcuda.dll" );
          m_rootbeerRuntimes.add(prefix + "rootbeer_x64.dll" );
          m_rootbeerCudas.add(prefix + "rootbeer_cuda_x64.dll" );
          extract( "rootbeer_x64.dll" );
          extract( "rootbeer_cuda_x64.dll" );
        }
      }
  }

  private boolean is32Bit()
  {
      //http://mark.koli.ch/2009/10/javas-osarch-system-property-is-the-bitness-of-the-jre-not-the-operating-system.html
      // The os.arch property will also say "x86" on a
      // 64-bit machine using a 32-bit runtime
      String arch = System.getProperty( "os.arch" );
      if ( arch.equals( "x86" ) || arch.equals( "i386" ) ) {
          return true;
      } else {
          return false;
      }
  }

  public void load(){
    doLoad(m_libCudas);
    doLoad(m_rootbeerRuntimes);
    doLoad(m_rootbeerCudas);
  }

  /**
   * Loads all dynamic libraries in paths into memory for use in this program
   **/
  private void doLoad(List<String> paths) {
    for(String path : paths){
      File file = new File(path);
      if(file.exists()){
        System.load(file.getAbsolutePath());
        return;
      }
    }
  }

  /**
   * this function extracts the shared libraries from inside the .jar file
   * compiled with Rootbeer.jar to ~/.rootbeer/ in order to use them
   *
   * @param[in] filename binary to extract. Must be in
   *                     /org/trifort/rootbeer/runtime/binaries/
   **/
  private void extract(String filename) {
    String path = "/org/trifort/rootbeer/runtime/binaries/"+filename;
    try {
      InputStream is = CUDALoader.class.getResourceAsStream(path);
      if(is == null){
        path = "src"+path;
        is = new FileInputStream(path);
      }
      OutputStream os = new FileOutputStream(RootbeerPaths.v().getRootbeerHome()+filename);
      while(true){
        byte[] buffer = new byte[32*1024];
        int len = is.read(buffer);
        if(len == -1)
          break;
        os.write(buffer, 0, len);
      }
      os.flush();
      os.close();
      is.close();
    } catch(Exception ex){
      ex.printStackTrace();
      throw new RuntimeException(ex);
    }
  }
}
