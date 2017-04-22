/*
 * Copyright 2013 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.testcases.rootbeertest.serialization;

import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.test.TestSerialization;

import java.util.ArrayList;
import java.util.List;

public class InterfaceTest implements TestSerialization {

  public List<Kernel> create() {
    List<Kernel> ret = new ArrayList<Kernel>();
    for(int i = 0; i < 2; ++i){
      ret.add(new InterfaceRunOnGpu());
    }
    return ret;
  }

  public boolean compare(Kernel original, Kernel from_heap) {
    InterfaceRunOnGpu lhs = (InterfaceRunOnGpu) original;
    InterfaceRunOnGpu rhs = (InterfaceRunOnGpu) from_heap;
    return lhs.compare(rhs);
  }
}
