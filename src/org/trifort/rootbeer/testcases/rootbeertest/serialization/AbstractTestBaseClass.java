/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.testcases.rootbeertest.serialization;

public abstract class AbstractTestBaseClass {

  public abstract int op(int x, int y);

  public static abstract class AbstractTestBaseClassExtension extends AbstractTestBaseClass {
    @Override
    public int op(int x, int y) {
      return abstractOp(x, y);
    }

    public abstract int abstractOp(int x, int y);
  }
}
