/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.testcases.rootbeertest.serialization;

class AbstractTestDerivedClass2 extends AbstractTestBaseClass.AbstractTestBaseClassExtension {

  public AbstractTestDerivedClass2() {
  }

  @Override
  public int abstractOp(int x, int y) {
    return x * y;
  }

  @Override
  public int op(int x, int y) {
    return super.op(x, y);
  }
}
