/*
 * Copyright 2012 Phil Pratt-Szeliga and other contributors
 * http://chirrup.org/
 *
 * See the file LICENSE for copying permission.
 */

package org.trifort.rootbeer.testcases.rootbeertest.serialization;

class AbstractTestDerivedClass2 extends AbstractTestBaseClass {

  public AbstractTestDerivedClass2() {
  }

  @Override
  public int add(int x, int y) {
    return x - y;
  }

}
