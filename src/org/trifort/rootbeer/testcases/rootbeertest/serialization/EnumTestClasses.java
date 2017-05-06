package org.trifort.rootbeer.testcases.rootbeertest.serialization;

import org.trifort.rootbeer.runtime.Kernel;

/**
 * Created by BElsn on 02/05/2017.
 */
public class EnumTestClasses {
    public static enum TestEnum1 {
        one("ONE"),
        two("TWO"),
        three("THREE");

        private String val;

        TestEnum1(String val) {
            this.val = val;
        }
    }

    public static enum TestEnum2 {
        a,
        b,
        c
    }

    public static class TestRunOnGPU implements Kernel {
        private TestEnum1 e1;
        private TestEnum2 e2;
        private int idx;

        public TestRunOnGPU(int idx) {
            this.idx = idx;
        }

        @Override
        public void gpuMethod() {
            if(TestEnum1.values() == null)
                e1 = TestEnum1.one;
            else
                e1 = TestEnum1.values()[idx];
            if(TestEnum2.values() == null)
                e2 = TestEnum2.a;
            else
                e2 = TestEnum2.values()[idx];
        }

        public boolean compare(TestRunOnGPU other) {
            if(e1 != other.e1) {
                System.out.println(e1.toString() + "(" + e1.val + ") != " + other.e1.toString() + "(" + other.e1.val + ")");
                return false;
            }
            if(e2 != other.e2) {
                System.out.println(e2.toString() + " != " + other.e2.toString());
                return false;
            }

            System.out.println(e1.toString() + "(" + e1.val + ") == " + other.e1.toString() + "(" + other.e1.val + ")");
            System.out.println(e2.toString() + " == " + other.e2.toString());
            return true;
        }
    }
}
