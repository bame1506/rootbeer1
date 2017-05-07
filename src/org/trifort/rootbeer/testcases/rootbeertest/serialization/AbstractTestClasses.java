package org.trifort.rootbeer.testcases.rootbeertest.serialization;

import org.trifort.rootbeer.runtime.Kernel;

/**
 * Created by BElsn on 07/05/2017.
 */
public class AbstractTestClasses {
    public static abstract class AbstractTestBaseClass {

        public abstract int op(int x, int y);

        public static abstract class AbstractTestBaseClassExtension extends AbstractTestBaseClass {
            @Override
            public int op(int x, int y) {
                return abstractOp(x, y);
            }

            public abstract int abstractOp(int x, int y);
        }
    }

   public static class AbstractTestDerivedClass extends AbstractTestBaseClass {

        public AbstractTestDerivedClass() {
        }

        @Override
        public int op(int x, int y) {
            return x + y;
        }

        public static class AbstractTestDerivedClassOverride extends AbstractTestDerivedClass {
            @Override
            public int op(int x, int y) {
                return x << y;
            }
        }

    }

    public static class AbstractTestBaseClass2 {
        public int op(int x, int y) {
            return doOp(x,y);
        }

        private int doOp(int x, int y) {
            int ret = 1;
            while(y-- > 0)
                ret *= x;
            return ret;
        }
    }

    public static class AbstractTestDerivedClass3 extends AbstractTestBaseClass2 {

    }

    public static class AbstractTestDerivedClass2 extends AbstractTestBaseClass.AbstractTestBaseClassExtension {

        public AbstractTestDerivedClass2() {
        }

        @Override
        public int abstractOp(int x, int y) {
            return x * y;
        }
    }

    public static class AbstractRunOnGpu implements Kernel {

        private int m_result;
        private int m_result2;
        private int m_result3;
        private int m_result4;
        private int m_result5;

        public void gpuMethod() {
            AbstractTestBaseClass base_class = new AbstractTestDerivedClass();
            AbstractTestBaseClass base_class2 = new AbstractTestDerivedClass2();
            AbstractTestBaseClass base_class3 = new AbstractTestDerivedClass.AbstractTestDerivedClassOverride();
            AbstractTestDerivedClass2 derivedClass2 = (AbstractTestDerivedClass2) base_class2;
            AbstractTestDerivedClass3 derivedClass3 = new AbstractTestDerivedClass3();
            m_result = base_class.op(10, 10);
            m_result2 = base_class2.op(10, 10);
            m_result3 = base_class3.op(10, 10);
            m_result4 = derivedClass2.op(10, 10);
            m_result5 = derivedClass3.op(5, 5);
        }

        public boolean compare(AbstractRunOnGpu rhs) {
            if(m_result != rhs.m_result){
                System.out.println("m_result: " + m_result + " != " + rhs.m_result);
                return false;
            }
            if(m_result2 != rhs.m_result2){
                System.out.println("m_result2: " + m_result2 + " != " + rhs.m_result2);
                return false;
            }
            if(m_result3 != rhs.m_result3){
                System.out.println("m_result3: " + m_result3 + " != " + rhs.m_result3);
                return false;
            }
            if(m_result4 != rhs.m_result4){
                System.out.println("m_result4: " + m_result4 + " != " + rhs.m_result4);
                return false;
            }
            if(m_result5 != rhs.m_result5){
                System.out.println("m_result4: " + m_result5 + " != " + rhs.m_result5);
                return false;
            }
            System.out.println("m_result: " + m_result + " == " + rhs.m_result);
            System.out.println("m_result2: " + m_result2 + " == " + rhs.m_result2);
            System.out.println("m_result3: " + m_result3 + " == " + rhs.m_result3);
            System.out.println("m_result4: " + m_result4 + " == " + rhs.m_result4);
            System.out.println("m_result4: " + m_result5 + " == " + rhs.m_result5);
            return true;
        }
    }


}
