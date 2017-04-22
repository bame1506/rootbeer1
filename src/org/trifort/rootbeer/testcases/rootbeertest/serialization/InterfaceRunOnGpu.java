package org.trifort.rootbeer.testcases.rootbeertest.serialization;

import org.trifort.rootbeer.runtime.Kernel;

public class InterfaceRunOnGpu implements Kernel {

    private int m_result;
    private int m_result2;
    private int m_result3;
    private int m_result4;
    private int m_result5;
    private int m_result6;

    public void gpuMethod() {
        I1 i1 = new C1();
        I2 i2 = new C2();
        I12 i12 = new C12();
        I1 i121 = i12;
        I2 i122 = i12;

        m_result = i1.op1(10, 10);
        m_result2 = i2.op2(10, 10);
        m_result3 = i12.op1(10, 10);
        m_result4 = i12.op2(10, 10);
        m_result5 = i121.op1(10, 10);
        m_result6 = i122.op2(10, 10);
    }

    public boolean compare(InterfaceRunOnGpu rhs) {
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
            System.out.println("m_result5: " + m_result5 + " != " + rhs.m_result5);
            return false;
        }
        if(m_result6 != rhs.m_result6){
            System.out.println("m_result6: " + m_result6 + " != " + rhs.m_result6);
            return false;
        }
        System.out.println("m_result: " + m_result + " == " + rhs.m_result);
        System.out.println("m_result2: " + m_result2 + " == " + rhs.m_result2);
        System.out.println("m_result3: " + m_result3 + " == " + rhs.m_result3);
        System.out.println("m_result4: " + m_result4 + " == " + rhs.m_result4);
        System.out.println("m_result5: " + m_result5 + " == " + rhs.m_result5);
        System.out.println("m_result6: " + m_result6 + " == " + rhs.m_result6);
        return true;
    }


    private interface I1 {
        int op1(int x, int y);
    }


    private interface I2 {
        int op2(int x, int y);
    }

    interface I12 extends I1, I2 {}

    private class C1 implements I1 {
        @Override
        public int op1(int x, int y) {
            return x + y;
        }
    }

    private class C2 implements I2 {
        @Override
        public int op2(int x, int y) {
            return x * y;
        }
    }

    private class C12 implements I12 {
        @Override
        public int op1(int x, int y) {
            return x << y;
        }
        @Override
        public int op2(int x, int y) {
            return (int) Math.pow(x,y);
        }
    }
}
