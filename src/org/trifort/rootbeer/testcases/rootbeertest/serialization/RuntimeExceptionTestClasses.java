package org.trifort.rootbeer.testcases.rootbeertest.serialization;

import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.RootbeerGpu;

/**
 * Created by BElsn on 02/05/2017.
 */
public class RuntimeExceptionTestClasses {
    public static class TestException extends RuntimeException {
        public TestException() {
        }

        public TestException(String message) {
            super(message);
        }

        public TestException(String message, Throwable cause) {
            super(message, cause);
        }

        public TestException(Throwable cause) {
            super(cause);
        }

        public TestException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
            super(message, cause, enableSuppression, writableStackTrace);
        }
    }


    public static  class TestRunOnGPU implements Kernel {
        private String message = "fail";

        @Override
        public void gpuMethod() {
            try {
                throw new TestException("Success");
            } catch (Exception e) {
                message = e.getMessage();
            }
        }

        public boolean compare(TestRunOnGPU other) {
            if(!message.equals(other.message)) {
                System.out.println(message + " != " + other.message);
                return false;
            }

            System.out.println(message + " = " + other.message);
            return true;
        }
    }
}
