package org.trifort.rootbeer.testcases.rootbeertest.serialization;

import java.util.ArrayList;
import java.util.List;

import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.test.TestSerialization;

public class RuntimeExceptionTest implements TestSerialization {

    public List<Kernel> create() {
        List<Kernel> ret = new ArrayList<Kernel>();
        for(int i = 0; i < 2; ++i){
            ret.add(new RuntimeExceptionTestClasses.TestRunOnGPU());
        }
        return ret;
    }

    public boolean compare(Kernel original, Kernel from_heap) {
        RuntimeExceptionTestClasses.TestRunOnGPU lhs = (RuntimeExceptionTestClasses.TestRunOnGPU) original;
        RuntimeExceptionTestClasses.TestRunOnGPU rhs = (RuntimeExceptionTestClasses.TestRunOnGPU) from_heap;
        return lhs.compare(rhs);
    }
}
