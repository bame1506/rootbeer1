ROOTBEER_ROOT := ../..

.phony: clean all

all: Count.jar
clean:
	rm -f *.class *.jar *.tmp

CountKernel.class: CountKernel.java $(ROOTBEER_ROOT)/Rootbeer.jar
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.
Count.class: Count.java $(ROOTBEER_ROOT)/Rootbeer.jar
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.

gpu.jar: CountKernel.class
	jar -cvf $@.tmp.jar *.class
	java -jar $(ROOTBEER_ROOT)/Rootbeer.jar $@.tmp.jar gpu.jar -64bit -computecapability=sm_30

cpu.jar: manifest.txt Count.class
	jar -cvfm $@ $^

Count.jar: Count.class $(ROOTBEER_ROOT)/Rootbeer.jar gpu.jar cpu.jar
	zipmerge $@ $(ROOTBEER_ROOT)/Rootbeer.jar gpu.jar cpu.jar
