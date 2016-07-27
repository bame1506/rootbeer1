# ( cd ../.. && ( cd csrc && ./compile_linux_x64 ) && ant clean && 'rm'f dist/Rootbeer1.jar Rootbeer.jar && ant jar && ./pack-rootbeer ) && make clean && make -B && java -jar Count.jar
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
	time -p java -jar $(ROOTBEER_ROOT)/Rootbeer.jar -nofatjar $@.tmp.jar gpu.jar

cpu.jar: manifest.txt Count.class
	jar -cvfm $@ $^

Count.jar: Count.class $(ROOTBEER_ROOT)/Rootbeer.jar gpu.jar cpu.jar
	zipmerge $@ $(ROOTBEER_ROOT)/Rootbeer.jar gpu.jar cpu.jar
