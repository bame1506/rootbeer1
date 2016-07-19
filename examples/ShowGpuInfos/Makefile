ROOTBEER_ROOT := ../..

.phony: clean all

all: ShowGpuInfos.jar
clean:
	rm -f *.class *.jar *.tmp

ShowGpuInfos.class: ShowGpuInfos.java $(ROOTBEER_ROOT)/Rootbeer.jar
	javac $< -classpath $(ROOTBEER_ROOT)/Rootbeer.jar:.

cpu.jar: manifest.txt ShowGpuInfos.class
	jar -cvfm $@ $^

ShowGpuInfos.jar: ShowGpuInfos.class $(ROOTBEER_ROOT)/Rootbeer.jar cpu.jar
	zipmerge $@ $(ROOTBEER_ROOT)/Rootbeer.jar cpu.jar
