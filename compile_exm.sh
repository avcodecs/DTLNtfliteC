rm -rf bin
mkdir bin
g++ \
	examples/ns.c \
	-Iinc \
	-Ithirdpart/include \
	-Llib \
	-Lthirdpart/lib \
	-lathena \
	-lsndfile \
	-lpthread \
	-ldl \
	-lm \
	-Wl,-rpath,./lib \
	-o bin/dtln
