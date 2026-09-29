###########################################################################
## Makefile generated for component 'yolov2_detect'. 
## 
## Makefile     : yolov2_detect_rtw.mk
## Generated on : Thu Sep 24 21:15:33 2026
## Final product: ./yolov2_detect.lib
## Product type : static-library
## 
###########################################################################

###########################################################################
## MACROS
###########################################################################

# Macro Descriptions:
# PRODUCT_NAME            Name of the system to build
# MAKEFILE                Name of this makefile
# COMPILER_COMMAND_FILE   Compiler command listing model reference header paths
# CMD_FILE                Command file
# MODELLIB                Static library target

PRODUCT_NAME              = yolov2_detect
MAKEFILE                  = yolov2_detect_rtw.mk
MATLAB_ROOT               = C:/PROGRA~1/MATLAB/R2026a
MATLAB_BIN                = C:/PROGRA~1/MATLAB/R2026a/bin
MATLAB_ARCH_BIN           = $(MATLAB_BIN)/win64
START_DIR                 = C:/Users/mrese/OneDrive/Documentos/MATLAB
TGT_FCN_LIB               = ISO_C++11
SOLVER_OBJ                = 
CLASSIC_INTERFACE         = 0
MODEL_HAS_DYNAMICALLY_LOADED_SFCNS = 
RELATIVE_PATH_TO_ANCHOR   = ../../..
COMPILER_COMMAND_FILE     = yolov2_detect_rtw_comp.rsp
CMD_FILE                  = yolov2_detect_rtw.rsp
C_STANDARD_OPTS           = -fwrapv
CPP_STANDARD_OPTS         = -fwrapv
MODELLIB                  = yolov2_detect.lib

###########################################################################
## TOOLCHAIN SPECIFICATIONS
###########################################################################

# Toolchain Name:          MinGW64 | gmake (64-bit Windows)
# Supported Version(s):    14.x
# ToolchainInfo Version:   2026a
# Specification Revision:  1.0
# 
#-------------------------------------------
# Macros assumed to be defined elsewhere
#-------------------------------------------

# C_STANDARD_OPTS
# CPP_STANDARD_OPTS
# MINGW_ROOT
# MINGW_C_STANDARD_OPTS

#-----------
# MACROS
#-----------

WARN_FLAGS            = -Wall -W -Wwrite-strings -Winline -Wstrict-prototypes -Wnested-externs -Wpointer-arith -Wcast-align -Wno-stringop-overflow
WARN_FLAGS_MAX        = $(WARN_FLAGS) -Wcast-qual -Wshadow
CPP_WARN_FLAGS        = -Wall -W -Wwrite-strings -Winline -Wpointer-arith -Wcast-align -Wno-stringop-overflow
CPP_WARN_FLAGS_MAX    = $(CPP_WARN_FLAGS) -Wcast-qual -Wshadow
MW_EXTERNLIB_DIR      = $(MATLAB_ROOT)/extern/lib/win64/mingw64
SHELL                 = %SystemRoot%/system32/cmd.exe

TOOLCHAIN_SRCS = 
TOOLCHAIN_INCS = 
TOOLCHAIN_LIBS = -lws2_32

FORMAT_FOR_ECHO_CMD              = $(strip $(subst >,^>,\
	$(subst <,^<,\
	$(subst |,^|,\
	$(subst &,^&,\
	$(subst ",^",\
	$(subst ^,^^,\
	$1)))))))
FORMAT_FOR_ECHO                  = $(FORMAT_FOR_ECHO_CMD)
HASH                             = \#
SEMICOLON                        = ;
UNESCAPE_SEMICOLONS              = $(subst \;,;,$1)
ADD_QUOTES                       = $(foreach aPath,$1,"$(aPath)")

#------------------------
# BUILD TOOL COMMANDS
#------------------------

# C Compiler: GNU C Compiler
CC_PATH = $(MINGW_ROOT)
CC = "$(CC_PATH)/gcc"

# Linker: GNU Linker
LD_PATH = $(MINGW_ROOT)
LD = "$(LD_PATH)/g++"

# C++ Compiler: GNU C++ Compiler
CPP_PATH = $(MINGW_ROOT)
CPP = "$(CPP_PATH)/g++"

# C++ Linker: GNU C++ Linker
CPP_LD_PATH = $(MINGW_ROOT)
CPP_LD = "$(CPP_LD_PATH)/g++"

# Archiver: GNU Archiver
AR_PATH = $(MINGW_ROOT)
AR = "$(AR_PATH)/ar"

# MEX Tool: MEX Tool
MEX_PATH = $(MATLAB_ARCH_BIN)
MEX = "$(MEX_PATH)/mex"

# Download: Download
DOWNLOAD =

# Execute: Execute
EXECUTE = $(PRODUCT)

# Builder: GMAKE Utility
MAKE_PATH = $(MINGW_ROOT)
MAKE = "$(MAKE_PATH)/mingw32-make.exe"


#-------------------------
# Directives/Utilities
#-------------------------

CDEBUG              = -g
C_OUTPUT_FLAG       = -o
LDDEBUG             = -g
OUTPUT_FLAG         = -o
CPPDEBUG            = -g
CPP_OUTPUT_FLAG     = -o
CPPLDDEBUG          = -g
OUTPUT_FLAG         = -o
ARDEBUG             =
STATICLIB_OUTPUT_FLAG =
MEX_DEBUG           = -g
RM                  = @del
ECHO                = @echo
MV                  = @move
RUN                 =

#--------------------------------------
# "Faster Runs" Build Configuration
#--------------------------------------

ARFLAGS              = ruvs
CFLAGS               = -c $(MINGW_C_STANDARD_OPTS) -m64 -Wno-error=incompatible-pointer-types -Wno-error=stringop-overflow \
                       -O3
CPPFLAGS             = -c $(CPP_STANDARD_OPTS) -m64 -Wno-error=incompatible-pointer-types -Wno-error=stringop-overflow \
                       -O3
CPP_LDFLAGS          =  -static -m64
CPP_SHAREDLIB_LDFLAGS  = -shared -Wl,--no-undefined \
                         -Wl,--out-implib,$(basename $(PRODUCT)).lib
DOWNLOAD_FLAGS       =
EXECUTE_FLAGS        =
LDFLAGS              =  -static -m64
MEX_CPPFLAGS         =
MEX_CPPLDFLAGS       =
MEX_CFLAGS           =
MEX_LDFLAGS          =
MAKE_FLAGS           = -j $(MAX_MAKE_JOBS) -l $(MAX_MAKE_LOAD_AVG) -Oline -f $(MAKEFILE)
SHAREDLIB_LDFLAGS    = -shared -Wl,--no-undefined \
                       -Wl,--out-implib,$(basename $(PRODUCT)).lib



###########################################################################
## OUTPUT INFO
###########################################################################

PRODUCT = ./yolov2_detect.lib
PRODUCT_TYPE = "static-library"
BUILD_TYPE = "Static Library"

###########################################################################
## INCLUDE PATHS
###########################################################################

INCLUDES_BUILDINFO = 

INCLUDES = $(INCLUDES_BUILDINFO)

###########################################################################
## DEFINES
###########################################################################

DEFINES_ = -D__USE_MINGW_ANSI_STDIO=1
DEFINES_CUSTOM = 
DEFINES_STANDARD = -DMODEL=yolov2_detect

DEFINES = $(DEFINES_) $(DEFINES_CUSTOM) $(DEFINES_STANDARD)

###########################################################################
## SOURCE FILES
###########################################################################

SRCS = $(START_DIR)/codegen/lib/yolov2_detect/yolov2_detect_data.cpp $(START_DIR)/codegen/lib/yolov2_detect/rt_nonfinite.cpp $(START_DIR)/codegen/lib/yolov2_detect/rtGetNaN.cpp $(START_DIR)/codegen/lib/yolov2_detect/rtGetInf.cpp $(START_DIR)/codegen/lib/yolov2_detect/yolov2_detect_initialize.cpp $(START_DIR)/codegen/lib/yolov2_detect/yolov2_detect_terminate.cpp $(START_DIR)/codegen/lib/yolov2_detect/yolov2_detect.cpp $(START_DIR)/codegen/lib/yolov2_detect/YOLOv2Network.cpp $(START_DIR)/codegen/lib/yolov2_detect/imresize.cpp $(START_DIR)/codegen/lib/yolov2_detect/conv2dDirectOptimizedColMajor.cpp $(START_DIR)/codegen/lib/yolov2_detect/callPredict.cpp $(START_DIR)/codegen/lib/yolov2_detect/YOLOv2TransformLayer.cpp $(START_DIR)/codegen/lib/yolov2_detect/internal_softmax.cpp $(START_DIR)/codegen/lib/yolov2_detect/findHelper.cpp $(START_DIR)/codegen/lib/yolov2_detect/strtrim.cpp $(START_DIR)/codegen/lib/yolov2_detect/cellstr_unique.cpp $(START_DIR)/codegen/lib/yolov2_detect/cellstr_sort.cpp $(START_DIR)/codegen/lib/yolov2_detect/selectStrongestBboxCodegen.cpp $(START_DIR)/codegen/lib/yolov2_detect/elementwiseOperationInPlace.cpp $(START_DIR)/codegen/lib/yolov2_detect/nullAssignment.cpp $(START_DIR)/codegen/lib/yolov2_detect/sort.cpp $(START_DIR)/codegen/lib/yolov2_detect/sortIdx.cpp $(START_DIR)/codegen/lib/yolov2_detect/insertionsort.cpp $(START_DIR)/codegen/lib/yolov2_detect/introsort.cpp $(START_DIR)/codegen/lib/yolov2_detect/heapsort.cpp $(START_DIR)/codegen/lib/yolov2_detect/poolingOperation.cpp

ALL_SRCS = $(SRCS)

###########################################################################
## OBJECTS
###########################################################################

OBJS = yolov2_detect_data.obj rt_nonfinite.obj rtGetNaN.obj rtGetInf.obj yolov2_detect_initialize.obj yolov2_detect_terminate.obj yolov2_detect.obj YOLOv2Network.obj imresize.obj conv2dDirectOptimizedColMajor.obj callPredict.obj YOLOv2TransformLayer.obj internal_softmax.obj findHelper.obj strtrim.obj cellstr_unique.obj cellstr_sort.obj selectStrongestBboxCodegen.obj elementwiseOperationInPlace.obj nullAssignment.obj sort.obj sortIdx.obj insertionsort.obj introsort.obj heapsort.obj poolingOperation.obj

ALL_OBJS = $(OBJS)

###########################################################################
## PREBUILT OBJECT FILES
###########################################################################

PREBUILT_OBJS = 

###########################################################################
## LIBRARIES
###########################################################################

LIBS = 

###########################################################################
## SYSTEM LIBRARIES
###########################################################################

SYSTEM_LIBS = 

###########################################################################
## ADDITIONAL TOOLCHAIN FLAGS
###########################################################################

#---------------
# C Compiler
#---------------

CFLAGS_OPTS = -fopenmp
CFLAGS_BASIC = $(DEFINES) $(INCLUDES) @$(COMPILER_COMMAND_FILE)

CFLAGS += $(CFLAGS_OPTS) $(CFLAGS_BASIC)

#-----------------
# C++ Compiler
#-----------------

CPPFLAGS_OPTS = -fopenmp
CPPFLAGS_BASIC = $(DEFINES) $(INCLUDES) @$(COMPILER_COMMAND_FILE)

CPPFLAGS += $(CPPFLAGS_OPTS) $(CPPFLAGS_BASIC)

#---------------
# C++ Linker
#---------------

CPP_LDFLAGS_ = -fopenmp

CPP_LDFLAGS += $(CPP_LDFLAGS_)

#------------------------------
# C++ Shared Library Linker
#------------------------------

CPP_SHAREDLIB_LDFLAGS_ = -fopenmp

CPP_SHAREDLIB_LDFLAGS += $(CPP_SHAREDLIB_LDFLAGS_)

#-----------
# Linker
#-----------

LDFLAGS_ = -fopenmp

LDFLAGS += $(LDFLAGS_)

#---------------------
# MEX C++ Compiler
#---------------------

MEX_CPP_Compiler_BASIC =  @$(COMPILER_COMMAND_FILE)

MEX_CPPFLAGS += $(MEX_CPP_Compiler_BASIC)

#-----------------
# MEX Compiler
#-----------------

MEX_Compiler_BASIC =  @$(COMPILER_COMMAND_FILE)

MEX_CFLAGS += $(MEX_Compiler_BASIC)

#--------------------------
# Shared Library Linker
#--------------------------

SHAREDLIB_LDFLAGS_ = -fopenmp

SHAREDLIB_LDFLAGS += $(SHAREDLIB_LDFLAGS_)

###########################################################################
## INLINED COMMANDS
###########################################################################


MINGW_C_STANDARD_OPTS = $(C_STANDARD_OPTS)


###########################################################################
## PHONY TARGETS
###########################################################################

.PHONY : all build clean info prebuild download execute


all : build
	@echo $(call FORMAT_FOR_ECHO,### Successfully generated all binary outputs.)


build : prebuild $(PRODUCT)


prebuild : 


download : $(PRODUCT)


execute : download


###########################################################################
## FINAL TARGET
###########################################################################

#---------------------------------
# Create a static library         
#---------------------------------

$(PRODUCT) : $(OBJS) $(PREBUILT_OBJS)
	@echo $(call FORMAT_FOR_ECHO,### Creating static library "$(PRODUCT)" ...)
	$(AR) $(ARFLAGS)  $(PRODUCT) @$(CMD_FILE)
	@echo $(call FORMAT_FOR_ECHO,### Created: "$(PRODUCT)")


###########################################################################
## INTERMEDIATE TARGETS
###########################################################################

#---------------------
# SOURCE-TO-OBJECT
#---------------------

%.obj : %.c
	$(CC) $(CFLAGS) -o "$@" "$<"


%.obj : %.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : %.cc
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : %.cp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : %.cxx
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : %.CPP
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : %.c++
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : %.C
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(RELATIVE_PATH_TO_ANCHOR)/%.c
	$(CC) $(CFLAGS) -o "$@" "$<"


%.obj : $(RELATIVE_PATH_TO_ANCHOR)/%.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(RELATIVE_PATH_TO_ANCHOR)/%.cc
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(RELATIVE_PATH_TO_ANCHOR)/%.cp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(RELATIVE_PATH_TO_ANCHOR)/%.cxx
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(RELATIVE_PATH_TO_ANCHOR)/%.CPP
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(RELATIVE_PATH_TO_ANCHOR)/%.c++
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(RELATIVE_PATH_TO_ANCHOR)/%.C
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(START_DIR)/codegen/lib/yolov2_detect/%.c
	$(CC) $(CFLAGS) -o "$@" "$<"


%.obj : $(START_DIR)/codegen/lib/yolov2_detect/%.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(START_DIR)/codegen/lib/yolov2_detect/%.cc
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(START_DIR)/codegen/lib/yolov2_detect/%.cp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(START_DIR)/codegen/lib/yolov2_detect/%.cxx
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(START_DIR)/codegen/lib/yolov2_detect/%.CPP
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(START_DIR)/codegen/lib/yolov2_detect/%.c++
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(START_DIR)/codegen/lib/yolov2_detect/%.C
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(START_DIR)/%.c
	$(CC) $(CFLAGS) -o "$@" "$<"


%.obj : $(START_DIR)/%.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(START_DIR)/%.cc
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(START_DIR)/%.cp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(START_DIR)/%.cxx
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(START_DIR)/%.CPP
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(START_DIR)/%.c++
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


%.obj : $(START_DIR)/%.C
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


yolov2_detect_data.obj : $(START_DIR)/codegen/lib/yolov2_detect/yolov2_detect_data.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


rt_nonfinite.obj : $(START_DIR)/codegen/lib/yolov2_detect/rt_nonfinite.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


rtGetNaN.obj : $(START_DIR)/codegen/lib/yolov2_detect/rtGetNaN.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


rtGetInf.obj : $(START_DIR)/codegen/lib/yolov2_detect/rtGetInf.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


yolov2_detect_initialize.obj : $(START_DIR)/codegen/lib/yolov2_detect/yolov2_detect_initialize.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


yolov2_detect_terminate.obj : $(START_DIR)/codegen/lib/yolov2_detect/yolov2_detect_terminate.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


yolov2_detect.obj : $(START_DIR)/codegen/lib/yolov2_detect/yolov2_detect.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


YOLOv2Network.obj : $(START_DIR)/codegen/lib/yolov2_detect/YOLOv2Network.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


imresize.obj : $(START_DIR)/codegen/lib/yolov2_detect/imresize.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


conv2dDirectOptimizedColMajor.obj : $(START_DIR)/codegen/lib/yolov2_detect/conv2dDirectOptimizedColMajor.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


callPredict.obj : $(START_DIR)/codegen/lib/yolov2_detect/callPredict.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


YOLOv2TransformLayer.obj : $(START_DIR)/codegen/lib/yolov2_detect/YOLOv2TransformLayer.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


internal_softmax.obj : $(START_DIR)/codegen/lib/yolov2_detect/internal_softmax.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


findHelper.obj : $(START_DIR)/codegen/lib/yolov2_detect/findHelper.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


strtrim.obj : $(START_DIR)/codegen/lib/yolov2_detect/strtrim.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


cellstr_unique.obj : $(START_DIR)/codegen/lib/yolov2_detect/cellstr_unique.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


cellstr_sort.obj : $(START_DIR)/codegen/lib/yolov2_detect/cellstr_sort.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


selectStrongestBboxCodegen.obj : $(START_DIR)/codegen/lib/yolov2_detect/selectStrongestBboxCodegen.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


elementwiseOperationInPlace.obj : $(START_DIR)/codegen/lib/yolov2_detect/elementwiseOperationInPlace.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


nullAssignment.obj : $(START_DIR)/codegen/lib/yolov2_detect/nullAssignment.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


sort.obj : $(START_DIR)/codegen/lib/yolov2_detect/sort.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


sortIdx.obj : $(START_DIR)/codegen/lib/yolov2_detect/sortIdx.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


insertionsort.obj : $(START_DIR)/codegen/lib/yolov2_detect/insertionsort.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


introsort.obj : $(START_DIR)/codegen/lib/yolov2_detect/introsort.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


heapsort.obj : $(START_DIR)/codegen/lib/yolov2_detect/heapsort.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


poolingOperation.obj : $(START_DIR)/codegen/lib/yolov2_detect/poolingOperation.cpp
	$(CPP) $(CPPFLAGS) -o "$@" "$<"


###########################################################################
## DEPENDENCIES
###########################################################################

$(ALL_OBJS) : rtw_proj.tmw $(COMPILER_COMMAND_FILE) $(MAKEFILE)


###########################################################################
## MISCELLANEOUS TARGETS
###########################################################################

info : 
	@echo $(call FORMAT_FOR_ECHO,### PRODUCT = $(PRODUCT))
	@echo $(call FORMAT_FOR_ECHO,### PRODUCT_TYPE = $(PRODUCT_TYPE))
	@echo $(call FORMAT_FOR_ECHO,### BUILD_TYPE = $(BUILD_TYPE))
	@echo $(call FORMAT_FOR_ECHO,### INCLUDES = $(INCLUDES))
	@echo $(call FORMAT_FOR_ECHO,### DEFINES = $(DEFINES))
	@echo $(call FORMAT_FOR_ECHO,### ALL_SRCS = $(ALL_SRCS))
	@echo $(call FORMAT_FOR_ECHO,### ALL_OBJS = $(ALL_OBJS))
	@echo $(call FORMAT_FOR_ECHO,### LIBS = $(LIBS))
	@echo $(call FORMAT_FOR_ECHO,### MODELREF_LIBS = $(MODELREF_LIBS))
	@echo $(call FORMAT_FOR_ECHO,### SYSTEM_LIBS = $(SYSTEM_LIBS))
	@echo $(call FORMAT_FOR_ECHO,### TOOLCHAIN_LIBS = $(TOOLCHAIN_LIBS))
	@echo $(call FORMAT_FOR_ECHO,### CFLAGS = $(CFLAGS))
	@echo $(call FORMAT_FOR_ECHO,### LDFLAGS = $(LDFLAGS))
	@echo $(call FORMAT_FOR_ECHO,### SHAREDLIB_LDFLAGS = $(SHAREDLIB_LDFLAGS))
	@echo $(call FORMAT_FOR_ECHO,### CPPFLAGS = $(CPPFLAGS))
	@echo $(call FORMAT_FOR_ECHO,### CPP_LDFLAGS = $(CPP_LDFLAGS))
	@echo $(call FORMAT_FOR_ECHO,### CPP_SHAREDLIB_LDFLAGS = $(CPP_SHAREDLIB_LDFLAGS))
	@echo $(call FORMAT_FOR_ECHO,### ARFLAGS = $(ARFLAGS))
	@echo $(call FORMAT_FOR_ECHO,### MEX_CFLAGS = $(MEX_CFLAGS))
	@echo $(call FORMAT_FOR_ECHO,### MEX_CPPFLAGS = $(MEX_CPPFLAGS))
	@echo $(call FORMAT_FOR_ECHO,### MEX_LDFLAGS = $(MEX_LDFLAGS))
	@echo $(call FORMAT_FOR_ECHO,### MEX_CPPLDFLAGS = $(MEX_CPPLDFLAGS))
	@echo $(call FORMAT_FOR_ECHO,### DOWNLOAD_FLAGS = $(DOWNLOAD_FLAGS))
	@echo $(call FORMAT_FOR_ECHO,### EXECUTE_FLAGS = $(EXECUTE_FLAGS))
	@echo $(call FORMAT_FOR_ECHO,### MAKE_FLAGS = $(MAKE_FLAGS))


clean : 
	$(ECHO) "### Deleting all derived files ..."
	$(RM) $(subst /,\,$(PRODUCT))
	$(RM) $(subst /,\,$(ALL_OBJS))
	$(ECHO) "### Deleted all derived files."


