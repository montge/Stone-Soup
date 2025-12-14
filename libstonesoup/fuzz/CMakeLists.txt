# CMakeLists.txt for Stone Soup C Library Fuzzing
#
# Build fuzzing targets with:
#   cmake -DCMAKE_C_COMPILER=clang -DENABLE_FUZZING=ON ..
#   make fuzz_kalman
#
# Run fuzzing:
#   ./fuzz_kalman -max_len=1024 corpus/

cmake_minimum_required(VERSION 3.14)

option(ENABLE_FUZZING "Enable fuzzing targets" OFF)
option(ENABLE_ASAN "Enable AddressSanitizer" ON)
option(ENABLE_UBSAN "Enable UndefinedBehaviorSanitizer" ON)

if(ENABLE_FUZZING)
    # Require Clang for libFuzzer support
    if(NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
        message(WARNING "Fuzzing requires Clang compiler for libFuzzer support")
    endif()

    # Sanitizer flags
    set(SANITIZER_FLAGS "")
    if(ENABLE_ASAN)
        set(SANITIZER_FLAGS "${SANITIZER_FLAGS},address")
    endif()
    if(ENABLE_UBSAN)
        set(SANITIZER_FLAGS "${SANITIZER_FLAGS},undefined")
    endif()

    # Fuzz targets
    add_executable(fuzz_kalman fuzz_kalman.c)
    target_include_directories(fuzz_kalman PRIVATE ${CMAKE_SOURCE_DIR}/include)
    target_link_libraries(fuzz_kalman stonesoup_static m)

    # Add fuzzer and sanitizer flags
    if(CMAKE_C_COMPILER_ID STREQUAL "Clang")
        target_compile_options(fuzz_kalman PRIVATE
            -g -O1
            -fsanitize=fuzzer${SANITIZER_FLAGS}
            -fno-omit-frame-pointer
        )
        target_link_options(fuzz_kalman PRIVATE
            -fsanitize=fuzzer${SANITIZER_FLAGS}
        )
    endif()

    # Create corpus directory
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/corpus)

    # Create seed corpus with basic inputs
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/corpus/seed_small
        "\x02\x00\x00\x00\x00\x00\x00\x00")
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/corpus/seed_medium
        "\x04\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xf0\x3f")

    # Custom target to run fuzzing
    add_custom_target(run_fuzz
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/fuzz_kalman
                -max_len=1024
                -max_total_time=60
                ${CMAKE_CURRENT_BINARY_DIR}/corpus
        DEPENDS fuzz_kalman
        COMMENT "Running fuzzing for 60 seconds..."
    )
endif()
