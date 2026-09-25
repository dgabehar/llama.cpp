set(TARGET ggml-rpc-server)
add_executable(${TARGET} rpc-server.cpp)
target_link_libraries(${TARGET} PRIVATE ggml)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if (LLAMA_BUILD_TESTS AND UNIX AND NOT GGML_BACKEND_DL)
    add_executable(test-rpc-multi-server ${PROJECT_SOURCE_DIR}/tests/test-rpc-multi-server.cpp)
    target_link_libraries(test-rpc-multi-server PRIVATE ggml ggml-rpc)
    target_include_directories(test-rpc-multi-server PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
    add_test(
        NAME test-rpc-multi-server
        COMMAND bash ${PROJECT_SOURCE_DIR}/tests/test-rpc-multi-server.sh
            $<TARGET_FILE:ggml-rpc-server>
            $<TARGET_FILE:test-rpc-multi-server>)
    set_property(TEST test-rpc-multi-server PROPERTY LABELS main)

    add_executable(test-rpc-server-multiclient ${PROJECT_SOURCE_DIR}/tests/test-rpc-server-multiclient.cpp)
    target_link_libraries(test-rpc-server-multiclient PRIVATE ggml)
    add_test(
        NAME test-rpc-server-multiclient
        COMMAND bash ${PROJECT_SOURCE_DIR}/tests/test-rpc-server-multiclient.sh
            $<TARGET_FILE:ggml-rpc-server>
            $<TARGET_FILE:test-rpc-server-multiclient>)
    set_property(TEST test-rpc-server-multiclient PROPERTY LABELS main)
    set_property(TEST test-rpc-server-multiclient PROPERTY TIMEOUT 120)
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
