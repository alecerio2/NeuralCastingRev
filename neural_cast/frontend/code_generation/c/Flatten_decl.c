
$(DEFINE_CONNECTED_OUTPUT)
#ifdef CONNECTED_OUTPUT
float* tensor_$(OUTPUT_NAME);
#undef CONNECTED_OUTPUT
#endif