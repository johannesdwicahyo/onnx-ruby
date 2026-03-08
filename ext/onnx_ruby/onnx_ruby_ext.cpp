#include <rice/rice.hpp>
#include <rice/stl.hpp>
#include <onnxruntime_cxx_api.h>
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif
#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <memory>
#include <unordered_map>

using namespace Rice;

// Global ORT environment (initialized once)
static Ort::Env& get_env(int log_level = ORT_LOGGING_LEVEL_WARNING) {
  static Ort::Env env(static_cast<OrtLoggingLevel>(log_level), "onnx_ruby");
  return env;
}

// Map ORT element type to Ruby symbol name
static std::string ort_type_to_string(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return "float64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:  return "string";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return "bool";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:  return "uint16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:   return "int16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
    default: return "unknown";
  }
}

// Flatten a nested Ruby array into a flat vector and compute shape
static void flatten_ruby_array(VALUE arr, std::vector<VALUE>& flat, std::vector<int64_t>& shape, int depth) {
  if (!RB_TYPE_P(arr, T_ARRAY)) {
    flat.push_back(arr);
    return;
  }

  long len = RARRAY_LEN(arr);
  if (depth == (int)shape.size()) {
    shape.push_back(len);
  }

  for (long i = 0; i < len; i++) {
    flatten_ruby_array(rb_ary_entry(arr, i), flat, shape, depth + 1);
  }
}

// Convert an ORT output tensor to a nested Ruby array
static Rice::Object tensor_to_ruby(const Ort::Value& tensor) {
  auto type_info = tensor.GetTensorTypeAndShapeInfo();
  auto shape = type_info.GetShape();
  auto elem_type = type_info.GetElementType();
  size_t total = type_info.GetElementCount();

  // Build flat Ruby array first
  Rice::Array flat;

  switch (elem_type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      const float* data = tensor.GetTensorData<float>();
      for (size_t i = 0; i < total; i++) {
        flat.push(Rice::Object(rb_float_new(static_cast<double>(data[i]))));
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
      const double* data = tensor.GetTensorData<double>();
      for (size_t i = 0; i < total; i++) {
        flat.push(Rice::Object(rb_float_new(data[i])));
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
      const int32_t* data = tensor.GetTensorData<int32_t>();
      for (size_t i = 0; i < total; i++) {
        flat.push(Rice::Object(INT2NUM(data[i])));
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      const int64_t* data = tensor.GetTensorData<int64_t>();
      for (size_t i = 0; i < total; i++) {
        flat.push(Rice::Object(LONG2NUM(data[i])));
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
      const bool* data = tensor.GetTensorData<bool>();
      for (size_t i = 0; i < total; i++) {
        flat.push(Rice::Object(data[i] ? Qtrue : Qfalse));
      }
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: {
      size_t count = total;
      std::vector<std::string> strings(count);
      // GetStringTensorContent approach
      size_t total_len = tensor.GetStringTensorDataLength();
      std::vector<char> buffer(total_len);
      std::vector<size_t> offsets(count);
      tensor.GetStringTensorContent(buffer.data(), total_len, offsets.data(), count);
      for (size_t i = 0; i < count; i++) {
        size_t start = offsets[i];
        size_t end = (i + 1 < count) ? offsets[i + 1] : total_len;
        flat.push(Rice::Object(rb_str_new(buffer.data() + start, end - start)));
      }
      break;
    }
    default:
      throw std::runtime_error("Unsupported output tensor element type: " + ort_type_to_string(elem_type));
  }

  // Reshape flat array into nested arrays according to shape
  if (shape.empty() || shape.size() == 1) {
    return flat;
  }

  // Reshape from innermost dimension outward
  Rice::Array current = flat;
  for (int d = (int)shape.size() - 1; d >= 1; d--) {
    int64_t dim_size = shape[d];
    Rice::Array reshaped;
    long total_items = RARRAY_LEN(current.value());
    for (long i = 0; i < total_items; i += dim_size) {
      Rice::Array slice;
      for (int64_t j = 0; j < dim_size; j++) {
        slice.push(Rice::Object(rb_ary_entry(current.value(), i + j)));
      }
      reshaped.push(slice);
    }
    current = reshaped;
  }

  return current;
}

// Map Ruby optimization level symbol to ORT enum
static GraphOptimizationLevel parse_opt_level(const std::string& level) {
  if (level == "none" || level == "disabled") return GraphOptimizationLevel::ORT_DISABLE_ALL;
  if (level == "basic") return GraphOptimizationLevel::ORT_ENABLE_BASIC;
  if (level == "extended") return GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
  return GraphOptimizationLevel::ORT_ENABLE_ALL; // "all" or default
}

// Optimize a model and save to disk
static Rice::Object optimize_model(const std::string& input_path, const std::string& output_path,
                                   const std::string& opt_level) {
  Ort::SessionOptions opts;
  opts.SetGraphOptimizationLevel(parse_opt_level(opt_level));
  opts.SetOptimizedModelFilePath(output_path.c_str());

  // Creating the session triggers optimization and saves the optimized model
  Ort::Session session(get_env(), input_path.c_str(), opts);
  return Rice::Object(Qtrue);
}

// Get available execution providers
static Rice::Array available_providers() {
  Rice::Array result;
  auto providers = Ort::GetAvailableProviders();
  for (const auto& p : providers) {
    result.push(Rice::String(p));
  }
  return result;
}

class SessionWrapper {
public:
  SessionWrapper(const std::string& model_path, int log_level, int intra_threads, int inter_threads,
                 const std::string& opt_level, bool memory_pattern, bool cpu_mem_arena,
                 const std::string& execution_mode, Rice::Array providers) {
    Ort::SessionOptions opts;

    if (intra_threads > 0) opts.SetIntraOpNumThreads(intra_threads);
    if (inter_threads > 0) opts.SetInterOpNumThreads(inter_threads);
    opts.SetGraphOptimizationLevel(parse_opt_level(opt_level));

    if (!memory_pattern) opts.DisableMemPattern();
    if (!cpu_mem_arena) opts.DisableCpuMemArena();

    if (execution_mode == "parallel") {
      opts.SetExecutionMode(ExecutionMode::ORT_PARALLEL);
    } else {
      opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
    }

    // Append execution providers
    for (long i = 0; i < RARRAY_LEN(providers.value()); i++) {
      std::string provider = Rice::detail::From_Ruby<std::string>().convert(
          rb_ary_entry(providers.value(), i));

      if (provider == "coreml") {
#ifdef __APPLE__
        uint32_t coreml_flags = COREML_FLAG_USE_NONE;
        auto status = OrtSessionOptionsAppendExecutionProvider_CoreML(opts, coreml_flags);
        if (status) {
          std::string msg = Ort::GetApi().GetErrorMessage(status);
          Ort::GetApi().ReleaseStatus(status);
          throw std::runtime_error("CoreML provider error: " + msg);
        }
#else
        throw std::runtime_error("CoreML provider is only available on macOS/iOS");
#endif
      } else if (provider == "cuda") {
        OrtCUDAProviderOptions cuda_opts;
        memset(&cuda_opts, 0, sizeof(cuda_opts));
        opts.AppendExecutionProvider_CUDA(cuda_opts);
      } else if (provider == "tensorrt") {
        OrtTensorRTProviderOptions trt_opts;
        memset(&trt_opts, 0, sizeof(trt_opts));
        opts.AppendExecutionProvider_TensorRT(trt_opts);
      } else if (provider == "cpu") {
        // CPU is always available as fallback, no-op
      } else {
        throw std::runtime_error("Unknown execution provider: " + provider);
      }
    }

    session_ = std::make_unique<Ort::Session>(get_env(log_level), model_path.c_str(), opts);
    allocator_ = Ort::AllocatorWithDefaultOptions();
  }

  // Get input metadata
  Rice::Array input_info() {
    Rice::Array result;
    size_t count = session_->GetInputCount();

    for (size_t i = 0; i < count; i++) {
      auto name = session_->GetInputNameAllocated(i, allocator_);
      auto type_info = session_->GetInputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      auto shape = tensor_info.GetShape();
      auto elem_type = tensor_info.GetElementType();

      Rice::Hash info;
      info[Rice::Symbol("name")] = Rice::String(name.get());
      info[Rice::Symbol("type")] = Rice::Symbol(ort_type_to_string(elem_type));

      Rice::Array rb_shape;
      for (auto dim : shape) {
        rb_shape.push(Rice::Object(LONG2NUM(dim)));
      }
      info[Rice::Symbol("shape")] = rb_shape;

      result.push(Rice::Object(info.value()));
    }
    return result;
  }

  // Get output metadata
  Rice::Array output_info() {
    Rice::Array result;
    size_t count = session_->GetOutputCount();

    for (size_t i = 0; i < count; i++) {
      auto name = session_->GetOutputNameAllocated(i, allocator_);
      auto type_info = session_->GetOutputTypeInfo(i);
      auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
      auto shape = tensor_info.GetShape();
      auto elem_type = tensor_info.GetElementType();

      Rice::Hash info;
      info[Rice::Symbol("name")] = Rice::String(name.get());
      info[Rice::Symbol("type")] = Rice::Symbol(ort_type_to_string(elem_type));

      Rice::Array rb_shape;
      for (auto dim : shape) {
        rb_shape.push(Rice::Object(LONG2NUM(dim)));
      }
      info[Rice::Symbol("shape")] = rb_shape;

      result.push(Rice::Object(info.value()));
    }
    return result;
  }

  // Run inference
  Rice::Object run(Rice::Array input_specs, Rice::Array output_names_filter) {
    std::vector<const char*> input_names;
    std::vector<Ort::Value> input_tensors;
    std::vector<std::string> input_name_strs;

    // Storage for tensor data (must outlive the Run call)
    std::vector<std::vector<float>> float_buffers;
    std::vector<std::vector<double>> double_buffers;
    std::vector<std::vector<int32_t>> int32_buffers;
    std::vector<std::vector<int64_t>> int64_buffers;
    // Use uint8_t instead of bool because std::vector<bool> is bit-packed and has no .data()
    std::vector<std::vector<uint8_t>> bool_buffers;

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    for (long idx = 0; idx < RARRAY_LEN(input_specs.value()); idx++) {
      Rice::Hash spec(rb_ary_entry(input_specs.value(), idx));

      std::string name = Rice::detail::From_Ruby<std::string>().convert(
          spec[Rice::Symbol("name")].value());
      input_name_strs.push_back(name);

      Rice::Array data(spec[Rice::Symbol("data")].value());
      Rice::Array rb_shape(spec[Rice::Symbol("shape")].value());
      std::string dtype = Rice::detail::From_Ruby<std::string>().convert(
          spec[Rice::Symbol("dtype")].value());

      std::vector<int64_t> shape;
      for (long s = 0; s < RARRAY_LEN(rb_shape.value()); s++) {
        shape.push_back(NUM2LONG(rb_ary_entry(rb_shape.value(), s)));
      }

      size_t total_elements = 1;
      for (auto dim : shape) total_elements *= dim;

      if (dtype == "float") {
        float_buffers.emplace_back(total_elements);
        auto& buf = float_buffers.back();
        for (size_t i = 0; i < total_elements; i++) {
          buf[i] = static_cast<float>(NUM2DBL(rb_ary_entry(data.value(), i)));
        }
        input_tensors.push_back(
          Ort::Value::CreateTensor<float>(memory_info, buf.data(), total_elements,
                                          shape.data(), shape.size()));
      } else if (dtype == "double") {
        double_buffers.emplace_back(total_elements);
        auto& buf = double_buffers.back();
        for (size_t i = 0; i < total_elements; i++) {
          buf[i] = NUM2DBL(rb_ary_entry(data.value(), i));
        }
        input_tensors.push_back(
          Ort::Value::CreateTensor<double>(memory_info, buf.data(), total_elements,
                                           shape.data(), shape.size()));
      } else if (dtype == "int32") {
        int32_buffers.emplace_back(total_elements);
        auto& buf = int32_buffers.back();
        for (size_t i = 0; i < total_elements; i++) {
          buf[i] = static_cast<int32_t>(NUM2INT(rb_ary_entry(data.value(), i)));
        }
        input_tensors.push_back(
          Ort::Value::CreateTensor<int32_t>(memory_info, buf.data(), total_elements,
                                            shape.data(), shape.size()));
      } else if (dtype == "int64") {
        int64_buffers.emplace_back(total_elements);
        auto& buf = int64_buffers.back();
        for (size_t i = 0; i < total_elements; i++) {
          buf[i] = NUM2LONG(rb_ary_entry(data.value(), i));
        }
        input_tensors.push_back(
          Ort::Value::CreateTensor<int64_t>(memory_info, buf.data(), total_elements,
                                            shape.data(), shape.size()));
      } else if (dtype == "bool") {
        bool_buffers.emplace_back(total_elements);
        auto& buf = bool_buffers.back();
        for (size_t i = 0; i < total_elements; i++) {
          buf[i] = RTEST(rb_ary_entry(data.value(), i)) ? 1 : 0;
        }
        input_tensors.push_back(
          Ort::Value::CreateTensor(memory_info, reinterpret_cast<bool*>(buf.data()), total_elements,
                                   shape.data(), shape.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL));
      } else {
        throw std::runtime_error("Unsupported input dtype: " + dtype);
      }
    }

    for (auto& s : input_name_strs) {
      input_names.push_back(s.c_str());
    }

    // Determine output names
    std::vector<std::string> output_name_strs;
    std::vector<const char*> output_names;

    if (RARRAY_LEN(output_names_filter.value()) > 0) {
      for (long i = 0; i < RARRAY_LEN(output_names_filter.value()); i++) {
        output_name_strs.push_back(
          Rice::detail::From_Ruby<std::string>().convert(
              rb_ary_entry(output_names_filter.value(), i)));
      }
    } else {
      size_t output_count = session_->GetOutputCount();
      for (size_t i = 0; i < output_count; i++) {
        auto name = session_->GetOutputNameAllocated(i, allocator_);
        output_name_strs.push_back(name.get());
      }
    }

    for (auto& s : output_name_strs) {
      output_names.push_back(s.c_str());
    }

    // Run inference
    auto results = session_->Run(
      Ort::RunOptions{nullptr},
      input_names.data(), input_tensors.data(), input_names.size(),
      output_names.data(), output_names.size());

    // Convert results to Ruby Hash
    Rice::Hash output;
    for (size_t i = 0; i < results.size(); i++) {
      Rice::Object rb_tensor = tensor_to_ruby(results[i]);
      output[Rice::String(output_name_strs[i])] = rb_tensor;
    }

    return output;
  }

private:
  std::unique_ptr<Ort::Session> session_;
  Ort::AllocatorWithDefaultOptions allocator_;
};

extern "C" void Init_onnx_ruby_ext() {
  Module rb_mOnnxRuby = define_module("OnnxRuby");
  Module rb_mExt = define_module_under(rb_mOnnxRuby, "Ext");

  define_class_under<SessionWrapper>(rb_mExt, "SessionWrapper")
    .define_constructor(Constructor<SessionWrapper, const std::string&, int, int, int,
                        const std::string&, bool, bool, const std::string&, Rice::Array>(),
                        Arg("model_path"), Arg("log_level"), Arg("intra_threads"), Arg("inter_threads"),
                        Arg("opt_level"), Arg("memory_pattern"), Arg("cpu_mem_arena"),
                        Arg("execution_mode"), Arg("providers"))
    .define_method("input_info", &SessionWrapper::input_info)
    .define_method("output_info", &SessionWrapper::output_info)
    .define_method("run", &SessionWrapper::run);

  rb_mExt.define_module_function("optimize_model", &optimize_model,
                                  Arg("input_path"), Arg("output_path"), Arg("opt_level"));
  rb_mExt.define_module_function("available_providers", &available_providers);
}
